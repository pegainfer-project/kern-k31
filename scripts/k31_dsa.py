"""K3.1's DSA layer for gen_stage.py: the indexer, its top 2048 and the sparse attention over the fp8 latent
pages, as vLLM's Kimi K3.1 runs them on SM100 (FlashInferMLASparse over the plain fp8 cache).

Both sides project the indexer from the layer's input (`wk | weights_proj`, 160 bf16 columns) and from q_norm
(`wq_b`, 32 heads of 128), bf16 GEMMs as vLLM's unquantized ones. The key goes through its LayerNorm into
`idx.<l>`, a token-major state of 144-byte rows (128 e4m3, the f32 scale, padding) on the latent pages' slots;
the query to e4m3 with its head weights folded (source/k31_dsa.cu).

P scores every row against its sequence's keys up to its own position: the keys gathered sequence after
sequence into one stream (k31_idx_plan / k31_idx_gather), DeepGEMM's MQA logits over it, vLLM's prefill top-k
per row, the picks as cache slots. D scores its rows against their paged keys (DeepGEMM's paged MQA logits and
its scheduler), the same top-k, the same slots. Then both run the absorbed query (k31_absorb on P,
k3_mla_absorb_mma's OUT_HEADS=128 build on D) through TRT-LLM gen's sparse MLA and the latent output through
W_UV and the gate (k31_vup_gate): the 96 heads ride as 128, the pad heads' query never written and their
output never read.
"""
import math

import k31_sparse_mla_abi as sparse
from k3_common import B, CTX, HEADS, KV_LORA, Q_LORA, T, b, i32, i64, launch
from pinned import module

IH, ID, IDX_ROW, TOPK, PAGE = 32, 128, 144, 2048, 64
IDX_KW = ID + IH  # wk | weights_proj
IDX_Q = IH * ID
SMS = 152
MQA = ("_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb0ELb1ELb0ELb0ELj4ELj256ELj3ELj5ELj152ELj128ELj256E"
       "N7cutlass12float_e4m3_tEffLj2EEEvjjjPKjS4_S4_PT13_14CUtensorMap_stS7_S7_S7_S7_")
MQA_SMEM = 220160
PAGED = ("_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj32ELj128ELj64ELb0ELb1ELb0ELj3ELj4ELj256ELj16ELj128ELj256E"
         "N7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")
PAGED_SMEM = 198656
META = "_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb1ELb0ELj256ELj152EEEvjjPKjS3_Pj"
ABSORB_GRID_X = 3  # k31_absorb's row walkers a (head, column block): 8192 rows at 389 us, flat from 3 on
VUP_SMEM = 3 * (64 + 128) * 72 * 2
TMAP = "bytes<128>"
HEADC_BYTES = (Q_LORA + KV_LORA + 64) * 2  # the gate's columns in P's fused projection (q_a | kv_a | rope | gate)


def stride(ctx):
    """A logits row: DeepGEMM's blocks of 256 keys."""
    return -(-ctx // 256) * 256


def tmap(param, dtype, dims, strides, box, swizzle):
    return {"pack": {"size": 128, "fields": [{"at": 0, "tensormap": {
        "param": param, "dtype": dtype, "dims": dims, "strides": strides, "box": box, "swizzle": swizzle}}]}}


def ops(chunk, rows_max, seqs_max, max_ctx, max_pages):
    """The layer's ops; `rows_max` a call's rows (P) or the batch (D), `max_ctx` a row's keys at most."""
    dsa = lambda entry, grid, block, **kw: launch("k31_dsa", entry, grid=grid, block=block, **kw)
    o = {
        "idx_key": {
            "params": ["in buffer<bf16>", "i32", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>",
                       "inout state", "i32"],
            "impl": {"launches": [dsa("k31_idx_key", [{"ceil_div": [T, 4]}, 1, 1], [128, 1, 1])]},
        },
        "idx_q": {
            "params": ["in buffer<bf16>", "i32", "in buffer<bf16>", "i32", "out buffer<u8>", "out buffer<f32>", "i32"],
            "impl": {"launches": [dsa("k31_idx_q", [T, 1, 1], [256, 1, 1])]},
        },
        # vLLM's prefill top-k over [0, end) of a row: (logits, starts, ends, out, stride)
        "idx_topk": {
            "params": ["in buffer<f32>", "in buffer<i32>", "in buffer<i32>", "out buffer<i32>", "i32"],
            "impl": {"launches": [launch(
                "vllm_topk", "glm_topk_prefill", grid=[T, 1, 1], block=[512, 1, 1], smem=TOPK * 4,
                params=["in buffer<f32>", "in buffer<i32>", "in buffer<i32>", "out buffer<i32>", "i32", "i32", "i32",
                        "i32"],
                args=[*({"param": i} for i in range(5)), i32(1), i32(TOPK), i32(0)])]},
        },
        "absorb": {
            "params": ["in buffer<bf16>" if chunk else "in buffer<f32>", "in buffer<bf16>", "out buffer<fp8e4m3>",
                       "i32"],
            "impl": {"launches": [
                launch("k31_absorb", "kern_k31_absorb", grid=[ABSORB_GRID_X, HEADS, 4], block=[256, 1, 1])
                if chunk else
                launch("k3_mla_absorb_mma", "kern_k3_mla_absorb_mma", grid=[1, HEADS, 4], block=[256, 1, 1],
                       defines={"Q_FP8": 1, "OUT_HEADS": sparse.HEADS})]},
        },
        "sparse_mla": sparse.op(rows_max, T),
        # (o, w_kv_b, gate, gate_stride, gated, B)
        "vup_gate": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k31_vup_gate", "kern_k31_vup_gate", grid=[{"ceil_div": [T, 64]}, HEADS, 1],
                                         block=[256, 1, 1], smem=VUP_SMEM)]},
        },
    }
    if chunk:
        n = stride(max_ctx)
        o.update({
            # (cu_seqlens, seq_lens, seqs, ks, ke, klen, row_seq, B)
            "idx_plan": {
                "params": ["in buffer<i64>", "in buffer<i32>", "i32", "out buffer<u32>", "out buffer<u32>",
                           "out buffer<i32>", "out buffer<i32>", "i32"],
                "impl": {"launches": [dsa("k31_idx_plan", [{"ceil_div": [T, 256]}, 1, 1], [256, 1, 1])]},
            },
            # (cache, block_table, max_pages, seq_lens, seqs, kg, ksf)
            "idx_gather": {
                "params": ["in state", "in buffer<i32>", "i32", "in buffer<i32>", "i32", "out buffer<u8>",
                           "out buffer<f32>"],
                "impl": {"launches": [dsa("k31_idx_gather", [{"add": [{"ceil_div": [CTX, PAGE]}, "seqs"]}, 1, 1],
                                          [256, 1, 1])]},
            },
            # (iq, wout, kg, ksf, ks, ke, logits, rows, keys): row r's logits at [r][0, ke - ks), rows n apart
            "idx_logits": {
                "params": ["in buffer<u8>", "in buffer<f32>", "in buffer<u8>", "in buffer<f32>", "in buffer<u32>",
                           "in buffer<u32>", "out buffer<f32>", "i32", "i32"],
                "impl": {"launches": [{
                    **module("k31_mqa_logits"), "entry": MQA, "grid": [SMS, 1, 1], "block": [384, 1, 1],
                    "shared_mem": MQA_SMEM,
                    "params": ["i32", "i32", "i32", "in buffer<u32>", "in buffer<u32>", "i64", "out buffer<f32>",
                               *[TMAP] * 5],
                    "args": [{"param": 7}, {"param": 8}, i32(n), {"param": 4}, {"param": 5}, i64(0), {"param": 6},
                             tmap(0, "u8", [ID, 0], [ID], [ID, 128], 128),
                             tmap(3, "f32", [n, 1], [n * 4], [256, 1], 0),
                             tmap(2, "u8", [ID, 0], [ID], [ID, 256], 128),
                             tmap(3, "f32", [n, 1], [n * 4], [256, 1], 0),
                             tmap(1, "f32", [IH, 0], [IH * 4], [IH, 4], 0)],
                }]},
            },
            # (ind, block_table, max_pages, row_seq, klen, tlen, B)
            "idx_slots": {
                "params": ["inout buffer<i32>", "in buffer<i32>", "i32", "in buffer<i32>", "in buffer<i32>",
                           "out buffer<i32>", "i32"],
                "impl": {"launches": [dsa("k31_topk_slots", [T, 1, 1], [256, 1, 1])]},
            },
        })
    else:
        n = stride(max_ctx)
        page_bytes = PAGE * IDX_ROW
        o.update({
            # (seq_lens, schedule, rows): DeepGEMM's split of the batch's pages over the SMs
            "idx_sched": {
                "params": ["in buffer<i32>", "out buffer<i32>", "i32"],
                "impl": {"launches": [{
                    **module("k31_paged_mqa_logits"), "entry": META, "grid": [1, 1, 1], "block": [1024, 1, 1],
                    "shared_mem": (seqs_max + 33) * 4,
                    "params": ["i32", "i32", "in buffer<i32>", "i64", "out buffer<i32>"],
                    "args": [{"param": 2}, {"param": 2}, {"param": 0}, i64(0), {"param": 1}],
                }]},
            },
            # (iq, wout, cache, cache + 128, seq_lens, block_table, schedule, logits, rows)
            "idx_logits": {
                "params": ["in buffer<u8>", "in buffer<f32>", "in state", "in state", "in buffer<i32>",
                           "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "i32"],
                "impl": {"launches": [{
                    **module("k31_paged_mqa_logits"), "entry": PAGED, "grid": [SMS, 1, 1], "block": [384, 1, 1],
                    "shared_mem": PAGED_SMEM,
                    "params": ["i32", "i32", "i32", "in buffer<i32>", "out buffer<f32>", "in buffer<i32>", "i64",
                               "in buffer<i32>", *[TMAP] * 5],
                    "args": [{"param": 8}, i32(n), i32(max_pages), {"param": 4}, {"param": 7}, {"param": 5}, i64(0),
                             {"param": 6},
                             tmap(0, "u8", [ID, 0], [ID], [ID, 128], 128),
                             tmap(3, "f32", [4, PAGE, 0], [IDX_ROW, page_bytes], [4, PAGE, 1], 0),
                             tmap(2, "u8", [ID, PAGE, 0], [IDX_ROW, page_bytes], [ID, PAGE, 1], 128),
                             tmap(3, "f32", [4, PAGE, 0], [IDX_ROW, page_bytes], [4, PAGE, 1], 0),
                             tmap(1, "f32", [IH, 0], [IH * 4], [IH, 4], 0)],
                }]},
            },
            # (ind, block_table, max_pages, seq_lens, tlen, B): row r is sequence r
            "idx_slots": {
                "params": ["inout buffer<i32>", "in buffer<i32>", "i32", "in buffer<i32>", "out buffer<i32>", "i32"],
                "impl": {"launches": [dsa(
                    "k31_topk_slots", [T, 1, 1], [256, 1, 1],
                    params=["inout buffer<i32>", "in buffer<i32>", "i32", "i64", "in buffer<i32>", "out buffer<i32>",
                            "i32"],
                    args=[{"param": 0}, {"param": 1}, {"param": 2}, i64(0), {"param": 3}, {"param": 4},
                          {"param": 5}])]},
            },
        })
    return o


def weights(weight, seg, a, n):
    """The layer's indexer weights, checkpoint prefix `a`, manifest prefix `n`."""
    weight(n + "idx_kw", [IDX_KW, 7168], [seg(a + "indexer.wk.weight"), seg(a + "indexer.weights_proj.weight")])
    weight(n + "idx_qw", [IDX_Q, Q_LORA], [seg(a + "indexer.wq_b.weight")])
    weight(n + "idx_kn_w", [ID], [seg(a + "indexer.k_norm.weight")])
    weight(n + "idx_kn_b", [ID], [seg(a + "indexer.k_norm.bias")])


def buffers(work, bufs, chunk, rows_max, max_ctx, seqs_max):
    work("idx_kwo", IDX_KW)
    work("idx_qb", IDX_Q)
    work("idx_iq", IDX_Q, "u8")
    work("idx_w", IH, "f32")
    work("idx_ind", TOPK, "i32")
    work("idx_tlen", 1, "i32")
    work("mla_q_abs", sparse.HEADS * sparse.D_QK, "fp8e4m3")
    work("mla_o", sparse.HEADS * sparse.D_V)
    work("mla_md", sparse.HEADS * 2, "f32")
    work("idx_logits", stride(max_ctx), "f32")
    # vLLM's top-k scores [start, end) of a row: start 0, filled once
    bufs["idx_zero"] = {"dtype": "i32", "shape": [rows_max], "kind": "carry"}
    if chunk:
        for x in ("idx_ks", "idx_ke"):
            work(x, 1, "u32")
        work("idx_klen", 1, "i32")
        work("idx_row_seq", 1, "i32")
        bufs["idx_kg"] = {"dtype": "u8", "shape": [stride(max_ctx), ID], "kind": "workspace"}
        bufs["idx_ksf"] = {"dtype": "f32", "shape": [stride(max_ctx)], "kind": "workspace"}
    else:
        bufs["idx_sched"] = {"dtype": "i32", "shape": [(SMS + 1) * 2], "kind": "workspace"}


def steps(step, w, L, kv, idx, chunk, gate, gate_stride, max_pages, max_ctx, q):
    """The layer from its input projections to `gated`: `gate` / `gate_stride` the gate's columns, `q` the
    q_b output (P's bf16, D's f32), the latent row and q_norm already in place."""
    step(L + "idx_kw", "gemm_bf16", b("normed"), w("idx_kw"), b("idx_kwo"), B, i32(IDX_KW), i32(7168), i32(IDX_KW))
    step(L + "idx_key", "idx_key", b("idx_kwo"), i32(IDX_KW), w("idx_kn_w"), w("idx_kn_b"), b("slot_mapping"),
         {"state": idx}, B)
    step(L + "idx_qb", "gemm_bf16", b("q_norm"), w("idx_qw"), b("idx_qb"), B, i32(IDX_Q), i32(Q_LORA), i32(IDX_Q))
    step(L + "idx_q", "idx_q", b("idx_qb"), i32(IDX_Q), b("idx_kwo", ID * 2), i32(IDX_KW), b("idx_iq"), b("idx_w"), B)
    if chunk:
        step(L + "idx_plan", "idx_plan", b("cu_seqlens"), b("seq_lens"), {"var": "seqs"}, b("idx_ks"), b("idx_ke"),
             b("idx_klen"), b("idx_row_seq"), B)
        step(L + "idx_gather", "idx_gather", {"state": idx}, b("block_table"), i32(max_pages), b("seq_lens"),
             {"var": "seqs"}, b("idx_kg"), b("idx_ksf"))
        step(L + "idx_logits", "idx_logits", b("idx_iq"), b("idx_w"), b("idx_kg"), b("idx_ksf"), b("idx_ks"),
             b("idx_ke"), b("idx_logits"), B, {"var": CTX})
        ends = b("idx_klen")
    else:
        step(L + "idx_sched", "idx_sched", b("seq_lens"), b("idx_sched"), B)
        step(L + "idx_logits", "idx_logits", b("idx_iq"), b("idx_w"), {"state": idx}, {"state": idx, "offset": ID},
             b("seq_lens"), b("block_table"), b("idx_sched"), b("idx_logits"), B)
        ends = b("seq_lens")
    step(L + "idx_topk", "idx_topk", b("idx_logits"), b("idx_zero"), ends, b("idx_ind"), i32(stride(max_ctx)))
    if chunk:
        step(L + "idx_slots", "idx_slots", b("idx_ind"), b("block_table"), i32(max_pages), b("idx_row_seq"),
             b("idx_klen"), b("idx_tlen"), B)
    else:
        step(L + "idx_slots", "idx_slots", b("idx_ind"), b("block_table"), i32(max_pages), b("seq_lens"),
             b("idx_tlen"), B)
    step(L + "absorb", "absorb", q, w("w_kv_b"), b("mla_q_abs"), B)
    step(L + "attn", "sparse_mla", b("mla_q_abs"), {"state": kv}, b("idx_ind"), b("idx_tlen"), b("mla_o"),
         b("mla_md"))
    step(L + "vup_gate", "vup_gate", b("mla_o"), w("w_kv_b"), gate, i32(gate_stride), b("gated"), B)
