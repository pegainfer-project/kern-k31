"""K3.1's DSA layer for gen_stage.py: the indexer, its top 2048 and the sparse attention over the fp8 latent
pages, as vLLM's Kimi K3.1 runs them on SM100 (FlashInferMLASparse over the plain fp8 cache).

Both sides project the indexer from the layer's input (`wk | weights_proj`, 160 bf16 columns) and from q_norm
(`wq_b`, 32 heads of 128), bf16 GEMMs as vLLM's unquantized ones. The key goes through its LayerNorm into
`idx.<l>`, a token-major state of 144-byte rows (128 e4m3, a UE8M0 scale word, padding) on the latent pages'
slots; the query to e4m3 with UE8M0 scales and bf16 head weights (source/k31_dsa.cu).

P scores every row against its sequence's keys up to its own position: the keys gathered sequence after
sequence into one stream (k31_idx_plan / k31_idx_gather), DeepGEMM's bf16 MQA logits over it, a top-k per bf16
row, the picks as cache slots. D scores its rows against their paged keys (DeepGEMM's paged MQA logits and
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
MQA = ("_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb0ELj4ELj384ELj128ELj3ELj3ELj3ELj152ELj128ELj384E"
       "N7cutlass12float_e4m3_tELj3EEEvjjjPKjS4_S4_P13__nv_bfloat1614CUtensorMap_stS7_S7_S7_S7_")
MQA_SMEM = 203776
PAGED = ("_ZN9deep_gemm22sm100_paged_mqa_logitsILj32ELj128ELj64ELj4ELj128ELj3ELj3ELj3ELj384ELj8ELj128ELj384E"
         "N7cutlass12float_e4m3_tELj3EEEvjjjPKjP13__nv_bfloat16S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")
PAGED_SMEM = 218112
META = "_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj384ELj152ELj4ELj256EEEvjPKjS3_Pj"
META_THREADS = 256
SPLIT_KV = 384  # DeepGEMM's KV split: the paged kernel's logits rows are whole splits
# k31_absorb_tc's grid is odd so a block's items alternate halves (the rope rides on the first)
ABSORB_TC_SMEM = 1024 + 3 * (16384 + 4 * 8192) + 32768 + 16384 + 8192 + 256  # its SMEM_BYTES
VUP_SMEM = 3 * (64 + 128) * 72 * 2
VUP_TC_SMEM = 1024 + 6 * 2 * 128 * 64 * 2 + 256  # k31_vup_gate_tc SMEM_BYTES
TOPK_SMEM = 93408  # k31_topk_smem
TMAP = "bytes<128>"
HEADC_BYTES = (Q_LORA + KV_LORA + 64) * 2  # the gate's columns in P's fused projection (q_a | kv_a | rope | gate)


def stride(ctx):
    """A logits row: DeepGEMM's KV splits."""
    return -(-ctx // SPLIT_KV) * SPLIT_KV


def tmap(param, dtype, dims, strides, box, swizzle):
    return {"pack": {"size": 128, "fields": [{"at": 0, "tensormap": {
        "param": param, "dtype": dtype, "dims": dims, "strides": strides, "box": box, "swizzle": swizzle}}]}}


def ops(chunk, rows_max, seqs_max, max_ctx, max_pages, gate_stride, tp=1):
    """The layer's ops; `rows_max` a call's rows (P) or the batch (D), `max_ctx` a row's keys at most (a
    member's, under DCP), `gate_stride` the gate's row stride in elements, `tp` D's DCP group."""
    dsa = lambda entry, grid, block, **kw: launch("k31_dsa", entry, grid=grid, block=block, **kw)
    # D's chain from the prep to the exchange launches as programmatic dependents (each kernel waits)
    pdl = {} if chunk else {"pdl": True}
    o = {
        "idx_key": {
            "params": ["in buffer<bf16>", "i32", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>",
                       "inout state", "i32"],
            "impl": {"launches": [dsa("k31_idx_key", [{"ceil_div": [T, 4]}, 1, 1], [128, 1, 1], **pdl)]},
        },
        # (q, ldq, weights_proj, ldw, iq, sfq, w, rid, B); P has no request ids
        "idx_q": {
            "params": ["in buffer<bf16>", "i32", "in buffer<bf16>", "i32", "out buffer<u8>", "out buffer<u32>",
                       "out buffer<bf16>", *([] if chunk else ["out buffer<u32>"]), "i32"],
            "impl": {"launches": [dsa(
                "k31_idx_q", [T, 1, 1], [256, 1, 1],
                params=["in buffer<bf16>", "i32", "in buffer<bf16>", "i32", "out buffer<u8>", "out buffer<u32>",
                        "out buffer<bf16>", "i64" if chunk else "out buffer<u32>", "i32"],
                args=[*({"param": i} for i in range(7)), i64(0) if chunk else {"param": 7},
                      {"param": 7 if chunk else 8}], **pdl)]},
        },
        # each row's top TOPK over [0, len) as ascending positions (k31_topk, vLLM's set, deterministic);
        # D's few rows take one block an SM, a row of up to 32k keys read once into registers
        "idx_topk": {
            "params": ["in buffer<bf16>", "in buffer<i32>", "out buffer<i32>", "i32"],
            "impl": {"launches": [launch(
                "k31_topk", "k31_topk" if chunk else "k31_topk_held", grid=[T, 1, 1], block=[512, 1, 1],
                smem=TOPK_SMEM,
                params=["in buffer<bf16>", "i32", "in buffer<i32>", "out buffer<i32>"],
                args=[{"param": 0}, {"param": 3}, {"param": 1}, {"param": 2}], **pdl)]},
        },
        "absorb": {
            "params": ["in buffer<bf16>" if chunk else "in buffer<f32>", "in buffer<bf16>", "out buffer<fp8e4m3>",
                       "i32"],
            "impl": {"launches": [
                launch("k31_absorb_tc", "kern_k31_absorb_tc", grid=[SMS - 1, 1, 1], block=[384, 1, 1],
                       smem=ABSORB_TC_SMEM, params=[*[TMAP] * 4, "i32"],
                       args=[tmap(0, "bf16", [192, HEADS, 0], [192 * 2, HEADS * 192 * 2], [64, 1, 128], 128),
                             tmap(1, "bf16", [KV_LORA, 0], [KV_LORA * 2], [64, 64], 128),
                             tmap(2, "u8", [sparse.D_QK, sparse.HEADS, 0], [sparse.D_QK, sparse.HEADS * sparse.D_QK],
                                  [128, 1, 128], 128),
                             tmap(2, "u8", [sparse.D_QK, sparse.HEADS, 0], [sparse.D_QK, sparse.HEADS * sparse.D_QK],
                                  [64, 1, 128], 0),
                             {"param": 3}])
                if chunk else
                launch("k3_mla_absorb_mma", "kern_k3_mla_absorb_mma", grid=[1, HEADS, 4], block=[256, 1, 1],
                       defines={"Q_FP8": 1, "OUT_HEADS": sparse.HEADS})]},
        },
        "sparse_mla": sparse.op(rows_max, T, pdl=not chunk),
        # (o_lat, w_kv_b, gate, gated, B): P's chunk on tcgen05, D's batch on mma.sync
        "vup_gate": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch(
                "k31_vup_gate_tc", "kern_k31_vup_gate_tc", grid=[SMS, 1, 1], block=[384, 1, 1], smem=VUP_TC_SMEM,
                params=[*[TMAP] * 4, "i32"],
                args=[tmap(0, "bf16", [sparse.D_V, sparse.HEADS, 0], [sparse.D_V * 2, sparse.HEADS * sparse.D_V * 2],
                           [64, 1, 128], 128),
                      tmap(1, "bf16", [KV_LORA, 0], [KV_LORA * 2], [64, 128], 128),
                      tmap(2, "bf16", [HEADS * 128, 0], [gate_stride * 2], [64, 128], 128),
                      tmap(3, "bf16", [HEADS * 128, 0], [HEADS * 128 * 2], [64, 128], 128),
                      {"param": 4}])] if chunk else [launch(
                "k31_vup_gate", "kern_k31_vup_gate", grid=[{"ceil_div": [T, 64]}, HEADS, 1], block=[256, 1, 1],
                smem=VUP_SMEM, params=["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32",
                                       "out buffer<bf16>", "i32"],
                args=[*({"param": i} for i in range(3)), i32(gate_stride), {"param": 3}, {"param": 4}])]},
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
                           "out buffer<u32>"],
                "impl": {"launches": [dsa("k31_idx_gather", [{"add": [{"ceil_div": [CTX, PAGE]}, "seqs"]}, 1, 1],
                                          [256, 1, 1])]},
            },
            # (iq, sfq, w, kg, ksf, ks, ke, logits, rows, keys): row r's logits at [r][0, ke - ks), rows n apart
            "idx_logits": {
                "params": ["in buffer<u8>", "in buffer<u32>", "in buffer<bf16>", "in buffer<u8>", "in buffer<u32>",
                           "in buffer<u32>", "in buffer<u32>", "out buffer<bf16>", "i32", "i32"],
                "impl": {"launches": [{
                    **module("k31_mqa_logits"), "entry": MQA, "grid": [SMS, 1, 1], "block": [512, 1, 1],
                    "shared_mem": MQA_SMEM,
                    "params": ["i32", "i32", "i32", "in buffer<u32>", "in buffer<u32>", "i64", "out buffer<bf16>",
                               *[TMAP] * 5],
                    "args": [{"param": 8}, {"param": 9}, i32(n), {"param": 5}, {"param": 6}, i64(0), {"param": 7},
                             tmap(0, "u8", [ID, 0], [ID], [ID, 128], 128),
                             tmap(1, "u32", [IH, 0], [IH * 4], [IH, 4], 0),
                             tmap(3, "u8", [ID, 0], [ID], [ID, 128], 128),
                             tmap(4, "u32", [n, 1], [n * 4], [128, 1], 0),
                             tmap(2, "bf16", [IH, 0], [IH * 2], [IH, 4], 0)],
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
            # (seq_lens, rid, schedule, rows): DeepGEMM's split of the batch's KV splits over the SMs
            "idx_sched": {
                "params": ["in buffer<i32>", "in buffer<u32>", "out buffer<i32>", "i32"],
                "impl": {"launches": [{
                    **module("k31_paged_mqa_logits"), "entry": META, "grid": [1, 1, 1], "block": [META_THREADS, 1, 1],
                    "shared_mem": (2 * seqs_max + META_THREADS // 32 + 1) * 4,
                    "params": ["i32", "in buffer<i32>", "in buffer<u32>", "out buffer<i32>"],
                    "args": [{"param": 3}, {"param": 0}, {"param": 1}, {"param": 2}], "pdl": True,
                }]},
            },
            # (iq, sfq, w, cache, cache + 128, seq_lens, block_table, rid, schedule, logits, rows)
            "idx_logits": {
                "params": ["in buffer<u8>", "in buffer<u32>", "in buffer<bf16>", "in state", "in state",
                           "in buffer<i32>", "in buffer<i32>", "in buffer<u32>", "in buffer<i32>", "out buffer<bf16>",
                           "i32"],
                "impl": {"launches": [{
                    **module("k31_paged_mqa_logits"), "entry": PAGED, "grid": [SMS, 1, 1], "block": [512, 1, 1],
                    "shared_mem": PAGED_SMEM,
                    "params": ["i32", "i32", "i32", "in buffer<i32>", "out buffer<bf16>", "in buffer<i32>",
                               "in buffer<u32>", "in buffer<i32>", *[TMAP] * 5],
                    "args": [{"param": 10}, i32(n), i32(max_pages), {"param": 5}, {"param": 9}, {"param": 6},
                             {"param": 7}, {"param": 8},
                             tmap(0, "u8", [ID, 0], [ID], [ID, 128], 128),
                             tmap(1, "u32", [IH, 0], [IH * 4], [IH, 4], 0),
                             tmap(3, "u8", [ID, PAGE, 0], [IDX_ROW, page_bytes], [ID, PAGE, 1], 128),
                             tmap(4, "u32", [4, PAGE, 0], [IDX_ROW, page_bytes], [4, PAGE, 1], 0),
                             tmap(2, "bf16", [IH, 0], [IH * 2], [IH, 4], 0)], "pdl": True,
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
    if not chunk and tp > 1:
        if max_ctx * tp >= 1 << 24:
            raise SystemExit(f"k31_dcp_select keys a global position in 24 bits; a context of {max_ctx * tp} is past it")
        o.update({
            # (logits, stride, ind, rows, peers, state, stage bytes): the local top-k as candidates, pushed into
            # every member's stage
            "idx_pack": {
                "params": ["in buffer<bf16>", "i32", "in buffer<i32>", "i32", "in buffer<u64>", "in buffer<i32>",
                           "i64"],
                "impl": {"launches": [dsa(
                    "k31_dcp_pack", [T, 1, 1], [256, 1, 1],
                    params=["in buffer<bf16>", "i32", "in buffer<i32>", "i32", "i32", "i32", "in buffer<u64>",
                            "in buffer<i32>", "i64"],
                    args=[{"param": 0}, {"param": 1}, {"param": 2}, {"rank": "tp"}, i32(tp),
                          *({"param": i} for i in range(3, 7))], **pdl)]},
            },
            # (stages, state, err, stage bytes, timeout, rows, block_table, max_pages, ind, lens, live): the
            # group's top TOPK once every member's candidates are in, this member's share of it as cache slots
            "idx_select": {
                "params": ["inout buffer<u8>", "inout buffer<i32>", "out buffer<i32>", "i64", "i64", "i32",
                           "in buffer<i32>", "i32", "out buffer<i32>", "out buffer<i32>", "out buffer<i32>"],
                "impl": {"launches": [launch(
                    "k31_dsa", "k31_dcp_select", grid=[T, 1, 1], block=[1024, 1, 1],
                    params=["inout buffer<u8>", "inout buffer<i32>", "out buffer<i32>", "i64", "i64", "i32", "i32",
                            "in buffer<i32>", "i32", "out buffer<i32>", "out buffer<i32>", "out buffer<i32>"],
                    args=[*({"param": i} for i in range(6)), {"rank": "tp"}, *({"param": i} for i in range(6, 11))],
                    **pdl)]},
            },
            # every stage of the candidates poisoned, once before the first step
            "idx_init": {
                "params": ["inout buffer<u8>", "i64"],
                "impl": {"launches": [dsa("k31_dcp_poison", [SMS, 1, 1], [256, 1, 1],
                                          params=["inout buffer<u8>", "i64"], args=[{"param": 0}, {"param": 1}])]},
            },
        })
    return o


def weights(weight, seg, a, n):
    """The layer's indexer weights, checkpoint prefix `a`, manifest prefix `n`."""
    weight(n + "idx_kw", [IDX_KW, 7168], [seg(a + "indexer.wk.weight"), seg(a + "indexer.weights_proj.weight")])
    weight(n + "idx_qw", [IDX_Q, Q_LORA], [seg(a + "indexer.wq_b.weight")])
    weight(n + "idx_kn_w", [ID], [seg(a + "indexer.k_norm.weight")])
    weight(n + "idx_kn_b", [ID], [seg(a + "indexer.k_norm.bias")])


def cand_stage(seqs_max, tp):
    """Bytes of one stage of D's candidates under DCP: every member's rows, 2048 of 8 bytes each."""
    return tp * seqs_max * TOPK * 8


def buffers(work, bufs, chunk, max_ctx, seqs_max, tp=1):
    work("idx_kwo", IDX_KW)
    work("idx_qb", IDX_Q)
    work("idx_iq", IDX_Q, "u8")
    work("idx_sfq", IH, "u32")
    work("idx_w", IH)
    work("idx_ind", TOPK, "i32")
    work("idx_tlen", 1, "i32")
    work("mla_q_abs", sparse.HEADS * sparse.D_QK, "fp8e4m3")
    work("mla_o", sparse.HEADS * sparse.D_V)
    work("mla_md", sparse.HEADS * 2, "f32")
    work("idx_logits", stride(max_ctx))
    if chunk:
        for x in ("idx_ks", "idx_ke"):
            work(x, 1, "u32")
        work("idx_klen", 1, "i32")
        work("idx_row_seq", 1, "i32")
        bufs["idx_kg"] = {"dtype": "u8", "shape": [stride(max_ctx), ID], "kind": "workspace"}
        bufs["idx_ksf"] = {"dtype": "u32", "shape": [stride(max_ctx)], "kind": "workspace"}
    else:
        work("idx_rid", 1, "u32")
        bufs["idx_sched"] = {"dtype": "i32", "shape": [(SMS + 1) * 2], "kind": "workspace"}
        if tp > 1:
            # three stages of every member's candidates (k31_dcp_pack), the stage they rotate through
            bufs["idx_lamport"] = {"dtype": "u8", "shape": [3 * cand_stage(seqs_max, tp)], "kind": "carry",
                                   "export": True}
            bufs["idx_lamport_peers"] = {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "idx_lamport",
                                         "group": "tp"}
            bufs["idx_state"] = {"dtype": "i32", "shape": [8], "kind": "carry"}
            work("idx_live", 1, "i32")


def steps(step, w, L, kv, idx, chunk, gate, max_pages, max_ctx, q, exchange=None, select=None):
    """The layer from its input projections to `gated`: `gate` the gate's columns (at ops' stride), `q` the
    q_b output (P's bf16, D's f32), the latent row and q_norm already in place. Under DCP, `exchange`
    (the dcp_exchange call's trailing args) merges the members' partials into this member's heads, and
    `select` (err, timeout, stage bytes) is the candidates' exchange."""
    step(L + "idx_kw", "gemm_bf16", b("normed"), w("idx_kw"), b("idx_kwo"), B, i32(IDX_KW), i32(7168), i32(IDX_KW))
    step(L + "idx_key", "idx_key", b("idx_kwo"), i32(IDX_KW), w("idx_kn_w"), w("idx_kn_b"), b("slot_mapping"),
         {"state": idx}, B)
    step(L + "idx_qb", "gemm_bf16", b("q_norm"), w("idx_qw"), b("idx_qb"), B, i32(IDX_Q), i32(Q_LORA), i32(IDX_Q))
    step(L + "idx_q", "idx_q", b("idx_qb"), i32(IDX_Q), b("idx_kwo", ID * 2), i32(IDX_KW), b("idx_iq"), b("idx_sfq"),
         b("idx_w"), *([] if chunk else [b("idx_rid")]), B)
    if chunk:
        step(L + "idx_plan", "idx_plan", b("cu_seqlens"), b("seq_lens"), {"var": "seqs"}, b("idx_ks"), b("idx_ke"),
             b("idx_klen"), b("idx_row_seq"), B)
        step(L + "idx_gather", "idx_gather", {"state": idx}, b("block_table"), i32(max_pages), b("seq_lens"),
             {"var": "seqs"}, b("idx_kg"), b("idx_ksf"))
        step(L + "idx_logits", "idx_logits", b("idx_iq"), b("idx_sfq"), b("idx_w"), b("idx_kg"), b("idx_ksf"),
             b("idx_ks"), b("idx_ke"), b("idx_logits"), B, {"var": CTX})
        ends = b("idx_klen")
    else:
        step(L + "idx_sched", "idx_sched", b("seq_lens"), b("idx_rid"), b("idx_sched"), B)
        step(L + "idx_logits", "idx_logits", b("idx_iq"), b("idx_sfq"), b("idx_w"), {"state": idx},
             {"state": idx, "offset": ID}, b("seq_lens"), b("block_table"), b("idx_rid"), b("idx_sched"),
             b("idx_logits"), B)
        ends = b("seq_lens")
    step(L + "idx_topk", "idx_topk", b("idx_logits"), ends, b("idx_ind"), i32(stride(max_ctx)))
    if chunk:
        step(L + "idx_slots", "idx_slots", b("idx_ind"), b("block_table"), i32(max_pages), b("idx_row_seq"),
             b("idx_klen"), b("idx_tlen"), B)
    elif exchange:
        err, timeout, stage = select
        step(L + "idx_pack", "idx_pack", b("idx_logits"), i32(stride(max_ctx)), b("idx_ind"), B,
             b("idx_lamport_peers"), b("idx_state"), stage)
        step(L + "idx_select", "idx_select", b("idx_lamport"), b("idx_state"), err, stage, timeout, B,
             b("block_table"), i32(max_pages), b("idx_ind"), b("idx_tlen"), b("idx_live"))
    else:
        step(L + "idx_slots", "idx_slots", b("idx_ind"), b("block_table"), i32(max_pages), b("seq_lens"),
             b("idx_tlen"), B)
    step(L + "absorb", "absorb", q, w("w_kv_b"), b("mla_q_abs"), B)
    step(L + "attn", "sparse_mla", b("mla_q_abs"), {"state": kv}, b("idx_ind"), b("idx_tlen"), b("mla_o"),
         b("mla_md"))
    if exchange:
        step(L + "dcp", "dcp_exchange", b("mla_o"), b("mla_md"), b("idx_live"), w("w_kv_b_l"), gate, b("gated"),
             *exchange)
    else:
        step(L + "vup_gate", "vup_gate", b("mla_o"), w("w_kv_b"), gate, b("gated"), B)
