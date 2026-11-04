"""gen_stage.py's bf16 MLA cache (`--kv bf16`): 64-token pages of bf16 [576] rows.

D attends them with FlashInfer's CuTe-DSL MLA decode kernel (or k3_mla_decode.cu from
`--mla-ours-from` rows), the query absorbed on tcgen05 in bf16; over a group the splits go to the
exchange (source/k3_dcp.cu) unreduced. P keeps an absorbed short path: a call of at most
`--mla-short` rows attends in the latent space with the same DSL kernel instead of expanding its
context.
"""
import math

from pinned import module
import trtllm_fmha_abi
from k3_common import (B, HEADS, KV_LORA, LATENT_ROW, MLA_SHORT_SPLITS, NOPE_DIM, PAGE, ROPE, T, TP_TIMEOUT_NS,
                       b, bf16, dim, i32, i64, launch, pack)

BYTES = 2
P_GLUE = None
P_FP8_ATTN = False
P_MXFP8 = False
D_PREP = {}
Q_ABS = "bf16"

# MLA decode attention: FlashInfer's CuTe-DSL Blackwell kernel, prebuilt
# (tools/kernels-bin/README.md), two entries in one module. It walks a row
# in 128-token tiles; each row runs as `mla_bsk[b]` splits of a 2-CTA
# cluster and a reduction merges them. The parameter ABI packed below is
# the DSL's flattened struct layout (docs/k3-kernel-abi.md K5).
MLA_MODULE = "mla_decode_h96_p64"
MLA_MAIN = ("kernel_cutlass_split_kv_kernel_flashinfercute_dslattentionmonolithicmla_decode_fp16"
            "BlackwellMultiHeadLatentAttentionForwardFP16_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____0")
MLA_REDUCE = ("kernel_cutlass_reduction_kernel_flashinfercute_dslattentionmonolithicmla_decode_fp16"
              "BlackwellMultiHeadLatentAttentionForwardFP16_object_at__tensorptrbf16gmemalign16odiv16i64div161i64div16_1")
MLA_M_TILE = 128          # the MMA's row tile: 96 heads pad to one
MLA_MAIN_SMEM = 232448
MLA_REDUCE_SMEM = 1024    # 256-split reducer scratch
ABSORB_SMEM = 1024 + 2 * 4 * 64 * 128 * 2 + 2 * 64 * 128 + 128  # k3_mla_absorb_tc.cu SMEM_BYTES


check = lambda decode, xchg, seqs_max, dsa: None


def fast_divmod(d):
    """The DSL's 32-bit FastDivmod divisor image: {divisor, multiplier, shift1, shift2},
    q = ((x - mulhi(x, m)) >> s1 + mulhi(x, m)) >> s2."""
    if d == 1:
        return [{"at": 0, "i32": 1}, {"at": 4, "i32": 1}]
    l = (d - 1).bit_length()
    m = ((1 << (32 + l)) + d - 1) // d - (1 << 32)
    return [{"at": 0, "i32": d}, {"at": 4, "i32": m - (1 << 32) if m >= 1 << 31 else m}, {"at": 8, "u8": 1},
            {"at": 9, "u8": l - 1}]


def mla_attn_op(batch_max, split_max, reduce=True, scale_log2=bf16((NOPE_DIM + ROPE) ** -0.5) * math.log2(math.e),
                ours_from=None):
    """The DSL attention as one op: split kernel + reduction, structs packed from the interface.
    Interface: q_abs latent | q_abs rope (+1024 B) | kv latent | kv rope (+1024 B) | block_table |
    seq_lens | mla_bsk | o_lat | lse | acc_o | acc_lse | B | max_pages, at most `batch_max` rows.
    Without `reduce` the splits stay in acc_o / acc_lse for their merge (D's exchange, P's gate), and with
    `ours_from` batches of that many rows take k3_mla_decode.cu instead (mla_decode_launch).
    `scale_log2`: the softmax scale times log2(e)."""
    batch = T
    V = {"at": 0, **dim(batch)}
    tmap = lambda param, d0, page, box1, stride1: pack(128, {"at": 0, "tensormap": {
        "param": param, "dtype": "bf16", "dims": [d0, page, 0 if page == PAGE else batch_max],
        "strides": [LATENT_ROW * 2, stride1], "box": [64, box1, 1], "swizzle": 128, "l2_promotion": 128}})
    at = lambda off, f: {**f, "at": off}
    q_stride, kv_stride = HEADS * LATENT_ROW * 2, PAGE * LATENT_ROW * 2
    acc_o = pack(48, {"at": 0, "param": 9}, {"at": 8, "i32": MLA_M_TILE}, {"at": 12, "i32": split_max}, {"at": 16, "i32": KV_LORA},
                 {"at": 20, "i32": 1}, at(24, V), {"at": 28, "i32": split_max * KV_LORA}, {"at": 32, "i32": KV_LORA},
                 {"at": 36, "i32": split_max * MLA_M_TILE * KV_LORA}, {"at": 40, "i32": MLA_M_TILE * split_max * KV_LORA})
    acc_lse = pack(40, {"at": 0, "param": 10}, {"at": 8, "i32": MLA_M_TILE}, {"at": 12, "i32": split_max}, {"at": 16, "i32": 1},
                   at(20, V), {"at": 24, "i32": split_max}, {"at": 28, "i32": MLA_M_TILE * split_max},
                   {"at": 32, "i32": MLA_M_TILE * split_max})
    seqs = pack(16, {"at": 0, "param": 5}, {"at": 8, **dim(batch), "width": 8})
    bsk = pack(16, {"at": 0, "param": 6}, {"at": 8, **dim(batch), "width": 8})
    # the tiled-MMA descriptors and the TMA coordinate shapes are not read by this build; zero
    main_params = ["bytes<64>", "bytes<64>", "bytes<128>", "bytes<8>", "bytes<128>", "bytes<8>", "bytes<128>", "bytes<12>",
                   "bytes<128>", "bytes<12>", "bytes<128>", "bytes<12>", "bytes<24>", "bytes<48>", "bytes<24>", "bytes<48>",
                   "bytes<40>", "i32", "bytes<16>", "bytes<16>", "f32", "f32", "i32", "i32", "i32", "bytes<12>", "bytes<12>",
                   "bytes<12>"]
    main_args = [
        pack(64), pack(64),
        tmap(0, KV_LORA, HEADS, 64, q_stride), pack(8, {"at": 0, "i32": KV_LORA}, at(4, V)),
        tmap(1, ROPE, HEADS, 64, q_stride), pack(8, {"at": 0, "i32": ROPE}, at(4, V)),
        tmap(2, KV_LORA, PAGE, 64, kv_stride), pack(12, {"at": 0, "i32": PAGE}, {"at": 4, "i32": KV_LORA}),
        tmap(3, ROPE, PAGE, 64, kv_stride), pack(12, {"at": 0, "i32": PAGE}, {"at": 4, "i32": ROPE}),
        tmap(2, KV_LORA, PAGE, 32, kv_stride), pack(12, {"at": 0, "i32": KV_LORA}, {"at": 4, "i32": PAGE}),
        # page table [max_pages, B] strides (1, max_pages)
        pack(24, {"at": 0, "param": 4}, {"at": 8, "param": 12}, at(12, V), {"at": 16, "param": 12, "width": 8}),
        # o as (128, 512, tiles=1, B); lse as (128, 1, B): the split path never stores through these
        pack(48, {"at": 0, "param": 7}, {"at": 8, "i32": KV_LORA}, {"at": 12, "i32": 1}, at(16, V), {"at": 24, "i64": KV_LORA},
             {"at": 32, "i64": MLA_M_TILE * KV_LORA}, {"at": 40, "i64": HEADS * KV_LORA}),
        pack(24, {"at": 0, "param": 8}, {"at": 8, "i32": 1}, at(12, V), {"at": 16, "i64": HEADS}),
        acc_o, acc_lse,
        {"i32": split_max}, seqs, bsk,
        {"f32": scale_log2}, {"f32": 1.0},
        {"param": 11}, {"i32": 1}, {"i32": split_max},
        pack(12), pack(12, *fast_divmod(1)), pack(12, *fast_divmod(split_max)),
    ]
    reduce_params = ["bytes<48>", "bytes<40>", "bytes<48>", "bytes<40>", "i32", "bytes<16>", "bytes<16>"]
    reduce_args = [
        # o as (H, 512, S=1, B), lse as (H, 1, B)
        pack(48, {"at": 0, "param": 7}, {"at": 8, "i32": HEADS}, {"at": 12, "i32": KV_LORA}, {"at": 16, "i32": 1}, at(20, V),
             {"at": 24, "i64": KV_LORA}, {"at": 32, "i64": HEADS * KV_LORA}, {"at": 40, "i64": HEADS * KV_LORA}),
        pack(40, {"at": 0, "param": 8}, {"at": 8, "i32": HEADS}, {"at": 12, "i32": 1}, at(16, V), {"at": 24, "i64": HEADS},
             {"at": 32, "i64": HEADS}),
        acc_o, acc_lse, {"i32": split_max}, seqs, bsk,
    ]
    mod = module(MLA_MODULE)
    main = {**mod, "entry": MLA_MAIN, "params": main_params, "args": main_args, "block": [384, 1, 1],
            "grid": [2, batch, split_max], "cluster": [2, 1, 1], "shared_mem": MLA_MAIN_SMEM}
    ours = [] if reduce or ours_from is None else [
        mla_decode_launch(batch_max, split_max, scale_log2, {"var": T, "min": ours_from})]
    if ours:
        main["when"] = {"var": T, "max": ours_from - 1}
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in state", "in state", "in buffer<i32>", "in buffer<i32>",
                   "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>", "out buffer<f32>", "out buffer<f32>", "i32", "i32"],
        "impl": {"launches": [main, *ours] + [
            {**mod, "entry": MLA_REDUCE, "params": reduce_params, "args": reduce_args, "block": [128, 1, 1],
             "grid": [HEADS, 1, batch], "shared_mem": MLA_REDUCE_SMEM},
        ][:1 if reduce else 0]},
    }


# D's MLA decode over the DCP member's pages, its splits left for the exchange, from `--mla-ours-from` rows
# (off by default): k3_mla_decode.cu, the DSL split kernel's grid and output (acc_o / acc_lse, `mla_bsk`
# splits), a CTA pair a (row, split). It is the faster for a batch of one wave of items from 24 rows, the
# slower below and over several waves (a skewed batch's): kern-k3-pp-prefill's notes/r-attn.md.
MLA_DECODE_SMEM = 232320  # k3_mla_decode.cu SMEM_BYTES


def mla_decode_launch(batch_max, split_max, scale_log2, when):
    """k3_mla_decode.cu over mla_attn_op's interface: q and the pages by 64 x 64 boxes of their 576-wide rows,
    128 B swizzle."""
    tmap = lambda param, d1, d2, stride: pack(128, {"at": 0, "tensormap": {
        "param": param, "dtype": "bf16", "dims": [LATENT_ROW, d1, d2], "strides": [LATENT_ROW * 2, stride],
        "box": [64, 64, 1], "swizzle": 128, "l2_promotion": 128}})
    return {**module("k3_mla_decode"), "entry": "kern_k3_mla_decode",
            "params": ["bytes<128>", "bytes<128>", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>", "out buffer<f32>",
                       "out buffer<f32>", "i32", "i32", "f32"],
            "args": [tmap(0, HEADS, batch_max, HEADS * LATENT_ROW * 2), tmap(2, PAGE, 0, PAGE * LATENT_ROW * 2),
                     {"param": 4}, {"param": 5}, {"param": 6}, {"param": 9}, {"param": 10}, {"i32": split_max},
                     {"param": 12}, {"f32": scale_log2}],
            "block": [224, 1, 1], "grid": [2, T, split_max], "cluster": [2, 1, 1], "shared_mem": MLA_DECODE_SMEM,
            "when": when}


# ---- P: the absorbed short path, a call of at most `mla_short` rows


def p_short(mla_short):
    return mla_short


def p_short_ops(mla_short):
    short = {"when": {"var": T, "max": mla_short}}
    ops = {
        "mla_absorb_short": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_absorb", "kern_k3_mla_absorb", grid=[{"ceil_div": [T, 32]}, HEADS, 8],
                                         block=[128, 1, 1], defines={"QBF16": 1}, **short)]},
        },
        # the FMHA's softmax scale (f32 192^-0.5), not the decode step's bf16-rounded one;
        # the splits merged by the gate kernel, not the DSL's reduction launch
        "mla_attn_short": mla_attn_op(mla_short, MLA_SHORT_SPLITS, reduce=False, scale_log2=trtllm_fmha_abi.SCALE_LOG2),
    }
    for l in ops["mla_attn_short"]["impl"]["launches"]:
        l.update(short)
    return ops


def p_short_buffers(work, buffers, mla_short, max_pages):
    for name, width, dt in (("mla_row_table", max_pages, "i32"), ("mla_q_abs", HEADS * LATENT_ROW, "bf16"),
                            ("mla_o_lat", HEADS * KV_LORA, "bf16"), ("mla_s_lse", HEADS, "f32"),
                            ("mla_s_acc_o", MLA_SHORT_SPLITS * MLA_M_TILE * KV_LORA, "f32"),
                            ("mla_s_acc_lse", MLA_SHORT_SPLITS * MLA_M_TILE, "f32")):
        work(name, width, dt, var=mla_short)
    for name in ("mla_row_lens", "mla_row_bsk"):
        buffers[name] = {"dtype": "i32", "shape": [mla_short], "kind": "workspace"}


def p_short_args():
    """The prep's and the gate's short-path tables and partials."""
    return b("mla_row_table"), b("mla_row_lens"), b("mla_row_bsk"), b("mla_s_acc_o"), b("mla_s_acc_lse")


def p_short_steps(step, w, L, kv, max_pages):
    step(L + "absorb", "mla_absorb_short", b("q_bf16"), w("w_kv_b"), b("mla_q_abs"), B)
    step(L + "attn_short", "mla_attn_short", b("mla_q_abs"), b("mla_q_abs", KV_LORA * 2),
         {"state": kv, "offset": 0}, {"state": kv, "offset": KV_LORA * 2},
         b("mla_row_table"), b("mla_row_lens"), b("mla_row_bsk"), b("mla_o_lat"), b("mla_s_lse"),
         b("mla_s_acc_o"), b("mla_s_acc_lse"), B, i32(max_pages))


# ---- D


def d_attn_op(g):
    return mla_attn_op(g.seqs_max, g.mla_split_max, reduce=not g.xchg, ours_from=g.mla_ours_from if g.xchg else None)


def d_absorb_op():
    """On tcgen05, a block a head, W_UK by a tensor map (64 columns x 128 rows a box, the 128 B swizzle its MMA
    reads)."""
    return {
        "params": ["in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
        "impl": {"launches": [launch(
            "k3_mla_absorb_tc", "kern_k3_mla_absorb_tc", grid=[HEADS, 1, 1], block=[256, 1, 1], smem=ABSORB_SMEM,
            params=["bytes<128>", "in buffer<f32>", "out buffer<bf16>", "i32"],
            args=[{"pack": {"size": 128, "fields": [{"at": 0, "tensormap": {
                "param": 1, "dtype": "bf16", "dims": [KV_LORA, HEADS * 256], "strides": [KV_LORA * 2],
                "box": [64, NOPE_DIM], "swizzle": 128, "l2_promotion": 128}}]}},
                  {"param": 0}, {"param": 2}, {"param": 3}])]},
    }


def d_exchange_op(g, xg, tail):
    return {
        "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<i32>", "in buffer<i32>", "i32", *tail],
        "impl": {"launches": [xg()]},
    }


def d_buffers(work, buffers, g):
    work("o_lat", HEADS * KV_LORA)
    work("mla_lse", HEADS, "f32")
    work("mla_acc_o", g.mla_split_max * MLA_M_TILE * KV_LORA, "f32", var=g.seqs_max)
    work("mla_acc_lse", g.mla_split_max * MLA_M_TILE, "f32", var=g.seqs_max)


def d_attn_steps(step, w, L, kv, g):
    step(L + "attn", "mla_attn", b("q_abs"), b("q_abs", KV_LORA * 2), {"state": kv, "offset": 0},
         {"state": kv, "offset": KV_LORA * 2}, b("block_table"), b("seq_lens"), b("mla_bsk"),
         b("o_lat"), b("mla_lse"), b("mla_acc_o"), b("mla_acc_lse"), B, i32(g.max_pages))
    if g.xchg:
        # every head over this member's positions → this member's heads over all of them
        step(L + "dcp", "dcp_exchange", b("mla_acc_o"), b("mla_acc_lse"), b("seq_lens"),
             b("mla_bsk"), i32(g.mla_split_max), w("w_kv_b_l"), b("mla_gate"), b("gated"),
             b("dcp_lamport"), b("dcp_lamport_peers"), b("dcp_state"), b("tp_err"), {"rank": "tp"},
             B, i64(g.dcp_stage), i64(TP_TIMEOUT_NS))
    else:
        step(L + "vup", "mla_vup_gate", b("o_lat"), w("w_kv_b"), b("mla_gate"), b("gated"), B)
