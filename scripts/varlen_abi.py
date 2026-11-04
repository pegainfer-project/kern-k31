"""The packed prefill's two vendored kernels over several sequences at once:
FlashKDA's varlen build and TRT-LLM gen's context FMHA at a batch of
`seqs`. Both reuse kern's captured ABIs (flash_kda_abi, trtllm_fmha_abi);
only what the batch changes is spelled here.

FlashKDA varlen (`flash_kda_d128_varlen`, the same sources built with
-DKERN_VARLEN): the template's IsVarlen flag flips and nothing else in the
parameter types, so the TiledCopy packs are the non-varlen ones. Sequence j
is rows [cu_seqlens[j], cu_seqlens[j + 1]) (int64) of q / k / v / g / beta.
Kernel 1 runs a tile per 16 rows of a sequence: a sequence adds at most
one partial tile, so `ceil(rows / 16) + seqs` bounds the grid, the
workspace and `total_tiles` (the workspace's head stride, the same in both
kernels); excess CTAs return. The tile prefix kernel 1 searches comes
filled (upstream's `_flash_kda_build_tile_prefix`, folded into the packed
gather). Kernel 2 is replaced by source/k3_kda_rec.cu (one CTA per
(sequence, head)), which reads its workspace.
"""
import flash_kda_abi as kda
import trtllm_fmha_abi as fmha

KDA_MODULE = "flash_kda_d128_varlen"
PREPARE = kda.PREPARE.replace("Li16ELi128ELi256ELb0EE", "Li16ELi128ELi256ELb1EE")
assert PREPARE != kda.PREPARE


def tiles_max(rows_max, seqs_max):
    return -(-rows_max // kda.CHUNK) + seqs_max


def kda_workspace(hl, rows_max, seqs_max):
    """The six workspace arrays at the packed tile bound and the tile prefix."""
    n = tiles_max(rows_max, seqs_max) * hl
    c, d = kda.CHUNK, kda.HEAD_DIM
    return {
        "span_ws_kd": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_qd": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_kr": {"dtype": "bf16", "shape": [n, c, d], "kind": "workspace"},
        "span_ws_gt": {"dtype": "f32", "shape": [n, d], "kind": "workspace"},
        "span_ws_inv": {"dtype": "bf16", "shape": [n, c, c], "kind": "workspace"},
        "span_ws_mqk": {"dtype": "bf16", "shape": [n, c, c], "kind": "workspace"},
        "span_tile_prefix": {"dtype": "i32", "shape": [seqs_max + 1], "kind": "workspace"},
    }


# source/k3_kda_rec.cu: sizeof(K12Smem), its block
REC_SMEM = 182272
REC_ROWS_MAX = 8192
REC_BLOCK = 512
REC_GATE_CTAS = 56  # source/k3_kda_rec.cu K12_GATE_CTAS


def kda_op(hl, rows_max, seqs_max, module, rec_module, rows, seqs, scale=kda.QSCALE):
    """Interface: q | k | v | g (bf16 [rows, hl*128]) | beta (bf16 [hl, rows]) | dt_bias (f32 [hl*128]) |
    a_log (f32 [hl]) | ws_kd | ws_qd | ws_kr | ws_gt | ws_inv | ws_mqk | rows | cu_seqlens (i64 [seqs + 1]) |
    tile_prefix (i32 [seqs + 1], filled) | seqs | the KDA state | its line table | line bytes | the q|k|v|gate
    projections (bf16 [rows, 4*hl*128]) | gamma_o (f32 [128]) | gated (bf16 [rows, hl*128]) | span_at |
    progress (i32 [seqs * hl], zero). The recurrence's ungated output goes to q's buffer, which kernel 1
    has read. FlashKDA's kernel 1
    fills the workspace; k3_kda_rec runs the recurrence from and to every sequence's line on one CTA per
    (sequence, head), advances the conv windows, and its REC_GATE_CTAS more CTAs gate the output rows as
    the recurrence publishes them. `rows` / `seqs` are the call's vars."""
    assert rows_max <= REC_ROWS_MAX, "k3_kda_rec keeps a sequence's betas in shared memory"
    n = tiles_max(rows_max, seqs_max) * hl
    T, N, cu, tp = {"param": 13}, {"param": 16}, {"param": 14}, {"param": 15}
    tiles_expr = {"add": [{"ceil_div": [rows, kda.CHUNK]}, seqs]}
    tiles = {"pack": {"size": 4, "fields": [{"at": 0, "expr": tiles_expr}]}}
    tc = kda.tiled_copy
    D = kda.HEAD_DIM
    r = lambda param, box_rows: tc(param, "bf16", [D, rows_max, hl], [hl * 256, 256], [box_rows, kda.CHUNK, 1],
                                   dynamic=True)
    beta = tc(4, "bf16", [hl * rows_max], [], [32])
    dt_bias = tc(5, "f32", [D, hl], [512], [D, 1])
    ws_rows = lambda param: tc(param, "bf16", [D, kda.CHUNK, n], [256, 4096], [8, kda.CHUNK, 1])
    ws_gt = tc(10, "f32", [D, n], [512], [D, 1])
    ws_sq = lambda param: tc(param, "bf16", [kda.CHUNK, kda.CHUNK, n], [32, 512], [8, kda.CHUNK, 1])
    copies = ["bytes<256>"] * 11
    prepare = {
        **module, "entry": PREPARE, "block": [256, 1, 1], "grid": [tiles_expr, hl, 1], "shared_mem": 21248,
        "params": copies + ["f32", "i32", "i32", "i32", "in buffer<i64>", "bytes<4>", "in buffer<f32>", "f32",
                            "in buffer<i32>"],
        "args": [r(0, D), r(1, D), beta, r(3, D), dt_bias, ws_rows(7), ws_rows(8), ws_rows(9), ws_gt, ws_sq(11),
                 ws_sq(12), {"f32": scale}, T, {"i32": hl}, N, cu, tiles, {"param": 6}, {"f32": kda.GATE_SCALE}, tp],
    }
    tmap = lambda param, dims, strides, box, swizzle: {"pack": {"size": 128, "fields": [{"at": 0, "tensormap": {
        "param": param, "dtype": "bf16", "dims": dims, "strides": strides, "box": box, "swizzle": swizzle,
        "l2_promotion": 128}}]}}
    ws_tile = lambda param: tmap(param, [64, n * kda.CHUNK, 2], [256, 128], [64, kda.CHUNK, 2], 128)
    ws_tile_sq = lambda param: tmap(param, [kda.CHUNK, n * kda.CHUNK], [32], [kda.CHUNK, kda.CHUNK], 0)
    rec = {
        **rec_module, "entry": "kern_k3_kda_rec", "block": [REC_BLOCK, 1, 1],
        "grid": [{"add": [{"mul": [seqs, hl]}, REC_GATE_CTAS]}, 1, 1], "shared_mem": REC_SMEM,
        "params": ["bytes<128>"] * 6 + [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "inout state", "in buffer<i32>", "i64",
            "in buffer<bf16>", "in buffer<f32>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<i32>",
            "in buffer<i64>", "in buffer<i32>", "inout buffer<i32>", "bytes<4>", "i32", "i32"],
        "args": [ws_tile(7), ws_tile(8), ws_tile(9),
                 tmap(2, [64, rows_max, hl, 2], [hl * D * 2, 256, 128], [64, kda.CHUNK, 1, 2], 128),
                 ws_tile_sq(11), ws_tile_sq(12),
                 *({"param": k} for k in [2, 4, 10, 17, 18, 19, 20, 21, 0, 22, 23]), cu, tp, {"param": 24}, tiles,
                 T, N],
    }
    return {
        "params": ["inout buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                   "in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                   "out buffer<f32>", "out buffer<bf16>", "out buffer<bf16>", "i32", "in buffer<i64>",
                   "in buffer<i32>", "i32", "inout state", "in buffer<i32>", "i64", "in buffer<bf16>",
                   "in buffer<f32>", "out buffer<bf16>", "in buffer<i32>", "inout buffer<i32>"],
        "impl": {"launches": [prepare, rec]},
    }


def fmha_op(heads, q_max, kv_max, module, q, seqs, entry=fmha.ENTRY, fp8=False):
    """trtllm_fmha_abi.op at a batch of `seqs` (a var): grid z and mBatchSize are the sequences, the
    length tables ([seqs] / [seqs + 1] / [seqs + 1]) are the caller's; the softmax stats (float2 max | sum
    per row and head, extra param 9) written, for the merge of several passes. `entry` picks the
    causal kernel or its dense sibling of the same ABI; `fp8` its e4m3 Q / K / V build (o stays bf16), at
    scale 1: the same parameters over byte-wide rows, the Q and K / V boxes 128 elements (one 128 B
    swizzle span) wide, as FlashInfer's launcher sizes them."""
    op = fmha.op(heads, q_max, kv_max, module, q)
    launch = op["impl"]["launches"][0]
    fields = launch["args"][0]["pack"]["fields"]
    at = {f["at"]: k for k, f in enumerate(fields)}
    fields[at[1132]] = {"at": 1132, "var": seqs}
    fields.append({"at": 1112, "param": 9})  # ptrSoftmaxStats
    launch["entry"] = entry
    launch["grid"] = [launch["grid"][0], launch["grid"][1], seqs]
    op["params"].append("out buffer<f32>")
    if fp8:
        for off in (0, 128, 384):
            t = fields[at[off]]["tensormap"]
            t["dtype"] = "u8"
            t["strides"] = [s // 2 if s > 16 else s for s in t["strides"]]
            t["box"] = [128, *t["box"][1:]]
        launch["shared_mem"] = FMHA_FP8_SMEM
        op["params"][:3] = ["in buffer<fp8e4m3>"] * 3
    return op


FMHA_FP8_SMEM = 117376  # the e4m3 builds' shared memory, from the bundle's flashInferMetaInfo.h
