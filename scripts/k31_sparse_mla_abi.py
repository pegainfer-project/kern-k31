"""TRT-LLM gen's sparse MLA (kernels.toml `trtllm_sparse_mla_h576_v512_e4m3`) as one manifest op: every row
attends its own top-k rows of the fp8 MLA cache in the latent space, e4m3 q and KV 576 / 512, o bf16.

The ABI is data, not derivation: what tools/kernel-capture lifted from FlashInfer 0.7.2.post1's own launch
(harness/sparse_mla/probe.py, a call shaped as vLLM's FlashInferMLASparse backend makes it), field for field
against `KernelParams`, identical for 64 and 8192 rows but for the row count and the pool's page count.
Every row is one request of one query token: grid.z is the row, the two CTAs of a cluster split the 128
heads, one CTA a row's whole top-k (no KV split, so the partial and counter pointers are never followed).
K3's 96 heads ride as 128, the pad heads' q zero.

The kernel leaves a row's softmax as (max, sum) per head, max already scaled by bmm1; its base-2 LSE is
log2(e) * max + log2(sum) (FlashInfer's ComputeLSEFromMD). A row's count must be at least 1: a row with
nothing selected attends one dummy index and its result is discarded by whoever reads it.
"""
from pinned import module

ENTRY = ("fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseStaticTokenSparseP1VarSeqQ64"
         "Kv128Persistent2CtaKeepsAbForGen")
MODULE = "trtllm_sparse_mla_h576_v512_e4m3"
HEADS, D_QK, D_V, TOPK, PAGE = 128, 576, 512, 2048, 64
SMEM, PARAMS = 213128, 1344
MAX_ROWS = 65535  # grid.z
SCALE_LOG2 = 0.10411754995584488  # 192^-0.5 * log2(e), as FlashInfer rounds it
SCRATCH = 1 << 20


def op(rows_max, rows, kv="in state"):
    """Interface: q (e4m3 [rows, 128, 576]) | kv (the layer's fp8 pages, [., 64, 576]) | ind (i32 [rows, 2048],
    cache rows, -1 past the count) | lens (i32 [rows], each >= 1) | o (bf16 [rows, 128, 512]) |
    stats (f32 [rows, 128, 2]: max, sum). `rows` is the call's row count (a var name or an expression); `kv` the
    cache parameter's type, a buffer in the probe."""
    assert rows_max <= MAX_ROWS
    n = {"var": rows} if isinstance(rows, str) else {"expr": rows}
    tm = lambda at, param, dims, strides, box, **kw: {"at": at, "tensormap": {
        "param": param, "dtype": "u8", "dims": dims, "strides": strides, "box": box, "swizzle": 128,
        "l2_promotion": 128, **kw}}
    i32 = lambda at, v: {"at": at, "i32": v}
    f32 = lambda at, v: {"at": at, "f32": v}
    fields = [
        tm(0, 0, [D_QK, 64, 2, rows_max], [D_QK, 64 * D_QK, HEADS * D_QK], [128, 64, 1, 1]),
        tm(128, 1, [D_QK, 0], [D_QK], [128, 1]),  # FlashInfer: 2^31 - 1 rows, i.e. unbounded
        tm(384, 1, [D_QK, PAGE, 1, 0], [D_QK, PAGE * D_QK, PAGE * D_QK], [128, 1, 1, 1]),
        tm(512, 4, [D_V, rows_max, 1, HEADS, 1], [HEADS * D_V, D_V, D_V, 16], [128, 64, 1, 1, 1]),  # FlashInfer: 0 for the unit dim
        {"at": 912, "param": 4},       # ptrO
        {"at": 1000, "scratch": "s"},  # ptrMultiCtasKvCounter
        {"at": 1016, "param": 2},      # ptrPageIdxKv
        {"at": 1024, "scratch": "s"},  # ptrPartialO
        {"at": 1032, "scratch": "s"},  # ptrPartialStats
        {"at": 1096, "param": 3},      # ptrSeqLensKv
        {"at": 1112, "param": 5},      # ptrSoftmaxStats
        i32(1128, 0x7FFFFFFF),         # mAttentionWindowSize
        {"at": 1132, **n},             # mBatchSize
        i32(1172, 1), i32(1176, TOPK), i32(1180, 1), i32(1184, 1), i32(1188, TOPK),
        i32(1192, 1), i32(1196, HEADS), i32(1200, HEADS),
        i32(1204, HEADS), i32(1208, -(1 << 31)), i32(1216, 6),  # FastModDivInt32(128)
        {"at": 1224, "i64": HEADS * D_QK},  # mNumHiddenEltsO
        i32(1236, 1),                  # mNumTokensPerCtaQ
        i32(1240, 6),                  # mNumTokensPerPageLog2
        i32(1244, 1),                  # mReshapeFactorKv
        f32(1248, 1.0), f32(1252, SCALE_LOG2), f32(1256, 1.0), f32(1260, -1.0),
        i32(1268, TOPK),               # mSparseAttnTopK
        {"at": 1276, **n},             # mSumOfSeqLensQ
        {"at": 1285, "u8": 1},         # mUsesSharedPagedKvIdx
    ]
    launch = {
        **module(MODULE), "entry": ENTRY, "block": [512, 1, 1], "grid": [2, 1, n.get("var", n.get("expr"))],
        "cluster": [2, 1, 1], "shared_mem": SMEM, "params": [f"bytes<{PARAMS}>"],
        "args": [{"pack": {"size": PARAMS, "fields": fields}}],
    }
    return {
        "params": ["in buffer<fp8e4m3>", kv, "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
                   "out buffer<f32>"],
        "impl": {"scratch": {"s": {"dtype": "u8", "shape": [SCRATCH]}}, "launches": [launch]},
    }
