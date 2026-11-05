"""cuDNN frontend's FROST KDA prefill (source/cudnn-frost-kda, one cubin of two entries) as one manifest op over
P's packed call: every sequence's state read from and written to its KDA line in place.

The ABI below is what tools/kernel-capture lifted from the vendored build's own launch over buffers laid out as P
lays them out (bench_results 2026-11-04-k31-dsa kda-cudnn, scripts/ref_launch.py), field for field. A CuTe DSL
tensor parameter is a struct by value: the pointer, an i32 per mode of its shape, padding to 8, an i64 per
non-unit stride (in elements), the unit stride's mode last. The structs' tail padding is host stack the launcher
never initialises; the packs leave it zero.

The prologue (two CTAs) orders the (sequence, head) work items longest first and zeroes the scheduler (block 0),
and copies each base descriptor once per sequence into the descriptor workspace with its address and token extent
patched to the sequence's rows (block 1). The base descriptors are therefore encoded once at load over the
buffers' whole extent. The sixth is the checkpoints' and, with none, the output's again. The prefill then runs at
most 152 persistent CTAs over the work items. Neither reads a pool row's count (the pool's leading extent).
"""
HEADS = 96
DIM = 128
NUM_SM = 152
SMEM_PROLOGUE = 32776
SMEM_PREFILL = 208384
WORK_ITEM_FIELDS = 10
# the prologue's descriptor workspace: six arrays of a 128-byte CUtensorMap a sequence, and a 128-byte tail (i64s)
WS_WORDS_A_SEQ, WS_WORDS = 6 * 128 // 8, 128 // 8
PROLOGUE = ("cudnn_kernel_cutlass_frost_kda_prefill_prologue_16_152_1_TensorMapswizzleTensorMapSwizzles128b_3_dtype"
            "class_cutlassbase_dsltypingBFloat16_tma_formatTensorMapDataFormatDEFAULT_1_box_dims64_1_16__0")
PREFILL = ("cudnn_kernel_cutlass_frost_kda_prefill_KdaPrefillCfgio_dtypecutlassbase_dsltypingBFloat16state_dtype"
           "cutlassbase_dsltypingBFloat16gate_dtypecutlassbase_dsltypingBFloat16use_initial_stateTruestor_1")
# q / k / v / g / o (bf16 [rows, HEADS*DIM]) | beta (bf16 rows of `beta_ld`) | dt_bias (f32 [HEADS*DIM]) |
# a_log (f32 [HEADS]) | rows | cu_seqlens (i64 [seqs + 1]) | seqs | the KDA state | its line table (i32 [seqs])
PARAMS = ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
          "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "i32", "in buffer<i64>", "i32", "inout state",
          "in buffer<i32>"]
Q, K, V, G, O, BETA, DT_BIAS, A_LOG, ROWS, CU, SEQS, STATE, LINE = range(len(PARAMS))


def pack(size, *fields):
    return {"pack": {"size": size, "fields": list(fields)}}


def memref(ptr, shape, strides):
    """A CuTe DSL tensor struct: the pointer, the shape's i32s, the strides' i64s from the next 8-byte boundary."""
    at = 8 + 4 * len(shape)
    at += at % 8
    return pack(at + 8 * len(strides), {"at": 0, **ptr},
                *({"at": 8 + 4 * i, **s} for i, s in enumerate(shape)),
                *({"at": at + 8 * i, "i64": s} for i, s in enumerate(strides)))


def op(module, rows_max, seqs_max, line_bytes, beta_ld, rows, seqs, scale=DIM ** -0.5):
    """The prologue then the prefill, both programmatic dependents. `rows` / `seqs` are the call's vars, `line_bytes`
    a KDA line's stride (its state at offset 0, [HEADS][DIM v][DIM k] bf16), `beta_ld` beta's row stride in
    elements; o must not alias q / k / v / g, which the prefill reads to its end."""
    assert line_bytes % 16 == 0 and beta_ld % 8 == 0
    T, N = {"param": ROWS}, {"param": SEQS}
    i32 = lambda x: {"i32": x}
    rowwise = lambda p: memref({"param": p}, [T, i32(HEADS), i32(DIM)], [HEADS * DIM, DIM])
    tmap = lambda p: pack(128, {"at": 0, "tensormap": {
        "param": p, "dtype": "bf16", "dims": [DIM, HEADS, rows_max], "strides": [DIM * 2, HEADS * DIM * 2],
        "box": [64, 1, 16], "swizzle": 128, "l2_promotion": 0}})
    ws = memref({"scratch": "frost_ws"}, [{"expr": {"add": [{"mul": [seqs, WS_WORDS_A_SEQ]}, WS_WORDS]}}], [])
    cu = memref({"param": CU}, [{"expr": {"add": [seqs, 1]}}], [])
    work_items = memref({"scratch": "frost_work"}, [{"expr": {"mul": [seqs, HEADS]}}], [])
    count = memref({"scratch": "frost_count"}, [i32(1)], [])
    scheduler = memref({"scratch": "frost_scheduler"}, [i32(2)], [])
    pool = memref({"param": STATE}, [i32(0), i32(HEADS), i32(DIM), i32(DIM)], [line_bytes // 2, DIM * DIM, DIM])
    lines = memref({"param": LINE}, [N], [])
    divisor = pack(12, {"at": 0, "i32": 1}, {"at": 4, "i32": 1})  # FastDivmodDivisorV2(1): every head its own q / k / v
    prologue = {
        **module, "entry": PROLOGUE, "block": [1024, 1, 1], "grid": [2, 1, 1], "shared_mem": SMEM_PROLOGUE,
        "pdl": True,
        "params": ["bytes<128>"] * 6 + ["bytes<16>", "bytes<16>"] + ["bytes<40>"] * 5 + ["bytes<16>"] * 3
                  + ["i32", "i32"],
        "args": [tmap(Q), tmap(K), tmap(V), tmap(G), tmap(O), tmap(O), ws, cu, *(rowwise(p) for p in (Q, K, V, G, O)),
                 count, work_items, scheduler, N, i32(0)],
    }
    prefill = {
        **module, "entry": PREFILL, "block": [512, 1, 1], "grid": [{"clamp": [{"mul": [seqs, HEADS]}, 0, NUM_SM]}, 1, 1],
        "shared_mem": SMEM_PREFILL, "pdl": True,
        "params": ["bytes<12>", "i32"] * 3 + ["bytes<16>", "i32"] + ["bytes<40>"] * 4
                  + ["bytes<16>", "bytes<24>", "bytes<24>", "bytes<16>", "bytes<48>", "bytes<40>", "bytes<48>"]
                  + ["bytes<16>"] * 5 + ["f32", "i32"],
        "args": [divisor, i32(1), divisor, i32(1), divisor, i32(1), ws, N, *(rowwise(p) for p in (Q, K, V, G)),
                 memref({"param": A_LOG}, [i32(HEADS)], []),
                 memref({"param": DT_BIAS}, [i32(HEADS), i32(DIM)], [DIM]),
                 memref({"param": BETA}, [T, i32(HEADS)], [beta_ld]),
                 cu, pool, rowwise(O), pool, lines, lines, work_items, count, scheduler, {"f32": scale}, i32(0)],
    }
    return {
        "params": PARAMS,
        "impl": {
            "scratch": {
                "frost_ws": {"dtype": "i64", "shape": [seqs_max * WS_WORDS_A_SEQ + WS_WORDS]},
                "frost_work": {"dtype": "i32", "shape": [seqs_max * HEADS, WORK_ITEM_FIELDS]},
                "frost_count": {"dtype": "i32", "shape": [1]},
                "frost_scheduler": {"dtype": "i32", "shape": [2]},
            },
            "launches": [prologue, prefill],
        },
    }
