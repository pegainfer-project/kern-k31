"""Kimi-K3's shapes and the manifest helpers gen_stage.py and its MLA cache flavors (k3_kv_fp8.py,
k3_kv_bf16.py) share. Imported after gen_stage.py has put kern's tools/ on sys.path."""
import struct

from pinned import module
import trtllm_fmha_abi

H = 7168
V = 163840
HEADS, HEAD_DIM = 96, 128
INNER = HEADS * HEAD_DIM           # 12288
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
NOPE_DIM = 128  # per-head q/k dim before the rope part
KV_A = KV_LORA + ROPE              # 576
KV_EXP = HEADS * trtllm_fmha_abi.KV_ROW  # 30720, k 192 | v 128 per head
Q_B = HEADS * 192                  # 18432
WSM = 256                          # b_proj 96 | f_a 128 | pad
TOPK, LATENT, INTER = 16, 3584, 3072
EXPERTS = 896                      # routed
SHARED = 2 * INTER                 # 6144
DENSE_I = 33792
ATTN_RES_BLOCK = 12
NB_MAX = 8
LAYERS = 93
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}

# KDA state line: the recurrent state [96,128,128] (bf16, or f32 with --kda-state f32) then the three
# conv windows bf16 [3][12288] (q, k, v).
# MLA latent cache: 64-token pages of [mla_layers][64][576], e4m3 or bf16 (the flavor's BYTES).
PAGE = 64
LATENT_ROW = KV_A
# A short packed chunk's absorbed attention splits a row's KV at most this many ways (k3_mla_glue.cu's
# prep and gate take it whether or not the flavor has the short path).
MLA_SHORT_SPLITS = 16
# A long packed call's rows attend their own rows through the causal FMHA and their sequences' cached
# context through its dense sibling in rounds of `window` keys (k3_fmha_plan.cuh), each pass's o and
# softmax stats folded into an f32 accumulator (k3_mla_glue.cu). KV_PAD: the FMHA's KV tile, rows past
# the last sequence's end a pass reads, kept zero.
KV_PAD = 128
FMHA_DENSE = "trtllm_fmha_dense_h192_v128"
FMHA_DENSE_ENTRY = trtllm_fmha_abi.ENTRY.replace("Causal", "Dense")
assert FMHA_DENSE_ENTRY != trtllm_fmha_abi.ENTRY

TP_TIMEOUT_NS = 30_000_000_000  # a deadline that passes traps: long enough for any start skew

T = "tokens"
HF = "language_model."
ROWS = "rows"
CTX = "ctx"

# Launch geometry per entry, as the kernel headers document it
# (docs/k3-kernel-abi.md §1). grid.x is always the batch.
GEOM = {
    "kern_k3_land_add_attnres_rms": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_kda_core": ([T, HEADS, 1], [128, 1, 1], 0),
    "kern_k3_mla_prep": ([T, 4, 1], [512, 1, 1], 0),  # 1 norm/append block + 3 gate segments
    "kern_k3_mla_vup_gate": ([{"ceil_div": [T, 32]}, HEADS, 4], [256, 1, 1], 0),  # 32 rows x 32 dv per block
    "kern_k3_router_topk": ([T, 1, 1], [256, 1, 1], 0),
    "kern_k3_argmax_f32_partial": ([T, 64, 1], [1024, 1, 1], 0),
    "kern_k3_argmax_f32_final": ([T, 1, 1], [64, 1, 1], 0),
    "kern_k3_rms": ([T, 1, 1], [1024, 1, 1], 0),
}

B = {"var": T}
i32 = lambda v: {"i32": v}
i64 = lambda v: {"i64": v}
b = lambda name, off=0: {"buf": name, "offset": off} if off else {"buf": name}


def launch(cubin, entry, grid=None, block=None, smem=None, var=T, defines=None, **extra):
    """A launch of `entry`; `var` is the batch var its grid.x runs over,
    `defines` selects a variant build of the source (handwritten.hw)."""
    g, bl, s = GEOM.get(entry, (None, None, 0))
    g = [var if d == T else d for d in (grid or g)]
    l = {**module(cubin, **(defines or {})), "entry": entry, "block": block or bl, "grid": g, **extra}
    s = s if smem is None else smem
    if s:
        l["shared_mem"] = s
    return l


def bf16(x):
    """x rounded to bf16 (nearest even), as a float."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16))[0]


def pack(size, *fields):
    return {"pack": {"size": size, "fields": list(fields)}}


def dim(x):
    """A batch dimension as a scalar arg: a var by name or an expression over one."""
    return {"var": x} if isinstance(x, str) else {"expr": x}
