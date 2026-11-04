#!/usr/bin/env python3
"""Generate Kimi-K3's two manifests for kern: P, the packed prefill a
pipeline cuts into stages, and D, the TP / EP / DCP decode step.

A fork of kern's tools/gen_k3.py; the rest of kern's tools (the manifest
helpers, `once`, the kernel index, the FMHA and batched-GEMM ABI ports) are
imported from the kern checkout KERN names:

    KERN=<kern checkout> KERN_INDEX_DIR=<kern master>/kernel_index \
      python3 scripts/gen_stage.py --chunk 8192 > p.json
    ... --tp 8 --moe-variants FC1,FC2 --peer-ar > d.json

Both run layers [0, --layers) with the embedding and the head, and keep one
state per layer (`kv.<i>`, `kda.<i>`), so `kern cut` can drop the layers a
stage does not run.

P (`--chunk N`): program `prefill` runs up to --pack sequences' chunks, packed
back to back (`cu_seqlens`, at most N rows), on one GPU holding every expert,
and the head on each sequence's last row: `next_token` is one token a
sequence. A KDA layer runs FlashKDA's kernel 1 and source/k3_kda_rec.cu from
and to each sequence's line. An MLA layer expands the context's latent rows
by kv_b and runs TensorRT-LLM's context FMHA (docs/k3-kernel-abi.md K13), or,
over bf16 pages, for a call of at most --mla-short rows, attends in the latent space. The MoE
runs TRT-LLM gen's batched GEMMs over every expert (k3_moe_bmm.py), routed in
one call (source/k3_moe_route.cu).

D (`--tp R`): the decode step as SGLang's TP / EP / DCP decode runs it
(decode/sglang-tp8-dcp8-ep8), one batch replicated on the group's R ranks.
The KV of an MLA layer is dealt by position (`kv.<l>` shards `by: position`:
position p on member p % R), so every member runs the MLA's q side for every
row and all 96 heads over its own positions (q_a / kv_a, q_b and the absorb
replicated; the append writes through the owner's slot only, the others'
slots point at a pad page), and the members' partials change hands by heads
(source/k3_dcp.cu). From there a member holds HEADS / R heads of every row,
as it does in a KDA layer (`kda.<l>` shards `by: heads`): v-up, gate and
o_proj over its heads, the o_proj partial summed over the group (NCCL, or the
Lamport one-shot with --peer-ar).
The MoE routes every row through the member's experts — every R-th one from
its index, so the routing's first expert is `{"rank": "tp"}` and the slice a
stride, no rank arithmetic in the manifest — with FlashInfer's routing and
the batched GEMMs, and the routed latent's partial and the shared expert's
(column-split) go through one bf16 all-reduce side by side; the dense FFN's
down partial likewise. Router, lat_down / lat_up, the norms, the embedding
and the LM head (vocab-parallel) are replicated. `--tp 1` is the same step
on one GPU holding every expert, with nothing to exchange or sum: the oracle
the group is checked against (step/, scripts/compare_steps.py). `--layers 2`
(one KDA layer, one MoE layer) has no MLA layer and no paged state.

`--kv` picks the MLA cache: SGLang's fp8 (scripts/k3_kv_fp8.py, the default)
or bf16 (scripts/k3_kv_bf16.py). A flavor owns its page bytes, D's absorb,
attention and exchange, and P's absorbed short path (bf16's alone).

The kernels are kern's own (docs/k3-kernel-abi.md, source/k3_*.cu): B is a
runtime argument, every launch takes one row per block.x, and the landing /
residual / norm / append work is fused into its neighbours. The launch
sequence still follows pegainfer's certified `k3_step` operand for operand;
only the kernel boundaries moved.
"""
import argparse
import json
import math
import pathlib
import os
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(os.environ["KERN"]) / "tools"))
# This directory's forks first: kern's modules put kern's tools/ at the head of sys.path on import.
import k3_moe_bmm
from pinned import module
import trtllm_fmha_abi
import varlen_abi
import kern_manifest
from once import Once, buf, seg
import k3_once  # noqa: F401  K3's load-time ops, which kern's `once` no longer carries
from k3_common import *  # noqa: F403
import k3_kv_bf16
import k3_kv_fp8

KV = {"fp8": k3_kv_fp8, "bf16": k3_kv_bf16}

# k3_ar_fused.cu's KGRID, one block an SM: every block resident on 148 and 152 SMs, a multiple of the cluster of 8
# and under the 256-row flag table
TP_AR_GRID = 144
DCP_GRID = 2 * 144  # k3_dcp.cu bounds the exchange to two blocks an SM: every block resident on 148 and 152 SMs
KDA_SMEM_BLOCKS = 2 * 152  # ... and two of its blocks an SM: the batches of one wave of them
CAND_SMEM = NB_MAX * H * 2  # k3_ar_fused.cu: a row's snapshots prefetched into shared memory
LAND_CL = 4  # k3_ar_fused.cu: the closing add's row over a cluster of 4 blocks
PF_PARAMS = ["in buffer<bf16>", "i64"]  # k3_ar_fused.cu: the next GEMM's weight, prefetched into L2
PF_OPS = {"ar_attnres_rms", "ar_finalize_rms", "land_add2_attnres_rms_snap"}
base_op = lambda op: op.rsplit("_nb", 1)[0]  # an op without its snapshot count (k3_ar_fused.cu: an entry a count)
PF_MAX = 96 << 20  # of the GB300's 129 MB L2
LAND_PF_MAX = 16 << 20  # the closing add's prefetch hints: their issue holds the grid the next GEMM waits on
# ar_finalize_rms also loads the next o_proj's weight at evict_last; ar_attnres_rms after that o_proj demotes it
PF_LAST = ".o_proj"


# source/k3_gemm_f32.cu: D's K = 7168 f32 projections (qkvg 6400, front 6016) over 1-32 rows in cuBLAS's split-K
# order (cuBLAS splits K at these 64-column blocks below 49 rows) with the reduce in the same launch; from 33 rows
# cuBLAS keeps them (ours, at most 4 stages of a 40-48-row A, was 1-2 us a call slower there).
GEMM_T_SPLIT = (38, 76)
GEMM_T_BN = ((8, 8), (16, 16), (24, 24), (32, 32))  # (max rows, BN)
GEMM_T_BM = 64
GEMM_T_KB = {6016: 100}  # n -> a CTA's shared memory (KB): 100 two CTAs an SM where its clusters all fit, else 73


def gemm_t_op(n, k):
    """C[m, n] = A[m, k] @ W[n, k]^T, f32: one launch for each row bucket, cuBLAS past the last."""
    tiles = -(-n // GEMM_T_BM)
    kb = GEMM_T_KB.get(n, 73)
    tmap = lambda param, rows, box: pack(128, {"at": 0, "tensormap": {
        "param": param, "dtype": "bf16", "dims": [k, rows], "strides": [k * 2], "box": [64, box], "swizzle": 128,
        "l2_promotion": 256}})
    launches, lo = [], 1
    for hi, bn in GEMM_T_BN:
        launches.append({**module("k3_gemm_f32", BN=bn, CTA_KB=kb, SPLITS=3), "entry": "kern_k3_gemm_f32",
                         "block": [128, 1, 1], "grid": [tiles * 3, 1, 1], "cluster": [3, 1, 1],
                         "shared_mem": (kb + 1) * 1024, "pdl": True, "when": {"var": T, "min": lo, "max": hi},
                         "params": ["bytes<128>", "bytes<128>", "out buffer<f32>", "i32", "i32", "i32", "i32", "i32",
                                    "i32", "in buffer<bf16>", "i64", "in buffer<bf16>", "i64"],
                         "args": [tmap(1, n, GEMM_T_BM), tmap(0, 0, bn), {"param": 2}, {"param": 6}, {"param": 3},
                                  {"param": 4}, {"param": 5}, {"i32": GEMM_T_SPLIT[0]}, {"i32": GEMM_T_SPLIT[1]},
                                  *({"param": i} for i in range(7, 11))]})
        lo = hi + 1
    launches.append({"entry": "extern:cublas_bf16_tn_f32", "when": {"var": T, "min": lo},
                     "params": GEMM_PARAMS_F32, "args": [{"param": i} for i in range(7)]})
    return {"params": [*GEMM_PARAMS_F32, *PF_PARAMS, *PF_PARAMS], "impl": {"launches": launches}}


GEMM_PARAMS_F32 = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"]


def is_mla(i):
    return i in MLA_LAYERS



def build(end, max_ctx, seqs_max, tp=1, mla_split_max=16, chunk_max=0, moe_variants=None, pack=16,
          peer_ar=False, mla_short=128, kv="fp8", mla_ours_from=None, window=65536, kda_state="bf16"):
    """P with `chunk_max`, else D over a `tp` group (see the module doc), of layers [0, `end`). `pack`: P's sequences a call; `window`: the cached keys a
    P's MLA layer expands and attends at once; `peer_ar`: D's sums by the group's Lamport one-shot
    (source/peer_allreduce_bf16.cu) instead of NCCL; `kv`: the MLA cache, SGLang's fp8 (e4m3, scale 1; D
    attends it with the Cake kernel, P lifts it to bf16 for the expansion) or bf16; `kda_state`: the KDA
    recurrent state's storage, bf16 or f32."""
    chunk = chunk_max > 0
    decode = not chunk
    assert 0 < end <= LAYERS
    assert decode or (tp == 1 and pack > 1), "P is one GPU's packed prefill"
    assert chunk or (HEADS % tp == 0 and EXPERTS % tp == 0), "D deals the heads and the experts over its group"
    layers = range(end)
    # A D group of one is the step's own oracle: the same ops on one GPU
    # holding every expert, with nothing to exchange or sum.
    xchg = decode and tp > 1
    fused_ar = xchg and peer_ar
    flavor = KV[kv]
    assert kda_state in ("bf16", "f32")
    flavor.check(decode, xchg, seqs_max)
    if chunk:
        mla_short = flavor.p_short(mla_short)
    state_bytes = 2 if kda_state == "bf16" else 4
    kda_state_defs = {"KDA_STATE_BF16": 1} if kda_state == "bf16" else {}
    glue_defs_kv = flavor.P_GLUE
    # P's attention on e4m3 (q and the expanded k | v cast), the FMHA's fp8 builds
    fp8_attn = chunk and flavor.P_FP8_ATTN
    fmha_kernel = lambda name, entry: (module(name + "_e4m3" if fp8_attn else name),
                                       entry.replace("QkvBfloat16", "QkvE4m3") if fp8_attn else entry)
    attn_q, attn_sfx, attn_bytes = ("q_fp8", "_fp8", 1) if fp8_attn else ("q_bf16", "", 2)
    # P's MXFP8 projections, vLLM's online mxfp8 of in_proj_qkvgfab, fused_qkv_a_g_proj and the routed expert's
    # down / up: the weight quantized as it loads, the activation per call (k3_mxfp8.cu), cuBLASLt's GEMM over
    # both scales; a [rows, k] operand's scales in cuBLAS's 128x4 tiles
    mx = chunk and flavor.P_MXFP8
    sf_len = lambda rows, k: -(-rows // 128) * 128 * -(-k // 128) * 4
    if chunk:
        seqs_max = pack
    # A packed call's sequences' cached keys back to back (`ctx` is the sum of
    # their lengths with this call's rows) in rounds of `window`: round w
    # expands rows [w window, min((w + 1) window, ctx)) plus the FMHA's tile
    # past the last sequence's end. NS: a table row's words; a set is four rows.
    rounds = -(-max_ctx // window) if chunk else 0
    round_rows = lambda w: {"add": [{"clamp": [CTX, w * window, window]}, KV_PAD]}
    NS = pack + 1
    lens_set = lambda k: {"buf": "fmha_lens", "offset": k * 4 * NS * 4} if k else {"buf": "fmha_lens"}
    kda_layers = [i for i in layers if not is_mla(i)]
    mla_layers = [i for i in layers if is_mla(i)]
    mla = bool(mla_layers)
    assert mla or decode, "P needs at least one MLA layer (layer 3)"
    # A D member's page table covers its own positions: every tp-th of the context.
    max_pages = -(-max_ctx // tp // PAGE)
    page_stride = PAGE * LATENT_ROW  # elements
    blocks_total = -(-end // ATTN_RES_BLOCK)
    assert blocks_total <= NB_MAX
    # `tokens` bounds a D step's sequences or a P call's rows; P sizes its
    # residual stream, MLP and MoE workspaces for the whole call (`RW`).
    t_max = chunk_max if chunk else seqs_max
    RW = chunk_max if chunk else T
    # The MoE: TRT-LLM gen's batched GEMMs (k3_moe_bmm: the fp8 latent, its
    # scales and the routing) over every expert in P; over a D member's every
    # tp-th one for every row, its combine over every row the flat MoE
    # all-reduce sums (`seqs_max`), writing zeros past `tokens`.
    epr = EXPERTS // tp
    bp = (k3_moe_bmm.pieces(epr, T, seqs_max, seqs_max, prefix="moe.", names=moe_variants, stride=tp,
                            pdl=True) if decode else
          k3_moe_bmm.pieces(epr, T, chunk_max, {"mul": [T, 1]}, prefix="moe.", names=moe_variants))
    # This rank's KDA and MLA heads: whole, or a D member's shard (heads, the
    # per-head weights and the KDA state line all HEADS / tp wide; the kernels
    # are built for that width, the binds cut the checkpoint's tensors).
    hl = HEADS // tp
    inner_l, fused_l = hl * HEAD_DIM, 4 * hl * HEAD_DIM
    line_l = hl * HEAD_DIM * HEAD_DIM * state_bytes + 3 * (3 * inner_l * 2)
    kda_defs = {"HEADS": hl} if tp > 1 else None
    kda_smem_rows = KDA_SMEM_BLOCKS // hl
    # P's KDA projections, and a D member's, are one GEMM: its heads' q | k |
    # v | g, then beta's columns and f_a's, the wsm partial read from the same
    # row, padded to WSM columns with f_a's first rows again (nobody reads
    # them): at 6284 columns cuBLAS falls off its nvjet kernels to one twice as slow.
    kfused = fused_l + WSM if chunk or xchg else 0
    mla_fused_l = Q_LORA + KV_A + inner_l
    # This rank's columns of the shared expert and the dense FFN.
    sh_l, dn_l = SHARED // tp, DENSE_I // tp
    # A D member's MoE front is one GEMM over the router, lat_down and this
    # rank's shared gate | up rows (one read of 86 MB at 6016 columns instead of
    # three launches with their split-K reduces); each reader takes its
    # columns of the f32 row.
    front = EXPERTS + LATENT + 2 * sh_l if xchg else 0
    # P's projections land in their GEMM's epilogue (cuBLASLt, bf16 D); D's
    # KDA kernels read f32 projections.
    kpart = "bf16" if chunk else "f32"

    def per_row(n):
        return [T, -(-n // 1024), 1]

    # The mix and norm of P's rows run the prefill glue's many-rows-per-SM
    # forms (k3_prefill_glue.cu, bit for bit the decode kernels' values); a D
    # batch's the decode form, a row per 1024 threads: at 64 rows the glue's
    # 7-warp row leaves most SMs idle.
    def residual(entry):
        return launch("k3_prefill_glue", entry, grid=[T, 1, 1], block=[448, 1, 1])

    dcp_stage = tp * seqs_max * hl * (KV_LORA // 8 + 1) * 16
    g = types.SimpleNamespace(seqs_max=seqs_max, tp=tp, max_ctx=max_ctx, mla_split_max=mla_split_max, xchg=xchg,
                              mla_ours_from=mla_ours_from, max_pages=max_pages, dcp_stage=dcp_stage)
    ops = {
        "gemm_f32": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]},
        },
        **({f"gemm_t{n}": gemm_t_op(n, H) for n in (front, kfused)} if xchg else {}),
        # K1 residual stream
        # The embedding gather and the first layer's norm (nb == 0: the mix is the row) in one launch.
        **({"embed_rms": {
            "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                       "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_ar_fused", "kern_k3_embed_rms", grid=[T, 1, 1], block=[1024, 1, 1])]},
        }} if decode else {}),
        # P's: also its stage's setup, the MoE routing state zeroed and the FMHA's tables.
        **({"embed_rms_setup": {
            "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                       "out buffer<bf16>", "in buffer<i32>", "in buffer<i64>", "out buffer<i32>", "i32", "i32", "i32",
                       "i32", "i32", "out buffer<u32>", "i32", "i32"],
            "impl": {"launches": [residual("kern_k3g_embed_rms")]},
        }, "attnres_rms_setup": {
            "params": ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32", "out buffer<u32>", "i32", "i32"],
            "impl": {"launches": [residual("kern_k3g_attnres_rms_setup")]},
        }} if chunk else {}),
        # ... and D's MLA split plan (k3_mla_split_plan) in the same launch, a row's per block
        **({"embed_rms_plan": {
            "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                       "out buffer<bf16>", "in buffer<i32>", "out buffer<i32>", "i32", "i32"],
            "impl": {"launches": [launch("k3_ar_fused", "kern_k3_embed_rms_plan", grid=[{"add": [T, 1]}, 1, 1],
                                                 block=[1024, 1, 1])]},
        }} if decode and mla else {}),
        # The attention's landing is o_proj's bf16 output, the value the f32 form rounds to first.
        "land_add_attnres_rms_bf16": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",
                       "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [residual("kern_k3g_land_add_attnres_rms") if chunk else
                                  launch("k3_residual", "kern_k3_land_add_attnres_rms", defines={"LAND_BF16": 1})]},
        },
        # A layer's closing landing add fused into the next layer's mix and norm.
        **({"land_add2_attnres_rms": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32",
                       "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [residual("kern_k3g_land_add2_attnres_rms")]},
        }} if chunk else {}),
        # P's closing add on its own, before a snapshot layer.
        **({"land_add2": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_prefill_glue", "kern_k3g_land_add2", grid=[T, 4, 1], block=[224, 1, 1])]},
        }} if chunk else {}),
        **({
        # K2 + K3 KDA: the short conv in the delta rule's prologue
        "kda_core": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "in buffer<f32>",
                       "in buffer<f32>", "in buffer<f32>", "inout state", "in buffer<i32>", "i64", "out buffer<bf16>",
                       "i32", "in buffer<i32>", "i32"],
            # a batch whose blocks fit two an SM takes the build that copies the state into shared memory
            # at entry, a programmatic dependent of the projection (ours there, which lets it on at entry): its
            # cache and weight loads go out under the projection's tail; a larger one the register build, four
            # an SM (k3_kda_core.cu)
            "impl": {"launches": [
                launch("k3_kda_core", "kern_k3_kda_core", grid=[T, hl, 1], smem=smem, when={"var": T, **when},
                       defines={"KDA_CONV": 1, **(kda_defs or {}), **defs, **kda_state_defs,
                                **({"LDS": kfused, "WSM_LDS": kfused, "WSM_FA": hl} if kfused else {})}, **extra)
                for defs, smem, when, extra in (
                    ({"KDA_SMEM_STATE": 1, "ROWS_PER_ITER": 1, "KDA_PDL": 1}, HEAD_DIM * HEAD_DIM * state_bytes,
                     {"max": kda_smem_rows},
                     {"pdl": True}),
                    ({"KDA_MINB": 4, "ROWS_PER_ITER": 2}, 0, {"min": kda_smem_rows + 1}, {}))]},
        },
        "mla_attn": flavor.d_attn_op(g),
        } if decode else {}),
        "gemm_bf16": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]},
        },
        # a D group's vocab-parallel head (k3_head.cu): the slices to member 0, the members' keys exchanged
        **({"head_argmax": {
            "params": ["in buffer<f32>", "out buffer<f32>", "in buffer<u64>", "out buffer<i64>", "inout buffer<i32>",
                       "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>", "out buffer<i32>", "i32", "i32", "i64",
                       "i32"],
            "impl": {
                "scratch": {"pkey": {"dtype": "i64", "shape": [T, 16]}},
                "launches": [launch("k3_head", "kern_k3_head_argmax", grid=[T, 16, 1], block=[256, 1, 1],
                                    defines={"NRANKS": tp},
                                    params=["in buffer<f32>", "out buffer<f32>", "in buffer<u64>", "out buffer<i64>",
                                            "inout buffer<i32>", "out buffer<i64>", "inout buffer<u8>", "in buffer<u64>",
                                            "inout buffer<i32>", "out buffer<i32>", "i32", "i32", "i64", "i32"],
                                    args=[*({"param": i} for i in range(5)), {"scratch": "pkey"},
                                          *({"param": i} for i in range(5, 13))])],
            },
        }} if xchg else {}),
        # the head's argmax in one launch, the row's last block taking the max of its parts
        **({"argmax_f32": {
            "params": ["in buffer<f32>", "out buffer<i64>", "inout buffer<i32>", "i32"],
            "impl": {
                "scratch": {"pkey": {"dtype": "i64", "shape": [T, 16]}},
                "launches": [
                    launch("k3_router_argmax", "kern_k3_argmax_f32_fused", grid=[T, 16, 1], block=[256, 1, 1],
                           params=["in buffer<f32>", "out buffer<i64>", "inout buffer<i32>", "out buffer<i64>", "i32"],
                           args=[{"param": 0}, {"param": 1}, {"param": 2}, {"scratch": "pkey"}, {"param": 3}]),
                ],
            },
        }} if decode else {}),
        **({
            # P's KDA: the gather (source/k3_span_gather.cu: conv, beta / flow,
            # FlashKDA's tile prefix), then FlashKDA's kernel 1 and
            # source/k3_kda_rec.cu, which runs the recurrence from and to every
            # sequence's line (its column of the line table, its rows by
            # `cu_seqlens`), gates the output rows and advances the conv windows.
            "span_gather": {
                "params": ["in buffer<bf16>", "in buffer<f32>", "in state", "in buffer<i32>", "i64", "in buffer<bf16>",
                           "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                           "out buffer<bf16>", "in buffer<i64>", "i32", "i32", "out buffer<i32>", "out buffer<i32>"],
                "impl": {"launches": [
                    launch("k3_span_gather", "kern_k3_span_gather_packed",
                           grid=[{"add": [{"mul": [{"ceil_div": [T, 48]}, 3 * (inner_l // 1024) + 1]}, 1]}, 1, 1],
                           block=[128, 1, 1], defines={"PARTIAL_BF16": 1, "K9_LD": kfused, **kda_state_defs})]},
            },
            "flash_kda": varlen_abi.kda_op(hl, chunk_max, pack, module(varlen_abi.KDA_MODULE),
                                           module("k3_kda_rec", K12_LD=kfused, **kda_state_defs), T, "seqs"),
            # the own rows' causal pass and a round's dense pass, each writing its softmax stats
            "mla_fmha_own": (lambda mod, entry: varlen_abi.fmha_op(
                HEADS, chunk_max, chunk_max + KV_PAD, mod, T, "seqs", entry=entry, fp8=fp8_attn))(
                *fmha_kernel(trtllm_fmha_abi.MODULE, trtllm_fmha_abi.ENTRY)),
            "mla_fmha_round": (lambda mod, entry: varlen_abi.fmha_op(
                HEADS, chunk_max, window + KV_PAD, mod, T, "seqs", entry=entry, fp8=fp8_attn))(
                *fmha_kernel(FMHA_DENSE, FMHA_DENSE_ENTRY)),
            # k3_mla_glue.cu: a pass folded into the accumulator, the gate over it
            "mla_merge": {
                "params": ["in buffer<bf16>", "in buffer<f32>", "in buffer<i32>", "i32", "i32", "inout buffer<f32>",
                           "inout buffer<f32>", "i32", "i32", "i32"],
                "impl": {"launches": [launch("k3_mla_glue", "kern_k3g_mla_merge", grid=[{"mul": [T, inner_l // 2048]}, 1, 1],
                                             block=[256, 1, 1], defines=glue_defs_kv)]},
            },
            "mla_gate": {
                "params": ["out buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                           "in buffer<i32>", "in buffer<i32>", "i32", "in buffer<bf16>", "in buffer<f32>", "i32", "i32",
                           "i32"],
                "impl": {"launches": [launch("k3_mla_glue", "kern_k3g_mla_gate",
                                             grid=[{"add": [{"mul": [{"ceil_div": [T, 8]}, inner_l // 2048]},
                                                            -(-mla_short // 32) * HEADS * 4]}, 1, 1],
                                             block=[256, 1, 1], defines=glue_defs_kv)]},
            },
        } if chunk else {}),
        **({
        # K4 / K5 MLA. A head-sharded gate is this member's heads' columns: the
        # source's generic path (any gridDim.y but the fast path's 4) over that width.
        "mla_prep": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "inout state",
                       "i64", "i64", "out buffer<bf16>", "out buffer<bf16>", "i32"],
            # k3_mla_prep.cu's fast path: the norm / append block and a gate block per 4096 columns
            "impl": {"launches": [launch("k3_mla_prep", "kern_k3_mla_prep", grid=[T, 1 + -(-inner_l // 4096), 1],
                                         defines={"INNER": inner_l, "MLA_FUSED": mla_fused_l, **flavor.D_PREP})
                                  if tp > 1
                                  else launch("k3_mla_prep", "kern_k3_mla_prep")]},
        },
        # every member absorbs the whole batch's 96 heads, at most `seqs` rows
        "mla_absorb": flavor.d_absorb_op(),
        "mla_vup_gate": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_vup_gate", "kern_k3_mla_vup_gate",
                                         grid=[{"ceil_div": [T, 32]}, hl, 4], defines={"HEADS": hl})
                                  if tp > 1 else launch("k3_mla_vup_gate", "kern_k3_mla_vup_gate")]},
        },
        } if decode else {}),
        # K6 / K7
        "router_topk": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<i32>", "out buffer<f32>", "i32"],
            "impl": {"launches": [launch("k3_prefill_glue", "kern_k3g_router_topk", grid=[{"ceil_div": [T, 4]}, 1, 1],
                                         block=[128, 1, 1]) if chunk else
                                  launch("k3_router_argmax", "kern_k3_router_topk",
                                         block=[-(-EXPERTS // 256) * 256, 1, 1])]},
        },
        **({"moe_front": {
            # router top-k + latent land / mxfp8 + shared situ, then the last row's block builds the routing tables;
            # its blocks past `tokens` run the shared expert's down projection (64 output columns a block). A
            # programmatic dependent of the front GEMM (it waits at entry), as FC1 is of it.
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<i32>", "out buffer<f32>",
                       "out buffer<u8>", "out buffer<u8>", "out buffer<bf16>", *["inout buffer<i32>"] * 3,
                       *["out buffer<i32>"] * 6, "in buffer<bf16>", "out buffer<bf16>", "in buffer<u8>", "in buffer<u8>",
                       "i32", "i32"],
            "impl": {"launches": [{**launch("k3_moe_front", "kern_k3_moe_front", grid=[{"add": [T, H // 64]}, 1, 1],
                                            block=[1024, 1, 1], smem=1024 + 2 * sh_l * 128 + 64 * 64 * 2,
                                            defines={"EXPERTS": EXPERTS, "LDS": front, "SH": sh_l, "STRIDE": tp}),
                                   "pdl": True}]},
        }} if front else {}),
        **({"rms": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_land", "kern_k3_rms")]},
        }} if decode else {}),
    }
    if chunk:
        # The prep's head blocks and a short call's rows in one grid, the context gathered a round at a
        # time (k3_mla_glue.cu). A short call (`tokens` <= mla_short) attends in the latent space (the
        # decode step's absorb and FlashInfer's MLA decode kernel, each row its own causal length)
        # instead of expanding its context: the prep lays out the rows' page tables, lengths and KV
        # split plan, the gate first takes the attention's latent output through W_UV; the expansion,
        # the FMHA passes and the merges sit it out.
        short = {"when": {"var": T, "max": mla_short}}
        long_ = {"when": {"var": T, "min": mla_short + 1}} if mla_short else {}
        ops["mla_prep"] = {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "inout state", "i64",
                       "out buffer<bf16>", "in buffer<i32>", "i32", "in buffer<i32>", "i32", "i32", "out buffer<bf16>",
                       "out buffer<i32>", "out buffer<i32>", "out buffer<i32>", "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_mla_glue", "kern_k3g_mla_prep", grid=[{"add": [{"mul": [T, 2]}, 1]}, 1, 1],
                                         block=[512, 1, 1], defines=glue_defs_kv)]},
        }
        ops["mla_gather"] = {
            "params": ["in state", "i64", "in buffer<i32>", "i32", "in buffer<i32>", "i32", "i32", "out buffer<bf16>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_mla_glue", "kern_k3g_mla_gather", grid=[-(-(window + KV_PAD) // 28), 1, 1],
                                         block=[512, 1, 1], defines=glue_defs_kv, **long_)]},
        }
        ops["gemm_bf16_long"] = {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn", **long_}]},
        }
        if mx:
            ops["mxfp8_quant"] = {
                "params": ["in buffer<bf16>", "i32", "out buffer<fp8e4m3>", "out buffer<u8>", "i32", "i32"],
                "impl": {"launches": [launch("k3_mxfp8", "kern_k3_mxfp8_quant", grid=[8 * 152, 1, 1],
                                             block=[256, 1, 1])]},
            }
            ops["gemm_mxfp8"] = {
                "params": ["in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "out buffer<bf16>", "in buffer<u8>",
                           "in buffer<u8>", "i32", "i32", "i32", "i32"],
                "impl": {"launches": [{"entry": "extern:cublaslt_mxfp8_tn"}]},
            }
            ops["gemm_bf16_acc"] = {
                "params": ["in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>", "i32", "i32", "i32", "i32",
                           "i32"],
                "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn_acc"}]},
            }
        if fp8_attn:
            ops["mla_e4m3"] = {
                "params": ["in buffer<bf16>", "out buffer<fp8e4m3>", "i32", "i32"],
                "impl": {"launches": [launch("k3_mla_glue", "kern_k3g_e4m3", grid=[8 * 152, 1, 1], block=[256, 1, 1],
                                             defines=glue_defs_kv, **long_)]},
            }
        for name in ("mla_fmha_own", "mla_fmha_round", "mla_merge"):
            ops[name]["impl"]["launches"][0].update(long_)
        # the own pass's merge is the accumulator's first write
        ops["mla_merge_init"] = json.loads(json.dumps(ops["mla_merge"]).replace("inout buffer<f32>", "out buffer<f32>"))
        if mla_short:
            ops.update(flavor.p_short_ops(mla_short))
        # the head on each sequence's last row
        ops["last_rows"] = {
            "params": ["out buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "i32", "i32"],
            "impl": {"launches": [launch("copy_rows", "kern_last_rows_bf16", grid=["seqs", 1, 1], block=[1024, 1, 1])]},
        }
        ops["argmax_f32_seqs"] = {
            "params": ["in buffer<f32>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": ["seqs", 64]},
                            "pidx": {"dtype": "i32", "shape": ["seqs", 64]}},
                "launches": [
                    launch("k3_router_argmax", "kern_k3_argmax_f32_partial", var="seqs",
                           params=["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"],
                           args=[{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 2}]),
                    launch("k3_router_argmax", "kern_k3_argmax_f32_final", var="seqs",
                           params=["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                           args=[{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}]),
                ],
            },
        }
    if xchg:
        # The MLA partials change hands in one kernel (source/k3_dcp.cu): every
        # head's partial pushed into its owner's Lamport stage over NVLink, the
        # members' partials of this member's heads merged by their LSE, expanded by v-up and gated.
        tail = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "inout buffer<u8>", "in buffer<u64>",
                "inout buffer<i32>", "out buffer<i32>", "i32", "i32", "i64", "i64"]
        xg = lambda **extra: launch("k3_dcp", "kern_k3_dcp_exchange", grid=[DCP_GRID, 1, 1], block=[256, 1, 1],
                                    smem=HEAD_DIM // 2 * KV_LORA * 2, **extra)
        ops["dcp_exchange"] = flavor.d_exchange_op(g, xg, tail)
        ops["dcp_init"] = {
            "params": ["inout buffer<u8>", "i64"],
            "impl": {"launches": [launch("peer_allreduce_bf16", "kern_peer_lamport_init_bf16", defines={"NRANKS": tp},
                                         grid=[256, 1, 1], block=[256, 1, 1])]},
        }
    if xchg and not peer_ar:
        # `count` elements per rank, one NCCL launch
        io = ["in buffer<bf16>", "out buffer<bf16>", "i64"]
        ops["nccl_allreduce_bf16"] = {
            "params": io,
            "impl": {"launches": [{"entry": "extern:nccl_allreduce_bf16", "params": io + ["i32"],
                                   "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"rank": "tp"}]}]}}
    if fused_ar:
        # The layer's two sums with their neighbours (source/k3_ar_fused.cu): the attention's with the
        # landing, mix and norm after it; the MoE's with the combine before it and the latent norm after.
        lamport_params = ["inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>", "out buffer<i32>", "i32"]
        ops["ar_finalize_rms"] = {
            "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "in buffer<bf16>", *lamport_params,
                       "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i64", "i64",
                       *PF_PARAMS, *PF_PARAMS],
            "impl": {"launches": [launch("k3_ar_fused", "kern_k3_ar_finalize_rms", defines={"NRANKS": tp},
                                         grid=[TP_AR_GRID, 1, 1], block=[1024, 1, 1], pdl=True)]},
        }
        # `--peer-ar`: one kernel, every rank's partial pushed into every
        # peer's Lamport stage over NVLink (one hop, no ring), summed in f32.
        ops["tp_allreduce_bf16"] = {
            "params": ["in buffer<bf16>", "out buffer<bf16>", "inout buffer<u8>", "in buffer<u64>",
                       "inout buffer<i32>", "out buffer<i32>", "i32", "i64", "i64", "i64", "i64", "i64"],
            "impl": {"launches": [launch("peer_allreduce_bf16", "kern_peer_allreduce_bf16", defines={"NRANKS": tp},
                                         grid=[2 * TP_AR_GRID, 1, 1], block=[512, 1, 1])]},
        }
        ops["tp_lamport_init_bf16"] = {
            "params": ["inout buffer<u8>", "i64"],
            "impl": {"launches": [launch("peer_allreduce_bf16", "kern_peer_lamport_init_bf16", defines={"NRANKS": tp},
                                         grid=[256, 1, 1], block=[256, 1, 1])]},
        }

    # land / situ: grid.y depends on the width, one op per width.
    def land_op(n):
        name = f"land_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
                "impl": {"launches": [launch("k3_land", "kern_k3_land", grid=per_row(n), block=[1024, 1, 1])]},
            }
        return name

    def nb_op(base, nb):
        """k3_ar_fused.cu's residual rows with `nb` snapshots: an entry, so an op, a count. D's closing add
        (land_add2_attnres_rms_snap, a snapshot layer's too) and, with the attention's all-reduce before
        it, ar_attnres_rms."""
        name = f"{base}_nb{nb}"
        if name not in ops:
            ops[name] = {
                "land_add2_attnres_rms_snap": lambda: {
                    "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32",
                               "inout buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32",
                               "i32", *PF_PARAMS],
                    "impl": {"launches": [launch("k3_ar_fused", f"kern_k3_land_add2_attnres_rms_nb{nb}",
                                                 grid=[LAND_CL * TP_AR_GRID, 1, 1], block=[1024 // LAND_CL, 1, 1],
                                                 cluster=[LAND_CL, 1, 1], smem=CAND_SMEM // LAND_CL, pdl=True)]},
                },
                "ar_attnres_rms": lambda: {
                    "params": ["in buffer<bf16>", "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>",
                               "out buffer<i32>", "i32", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                               "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i64", "i64",
                               *PF_PARAMS, *PF_PARAMS],
                    "impl": {"launches": [launch("k3_ar_fused", f"kern_k3_ar_attnres_rms_nb{nb}",
                                                 defines={"NRANKS": tp}, grid=[TP_AR_GRID, 1, 1], block=[1024, 1, 1],
                                                 smem=CAND_SMEM, pdl=True)]},
                },
            }[base]()
        return name

    def situ_op(n):
        name = f"situ_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                "impl": {"launches": [launch("k3_prefill_glue", "kern_k3g_situ", grid=[T, -(-n // 2048), 1],
                                             block=[256, 1, 1])]},
            }
        return name

    ops.update(bp["ops"])
    if chunk and not mx:
        ops["k3_concat_cols"] = {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_moe_route", "kern_k3_concat_cols",
                                         grid=[-(-H * (LATENT + sh_l) // 8 // 256), 1, 1], block=[256, 1, 1])]},
        }

    # ---- buffers
    kv0, kda0 = f"kv.{mla_layers[0]}" if mla else None, f"kda.{kda_layers[0]}"
    buffers = {
        "token_ids": {"dtype": "i64", "shape": [ROWS if chunk else T], "kind": "input", "fill": "token",
                      "domain": {"index_into": "embed"}},
        **({"slot_mapping": {"dtype": "i64", "shape": [T], "kind": "input", "fill": "slot",
                             "domain": {"index_into": kv0}},
            "block_table": {"dtype": "i32", "shape": ["seqs", max_pages], "kind": "input",
                            "domain": {"index_into": kv0, "stride": PAGE}},
            # A D member's count of the sequence's positions it holds: 0 until the sequence reaches it.
            "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len",
                         "domain": {"min": 0 if decode else 1}}} if mla else {}),
        **({"cu_seqlens": {"dtype": "i64", "shape": [seqs_max + 1], "kind": "input", "fill": "cu_seqlens",
                           "domain": {"min": 0, "max": chunk_max, "monotone": True}}} if chunk else {}),
        # A call's KDA lines are its sequences' (one column each).
        "kda.line_index": {"dtype": "i32", "shape": [len(kda_layers), "seqs"], "kind": "input",
                           "domain": {"index_into": kda0, "stride": line_l}},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens",
                       "domain": {"index_into": "embed"}},
        **bp["buffers"],
    }
    # bytes: one Lamport stage, `tp` slots of D's widest bf16 sum: a MoE layer's, the routed latent's
    # `seqs_max` rows, then the shared expert's
    ar_stage = tp * seqs_max * (LATENT + H) * 2
    if fused_ar:
        buffers.update({
            # three stages, then the first call's handshake words (k3_ar_fused.cu), never poisoned
            "tp_ar_lamport": {"dtype": "u8", "shape": [3 * ar_stage + 64], "kind": "carry", "export": True},
            "tp_ar_lamport_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "tp_ar_lamport", "group": "tp"},
            # k3_ar_fused.cu: [6] the handshake, from [8] every block's stage and re-poison count
            "tp_ar_state": {"dtype": "i32", "shape": [8 + 2 * TP_AR_GRID], "kind": "carry"},
            "tp_err": {"dtype": "i32", "shape": [1], "kind": "output", "fill": "error"},
        })
    # D: a KV row lives on member p % tp; a KDA line is this member's heads of
    # the whole line, whose recurrent state and nine conv-window rows are each head-major.
    kv_shard = {"shard": {"group": "tp", "by": "position"}} if xchg else {}
    kda_shard = ({"shard": {"group": "tp", "by": "heads",
                            "segments": [[1, HEADS * HEAD_DIM * HEAD_DIM * state_bytes], [9, INNER * 2]]}}
                 if xchg else {})
    states = {**{f"kv.{i}": {"bytes_per_token": LATENT_ROW * flavor.BYTES, **kv_shard} for i in mla_layers},
              **{f"kda.{i}": {"bytes_per_seq": line_l, **kda_shard} for i in kda_layers}}

    def weight(name, shape, bind, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "weight", "bind": bind}

    def carry(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "carry"}

    def mxfp8(name):
        """A bf16 weight's MXFP8 copy, `name`.q and its scales `name`.sf, quantized while the weights load."""
        rows, k = buffers[name]["shape"]
        buffers[name]["kind"] = "source"
        carry(name + ".q", [rows, k], "fp8e4m3")
        carry(name + ".sf", [sf_len(rows, k)], "u8")
        derive.calls.append({"label": f"load.mxfp8.{name}", "op": "mxfp8_quant",
                             "args": [buf(name), i32(k), buf(name + ".q"), buf(name + ".sf"), i32(rows), i32(k)]})

    # This member's `per`-wide slice of a tp-split axis, or all of it.
    def shard(per):
        return {"group": "tp", "ranges": [[r * per, (r + 1) * per] for r in range(tp)]} if tp > 1 else None

    programs = {}
    once = Once({"buffers": buffers, "ops": ops, "programs": programs})
    derive = Once({"buffers": buffers, "ops": ops, "programs": programs})

    def scoring(name, tensor):
        """The folded attention-residual scoring vector f32(norm) * f32(proj)."""
        weight(name + ".norm", [H], [seg(tensor + "norm.weight")])
        weight(name + ".proj", [H], [seg(tensor + "proj.weight")])
        carry(name, [H], "f32")
        once.call("k3_scoring", [buf(name + ".norm"), buf(name + ".proj"), buf(name), i32(H)], H)

    def moe_weights(once, layer, prefix):
        """This rank's experts for the batched GEMMs: the checkpoint's mxfp4 tensors and UE8M0 scales, sources
        bound per expert (FC1 as [up; gate]) that the derive program shuffles into the GEMMs' layout while the
        weights load, so device memory never holds both; the gate's alpha and beta."""
        hf = lambda e, t: f"{HF}model.layers.{layer}.block_sparse_moe.experts.{e}.{t}"
        # Member r's j-th expert: every tp-th from r (the routing's stride).
        src = lambda t, j: ({"group": "tp", "tensors": [hf(r + tp * j, t) for r in range(tp)]} if tp > 1
                            else hf(j, t))
        raw = {"w13": [seg(src(f"{t}.weight_packed", j)) for j in range(epr) for t in ("w3", "w1")],
               "w13_sf": [seg(src(f"{t}.weight_scale", j)) for j in range(epr) for t in ("w3", "w1")],
               "w2": [seg(src("w2.weight_packed", j)) for j in range(epr)],
               "w2_sf": [seg(src("w2.weight_scale", j)) for j in range(epr)]}
        for op, name, scalars, threads in k3_moe_bmm.shuffles(epr):
            _, n, kg, _ = scalars
            weight(prefix + name + ".raw", [epr * n * kg], raw[name], "u8")
            buffers[prefix + name + ".raw"]["kind"] = "source"
            carry(prefix + name + "s", [epr * n * kg], "u8")
            derive.call(op, [buf(prefix + name + ".raw"), buf(prefix + name + "s"), *map(i32, scalars)], threads)
        for name, value in (("alpha", k3_moe_bmm.ALPHA), ("beta", k3_moe_bmm.BETA)):
            carry(prefix + name, [epr], "f32")
            once.call("fill_f32", [buf(prefix + name), i32(epr), {"f32": value}], epr)

    def gate_up(name, gate, up, per):
        """[gate; up], this member's rows of each."""
        weight(name, [2 * per, H], [seg(gate, rows=shard(per)), seg(up, rows=shard(per))])

    def work(name, width, dtype="bf16", var=T):
        buffers[name] = {"dtype": dtype, "shape": [var, width], "kind": "workspace"}

    weight("embed", [V, H], [seg(HF + "model.embed_tokens.weight")])
    weight("gamma_final", [H], [seg(HF + "model.norm.weight")])
    scoring("sw_out", HF + "model.output_attn_res_")
    weight("w_lm", [V // tp, H], [seg(HF + "lm_head.weight", rows=shard(V // tp))])
    for n in ["hidden", "prefix2", "normed"]:
        work(n, H, var=RW)
    buffers["blocks"] = {"dtype": "bf16", "shape": [RW, NB_MAX, H], "kind": "workspace"}
    if decode:
        work("hidden_partial", H)
    if xchg:
        work("o_partial", H)
    if xchg and mla:
        # the exchange's Lamport stages: a slot per member, a record of 65 vectors per (row, head) in it
        buffers.update({
            "dcp_lamport": {"dtype": "u8", "shape": [3 * dcp_stage], "kind": "carry", "export": True},
            "dcp_lamport_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "dcp_lamport", "group": "tp"},
            "dcp_state": {"dtype": "i32", "shape": [8], "kind": "carry"},
            "tp_err": {"dtype": "i32", "shape": [1], "kind": "output", "fill": "error"},
        })
    if decode:
        # the flat MoE all-reduce: the routed latent's rows at the bound, then the shared expert's
        for n in ["moe_flat", "moe_sum"] if xchg else ["moe_flat"]:
            buffers[n] = {"dtype": "bf16", "shape": [seqs_max * (LATENT + H)], "kind": "workspace"}
    if chunk:
        work("attn_part", H, var=chunk_max)
    work("kda_partial", kfused or fused_l, kpart)
    if not kfused:
        work("wsm_partial", WSM, kpart)
    wsm_part = b("kda_partial", fused_l * (2 if kpart == "bf16" else 4)) if kfused else b("wsm_partial")
    work("gated", inner_l)
    if decode and mla:
        work("mla_gate", inner_l)
    # The row a span starts at, which the KDA kernels skip past: always 0,
    # since kern feeds no span; a carry `load` zeroes.
    buffers["span_at"] = {"dtype": "i32", "shape": [1], "kind": "carry"}
    once.call("fill_i32", [buf("span_at"), i32(1), i32(0)], 1)
    if chunk:
        buffers["span_beta"] = {"dtype": "bf16", "shape": [hl, chunk_max], "kind": "workspace"}
        for n in ["span_q", "span_k", "span_v"]:
            work(n, inner_l, var=chunk_max)
        buffers["span_progress"] = {"dtype": "i32", "shape": [pack * hl * 32], "kind": "workspace"}  # a 128-byte line each
        work("span_flow", HEAD_DIM, var=chunk_max)
        work("span_g", inner_l, var=chunk_max)
        buffers.update(varlen_abi.kda_workspace(hl, chunk_max, pack))
    if mx:
        for q, k in (("normed", H), ("lat", LATENT)):
            work(q + "_q", k, "fp8e4m3", var=RW)
            buffers[q + "_sf"] = {"dtype": "u8", "shape": [sf_len(RW, k)], "kind": "workspace"}
    if mla:
        work("mla_fused_partial", mla_fused_l, kpart)
        work("q_norm", Q_LORA)
    if decode and mla:
        work("q_partial", Q_B, "f32")
        work("q_abs", HEADS * LATENT_ROW, flavor.Q_ABS)
        buffers["mla_bsk"] = {"dtype": "i32", "shape": [T], "kind": "workspace"}
        flavor.d_buffers(work, buffers, g)
    if chunk:
        work("normed_last", H, var="seqs")
        # the absorbed attention of a short call (at most mla_short rows)
        if mla_short:
            flavor.p_short_buffers(work, buffers, mla_short, max_pages)
        else:
            buffers["mla_stub"] = {"dtype": "i32", "shape": [8], "kind": "workspace"}
        work("q_bf16", Q_B)
        work("o_bf16", inner_l)
        # the chunk's own latent rows and a round's cached ones, each with their k | v expansion
        work("latent_own", KV_A, var=chunk_max + KV_PAD)
        work("kv_own", KV_EXP, var=chunk_max + KV_PAD)
        work("latent_g", KV_A, var=window + KV_PAD)
        work("kv_exp", KV_EXP, var=window + KV_PAD)
        if fp8_attn:
            work("q_fp8", Q_B, "fp8e4m3")
            work("kv_own_fp8", KV_EXP, "fp8e4m3", var=chunk_max + KV_PAD)
            work("kv_exp_fp8", KV_EXP, "fp8e4m3", var=window + KV_PAD)
        # the own pass's set and a set a round: seq_lens_kv | cum_q | cum_kv | aux, NS words each
        buffers["fmha_lens"] = {"dtype": "i32", "shape": [(1 + rounds) * 4 * NS], "kind": "workspace"}
        work("fmha_stats", 2 * HEADS, "f32")
        # the passes merged: o unnormalized and (max, sum) per row and head
        work("mla_acc", inner_l, "f32")
        work("mla_acc_ms", 2 * HEADS, "f32")
        buffers["fmha_scratch"] = {"dtype": "u8", "shape": [trtllm_fmha_abi.SCRATCH_BYTES], "kind": "workspace"}
    if front:
        work("front_partial", front, "f32", var=RW)
        buffers["moe_done"] = {"dtype": "i32", "shape": [1], "kind": "carry"}
        buffers["sh_ready"] = {"dtype": "i32", "shape": [H // 64, 64], "kind": "carry"}  # moe_front's shared-expert flags
        buffers["sh_ids"] = {"dtype": "i32", "shape": [H // 64, 64], "kind": "carry"}  # and its rows' ids, to them
    else:
        work("router_partial", EXPERTS, "f32", var=RW)
    work("topk_idx", TOPK, "i32", var=RW)
    work("topk_weight", TOPK, "f32", var=RW)
    if decode and not front:
        work("latent_partial", LATENT, "f32", var=RW)
    if chunk:
        # lat_down | the shared expert's gate | up, one GEMM's rows
        work("moe_front", LATENT + 2 * sh_l, var=RW)
    else:
        work("latent", LATENT, var=RW)
    work("latent_q", LATENT, "u8", var=RW)
    work("latent_sf", LATENT // 32, "u8", var=RW)
    if chunk:
        # the routed latent normed | the shared expert's activation, one GEMM's rows
        work("moe_back", LATENT + sh_l, var=RW)
    else:
        work("routed_latent_norm", LATENT, var=RW)
        work("shared_act", sh_l, var=RW)
    work("routed_partial", H, var=RW)
    if decode and not front:
        work("shared_partial", 2 * sh_l, var=RW)
    work("dense_partial", 2 * dn_l, var=RW)
    work("dense_act", dn_l, var=RW)
    if xchg:
        work("mlp_all", H)
    # A D step's logits are its sequences', a P call's its sequences' last rows'.
    work("logits", V, "f32", var="seqs")
    if decode:
        buffers["argmax_done"] = {"dtype": "i32", "shape": [seqs_max], "kind": "carry"}
    if xchg:
        # every member's slice of the logits lands in member 0's whole row, which the caller reads;
        # the members' best keys change hands through 3 stages of [tp, 64] u64
        buffers["logits"] = {"dtype": "f32", "shape": [seqs_max, V], "kind": "carry", "export": True}
        buffers["logits_peers"] = {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "logits", "group": "tp"}
        work("logits_part", V // tp, "f32", var="seqs")
        buffers["head_keys"] = {"dtype": "u8", "shape": [3 * tp * 64 * 8], "kind": "carry", "export": True}
        buffers["head_keys_peers"] = {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "head_keys",
                                      "group": "tp"}
        buffers["head_state"] = {"dtype": "i32", "shape": [8], "kind": "carry"}

    # ---- program
    prog = []

    def step(label, op, *args):
        prog.append({"label": label, "op": op, "args": list(args)})

    def f32_op(n, k):
        """D's own split-K GEMM (gemm_t_op) for its shapes, else cuBLAS."""
        return f"gemm_t{n}" if f"gemm_t{n}" in ops and k == H else "gemm_f32"

    def gemm(label, a, w, c, n, k, m=B):
        """c[m, n] = a[m, k] @ w[n, k]^T, f32."""
        step(label, f32_op(n, k), a, w, c, m, i32(n), i32(k), i32(n))

    # D's bf16 projections run over the rows rounded up to 8 (rows past `tokens` are scratch nobody reads): cuBLAS's
    # 1-7-row kernel reads a weight already in L2 at 0.6-0.7x its 8-row rate, and every split-K 1 kernel gives a row
    # the same bits (work/h/m1.cu)
    rows8 = dim({"mul": [{"ceil_div": [T, 8]}, 8]}) if xchg and seqs_max % 8 == 0 else B

    def proj(label, a, w, c, n, k, m=B, dt="bf16"):
        """A projection whose readers land it: the GEMM lands it (bf16) in P and for D's sums, else f32
        (`dt` the readers' dtype)."""
        if dt == "bf16" and m is B:
            m = rows8
        step(label, "gemm_bf16" if dt == "bf16" else f32_op(n, k), a, w, c, m, i32(n), i32(k), i32(n))

    def mx_gemm(label, q, a, ld, w, name, c, n, k, ldc):
        """`a` (rows `ld` apart) to MXFP8 in `q`_q / `q`_sf, then C[:, :n] (rows `ldc` apart) = a · `name`ᵀ."""
        step(label + "_mx", "mxfp8_quant", a, i32(ld), b(q + "_q"), b(q + "_sf"), B, i32(k))
        step(label, "gemm_mxfp8", b(q + "_q"), w(name + ".q"), c, b(q + "_sf"), w(name + ".sf"), B, i32(n), i32(k),
             i32(ldc))

    def land(label, p, o, n, off, ldc):
        step(label, land_op(n), p, o, i32(n), i32(off), i32(ldc), B)

    def land_situ(label, p, act, n):
        step(label, situ_op(n), p, act, i32(n), B)

    def lamport():
        return b("tp_ar_lamport"), b("tp_ar_lamport_peers"), b("tp_ar_state"), b("tp_err"), {"rank": "tp"}

    def summed(label, partial, whole, count, second=None):
        """A D group's bf16 sum of `partial` in `whole` by NCCL or the Lamport one-shot; `partial` itself
        alone. `second` (at, count): a second run of elements, summed by the one-shot in the same
        exchange; NCCL sums everything up to its end."""
        if not xchg:
            return partial
        at2, n2 = second or (0, 0)
        if peer_ar:
            step(label, "tp_allreduce_bf16", partial, whole, b("tp_ar_lamport"), b("tp_ar_lamport_peers"),
                 b("tp_ar_state"), b("tp_err"), {"rank": "tp"}, {"expr": count}, i64(at2), {"expr": n2} if second else i64(0),
                 i64(ar_stage), i64(TP_TIMEOUT_NS))
        else:
            step(label, "nccl_allreduce_bf16", partial, whole, {"expr": {"add": [at2, n2]} if second else count})
        return whole

    def kda_run(L, w, line, kda):
        """P's KDA layer: conv taps + beta/flow gathered (K9), g by one GEMM, FlashKDA and the recurrence
        over every sequence, the output gated (K11)."""
        N = {"var": "seqs"}
        step(L + "span_gather", "span_gather", b("kda_partial"), w("cw"), {"state": kda}, line, i64(line_l),
             wsm_part, b("span_q"), b("span_k"), b("span_v"), b("span_beta"), b("span_flow"),
             b("cu_seqlens"), N, B, b("span_tile_prefix"), b("span_progress"))
        step(L + "span_g", "gemm_bf16", b("span_flow"), w("w_f_b"), b("span_g"), B, i32(inner_l), i32(HEAD_DIM),
             i32(inner_l))
        step(L + "span_kda", "flash_kda", b("span_q"), b("span_k"), b("span_v"), b("span_g"), b("span_beta"),
             w("dt_bias"), w("a_log"),
             *(b(n) for n in ["span_ws_kd", "span_ws_qd", "span_ws_kr", "span_ws_gt", "span_ws_inv", "span_ws_mqk"]),
             B, b("cu_seqlens"), b("span_tile_prefix"), N, {"state": kda}, line, i64(line_l), b("kda_partial"),
             w("gamma_o"), b("gated"), b("span_at"), b("span_progress"))

    def emit():
        blocks, kda_k = 0, 0
        # A layer's closing landing add (label, p1, p2, two), for the next
        # layer's mix to fuse; P's snapshot layer, where a stage cut falls,
        # takes it on its own. ("ar", label, partial): D's dense MLP down
        # projection, left to the next mix's all-reduce.
        closing = None

        def summed_closing():
            """An "ar" closing summed by itself."""
            nonlocal closing
            if closing[0] == "ar":
                _, label, partial = closing
                mlp = summed(label.replace("hidden", "reduce_mlp"), partial, b("mlp_all"), {"mul": [T, H]})
                closing = (label, mlp, mlp, 0)

        def close():
            nonlocal closing
            label, p1, p2, two = closing
            step(label, "land_add2", p1, p2, b("prefix2"), b("hidden"), i32(two), B)
            closing = None

        def mix_in(label, sw, gamma, nb, snapshot=0):
            """The closing add fused into the next mix; D's takes a snapshot layer's too."""
            nonlocal closing
            if closing[0] == "ar" and not snapshot:
                # the sum, the closing add (K1b's, no snapshot) and the mix in one launch
                step(label, nb_op("ar_attnres_rms", nb), closing[2], *lamport(), b("prefix2"), b("blocks"), sw, gamma,
                     b("hidden"), b("normed"), i32(0), B, i64(ar_stage), i64(TP_TIMEOUT_NS))
                closing = None
                return
            summed_closing()
            _, p1, p2, two = closing
            if decode:
                step(label, nb_op("land_add2_attnres_rms_snap", nb), p1, p2, b("prefix2"), b("hidden"), i32(two),
                     b("blocks"), sw, gamma, b("normed"), i32(snapshot), B)
            else:
                step(label, "land_add2_attnres_rms", p1, p2, b("prefix2"), b("hidden"), i32(two), b("blocks"), sw,
                     gamma, b("normed"), i32(nb), B)
            closing = None

        for i in layers:
            L = f"l{i}."
            w = lambda n, off=0, i=i: b(f"layers.{i}.{n}", off)
            snapshot = i % ATTN_RES_BLOCK == 0
            nb_in = blocks
            if snapshot:
                blocks += 1
            nb_mlp = blocks

            # residual mix in + snapshot + norm → normed
            if i == 0 and chunk:
                step(L + "res_in", "embed_rms_setup", b("token_ids"), b("embed"), b("hidden"), b("blocks"),
                     w("gamma_in"), b("normed"), b("seq_lens"), b("cu_seqlens"), b("fmha_lens"), {"var": "seqs"},
                     i32(NS), i32(mla_short), i32(window), i32(rounds), b("moe.route_sync"),
                     i32(bp["route_sync_words"]), B)
            elif i == 0 and decode and mla:
                step(L + "res_in", "embed_rms_plan", b("token_ids"), b("embed"), b("hidden"), b("blocks"),
                     w("gamma_in"), b("normed"), b("seq_lens"), b("mla_bsk"), i32(mla_split_max), B)
            elif i == 0:
                step(L + "res_in", "embed_rms", b("token_ids"), b("embed"), b("hidden"), b("blocks"), w("gamma_in"),
                     b("normed"), B)
            elif chunk and snapshot:
                close()
                step(L + "res_in", "attnres_rms_setup", b("hidden"), b("blocks"), w("sw_attn"), w("gamma_in"),
                     b("normed"), i32(nb_in), i32(1), b("moe.route_sync"), i32(bp["route_sync_words"]), B)
            else:
                mix_in(L + "res_in", w("sw_attn"), w("gamma_in"), nb_in, int(snapshot))
            if is_mla(i):
                kv = f"kv.{i}"
                if mx:
                    mx_gemm(L + "wfu", "normed", b("normed"), H, w, "wfu", b("mla_fused_partial"), mla_fused_l, H,
                            mla_fused_l)
                else:
                    proj(L + "wfu", b("normed"), w("wfu"), b("mla_fused_partial"), mla_fused_l, H, dt=kpart)
                if chunk:
                    # q_norm and the latent row appended, the chunk's own latent rows expanded to k | v;
                    # a short call's q folded into the latent space, the attention over the latent pages
                    # row by row (each row its own causal length); a long call's own rows through the
                    # causal FMHA, then its context a round at a time (gathered, expanded while still in
                    # L2, the dense FMHA), every pass merged into the accumulator; the gate (a short
                    # call's out through W_UV first)
                    # without the absorbed path (fp8 pages) its tables and partials are stand-ins nobody touches
                    s_table, s_lens, s_bsk, s_acc_o, s_acc_lse = (
                        flavor.p_short_args() if mla_short else
                        (b("mla_stub"), b("mla_stub"), b("mla_stub"), b("mla_acc"), b("mla_acc_ms")))
                    step(L + "prep", "mla_prep", b("mla_fused_partial"), w("gamma_q_a"), w("gamma_kv_a"),
                         b("slot_mapping"), {"state": kv, "offset": 0}, i64(page_stride), b("q_norm"),
                         b("block_table"), i32(max_pages), b("fmha_lens"), i32(NS), {"var": "seqs"},
                         b("latent_own"), s_table, s_lens, s_bsk, i32(MLA_SHORT_SPLITS), i32(mla_short), B)
                    step(L + "expand_own", "gemm_bf16_long", b("latent_own"), w("w_aug"), b("kv_own"),
                         dim({"add": [T, KV_PAD]}), i32(KV_EXP), i32(KV_A), i32(KV_EXP))
                    step(L + "q_b", "gemm_bf16", b("q_norm"), w("w_q_b"), b("q_bf16"), B, i32(Q_B), i32(Q_LORA),
                         i32(Q_B))
                    if fp8_attn:
                        step(L + "own_cast", "mla_e4m3", b("kv_own"), b("kv_own_fp8"), dim({"add": [T, KV_PAD]}),
                             i32(KV_EXP))
                        step(L + "q_cast", "mla_e4m3", b("q_bf16"), b("q_fp8"), B, i32(Q_B))
                    if mla_short:
                        flavor.p_short_steps(step, w, L, kv, max_pages)
                    lens = lambda k, row: b("fmha_lens", (k * 4 + row) * NS * 4)

                    def fmha(label, op, kv_buf, k):
                        kv_buf += attn_sfx
                        step(label, op, b(attn_q), b(kv_buf), b(kv_buf, trtllm_fmha_abi.HQK * attn_bytes),
                             b("o_bf16"),
                             lens(k, 0), lens(k, 1), lens(k, 2), b("fmha_scratch"),
                             b("fmha_scratch", trtllm_fmha_abi.PARTIAL_O_OFFSET), b("fmha_stats"))
                        step(label + "_merge", "mla_merge_init" if k == 0 else "mla_merge", b("o_bf16"), b("fmha_stats"),
                             lens(k, 0), i32(NS), {"var": "seqs"}, b("mla_acc"), b("mla_acc_ms"), i32(int(k == 0)),
                             i32(inner_l), B)
                    fmha(L + "attn_own", "mla_fmha_own", "kv_own", 0)
                    for r in range(rounds):
                        step(L + f"gather{r}", "mla_gather", {"state": kv, "offset": 0}, i64(page_stride),
                             b("block_table"), i32(max_pages), b("fmha_lens"), i32(NS), {"var": "seqs"}, b("latent_g"),
                             i32(r), i32(window), dim(round_rows(r)))
                        step(L + f"expand{r}", "gemm_bf16_long", b("latent_g"), w("w_aug"), b("kv_exp"),
                             dim(round_rows(r)), i32(KV_EXP), i32(KV_A), i32(KV_EXP))
                        if fp8_attn:
                            step(L + f"cast{r}", "mla_e4m3", b("kv_exp"), b("kv_exp_fp8"), dim(round_rows(r)),
                                 i32(KV_EXP))
                        fmha(L + f"attn{r}", "mla_fmha_round", "kv_exp", 1 + r)
                    step(L + "gate", "mla_gate", b("gated"), b("mla_acc"), b("mla_fused_partial"), s_acc_o, s_acc_lse,
                         s_lens, s_bsk, i32(MLA_SHORT_SPLITS), w("w_kv_b"), b("mla_acc_ms"), i32(inner_l),
                         i32(mla_short), B)
                else:
                    step(L + "mla_prep", "mla_prep", b("mla_fused_partial"), w("gamma_q_a"), w("gamma_kv_a"),
                         b("slot_mapping"), {"state": kv}, i64(0), i64(page_stride), b("q_norm"), b("mla_gate"), B)
                    gemm(L + "q_b", b("q_norm"), w("w_q_b"), b("q_partial"), Q_B, Q_LORA)
                    step(L + "absorb", "mla_absorb", b("q_partial"), w("w_kv_b"), b("q_abs"), B)
                    flavor.d_attn_steps(step, w, L, kv, g)
            else:
                line = b("kda.line_index", kda_k * seqs_max * 4)
                kda_k += 1
                if mx:
                    mx_gemm(L + "qkvg", "normed", b("normed"), H, w, "wkda", b("kda_partial"), kfused, H, kfused)
                elif kfused:
                    proj(L + "qkvg", b("normed"), w("wkda"), b("kda_partial"), kfused, H, dt=kpart)
                else:
                    proj(L + "qkvg", b("normed"), w("wbig"), b("kda_partial"), fused_l, H, dt=kpart)
                    proj(L + "wsm", b("normed"), w("wsm"), b("wsm_partial"), WSM, H, dt=kpart)
                if chunk:
                    kda_run(L, w, line, f"kda.{i}")
                else:
                    step(L + "kda_core", "kda_core", b("kda_partial"), w("cw"), wsm_part, w("w_f_b"), w("dt_bias"),
                         w("a_log"), w("gamma_o"), {"state": f"kda.{i}"}, line, i64(line_l), b("gated"), B,
                         b("span_at"), i32(0))
            # this member's heads' o_proj, bf16, summed over a D group
            o = b("attn_part") if chunk else b("o_partial" if xchg else "hidden_partial")
            proj(L + "o_proj", b("gated"), w("w_o"), o, H, inner_l)
            # attn_out landing + residual (or snapshot replace) + mix + norm → prefix2, normed
            if fused_ar:
                step(L + "res_mlp", nb_op("ar_attnres_rms", nb_mlp), o, *lamport(), b("hidden"), b("blocks"),
                     w("sw_mlp"), w("gamma_post"), b("prefix2"), b("normed"), i32(int(snapshot)), B,
                     i64(ar_stage), i64(TP_TIMEOUT_NS))
            else:
                attn_out = summed(L + "reduce_attn", o, b("hidden_partial"), {"mul": [T, H]})
                step(L + "res_mlp", "land_add_attnres_rms_bf16", attn_out, b("hidden"), b("blocks"), w("sw_mlp"),
                     w("gamma_post"), b("prefix2"), b("normed"), i32(nb_mlp), i32(int(snapshot)), B)

            if i == 0:
                # the dense FFN, column-sharded over a D group (gate / up rows, down columns), the down
                # projection's partials summed
                proj(L + "wgu", b("normed"), w("wgu"), b("dense_partial"), 2 * dn_l, H)
                land_situ(L + "situ", b("dense_partial"), b("dense_act"), dn_l)
                proj(L + "w_dn", b("dense_act"), w("w_dn"), b("routed_partial"), H, dn_l)
                if fused_ar:
                    closing = ("ar", L + "hidden", b("routed_partial"))
                else:
                    mlp = summed(L + "reduce_mlp", b("routed_partial"), b("mlp_all"), {"mul": [T, H]})
                    closing = (L + "hidden", mlp, mlp, 0)
            elif decode:
                if front:
                    gemm(L + "front", b("normed"), w("w_front"), b("front_partial"), front, H)
                    step(L + "moe_front", "moe_front", b("front_partial"), w("bias"), w("rs"), b("topk_idx"),
                         b("topk_weight"), b("latent_q"), b("latent_sf"), b("shared_act"), b("sh_ready"),
                         b("sh_ids"), b("moe_done"), *(b("moe." + n) for n in ("cta_batch", "cta_limit", "num_non_exiting",
                                                                  "total_padded", "route_map", "exp2perm")),
                         w("sh_down"), b("moe_flat", seqs_max * LATENT * 2), w("moe.w13s"), w("moe.w13_sfs"),
                         {"rank": "tp"}, B)
                else:
                    gemm(L + "router", b("normed"), w("w_router"), b("router_partial"), EXPERTS, H)
                    step(L + "topk", "router_topk", b("router_partial"), w("bias"), w("rs"), b("topk_idx"),
                         b("topk_weight"), B)
                    gemm(L + "lat_down", b("normed"), w("w_lat_down"), b("latent_partial"), LATENT, H)
                    land(L + "latent", b("latent_partial"), b("latent"), LATENT, 0, LATENT)
                    step(L + "moe_quant", "moe_quant", b("latent"), b("latent_q"), b("latent_sf"), B, i32(LATENT))
                # Every row through this member's experts, the routed latent's partial and the shared
                # expert's side by side in one buffer, one bf16 all-reduce of both (rows past
                # `tokens` of the latent's block are the combine's zeros), the latent normed, up.
                prog.extend(bp["steps"](b("latent_q"), b("latent_sf"), b("topk_idx"), b("topk_weight"),
                                        w("moe.w13s"), w("moe.w13_sfs"), w("moe.w2s"), w("moe.w2_sfs"),
                                        w("moe.alpha"), w("moe.beta"), b("moe_flat"), {"rank": "tp"}, label=L,
                                        routed=bool(front), combined=not fused_ar))
                shared_at = seqs_max * LATENT * 2
                if not front:
                    proj(L + "wsh", b("normed"), w("wsh"), b("shared_partial"), 2 * sh_l, H)
                    land_situ(L + "shared_situ", b("shared_partial"), b("shared_act"), sh_l)
                if not front:
                    proj(L + "sh_down", b("shared_act"), w("sh_down"), b("moe_flat", shared_at), H, sh_l)
                if fused_ar:
                    # the combine, the sum of both and the latent norm in one launch; fc2's partial read
                    # before the grid wait unless fc2 is the kernel right before (it lets its dependents
                    # launch before its output is out)
                    early = b("moe.fc2_out") not in prog[-1]["args"]
                    step(L + "reduce_mlp", "ar_finalize_rms", b("moe.fc2_out"), b("moe.exp2perm"),
                         b("topk_weight"), b("moe_flat", shared_at), *lamport(), w("gamma_lat"),
                         b("routed_latent_norm"), b("moe_sum", shared_at), B, i32(int(early)), i64(ar_stage),
                         i64(TP_TIMEOUT_NS))
                    moe = "moe_sum"
                else:
                    # the routed latent's `tokens` rows, then the shared expert's
                    moe = summed(L + "reduce_mlp", b("moe_flat"), b("moe_sum"), {"mul": [T, LATENT]},
                                 (seqs_max * LATENT, {"mul": [T, H]}))["buf"]
                    step(L + "lat_norm", "rms", b(moe), w("gamma_lat"), b("routed_latent_norm"), i32(LATENT), B)
                proj(L + "lat_up", b("routed_latent_norm"), w("w_lat_up"), b("routed_partial"), H, LATENT)
                closing = (L + "hidden", b("routed_partial"), b(moe, shared_at), 1)
            else:
                # lat_down and the shared expert's gate | up one GEMM; the top-k, the latent's mxfp8, the
                # routing tables and the shared activation one call; the combine lands its row and norms
                # it (lat_norm) in one pass, next to the shared activation; lat_up and the shared down
                # projection one GEMM over both (K = 3584 + 6144), one partial
                gemm(L + "router", b("normed"), w("w_router"), b("router_partial"), EXPERTS, H)
                back = LATENT + sh_l
                if mx:
                    mx_gemm(L + "lat_down", "normed", b("normed"), H, w, "w_lat_down", b("moe_front"), LATENT, H,
                            LATENT + 2 * sh_l)
                    step(L + "sh_gu", "gemm_bf16", b("normed"), w("w_sh_gu"), b("moe_front", LATENT * 2), B,
                         i32(2 * sh_l), i32(H), i32(LATENT + 2 * sh_l))
                else:
                    proj(L + "front", b("normed"), w("w_moe_front"), b("moe_front"), LATENT + 2 * sh_l, H)
                prog.append(bp["route_step"](b("router_partial"), w("bias"), w("rs"), b("moe_front"),
                                             LATENT + 2 * sh_l, b("moe_front", LATENT * 2), LATENT + 2 * sh_l,
                                             b("latent_q"), b("latent_sf"), b("topk_idx"), b("topk_weight"),
                                             b("moe_back", LATENT * 2), back, label=L))
                prog.extend(bp["steps"](b("latent_q"), b("latent_sf"), b("topk_idx"), b("topk_weight"),
                                        w("moe.w13s"), w("moe.w13_sfs"), w("moe.w2s"), w("moe.w2_sfs"),
                                        w("moe.alpha"), w("moe.beta"), b("moe_back"),
                                        gamma=w("gamma_lat"), label=L, routed=True, ldo=back))
                if mx:
                    mx_gemm(L + "lat_up", "lat", b("moe_back"), back, w, "w_lat_up", b("routed_partial"), H, LATENT, H)
                    step(L + "sh_down", "gemm_bf16_acc", b("moe_back", LATENT * 2), w("sh_down"), b("routed_partial"),
                         B, i32(H), i32(sh_l), i32(H), i32(back))
                else:
                    proj(L + "back", b("moe_back"), w("w_moe_back"), b("routed_partial"), H, back)
                closing = (L + "hidden", b("routed_partial"), b("routed_partial"), 0)
        assert blocks == blocks_total
        mix_in("out.res", b("sw_out"), b("gamma_final"), blocks_total)
        if chunk:
            step("out.last", "last_rows", b("normed_last"), b("normed"), b("cu_seqlens"), i32(H), i32(H))
            gemm("out.lm_head", b("normed_last"), b("w_lm"), b("logits"), V, H, m={"var": "seqs"})
            step("out.argmax", "argmax_f32_seqs", b("logits"), b("next_token"), i32(V))
        elif xchg:
            gemm("out.lm_head", b("normed"), b("w_lm"), b("logits_part"), V // tp, H)
            step("out.argmax", "head_argmax", b("logits_part"), b("logits"), b("logits_peers"), b("next_token"),
                 b("argmax_done"), b("head_keys"), b("head_keys_peers"), b("head_state"), b("tp_err"),
                 {"rank": "tp"}, i32(V // tp), i64(TP_TIMEOUT_NS), B)
        else:
            gemm("out.lm_head", b("normed"), b("w_lm"), b("logits"), V, H)
            step("out.argmax", "argmax_f32", b("logits"), b("next_token"), b("argmax_done"), i32(V))
        return prog

    for i in layers:
        q = f"{HF}model.layers.{i}."
        a, n = q + "self_attn.", f"layers.{i}."
        weight(n + "gamma_in", [H], [seg(q + "input_layernorm.weight")])
        weight(n + "gamma_post", [H], [seg(q + "post_attention_layernorm.weight")])
        if i > 0:
            scoring(n + "sw_attn", q + "self_attention_res_")
        scoring(n + "sw_mlp", q + "mlp_res_")
        if is_mla(i):
            # every head's q side; this member's heads of the o side (g_proj, kv_b's v half, o_proj)
            weight(n + "wfu", [mla_fused_l, H],
                   [seg(a + "q_a_proj.weight"), seg(a + "kv_a_proj_with_mqa.weight"),
                    seg(a + "g_proj.weight", rows=shard(inner_l))])
            if mx:
                mxfp8(n + "wfu")
            weight(n + "gamma_q_a", [Q_LORA], [seg(a + "q_a_layernorm.weight")])
            weight(n + "gamma_kv_a", [KV_LORA], [seg(a + "kv_a_layernorm.weight")])
            weight(n + "w_q_b", [Q_B, Q_LORA], [seg(a + "q_b_proj.weight")])
            weight(n + "w_kv_b", [HEADS * 256, KV_LORA], [seg(a + "kv_b_proj.weight")])
            if xchg:
                # the v-up's heads: this member's of kv_b, whole for the absorb above
                weight(n + "w_kv_b_l", [hl * 256, KV_LORA], [seg(a + "kv_b_proj.weight", rows=shard(hl * 256))])
            if chunk:
                carry(n + "w_aug", [KV_EXP, KV_A])
                once.call("k3_kvb_aug", [buf(n + "w_kv_b"), buf(n + "w_aug"), i32(KV_EXP * KV_A)], KV_EXP * KV_A)
            weight(n + "w_o", [H, inner_l], [seg(a + "o_proj.weight", cols=shard(inner_l))])
        else:
            # This member's heads of every per-head axis.
            qkvg = [seg(a + f"{x}_proj.weight", rows=shard(inner_l)) for x in "qkvg"]
            if kfused:
                weight(n + "wkda", [kfused, H],
                       qkvg + [seg(a + "b_proj.weight", rows=shard(hl)), seg(a + "f_a_proj.weight"),
                               seg(a + "f_a_proj.weight", rows=[0, WSM - hl - HEAD_DIM])])
                if mx:
                    mxfp8(n + "wkda")
            else:
                weight(n + "wbig", [fused_l, H], qkvg)
                weight(n + "wsm.b", [hl, H], [seg(a + "b_proj.weight", rows=shard(hl))])
                weight(n + "wsm.f_a", [HEAD_DIM, H], [seg(a + "f_a_proj.weight")])
                carry(n + "wsm", [WSM, H])
                once.call("k3_wsm", [buf(n + "wsm.b"), buf(n + "wsm.f_a"), buf(n + "wsm"), i32(hl), i32(H)], WSM * H)
            weight(n + "w_f_b", [inner_l, HEAD_DIM], [seg(a + "f_b_proj.weight", rows=shard(inner_l))])
            for x in "qkv":
                weight(n + f"cw.{x}", [inner_l, 4], [seg(a + f"{x}_conv1d.weight", rows=shard(inner_l))], "f32")
            carry(n + "cw", [3, 4, inner_l], "f32")
            once.call("k3_conv_taps", [buf(n + "cw.q"), buf(n + "cw.k"), buf(n + "cw.v"), buf(n + "cw"), i32(inner_l)],
                      inner_l)
            weight(n + "dt_bias", [inner_l], [seg(a + "dt_bias", cols=shard(inner_l))], "f32")
            weight(n + "a_log", [hl], [seg(a + "A_log", cols=shard(hl) or [0, HEADS])], "f32")
            weight(n + "gamma_o", [HEAD_DIM], [seg(a + "o_norm.weight")], "f32")
            weight(n + "w_o", [H, inner_l], [seg(a + "o_proj.weight", cols=shard(inner_l))])
        if i == 0:
            d = q + "mlp."
            gate_up(n + "wgu", d + "gate_proj.weight", d + "up_proj.weight", dn_l)
            weight(n + "w_dn", [H, dn_l], [seg(d + "down_proj.weight", cols=shard(dn_l))])
        else:
            e = q + "block_sparse_moe."
            if front:
                weight(n + "w_front", [front, H],
                       [seg(e + "gate.weight"), seg(e + "routed_expert_down_proj.weight"),
                        seg(e + "shared_experts.gate_proj.weight", rows=shard(sh_l)),
                        seg(e + "shared_experts.up_proj.weight", rows=shard(sh_l))])
            else:
                weight(n + "w_router", [EXPERTS, H], [seg(e + "gate.weight")])
            weight(n + "bias", [EXPERTS], [seg(e + "gate.e_score_correction_bias")], "f32")
            carry(n + "rs", [1])
            once.call("fill_bf16", [buf(n + "rs"), i32(1), {"f32": 1.0}], 1)
            if chunk and mx:
                weight(n + "w_lat_down", [LATENT, H], [seg(e + "routed_expert_down_proj.weight")])
                mxfp8(n + "w_lat_down")
                gate_up(n + "w_sh_gu", e + "shared_experts.gate_proj.weight", e + "shared_experts.up_proj.weight", sh_l)
            elif chunk:
                weight(n + "w_moe_front", [LATENT + 2 * sh_l, H],
                       [seg(e + "routed_expert_down_proj.weight"), seg(e + "shared_experts.gate_proj.weight"),
                        seg(e + "shared_experts.up_proj.weight")])
            elif not front:
                weight(n + "w_lat_down", [LATENT, H], [seg(e + "routed_expert_down_proj.weight")])
            weight(n + "gamma_lat", [LATENT], [seg(e + "routed_expert_norm.weight")])
            if decode and not front:
                gate_up(n + "wsh", e + "shared_experts.gate_proj.weight", e + "shared_experts.up_proj.weight", sh_l)
            if chunk and mx:
                weight(n + "w_lat_up", [H, LATENT], [seg(e + "routed_expert_up_proj.weight")])
                mxfp8(n + "w_lat_up")
                weight(n + "sh_down", [H, sh_l], [seg(e + "shared_experts.down_proj.weight")])
            elif chunk:
                # [lat_up | shared down] by columns, made while the weights load
                for name, t, k in (("w_lat_up", "routed_expert_up_proj", LATENT), ("sh_down", "shared_experts.down_proj", sh_l)):
                    weight(n + name, [H, k], [seg(e + t + ".weight")])
                    buffers[n + name]["kind"] = "source"
                carry(n + "w_moe_back", [H, LATENT + sh_l])
                derive.calls.append({"label": f"load.k3_concat_cols.{i}", "op": "k3_concat_cols",
                                     "args": [buf(n + "w_lat_up"), buf(n + "sh_down"), buf(n + "w_moe_back"), i32(H),
                                              i32(LATENT), i32(sh_l)]})
            else:
                weight(n + "w_lat_up", [H, LATENT], [seg(e + "routed_expert_up_proj.weight")])
                weight(n + "sh_down", [H, sh_l], [seg(e + "shared_experts.down_proj.weight", cols=shard(sh_l))])
            moe_weights(once, i, n + "moe.")

    def prefetch_next(prog):
        """Every PF_OPS call names the weight of the GEMM after it (up to PF_MAX bytes) for L2; ar_finalize_rms
        also the next PF_LAST GEMM's, at evict_last, and the ar_attnres_rms after that GEMM the same range to
        put back to evict_normal. Our own GEMM (gemm_t*) for front pins the layer's sh_down weight the same way
        (for after the MoE), and the next of our GEMMs demotes it."""
        gemms = [(k, c) for k, c in enumerate(prog) if c["op"].startswith("gemm_")]
        size = lambda w: math.prod(buffers[w["buf"]]["shape"]) * {"bf16": 2, "f32": 4, "u8": 1}[
            buffers[w["buf"]]["dtype"]] - w.get("offset", 0)
        last = {}
        pinned = first = None  # the sh_down weight our front GEMM pinned, for the next of our GEMMs to demote
        for k, c in enumerate(prog):
            if c["op"].startswith("gemm_t"):
                j, g = next(((j, g) for j, g in gemms if j > k and g["label"].endswith(".sh_down")), (None, None))
                pin = g["args"][1] if g and c["label"].endswith(".front") else None
                own = c["args"][1]
                c["args"] += [pin or own, i64(size(pin) if pin else 0), pinned or own, i64(size(pinned) if pinned else 0)]
                pinned, first = pin, first or c
            if base_op(c["op"]) in PF_OPS:
                w = next(g["args"][1] for j, g in gemms if j > k)
                c["args"] += [w, i64(min(size(w), LAND_PF_MAX if c["op"].startswith("land") else PF_MAX))]
            if c["op"] == "ar_finalize_rms":
                j, g = next(((j, g) for j, g in gemms if j > k and g["label"].endswith(PF_LAST)), (None, None))
                c["args"] += [g["args"][1], i64(size(g["args"][1]))] if g else [w, i64(0)]
                if g:
                    last[j] = g["args"][1]
            if base_op(c["op"]) == "ar_attnres_rms":
                j = max(j for j, _ in gemms if j < k)
                c["args"] += [last[j], i64(size(last[j]))] if j in last else [w, i64(0)]
        if pinned:  # the step's last pin: the first of our GEMMs demotes it in the next replay
            first["args"][-2:] = [pinned, i64(size(pinned))]
        return prog

    if chunk:
        programs["prefill"] = kern_manifest.program(emit(), groups=seqs_max, rows=T, context=CTX)
    else:
        programs["decode"] = kern_manifest.program(prefetch_next(emit()), groups=seqs_max, rows=1, graph=True)
    # Run once after the peers are imported: the Lamport stages must read
    # -0.0 before the first allreduce, and a carry starts at zero.
    init = ([{"label": "tp_init", "op": "tp_lamport_init_bf16", "args": [b("tp_ar_lamport"), i64(3 * ar_stage)]}]
            if fused_ar else []) + (
        [{"label": "dcp_init", "op": "dcp_init", "args": [b("dcp_lamport"), i64(3 * dcp_stage)]}] if xchg and mla else [])
    if init:
        programs["tp_init"] = kern_manifest.program(init, once=True)
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"kimi-k3/{end}l/ep{tp}" + (f"-tp{tp}" if tp > 1 else "")
                 + ("-dcp" if decode else "-prefill"),
        "vars": {T: {"max": t_max, "axis": "rows"}, "seqs": {"max": seqs_max, "axis": "groups"},
                 **({ROWS: {"max": chunk_max, "axis": "ranks"}, CTX: {"max": max_ctx}} if chunk else {})},
        **({"topology": {"groups": {"tp": tp}}} if tp > 1 else {}),
        "states": states,
        "buffers": buffers,
        "ops": ops,
        "programs": programs,
    }
    once.finish()
    if derive.calls:
        derive.finish("derive", derive=True)
    # A manifest keeps the ops its programs call ...
    called = {c["op"] for p in programs.values() for c in p["calls"]}
    m["ops"] = {k: v for k, v in ops.items() if k in called}
    # ... and the workspaces they name.
    named = set()
    walk = lambda a: (named.add(a["buf"]) if "buf" in a else None) if isinstance(a, dict) else None
    for p in programs.values():
        for c in p["calls"]:
            for a in c["args"]:
                walk(a)
    m["buffers"] = {k: v for k, v in m["buffers"].items() if v["kind"] != "workspace" or k in named}
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=LAYERS, help="layers [0, N), with the embedding and the head")
    ap.add_argument("--chunk", type=int, default=0, help="P: the rows a prefill call packs at most (0: D)")
    ap.add_argument("--pack", type=int, default=16, help="P: the sequences a call packs back to back, split by "
                                                         "`cu_seqlens`")
    ap.add_argument("--mla-short", type=int, default=128,
                    help="P: a call of at most this many rows attends in the latent space (absorbed MLA) "
                         "instead of expanding its context")
    ap.add_argument("--tp", type=int, default=1,
                    help="D: the group the batch is replicated over: KV dealt by position, KDA by heads, every "
                         "tp-th expert a member")
    ap.add_argument("--seqs", type=int, default=64, help="D: sequences a step (the `tokens` / `seqs` bound)")
    ap.add_argument("--mla-split-max", type=int, default=16,
                    help="D: KV splits a row's attention may run as; the workspace is tokens x this x 256 KiB")
    ap.add_argument("--mla-ours-from", type=int, metavar="ROWS",
                    help="D: batches of at least ROWS rows take k3_mla_decode.cu for the MLA attention (default: the DSL)")
    ap.add_argument("--peer-ar", action="store_true",
                    help="D: the sums by the Lamport one-shot over the group's peers instead of NCCL")
    ap.add_argument("--max-ctx", type=int, default=1 << 20, help="the longest context a sequence reaches")
    ap.add_argument("--window", type=int, default=65536,
                    help="P: the cached keys an MLA layer expands and attends in one round")
    ap.add_argument("--kv", choices=tuple(KV), default="fp8",
                    help="the MLA cache: SGLang's fp8 (e4m3; D attends it with FlashInfer's Cake kernel, P lifts it to "
                         "bf16 for the expansion and has no absorbed short path) or bf16")
    ap.add_argument("--kda-state", choices=("bf16", "f32"), default="bf16", help="the KDA recurrent state's storage")
    ap.add_argument("--moe-variants", metavar="FC1,FC2",
                    help="the batched-GEMM variants of the MoE by name (default: the index's picks)")
    a = ap.parse_args()
    if a.chunk and (a.tp > 1 or a.peer_ar):
        ap.error("--tp and --peer-ar are D's; P runs on one GPU")
    if not a.chunk and a.seqs > 64:
        ap.error("D absorbs on tensor cores for at most 64 rows (k3_mla_absorb_mma BMAX)")
    names = tuple(a.moe_variants.split(",")) if a.moe_variants else None
    json.dump(build(a.layers, a.max_ctx, a.seqs, a.tp, a.mla_split_max, a.chunk, names, a.pack,
                    a.peer_ar, a.mla_short, a.kv, a.mla_ours_from, a.window, a.kda_state),
              sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
