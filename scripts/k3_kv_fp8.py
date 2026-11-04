"""gen_stage.py's fp8 MLA cache (`--kv fp8`, the default): SGLang's fp8 one, 64-token pages of e4m3 [576] rows
at scale 1.

The append and D's absorbed query land in e4m3 as SGLang casts them (k3_mla_prep's KV_FP8 build,
k3_mla_absorb_mma's Q_FP8 build). D, always under DCP, attends with FlashInfer's Cake kernel
(source/flashinfer-cake-k3-mla), its splits planned by FlashInfer's planner at 148 SMs for a member's share of
the capture context (scripts/cake_plan.py), and the exchange's CAKE build merges its splits with the members'.
P lifts the pages to bf16 as it gathers them (k3_mla_glue.cu's KV_FP8 build): every call expands its context,
no absorbed short path. Its attention runs on e4m3 as vLLM's FP8 prefill does (`use_prefill_query_quantization`):
q and the expanded k | v cast at scale 1 into the FMHA's e4m3 builds, o bf16. Its KDA and MLA input
projections, lat_down and lat_up are MXFP8 as vLLM's online mxfp8 makes them.
"""
import math

import cake_plan
from pinned import module
from k3_common import B, HEADS, KV_A, KV_LORA, NOPE_DIM, ROPE, T, TP_TIMEOUT_NS, b, i32, i64, launch, pack

BYTES = 1
P_GLUE = {"KV_FP8": 1}
P_FP8_ATTN = True
P_MXFP8 = True
D_PREP = {"KV_FP8": 1}
Q_ABS = "fp8e4m3"

# D's attention: FlashInfer's Cake-generated Kimi-K3 kernel, its wide route, a two-CTA cluster per (row, split).
# SGLang captures a graph planned for the longest context; Cake's planner (`plan_attention`, ported in
# scripts/cake_plan.py) for a member's share of it then sends every batch down this route with all rows as the
# same splits, (first batch, last batch, splits). It plans at 148 SMs, B300's count; GB300 runs the same plan.
CAKE_MAIN = "kernel_cake_kimi_k3_mla_fp8_paged_attention_ea17187e76fd47ab2653"
CAKE_SMEM = 232448
CAKE_PLAN_SMS = 148


def check(decode, xchg, seqs_max):
    assert not decode or (xchg and seqs_max <= 64), "D's fp8 cache is under DCP, at most 64 rows"


p_short = lambda mla_short: 0


def splits(g):
    """Cake's plan for a member's share of the capture context; the planner refuses fewer than 8192 positions."""
    return cake_plan.cake_splits(-(-g.max_ctx // g.tp), CAKE_PLAN_SMS)


def cake_attn_op(batch_max, splits):
    """The Cake attention: per batch range of `splits`, one launch whose
    every row runs as that range's splits, its partials left for the exchange.
    Interface: q_abs (e4m3 [rows][576]) | kv | block_table | seq_lens |
    cu_seqlens_q | part_o | part_max | part_sum | max_pages, at most `batch_max` rows."""
    tmap = lambda param, rows, box0, swizzle: pack(128, {"at": 0, "tensormap": {
        "param": param, "dtype": "u8", "dims": [KV_A, rows], "strides": [KV_A], "box": [box0, 64],
        "swizzle": swizzle}})
    q, kv = (lambda box0, sw: tmap(0, batch_max * HEADS, box0, sw)), (lambda box0, sw: tmap(1, 0, box0, sw))
    params = ["bytes<128>"] * 5 + ["out buffer<bf16>", "out buffer<f32>", "out buffer<f32>", "out buffer<bf16>",
                                   "in buffer<i32>", "in buffer<i32>", "in buffer<i32>", "f32", "f32", "i32", "i32",
                                   "i32", "i32", "i32"]
    args = lambda n: [q(128, 128), kv(128, 128), q(64, 64), kv(64, 64), kv(128, 128),
                      {"param": 5}, {"param": 6}, {"param": 7}, {"param": 5}, {"param": 3}, {"param": 4}, {"param": 2},
                      {"f32": (NOPE_DIM + ROPE) ** -0.5 * math.log2(math.e)}, {"f32": 1.0}, {"i32": HEADS},
                      {"i32": n}, {"param": 8}, {"i32": 1}, {"i32": 0}]
    return {
        "params": ["in buffer<fp8e4m3>", "in state", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>",
                   "out buffer<bf16>", "out buffer<f32>", "out buffer<f32>", "i32"],
        "impl": {"launches": [
            {**module("cake_k3_mla_wide"), "entry": CAKE_MAIN, "params": params, "args": args(n), "block": [384, 1, 1],
             "grid": [{"mul": [T, 2 * n]}, 1, 1], "cluster": [2, 1, 1], "shared_mem": CAKE_SMEM,
             "when": {"var": T, "min": lo, "max": hi}}
            for lo, hi, n in splits]},
    }


d_attn_op = lambda g: cake_attn_op(g.seqs_max, splits(g))


def d_absorb_op():
    """mma.sync's kernel, whose e4m3 casts (software, no saturation) go two at a time."""
    return {
        "params": ["in buffer<f32>", "in buffer<bf16>", "out buffer<fp8e4m3>", "i32"],
        "impl": {"launches": [launch("k3_mla_absorb_mma", "kern_k3_mla_absorb_mma", grid=[1, HEADS, 4],
                                     block=[256, 1, 1], defines={"Q_FP8": 1})]},
    }


def d_exchange_op(g, xg, tail):
    """Cake's partials, every row as its batch's splits: a launch per range of the plan, the count a literal."""
    return {
        "params": ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<i32>", *tail],
        "impl": {"launches": [
            xg(defines={"CAKE": 1}, params=["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<i32>",
                                            "i32", *tail],
               args=[*({"param": i} for i in range(4)), {"i32": n}, *({"param": i} for i in range(4, 15))],
               when={"var": T, "min": lo, "max": hi})
            for lo, hi, n in splits(g)]},
    }


def d_buffers(work, buffers, g):
    """Cake's partials: a batch's rows times its splits, each row's split-major."""
    rows = max(hi * n for _, hi, n in splits(g))
    work("mla_part_o", HEADS * KV_LORA, var=rows)
    work("mla_part_max", HEADS, "f32", var=rows)
    work("mla_part_sum", HEADS, "f32", var=rows)
    buffers["cu_seqlens_q"] = {"dtype": "i32", "shape": [g.seqs_max + 1], "kind": "input", "fill": "cu_seqlens",
                               "domain": {"min": 0, "max": g.seqs_max, "monotone": True}}


def d_attn_steps(step, w, L, kv, g):
    step(L + "attn", "mla_attn", b("q_abs"), {"state": kv}, b("block_table"), b("seq_lens"),
         b("cu_seqlens_q"), b("mla_part_o"), b("mla_part_max"), b("mla_part_sum"), i32(g.max_pages))
    step(L + "dcp", "dcp_exchange", b("mla_part_o"), b("mla_part_max"), b("mla_part_sum"),
         b("seq_lens"), w("w_kv_b_l"), b("mla_gate"), b("gated"), b("dcp_lamport"),
         b("dcp_lamport_peers"), b("dcp_state"), b("tp_err"), {"rank": "tp"}, B, i64(g.dcp_stage),
         i64(TP_TIMEOUT_NS))
