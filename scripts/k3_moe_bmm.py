"""K3's routed MoE as TRT-LLM gen batched GEMMs (kernel index families
`trtllm_bmm_mxe4m3_mxe2m1_mxe4m3` for FC1 with the fused SiTU gate and
`trtllm_bmm_bf16_mxe2m1_mxe4m3` for FC2, kern's tools/kernels/abi/trtllm_bmm.py)
around the glue of source/k3_moe_prefill.cu: the pieces gen_stage.py lays
out. A fork of kern's tools/k3_moe_bmm.py.

One rank holds `local` of the model's 896 experts: every one (P) or every
`stride`-th one (a D member: 112 of them, member r holding r, r + 8, ...). Its part of a layer: quantise the latent rows to
mxfp8, build the routing tables of its experts, FC1 (gate | up, SiTU, mxfp8
out), FC2 (bf16 out per permuted row), and the top-k combine into a bf16
partial whose rows past `tokens` are zero. P routes in one call with the
router top-k, the quant and the shared activation (source/k3_moe_route.cu);
a D member's tables come from FlashInfer's kernels
(flashinfer_moe_routing_abi.py), the same up to the order of rows within an
expert.
The expert weights are the checkpoint's mxfp4 tensors row-shuffled once
after load (`k3_moe_w_shuffle` / `k3_moe_sf_shuffle`, once.py): FC1's rows
are [up; gate] before the shuffle, so the kernel's pair (x0, x1) is (up, gate)
and its activation beta·tanh(x0/beta)·alpha·tanh(x1/alpha)·sigmoid(x1) is
K3's situ with alpha 4, beta 25.
"""
import flashinfer_moe_routing_abi as flashinfer_moe_routing
from kernels import index
from kernels.abi import trtllm_bmm
from pinned import module
from k3_common import EXPERTS

H, I, TOPK = 3584, 3072, 16
ALPHA, BETA = 4.0, 25.0
GLUE, PGLUE, ROUTING, ROUTE = "k3_moe_prefill", "k3_prefill_glue", "flashinfer_moe_routing", "k3_moe_route"
FC1, FC2 = "trtllm_bmm_mxe4m3_mxe2m1_mxe4m3", "trtllm_bmm_bf16_mxe2m1_mxe4m3"
SHAPE = "k3-prefill-16k-ep4"
# k3_moe_route.cu: 16 warps a block, two blocks an SM resident for its grid barrier (2 x 144 fits 148-SM B300
# and 152-SM GB300), a quarter of them up to 256 rows; its tables cover 8192 rows.
ROUTE_WARPS, ROUTE_BLOCKS, ROUTE_BLOCKS_SMALL, ROUTE_ROWS = 16, 2 * 144, 64, 8192


def ceil_div(a, b):
    return -(-a // b)


def variants(names=None):
    """The two GEMMs: the index's picks for SHAPE, or the (fc1, fc2) variant names given (a sweep, the probe's capture)."""
    v1, v2 = ((index.variant_by_name(FC1, names[0]), index.variant_by_name(FC2, names[1])) if names
              else (index.pick(FC1, "moe_fc1", SHAPE), index.pick(FC2, "moe_fc2", SHAPE)))
    assert v1.tags["sf_layout"][2] == v2.tags["sf_layout"][1], "FC1 writes its scales in the layout FC2 reads"
    assert v1.tags["act"] == "siTuGlu" and v1.tags["fused_act"] and v1.tags["route"] != "none"
    return v1, v2


def glue(entry, grid, block=256, mod=None):
    return {**(mod or index.variant(GLUE).module), "entry": entry, "grid": grid, "block": [block, 1, 1]}


def pieces(local, tokens, tokens_max, out_rows, prefix="", names=None, stride=None, pdl=False):
    """The ops and workspace buffers for `local` of the experts over `tokens` rows (a var, at most
    `tokens_max`); the combine writes `out_rows` rows (an expression). `steps` lists the calls of one
    layer. `stride`: the rank holds experts `rank + j * stride` (D), else every expert (P)."""
    v1, v2 = variants(names)
    # The combine + latent norm (source/k3_prefill_glue.cu).
    pglue = module(PGLUE)
    tile = v1.tags["tile"][1]
    assert v2.tags["tile"][1] == tile, "both GEMMs read one set of CTA tables"
    ctas = trtllm_bmm.ctas_bound(tokens, TOPK, local, tile)
    ctas_max = local + ceil_div(TOPK * tokens_max, tile)
    padded_max = ctas_max * tile
    n = lambda s: prefix + s
    i32 = lambda v: {"i32": v}
    dim = lambda x: {"var": x} if isinstance(x, str) else {"expr": x}
    whole = local == EXPERTS
    # P routes in one call (route_step) and keeps no FlashInfer counts.
    fused = stride is None
    assert not fused or tokens_max <= ROUTE_ROWS
    sync_words = 66 + 2 * EXPERTS * (1 + ROUTE_ROWS // 256 + ROUTE_ROWS // 32)
    buffers = {
        **({n("route_sync"): {"dtype": "u32", "shape": [sync_words], "kind": "workspace"}} if fused else
           {n("counts"): {"dtype": "i32", "shape": [flashinfer_moe_routing.counts_len(EXPERTS)], "kind": "workspace"}}),
        n("cta_batch"): {"dtype": "i32", "shape": [ctas_max], "kind": "workspace"},
        n("cta_limit"): {"dtype": "i32", "shape": [ctas_max], "kind": "workspace"},
        n("num_non_exiting"): {"dtype": "i32", "shape": [1], "kind": "workspace"},
        n("total_padded"): {"dtype": "i32", "shape": [1], "kind": "workspace"},
        n("route_map"): {"dtype": "i32", "shape": [padded_max], "kind": "workspace"},
        n("exp2perm"): {"dtype": "i32", "shape": [tokens_max, TOPK], "kind": "workspace"},
        n("fc1_out"): {"dtype": "u8", "shape": [padded_max, I], "kind": "workspace"},
        n("fc1_sf"): {"dtype": "u8", "shape": [ceil_div(padded_max, 128) * 128, I // 32], "kind": "workspace"},
        n("fc2_out"): {"dtype": "bf16", "shape": [padded_max, H], "kind": "workspace"},
    }
    ops = {
        "moe_quant": {
            "params": ["in buffer<bf16>", "out buffer<u8>", "out buffer<u8>", "i32", "i32"],
            "impl": {"launches": [glue("kern_k3_moe_quant", [{"ceil_div": [{"mul": [tokens, H // 32]}, 256]}, 1, 1])]},
        },
        **({} if fused else {"moe_routing": flashinfer_moe_routing.op(module(ROUTING), EXPERTS, tokens, tokens_max,
                                                                      tile, local=local, stride=stride or 1)}),
        **({"moe_route": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "i32", "in buffer<bf16>",
                       "i32", "out buffer<u8>",
                       "out buffer<u8>", "out buffer<i32>", "out buffer<f32>", "out buffer<bf16>", "i32",
                       "inout buffer<u32>",
                       *["out buffer<i32>"] * 6, "i32", "i32"],
            "impl": {"launches": [
                {**module(ROUTE), "entry": "kern_k3g_moe_route",
                 "block": [32 * ROUTE_WARPS, 1, 1], **g}
                for g in ({"grid": [ROUTE_BLOCKS_SMALL, 1, 1], "when": {"var": tokens, "max": 256}},
                          {"grid": [ROUTE_BLOCKS, 1, 1], "when": {"var": tokens, "min": 257}})]},
        }, "moe_finalize_rms_ld": {
            "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32"],
            "impl": {"launches": [{**module(ROUTE), "entry": "kern_k3g_moe_finalize_rms", "block": [H // 8, 1, 1],
                                   "grid": [out_rows, 1, 1]}]},
        }} if fused else {}),
        "moe_fc1": trtllm_bmm.op(v1, 2 * I, H, local, tokens, tokens_max, ctas, ctas_max),
        "moe_fc2": trtllm_bmm.op(v2, H, I, local, tokens, tokens_max, ctas, ctas_max),
        "moe_finalize": {
            "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "out buffer<bf16>", "i32", "i32"],
            # one 16-byte column vector per thread: at 256 a row took two passes over its 16 picks
            "impl": {"launches": [glue("kern_k3_moe_finalize", [out_rows, 1, 1], block=H // 8)]},
        },
        "moe_finalize_rms": {
            "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32"],
            "impl": {"launches": [glue("kern_k3g_finalize_rms", [out_rows, 1, 1], block=H // 8, mod=pglue)]},
        },
    }
    # The batched GEMMs wait on the grid before them (griddepcontrol.wait) and trigger their dependents
    # early: with `pdl` launched as programmatic dependents, their CTAs land while the kernel before
    # them finishes.
    for op in ("moe_fc1", "moe_fc2") if pdl else ():
        for l in ops[op]["impl"]["launches"]:
            l["pdl"] = True
    assert trtllm_bmm.params(v1) == ["a", "sf_a", "b", "sf_b", "c", "sf_c", "route_map", "alpha", "beta",
                                     "num_non_exiting", "total_padded", "cta_batch", "cta_limit"]
    assert trtllm_bmm.params(v2) == ["a", "sf_a", "b", "sf_b", "c", "num_non_exiting", "total_padded", "cta_batch",
                                     "cta_limit"]
    b = lambda s: {"buf": n(s)}
    tables = [b("num_non_exiting"), b("total_padded"), b("cta_batch"), b("cta_limit")]

    def routing(ids, rank, label):
        """The routing tables of this rank's experts; `rank` is unused when the rank holds them all, the first
        expert's id when it holds every `stride`-th one."""
        return {"label": label + "routing", "op": "moe_routing",
                "args": [ids, b("counts"), b("cta_batch"), b("cta_limit"), b("num_non_exiting"),
                         b("total_padded"), b("route_map"), b("exp2perm"), *([rank] if not whole else [])]}

    def route_step(S, bias, rs, x, ldx, gu, ldgu, q, sf, ids, wts, act, ldact, label=""):
        """The router top-k, the latent's mxfp8, the routing tables and the shared expert's activation (from
        its gate | up rows `gu`) in one call (k3_moe_route.cu)."""
        return {"label": label + "route", "op": "moe_route",
                "args": [S, bias, rs, x, i32(ldx), gu, i32(ldgu), q, sf, ids, wts, act, i32(ldact), b("route_sync"), b("cta_batch"), b("cta_limit"),
                         b("num_non_exiting"), b("total_padded"), b("route_map"), b("exp2perm"), i32(tile),
                         dim(tokens)]}

    def steps(q, sf, ids, wts, w13s, w13_sfs, w2s, w2_sfs, alpha, beta, out, rank=None, gamma=None, label="",
              routed=False, combined=True, ldo=None):
        """One layer's calls; with `gamma` the combine's row is normed by it (the latent norm) before `out`;
        `routed`: the tables are already built (route_step, or the DCP step's moe_front); not `combined`: the
        combine is the caller's (the DCP step's fused all-reduce)."""
        T = dim(tokens)
        combine = ({"label": label + "finalize", "op": "moe_finalize_rms_ld",
                    "args": [b("fc2_out"), b("exp2perm"), wts, gamma, out, i32(ldo), T]} if gamma and ldo else
                   {"label": label + "finalize", "op": "moe_finalize_rms",
                    "args": [b("fc2_out"), b("exp2perm"), wts, gamma, out, T, i32(H)]} if gamma else
                   {"label": label + "finalize", "op": "moe_finalize",
                    "args": [b("fc2_out"), b("exp2perm"), wts, out, T, i32(H)]})
        return [
            *([] if routed else [routing(ids, rank, label)]),
            {"label": label + "fc1", "op": "moe_fc1",
             "args": [w13s, w13_sfs, q, sf, b("fc1_out"), b("fc1_sf"), b("route_map"), alpha, beta, *tables]},
            {"label": label + "fc2", "op": "moe_fc2", "args": [w2s, w2_sfs, b("fc1_out"), b("fc1_sf"), b("fc2_out"), *tables]},
            *([combine] if combined else []),
        ]

    return {"buffers": buffers, "ops": ops, "route_step": route_step, "route_sync_words": sync_words, "steps": steps}


def shuffles(local):
    """The four weight-prep calls of one layer as (op, args after the buffers, threads): FC1's [up; gate] rows
    and scales with the gated interleave, FC2's plain."""
    return [
        ("k3_moe_w_shuffle", "w13", [local, 2 * I, H // 2, 1], local * 2 * I * (H // 2) // 16),
        ("k3_moe_sf_shuffle", "w13_sf", [local, 2 * I, H // 32, 1], local * 2 * I * (H // 32)),
        ("k3_moe_w_shuffle", "w2", [local, H, I // 2, 0], local * H * (I // 2) // 16),
        ("k3_moe_sf_shuffle", "w2_sf", [local, H, I // 32, 0], local * H * (I // 32)),
    ]
