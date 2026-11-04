"""FlashInfer's TRT-LLM gen MoE routing over precomputed top-k (the
`flashinfer_moe_routing` kernel index family) as one manifest op: the
multi-kernel path of `runPostTopKPipeline` (init the expert counts, histogram,
offsets), which builds the routing tables `trtllm_bmm` reads for any token
count with fixed grids, so it lowers to a static launch list.

Each kernel takes `routingPrecomputed::KernelParams<float, 1024, 16>` by value;
the offsets below are what
source/flashinfer-moe-routing/abi.cu prints for it (abi.json), `TOPK_DIV` the bytes of its
`IntFastDiv(16)`. The tables are the bmm's (`route_map`, `cta_batch`,
`cta_limit`, `num_non_exiting`, `total_padded`) plus `exp2perm`, and the
padding rows of `route_map` stay unwritten (the bmm stops at `cta_limit`).
Rows within an expert are ordered by shared-memory atomics, so the tables
differ between runs while every row's result, and the combine reading them
through `exp2perm`, does not.

Interface: `ids` i32 [tokens, 16] global expert ids | `counts` i32 [2 * experts,
at least 512] scratch | `cta_batch`, `cta_limit` i32 [ctas] | `num_non_exiting`,
`total_padded` i32 [1] | `route_map` i32 [padded] | `exp2perm` i32 [tokens, 16]
| and, for a rank holding a slice of `local` experts, `first` i32, the global id
of its first expert. The slice is `first + j * stride` for j in 0..local
(`stride` a power of two): a contiguous slice (stride 1, `first` = local *
rank) or a dealt one (stride = the group size, `first` = rank, which a
manifest passes as `{"rank": g}` with no arithmetic). The tables then cover
the slice only: `cta_batch` holds local expert indices j and `exp2perm` is -1
for every expanded id routed to another rank's expert, which kern's finalize
skips.
"""
SIZE = 192
MAX_EXPERTS, TOPK = 1024, 16
THREADS = 1024
OFF = {
    "mIsPow2": 1, "mPtrExpertCounts": 8, "mPtrPermutedIdxSize": 16, "mPtrExpandedIdxToPermutedIdx": 24,
    "mPtrPermutedIdxToTokenIdx": 40, "mPtrCtaIdxXyToBatchIdx": 48, "mPtrCtaIdxXyToMnLimit": 56,
    "mPtrNumNonExitingCtas": 64, "mPtrTopKIds": 80, "mNumTokens": 96, "mNumExperts": 100, "mPaddingLog2": 104,
    "mTileTokensDim": 108, "mLocalExpertsStartIdx": 112, "mLocalExpertsStrideLog2": 116,
    "mNumLocalExperts": 120, "mTopK": 176,
}
TOPK_DIV = [16, -2147483647, 3, 1]
PARAMS = ["ids", "counts", "cta_batch", "cta_limit", "num_non_exiting", "total_padded", "route_map", "exp2perm"]
_T = "_ZN3moe3dev7routing{}INS1_18routingPrecomputed12KernelParamsIfLi1024ELi16EEEEEvT_"
INIT, HISTOGRAM, OFFSETS = (_T.format(k) for k in ("23routingInitExpertCounts", "29routingIndicesHistogramKernel",
                                                   "27routingIndicesOffsetsKernel"))
# Expanded ids per CTA of the histogram and the offsets kernels (8 per thread each).
PER_CTA = 8 * THREADS


def counts_len(experts):
    return max(512, 2 * experts)


def op(module, experts, tokens, tokens_max, tile, local=None, stride=1):
    """The routing op of one rank holding `local` of the `experts` experts (all of them by default; a slice
    `stride` apart), over `tokens` (a var name or an expression, at most `tokens_max`) routed into
    `tile`-row CTA tiles."""
    local = local or experts
    sliced = local < experts
    assert experts <= MAX_EXPERTS and tile & (tile - 1) == 0 and 0 < local * stride <= experts
    assert stride & (stride - 1) == 0 and (sliced or stride == 1)
    assert -(-TOPK * tokens_max // PER_CTA) <= 1024, "the kernels' grids are capped at 1024 CTAs"
    P = {n: i for i, n in enumerate(PARAMS + ["first"] * sliced)}
    T = {int: "i32", str: "var"}.get(type(tokens), "expr")
    ptr = lambda f, p: {"at": OFF[f], "param": P[p]}
    i32 = lambda f, v: {"at": OFF[f], "i32": v}
    fields = [
        {"at": OFF["mIsPow2"], "u8": 1},
        ptr("mPtrExpertCounts", "counts"), ptr("mPtrPermutedIdxSize", "total_padded"),
        ptr("mPtrExpandedIdxToPermutedIdx", "exp2perm"), ptr("mPtrPermutedIdxToTokenIdx", "route_map"),
        ptr("mPtrCtaIdxXyToBatchIdx", "cta_batch"), ptr("mPtrCtaIdxXyToMnLimit", "cta_limit"),
        ptr("mPtrNumNonExitingCtas", "num_non_exiting"), ptr("mPtrTopKIds", "ids"),
        {"at": OFF["mNumTokens"], T: tokens}, i32("mNumExperts", experts), i32("mPaddingLog2", tile.bit_length() - 1),
        i32("mTileTokensDim", tile),
        {"at": OFF["mLocalExpertsStartIdx"], "param": P["first"]} if sliced else i32("mLocalExpertsStartIdx", 0),
        *([i32("mLocalExpertsStrideLog2", stride.bit_length() - 1)] if stride > 1 else []),
        i32("mNumLocalExperts", local),
        *({"at": OFF["mTopK"] + 4 * j, "i32": w} for j, w in enumerate(TOPK_DIV)),
    ]
    pack = {"pack": {"size": SIZE, "fields": sorted(fields, key=lambda f: f["at"])}}
    expanded = {"ceil_div": [{"mul": [tokens, TOPK]}, PER_CTA]}
    launch = lambda entry, grid: {**module, "entry": entry, "grid": [grid, 1, 1], "block": [THREADS, 1, 1],
                                  "params": [f"bytes<{SIZE}>"], "args": [pack]}
    return {
        "params": ["in buffer<i32>", *["out buffer<i32>"] * 7, *["i32"] * sliced],
        "impl": {"launches": [launch(INIT, -(-counts_len(experts) // THREADS)), launch(HISTOGRAM, expanded),
                              launch(OFFSETS, expanded)]},
    }
