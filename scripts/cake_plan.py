"""Cake's split plan for D's fp8 decode attention, without torch.

gen_stage.py plans the wide route at 148 SMs (B300's count; GB300 runs the
same plan) for a DCP member's share of the capture context.
harness/cake_plan/check.py compares this port with FlashInfer's own
`plan_attention` at 148 and 152 SMs.
"""


def cake_splits(local_context, sm_count):
    """q1/H96 wide plans for physical batches 1..64 at the capture context.

    Adapted from FlashInfer flashinfer/mla/cake_kimi_k3_mla.py @3637083a
    (Apache-2.0); specialized to q_len 1, 96 heads, batch <= 64, without torch.
    https://github.com/flashinfer-ai/flashinfer/blob/3637083a/flashinfer/mla/cake_kimi_k3_mla.py
    At B <= 64 < SM/2, plan_wide_work has no full unsplit wave; H96 occupies
    one 128-row tile. Shorter capture contexts need a different kernel ABI,
    which this generator does not provide. Short replay numerics are not
    qualified by this CPU planner comparison.
    """
    if local_context < 8192:
        raise ValueError(
            "FP8 Cake capture context must have at least 8192 positions per DCP member"
        )
    if sm_count not in (148, 152):
        raise ValueError("Cake policy is qualified only for 148 or 152 SMs")
    pairs = sm_count // 2
    tiles = (local_context + 127) // 128
    max_splits = max(1, min(256, tiles // 2))
    bands = []
    for batch in range(1, 65):

        def cost(split, batch=batch):
            items = batch * split
            return -(-items // pairs) * (-(-tiles // split) + 16) + 0.05 * items

        split = min(range(1, max_splits + 1), key=cost)
        if split > 1 and cost(1) / cost(split) < 1.15:
            split = 1
        if bands and bands[-1][2] == split:
            bands[-1] = (bands[-1][0], batch, split)
        else:
            bands.append((batch, batch, split))
    return tuple(bands)
