"""Execute the pinned, unmodified FlashInfer planner without Torch or CUDA."""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

SOURCE = Path(__file__).with_name("reference") / "cake_kimi_k3_mla.py"
SOURCE_SHA = "b5f612c81011566511979f6ef7de98526f8e12f8b2c37995c8fd7b38b32d386f"
CONTEXTS = tuple(
    sorted(
        set(
            [
                8192,
                8193,
                16384,
                32767,
                32768,
                32769,
                40960,
                65536,
                131071,
                131072,
                131073,
                262144,
                1 << 20,
                (1 << 20) + 1,
            ]
            + list(range(8192, 140001, 997))
        )
    )
)


def load_reference(path=SOURCE):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("FlashInfer planner source SHA-256 mismatch")
    name = "_cake_plan_flashinfer_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("torch")
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules[name] = module  # dataclasses resolves annotations through this
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned source verified above
    finally:
        if previous is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous
    return module


def reference_plans():
    reference = load_reference()
    plans = {}
    for sm in (148, 152):
        for length in CONTEXTS:
            bands = []
            for batch in range(1, 65):
                p = reference.plan_attention(
                    batch=batch,
                    max_q_len=1,
                    num_heads=96,
                    max_seq_len=length,
                    sm_count=sm,
                )
                if not (
                    p.wide
                    and p.n_full_items == 0
                    and p.m_tiles == 1
                    and p.grid_main[0] == 2 * batch * p.num_split
                ):
                    raise ValueError(
                        f"upstream route changed: {sm=}, {length=}, {batch=}"
                    )
                if bands and bands[-1][2] == p.num_split:
                    bands[-1] = (bands[-1][0], batch, p.num_split)
                else:
                    bands.append((batch, batch, p.num_split))
            plans[sm, length] = tuple(bands)
    return plans


def check_policy(candidate, plans):
    for length, sm in ((0, 148), (8191, 148), (131072, 80)):
        try:
            candidate(length, sm)
        except ValueError:
            pass
        else:
            raise ValueError(f"policy accepted unsupported input {sm=}, {length=}")
    for (sm, length), expected in plans.items():
        actual = candidate(length, sm)
        if actual != expected:
            raise ValueError(
                f"planner differs at {sm=}, {length=}: {actual} != {expected}"
            )
