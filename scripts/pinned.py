"""Kernel modules for the generator: the ones this repository pins
(`kernels.toml`) by their hash, every other one from kern's kernel index.
Both resolve to the same registry ref (`hf:Pegainfer/kern-kernels/blobs/<sha>`).
"""
from pathlib import Path
import tomllib

from kernels import index

PINNED = tomllib.loads((Path(__file__).resolve().parents[1] / "kernels.toml").read_text())["kernels"]


def module(family, **defines):
    name = index.variant_name(family, defines)
    if name in PINNED:
        sha = PINNED[name]["sha256"]
        return {"cubin": index.source(sha), "sha256": sha, "label": name}
    return index.variant(family, **defines).module
