#!/usr/bin/env python3
"""What one rank of a manifest holds in device memory, at the vars' bounds,
and how many KV tokens the rest leaves: weights and carries (the derived
expert layouts among them), the workspaces summed (an upper bound: the
runtime plans them by liveness), the per-sequence state slots (kern supplies
`seqs.max + 2`), then the paged states' bytes per token into what is left of
`--hbm` after `--reserve` (CUDA context, cuBLAS / NCCL buffers, graphs).
A state dealt by position over a group holds a token on one member only, so
the group's capacity is the members' summed.

    python3 scripts/rank_memory.py <manifest.json> [--hbm 274.2] [--reserve 8]
"""
import argparse
import json

SIZE = {"bf16": 2, "f16": 2, "f32": 4, "u8": 1, "i32": 4, "i64": 8, "u32": 4, "u64": 8, "fp8e4m3": 1}
GIB = 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--hbm", type=float, default=274.2, help="GiB free before load (SGLang's log on GB300: 274.22)")
    ap.add_argument("--reserve", type=float, default=8.0, help="GiB kept for the context, libraries and graphs")
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    bound = {k: v["max"] for k, v in m["vars"].items()}

    def nbytes(b):
        n = SIZE[b["dtype"]]
        for d in b["shape"]:
            n *= bound[d] if isinstance(d, str) else d
        return n

    by = {}
    for b in m["buffers"].values():
        by[b["kind"]] = by.get(b["kind"], 0) + nbytes(b)
    states = m["states"]
    slots = bound["seqs"] + 2
    per_seq = sum(s.get("bytes_per_seq", 0) for s in states.values())
    per_tok = sum(s.get("bytes_per_token", 0) for s in states.values())
    dealt = {s["shard"]["group"] for s in states.values() if s.get("shard", {}).get("by") == "position"}
    w = m["topology"]["groups"][dealt.pop()] if dealt else 1
    resident = by.get("weight", 0) + by.get("carry", 0)
    held = resident + by.get("workspace", 0) + slots * per_seq
    left = a.hbm * GIB - a.reserve * GIB - held
    tokens = int(left // per_tok) if per_tok else 0
    rows = [("weights (weight)", by.get("weight", 0)), ("derived + carries (carry)", by.get("carry", 0)),
            ("workspaces (sum, upper bound)", by.get("workspace", 0)),
            (f"per-seq slots: {slots} x {per_seq / 2**20:.2f} MiB", slots * per_seq),
            (f"reserve", a.reserve * GIB), ("left for paged states", left)]
    print(f"{m['model']}: one rank, vars at bounds {bound}")
    for k, v in rows:
        print(f"  {k:34s} {v / GIB:9.2f} GiB")
    print(f"  source buffers (load only)         {by.get('source', 0) / GIB:9.2f} GiB, staged per derive call")
    print(f"  KV: {per_tok} B per token per rank -> {tokens:,} tokens per rank"
          + (f", {tokens * w:,} over the {w} members a position is dealt to" if w > 1 else ""))


if __name__ == "__main__":
    main()
