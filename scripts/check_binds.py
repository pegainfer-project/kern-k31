#!/usr/bin/env python3
"""Resolve every rank's weight binds of a manifest against a checkpoint's
safetensors headers, without a GPU: what kern's load checks per rank
(docs/manifest.md "权重"), so a bad slice fails here and not after a
multi-minute load on eight GPUs.

Per rank of the bind's group (rank 0 alone when a manifest has no group):
every tensor is in the checkpoint, its dtype is the buffer's, a `rows` /
`cols` range lies inside the tensor seen as [dim0, product of the rest],
and the segments' bytes sum to the buffer's. Prints the bytes a rank binds
by buffer kind and exits 1 on any mismatch.

    python3 scripts/check_binds.py <manifest.json> <checkpoint dir>
"""
import json
import pathlib
import struct
import sys

DTYPES = {"BF16": "bf16", "F32": "f32", "U8": "u8", "I32": "i32", "I64": "i64", "F16": "f16", "F8_E4M3": "fp8e4m3"}
SIZE = {"bf16": 2, "f16": 2, "f32": 4, "u8": 1, "i32": 4, "i64": 8, "u32": 4, "u64": 8, "fp8e4m3": 1}


def headers(ckpt):
    """{tensor: (dtype, shape)} over every shard of the checkpoint."""
    out = {}
    for f in sorted(pathlib.Path(ckpt).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k, v in json.loads(fh.read(n)).items():
                if k != "__metadata__":
                    out[k] = (DTYPES.get(v["dtype"], v["dtype"]), v["shape"])
    return out


def pick(x, rank):
    """A bind field as rank `rank` sees it: a rank-selected one's entry, else itself."""
    if isinstance(x, dict) and "group" in x:
        return x.get("tensors", x.get("ranges"))[rank]
    return x


def members(bind, groups):
    gs = {f["group"] for s in bind for f in (s.get("tensor"), s.get("rows"), s.get("cols"))
          if isinstance(f, dict) and "group" in f}
    assert len(gs) <= 1, f"one bind selects by {gs}"
    return groups[gs.pop()] if gs else 1


def main():
    m = json.load(open(sys.argv[1]))
    hdr = headers(sys.argv[2])
    groups = m.get("topology", {}).get("groups", {})
    errs, total = [], {}
    for name, b in m["buffers"].items():
        if "bind" not in b:
            continue
        want = SIZE[b["dtype"]]
        for d in b["shape"]:
            want *= d
        for r in range(members(b["bind"], groups)):
            got = 0
            for s in b["bind"]:
                t = pick(s["tensor"], r)
                if t not in hdr:
                    errs.append(f"`{name}` rank {r}: no tensor `{t}`")
                    continue
                dt, shape = hdr[t]
                if dt != b["dtype"]:
                    errs.append(f"`{name}` rank {r}: `{t}` is {dt}, the buffer {b['dtype']}")
                rows, cols = shape[0], 1
                for d in shape[1:]:
                    cols *= d
                rr, cr = pick(s.get("rows"), r) or [0, rows], pick(s.get("cols"), r) or [0, cols]
                if len(shape) == 1 and "cols" in s:
                    rows, cols, rr = 1, shape[0], [0, 1]
                if not (0 <= rr[0] < rr[1] <= rows and 0 <= cr[0] < cr[1] <= cols):
                    errs.append(f"`{name}` rank {r}: `{t}` {shape} sliced rows {rr} cols {cr}")
                got += (rr[1] - rr[0]) * (cr[1] - cr[0]) * SIZE.get(dt, 1)
            if got != want:
                errs.append(f"`{name}` rank {r}: binds {got} bytes, the buffer holds {want}")
            if r == 0:
                total[b["kind"]] = total.get(b["kind"], 0) + want
    for k, v in sorted(total.items()):
        print(f"rank 0 binds {k:8s} {v / 2**30:9.3f} GiB")
    for e in errs[:40]:
        print("FAIL", e)
    print(f"{len(errs)} mismatches" if errs else "every rank's binds resolve")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
