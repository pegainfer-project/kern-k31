#!/usr/bin/env python3
"""The sparse MLA op (scripts/k31_sparse_mla_abi.py) through kern on probe.py's dumps: a manifest of just
the op, run by kern's program_io, its o compared with FlashInfer's byte for byte and its (max, sum) folded
to the base-2 LSE FlashInfer computes from them.

    KERN=<kern checkout> KERN_INDEX_DIR=<kernel index> \
      python3 harness/sparse_mla/gate.py <program_io> <probe dump dir> [--pool-pages N]
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(Path(os.environ["KERN"]) / "tools"))
import kern_manifest  # noqa: E402
import k31_sparse_mla_abi as sm  # noqa: E402


def manifest(rows, pages, pool_pages):
    buffers = {
        "q": {"dtype": "fp8e4m3", "shape": ["rows", sm.HEADS * sm.D_QK], "kind": "input"},
        "kv": {"dtype": "u8", "shape": [pages * sm.PAGE, sm.D_QK], "kind": "input"},
        "ind": {"dtype": "i32", "shape": ["rows", sm.TOPK], "kind": "input"},
        "lens": {"dtype": "i32", "shape": ["rows"], "kind": "input"},
        "o": {"dtype": "bf16", "shape": ["rows", sm.HEADS * sm.D_V], "kind": "output"},
        "stats": {"dtype": "f32", "shape": ["rows", sm.HEADS * 2], "kind": "output"},
    }
    op = sm.op(rows, "rows", kv="in buffer<u8>")
    if pool_pages is not None:
        op["impl"]["launches"][0]["args"][0]["pack"]["fields"].append({"at": 1232, "i32": pool_pages})
    args = [{"buf": b} for b in ("q", "kv", "ind", "lens", "o", "stats")]
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"k31-sparse-mla-probe/r{rows}-p{pages}",
        "vars": {"rows": {"max": rows}},
        "buffers": buffers,
        "ops": {"sparse_mla": op},
        "programs": {"attn": kern_manifest.program([{"label": "attn", "op": "sparse_mla", "args": args}])},
    }
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program_io")
    ap.add_argument("dump", type=Path)
    ap.add_argument("--pool-pages", type=int, help="mNumPagesInMemPool, left 0 by default")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    meta = json.loads((a.dump / "meta.json").read_text())
    rows, pages = meta["rows"], meta["pages"]
    tag = f"pool{a.pool_pages}" if a.pool_pages is not None else "pool0"
    out = a.dump / f"kern-{tag}"
    out.mkdir(exist_ok=True)
    (out / "probe.json").write_text(json.dumps(manifest(rows, pages, a.pool_pages)))
    r = subprocess.run([a.program_io, "--manifest", str(out / "probe.json"), "--cubins", str(ROOT / "build"),
                        "--program", "attn", "--vars", f"rows={rows}", "--graph", "--iters", str(a.iters),
                        *[x for n in ("q", "kv", "ind", "lens") for x in ("--in", f"{n}={a.dump / (n + '.bin')}")],
                        "--out", f"o={out / 'o.bin'}", "--out", f"stats={out / 'stats.bin'}"],
                       capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-2000:] if r.returncode else "")
    if r.returncode:
        raise SystemExit(r.returncode)
    want = np.fromfile(a.dump / "out.bin", dtype=np.uint16).reshape(rows, sm.HEADS, sm.D_V)
    got = np.fromfile(out / "o.bin", dtype=np.uint16).reshape(rows, sm.HEADS, sm.D_V)
    st = np.fromfile(out / "stats.bin", dtype=np.float32).reshape(rows, sm.HEADS, 2)
    lse = math.log2(math.e) * st[..., 0].astype(np.float64) + np.log2(st[..., 1].astype(np.float64))
    want_lse = np.fromfile(a.dump / "lse.bin", dtype=np.float32).reshape(rows, sm.HEADS)
    print(f"rows {rows}, {tag}: o bytes equal {float((got == want).mean()):.6f} "
          f"(real heads {float((got[:, :96] == want[:, :96]).mean()):.6f}), "
          f"lse max abs diff {float(np.abs(lse[:, :96] - want_lse[:, :96]).max()):.3e}")


if __name__ == "__main__":
    main()
