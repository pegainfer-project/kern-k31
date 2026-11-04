#!/usr/bin/env python3
"""Judge a k3_step run against its oracle (the same steps from the
manifest's group-of-one form): every step's logits by relative RMS, its
top-1 token, and at a flip the oracle's top-1 / top-2 margin, since a
near-tie that flips is kernel noise and a confident token that flips is a
bug (kern CLAUDE.md, "Gates"). Also checks that every rank of the run
handed back the same tokens, which a replicated batch must.

    python3 scripts/compare_steps.py <oracle out> <run out> --rows B [--tie 0.5]

Exit 1 when a flip has a margin above `--tie` (logits), a rank disagrees,
or a step's relative RMS exceeds `--rms`.
"""
import argparse
import array
import math
import pathlib
import sys


def load(path, code):
    a = array.array(code)
    a.frombytes(path.read_bytes())
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle", type=pathlib.Path)
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=163840)
    ap.add_argument("--tie", type=float, default=0.5, help="largest oracle top-1/top-2 margin a flip may have")
    ap.add_argument("--rms", type=float, default=0.05, help="largest relative RMS of a kept step's logits")
    a = ap.parse_args()
    nexts = {p.name: load(p, "q") for p in sorted(a.run.glob("next.r*.i64"))}
    lead = nexts["next.r0.i64"]
    bad = [f"{n} differs from rank 0 at {next(i for i, (x, y) in enumerate(zip(v, lead)) if x != y)}"
           for n, v in nexts.items() if v != lead]
    top_ref, top_got = load(a.oracle / "top.f64", "d"), load(a.run / "top.f64", "d")
    cells = len(top_ref) // 3
    assert len(top_got) == len(top_ref) and cells == len(lead), (len(top_ref), len(top_got), len(lead))
    flips = 0
    for c in range(cells):
        (t, v1, v2), u = top_ref[3 * c:3 * c + 3], top_got[3 * c]
        if t != u:
            flips += 1
            line = f"step {c // a.rows} row {c % a.rows}: oracle {int(t)} (margin {v1 - v2:.3f}), run {int(u)}"
            print("flip", line)
            if v1 - v2 > a.tie:
                bad.append("confident " + line)
    ref, got = load(a.oracle / "logits.f32", "f"), load(a.run / "logits.f32", "f")
    assert len(ref) == len(got), (len(ref), len(got))
    rms = []
    for k in range(len(ref) // a.vocab):
        x, y = ref[k * a.vocab:(k + 1) * a.vocab], got[k * a.vocab:(k + 1) * a.vocab]
        rms.append(math.sqrt(sum((q - p) ** 2 for p, q in zip(x, y)) / sum(p * p for p in x)))
    worst = max(rms, default=0.0)
    if worst > a.rms:
        bad.append(f"relative RMS {worst:.4f} at kept row-step {rms.index(worst)}")
    print(f"{cells // a.rows} steps x {a.rows} rows: top-1 agrees {1 - flips / cells:.4f}, "
          f"relRMS over {len(rms)} kept row-steps median {sorted(rms)[len(rms) // 2] if rms else 0:.5f} "
          f"max {worst:.5f}, {len(nexts)} ranks")
    for b in bad:
        print("FAIL", b)
    print("PASS" if not bad else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
