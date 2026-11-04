#!/usr/bin/env python3
"""dsteps.py REF RUN [ROWS]: per kept step (every step), the median over rows of the logits' relRMS and KL(ref||run)."""
import array, math, pathlib, statistics, sys
V = 163840
ref, run = map(pathlib.Path, sys.argv[1:3])
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 4
def load(p):
    a = array.array("f"); a.frombytes((p / "logits.f32").read_bytes()); return a
def lse(x):
    m = max(x); return m + math.log(math.fsum(math.exp(v - m) for v in x))
lr, lg = load(ref), load(run)
out = []
for s in range(len(lr) // V // rows):
    rr, kk = [], []
    for r in range(rows):
        k = s * rows + r
        x, y = lr[k * V:(k + 1) * V], lg[k * V:(k + 1) * V]
        rr.append(math.sqrt(math.fsum((q - p) ** 2 for p, q in zip(x, y)) / math.fsum(p * p for p in x)))
        zx, zy = lse(x), lse(y)
        kk.append(max(0.0, math.fsum(math.exp(p - zx) * ((p - zx) - (q - zy)) for p, q in zip(x, y))))
    out.append(f"{statistics.median(rr):.2e}/{statistics.median(kk):.1e}")
print(" ".join(out))
