#!/usr/bin/env python3
"""dcmp.py REF RUN [--rows 4]: one line comparing two k3_step outs (top.f64 every step, logits.f32 every 64th):
top-1 agreement, flips (confident = ref margin > 0.5), relRMS median/max and KL(ref||run) median/max of the kept rows."""
import array, math, pathlib, sys


def load(p, code):
    a = array.array(code)
    a.frombytes(pathlib.Path(p).read_bytes())
    return a


def softmax_log(x):
    m = max(x)
    s = math.fsum(math.exp(v - m) for v in x)
    return m + math.log(s)


def main():
    ref, run = map(pathlib.Path, sys.argv[1:3])
    V = 163840
    tr, tg = load(ref / "top.f64", "d"), load(run / "top.f64", "d")
    cells = len(tr) // 3
    flips = conf = 0
    for c in range(cells):
        if tr[3 * c] != tg[3 * c]:
            flips += 1
            conf += tr[3 * c + 1] - tr[3 * c + 2] > 0.5
    lr, lg = load(ref / "logits.f32", "f"), load(run / "logits.f32", "f")
    rms, kl = [], []
    for k in range(len(lr) // V):
        x, y = lr[k * V:(k + 1) * V], lg[k * V:(k + 1) * V]
        rms.append(math.sqrt(math.fsum((q - p) ** 2 for p, q in zip(x, y)) / math.fsum(p * p for p in x)))
        zx, zy = softmax_log(x), softmax_log(y)
        kl.append(max(0.0, math.fsum(math.exp(p - zx) * ((p - zx) - (q - zy)) for p, q in zip(x, y))))
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"top-1 {1 - flips / cells:.4f} ({flips} flips, {conf} confident) · relRMS median {med(rms):.5f} max {max(rms):.5f}"
          f" · KL median {med(kl):.2e} max {max(kl):.2e}")


main()
