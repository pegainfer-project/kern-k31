#!/usr/bin/env python3
"""mktokens.py IDS OUTDIR: from the D check's 4-row token file (640 steps a row), the files the floor runs use:
ids-r8x / ids-r16x (rows 0-3, then the same rows rotated: rows 0-3 are the check's) and ids-r4s16 / ids-r8s16
(the first 16 steps, for the step-level runs)."""
import array, pathlib, sys
a = array.array("q"); a.frombytes(pathlib.Path(sys.argv[1]).read_bytes())
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
S = 640
rows = [a[r * S:(r + 1) * S] for r in range(4)]
def write(name, rs, n=S):
    b = array.array("q")
    for r in rs: b.extend(r[:n])
    (out / name).write_bytes(b.tobytes())
rot = lambda k: rows[k:] + rows[:k]
write("ids-r8x.i64", rows + rot(1))
write("ids-r16x.i64", rows + rot(1) + rot(2) + rot(3))
write("ids-r4s16.i64", rows, 16)
write("ids-r8s16.i64", rows + rot(1), 16)
