#!/usr/bin/env python3
"""subrows.py IN OUT ROWS KEEP: rows [0, KEEP) of a ROWS-row k3_step out (top.f64, logits.f32) into OUT."""
import pathlib, sys
src, dst, rows, keep = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dst.mkdir(parents=True, exist_ok=True)
for name, width in (("top.f64", 3 * 8), ("logits.f32", 163840 * 4)):
    b = (src / name).read_bytes()
    per = rows * width
    (dst / name).write_bytes(b"".join(b[i:i + keep * width] for i in range(0, len(b), per)))
