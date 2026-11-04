#!/usr/bin/env python3
"""psum.py REPORT.json...: per kern test report, the end-to-end logit rows' KL(A||B) median / mean / max, flips
(with A's margin), and the verdict line's first word."""
import json, statistics, sys

for p in sys.argv[1:]:
    r = json.load(open(p))
    rows = r["detail"]["logit_rows"]
    kl = [x["kl"] for x in rows] or [0.0]
    flips = [f'{x["margin_a"]:.3f}' for x in rows if x["argmax_a"] != x["argmax_b"]]
    ident = all(x["cmp"]["n_diff"] == 0 for x in rows)
    name = p.rsplit("/", 1)[-1].removesuffix(".json")
    print(f"{name:22} rows {len(rows):2} · KL median {statistics.median(kl):.2e} mean {statistics.mean(kl):.2e} "
          f"max {max(kl):.2e} · flips {len(flips)} {' '.join(flips)}" + (" · bit-identical" if ident else ""))
