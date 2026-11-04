#!/usr/bin/env python3
"""pspan.py REPORT.json: kern test's local (same-input) span diffs grouped by buffer and program:
spans differing, worst max_abs, median max_abs, worst fraction of elements differing."""
import json, statistics, sys
from collections import defaultdict
r = json.load(open(sys.argv[1]))["detail"]["local"]
g = defaultdict(list)
for l in r:
    c = l.get("cmp") or {}
    if c.get("n_diff"):
        g[(l["program"], l["buffer"])].append((c.get("max_abs") or 0, c["n_diff"] / c["n"], l["span"]))
for (prog, buf), v in sorted(g.items()):
    worst = max(v)
    print(f"{prog:8} {buf:22} spans {len(v):3}  max_abs worst {worst[0]:.3g} median {statistics.median(x[0] for x in v):.3g}"
          f"  frac worst {max(x[1] for x in v):.3f}  ({worst[2]})")
