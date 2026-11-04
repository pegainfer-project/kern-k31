"""timeline.py <launches.jsonl>: launches grouped into bursts split by gaps > 0.5 s, with the most common symbols."""
import json, sys, collections, re
GAP = 0.5e9
bursts = []
cur = None
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        t = r["t_ns"]
        if cur is None or t - cur["t1"] > GAP:
            cur = {"i0": i, "t0": t, "t1": t, "n": 0, "syms": collections.Counter()}
            bursts.append(cur)
        cur["t1"] = t; cur["n"] += 1
        cur["syms"][re.sub(r"(.{60}).*", r"\1", r["symbol"])] += 1
t00 = bursts[0]["t0"]
for b in bursts:
    top = ", ".join(f"{s}×{c}" for s, c in b["syms"].most_common(2))
    print(f"line {b['i0']:>7} t+{(b['t0']-t00)/1e9:8.1f}s dur {(b['t1']-b['t0'])/1e9:6.2f}s n {b['n']:>7}  {top}")
