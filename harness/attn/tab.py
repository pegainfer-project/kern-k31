#!/usr/bin/env python3
"""harness/attn/tab.py REP.sqlite STDOUT KERNEL ARGS...: the median launch time of KERNEL per (rows, cubin), best of
the harness's rounds, from an nsys export of harness/attn/kda or absorb (KERNEL a name prefix)."""
import sqlite3, statistics, sys
db, txt, kernel, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
cubins = [a for a in args if ".cubin" in a]
lines = open(txt).read().splitlines()
reps, rounds = int(lines[0].split()[1]), int(lines[0].split()[3])
rows = [l.split() for l in lines[2:] if l.split() and l.split()[0].isdigit()]
d = [e - s for s, e in sqlite3.connect(db).execute(
    "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName = s.id "
    "where s.value like ? order by k.start", (kernel + "%",))]
i = 0
print("rows  floor  " + "  ".join(c.split("/")[-1][:12].rjust(12) for c in cubins))
for r in rows:
    best = [1e9] * len(cubins)
    for _ in range(rounds):
        for c in range(len(cubins)):
            best[c] = min(best[c], statistics.median(d[i:i + reps]) / 1e3)
            i += reps
    i += len(cubins)  # the comparison launches
    print(f"{r[0]:>4}  {float(r[1]):5.2f}  " + "  ".join(f"{b:9.2f} {m:>2}" for b, m in zip(best, ["b"] + [x for x in r[3:] if x in ("=", "DIFF")])))
