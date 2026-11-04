"""extract.py <capture dir> <out dir>: the decode CUDA graphs SGLang captured, one file per rank and batch size.

A forward starts at `_vocab_parallel_embedding_kernel`. In the graph-capture window SGLang runs each batch
size three times (two warmups, then the captured run), largest first; the third forward of each group, as
long as the warmups (it is checked to launch the same kernels at the same grids as the second), is the graph. The batch size is the
embedding kernel's grid x. Writes graph-bs<B>.rank<r>.jsonl (the launches, as captured) and graphs.json
(per graph: launch count, whether every rank launched the same symbols with the same geometry).
"""
import json
import sys
from pathlib import Path

EMBED = "_vocab_parallel_embedding_kernel"


def ranks(capture: Path):
    """Rank directories in node order (t06 = ranks 0-3, t07 = ranks 4-7), by pid within a node."""
    out = []
    for node in sorted(p for p in capture.iterdir() if p.is_dir()):
        pids = sorted((p for p in node.glob("pid*") if (p / "launches.jsonl").stat().st_size > 0),
                      key=lambda p: int(p.name[3:]))
        out += pids
    return out


def forwards(path: Path):
    """(start line, batch size) of every forward."""
    starts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if EMBED in line:
                r = json.loads(line)
                starts.append((i, r["grid"][0]))
    return starts


def graphs(path: Path):
    starts = forwards(path)
    # The capture window: nine forwards in a row at 48, 48, 48, 8, 8, 8, 1, 1, 1 rows. A warmup's length
    # is the distance to the next forward; the captured run is that many launches from its own start.
    pattern = [48] * 3 + [8] * 3 + [1] * 3
    k = next(k for k in range(len(starts) - 8) if [b for _, b in starts[k:k + 9]] == pattern)
    group = lambda g: [i for i, _ in starts[k + 3 * g:k + 3 * g + 3]]
    spans = {}
    for g, b in enumerate((48, 8, 1)):
        w0, w1, cap = group(g)
        n = w1 - w0
        assert starts[k + 3 * g + 2][0] - w1 >= n, (b, n)
        spans[b] = [(w1, w1 + n), (cap, cap + n)]
    lines = {lo: (b, which, hi) for b, two in spans.items() for which, (lo, hi) in enumerate(two)}
    got = {(b, which): [] for b in spans for which in (0, 1)}
    cur = None
    with open(path) as f:
        for i, line in enumerate(f):
            if i in lines:
                cur = lines[i]
            if cur is not None:
                b, which, hi = cur
                if i < hi:
                    got[(b, which)].append(json.loads(line))
                else:
                    cur = lines.get(i)
                    if cur is not None:
                        got[(cur[0], cur[1])].append(json.loads(line))
    sym = lambda xs: [(x["symbol"], tuple(x["grid"])) for x in xs]
    for b in spans:
        assert sym(got[(b, 0)]) == sym(got[(b, 1)]), f"bs{b}: the captured run differs from the second warmup"
    return {b: got[(b, 1)] for b in spans}


def main():
    capture, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    per_rank = []
    for r, d in enumerate(ranks(capture)):
        g = graphs(d / "launches.jsonl")
        per_rank.append(g)
        for b, launches in g.items():
            with open(out / f"graph-bs{b}.rank{r}.jsonl", "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in launches)
    shape = lambda x: (x["symbol"], tuple(x["grid"]), tuple(x["block"]), x["dynamic_shared_mem_bytes"])
    summary = {}
    for b in per_rank[0]:
        seqs = [[shape(x) for x in g[b]] for g in per_rank]
        summary[b] = {"launches": [len(s) for s in seqs], "same_on_every_rank": all(s == seqs[0] for s in seqs)}
    (out / "graphs.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
