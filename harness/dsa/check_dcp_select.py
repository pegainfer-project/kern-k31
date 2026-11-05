#!/usr/bin/env python3
"""D's candidate exchange under DCP (k31_dcp_pack -> k31_dcp_select) on one GPU, W = 8 members emulated
(their stages, states and lists side by side, every peer pointer local): each member's local top 2048
of indexer-like bf16 rows full of ties, its best `--hot` pushed into every member's stage, and every
member's share of the group's top 2048 as cache slots. Every output (ind, lens, live) must be the exact bytes
of a CPU merge in vLLM's order (higher score, then smaller global position), its slots in the kernel's fixed
candidate order, and, with --base, the same bytes as the base build's (`--base-hot 0`: main's ABI, every
candidate pushed). `--dist skew` puts one member's scores above the rest, so its list has to be pulled;
`--dist dealt` deals one row over the members as D does.
Calls rotate through the three stages. Prints each build's µs a call: pack alone, and select (pack + select
less pack), the mean over members, and how many rows pulled a list.

    python3 harness/dsa/check_dcp_select.py <cubin> [--word 4|8] [--hot 384] [--base <cubin> --base-word 8
        --base-hot 0] [--rows 1,64,128,256] [--keys 300,16384] [--dist ties|smooth|skew|dealt]

"""
import argparse
import ctypes

import numpy
import torch
from cuda.bindings import driver as cu

from check import Launcher, ok

W, TOPK, PAGE, SMS = 8, 2048, 64, 152


def scores(R, n, g, dist="ties", m=0):
    """[R, n] bf16 local logits of member m: `ties` on few distinct values (ties at every cut), some rows all
    equal, some -0.0; `smooth` indexer-like, a slow drift along the position plus noise, few ties past bf16's
    own, each member its own drift; `skew` smooth, member (row % W)'s raised so it holds most of the row's top.
    `dealt` (in main) is one smooth row of W n positions dealt position p to member p % W, as D's cache is."""
    if dist == "skew":
        x = scores(R, n, g, "smooth").float()
        x[torch.arange(R) % W == m] += 1.5
        return x.bfloat16()
    if dist == "smooth":
        drift = torch.nn.functional.interpolate(torch.randn(R, 1, n // 256 + 2, generator=g), size=n, mode="linear")
        return (2 + drift[:, 0] + 0.3 * torch.randn(R, n, generator=g)).relu().bfloat16()
    x = torch.randn(R, n, generator=g) * 3
    x = torch.where(torch.rand(R, n, generator=g) < 0.3, x.round(), x)
    x[0::7] = 1.0
    x[3::11, ::5] = -0.0
    return x.bfloat16()


def local_topk(v, n):
    """The held top-k's output: the top TOPK of v[:n] by (value, then smaller position), ascending, -1 after."""
    if n <= TOPK:
        return numpy.concatenate([numpy.arange(n), -numpy.ones(TOPK - n)]).astype(numpy.int32)
    order = numpy.lexsort((-numpy.arange(n), v[:n]))
    return numpy.sort(order[-TOPK:]).astype(numpy.int32)


def ordered(b):
    """bf16 bits -> the 16 ordered bits the merge sorts on (k31_dcp_pack)."""
    b = b.astype(numpy.int64) & 0xffff
    return numpy.where(b & 0x8000, ~b & 0xffff, b | 0x8000)


def reference(bits, inds, bts):
    """Every member's (ind row, lens, live) of one row: bits[k] the bf16 bits of member k's local logits,
    inds[k] its local top-k, bts[k] its page table row. The kept candidates go out in (c % 1024, c / 1024)
    order of c = member * 2048 + slot."""
    ind = numpy.stack(inds).astype(numpy.int64)                     # [W, 2048]
    k = numpy.repeat(numpy.arange(W), TOPK).reshape(W, TOPK)
    live = ind >= 0
    score = numpy.stack([ordered(bits[m][numpy.maximum(ind[m], 0)]) for m in range(W)])
    gpos = ind * W + k
    c = k * TOPK + numpy.arange(TOPK)[None]
    score, gpos, c, ind, k = (x[live] for x in (score, gpos, c, ind, k))
    top = numpy.lexsort((gpos, -score))[:TOPK]
    out = []
    for m in range(W):
        t = top[k[top] == m]
        t = t[numpy.lexsort((c[t] // 1024, c[t] % 1024))]
        row = numpy.full(TOPK, -1, dtype=numpy.int32)
        row[:len(t)] = bts[m][ind[t] // PAGE] * PAGE + ind[t] % PAGE
        if not len(t):
            row[0] = 0
        out.append((row, max(len(t), 1), len(t)))
    return out


class Build:
    def __init__(self, path, word, hot, R_max):
        self.mod = mod = ok(cu.cuModuleLoadData(open(path, "rb").read()))
        self.fn = {n: ok(cu.cuModuleGetFunction(mod, n.encode())) for n in ("k31_dcp_pack", "k31_dcp_poison",
                                                                            "k31_dcp_select")}
        self.hot = hot
        self.stage = W * R_max * (hot or TOPK) * word
        self.lam = [torch.empty(3 * self.stage, dtype=torch.uint8, device="cuda") for _ in range(W)]
        self.state = [torch.zeros(8, dtype=torch.int32, device="cuda") for _ in range(W)]
        self.err = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.peers = torch.tensor([x.data_ptr() for x in self.lam], dtype=torch.int64, device="cuda")
        L = Launcher()
        for x in self.lam:
            L(self.fn["k31_dcp_poison"], (SMS, 1, 1), (256, 1, 1), [x, ctypes.c_longlong(3 * self.stage)])

    def pack(self, L, members, R):
        for m, (logits, stride, ind, _bt, _mp) in enumerate(members):
            L(self.fn["k31_dcp_pack"], (R, 1, 1), (1024 if self.hot else 256, 1, 1),
              [logits, ctypes.c_int(stride), ind, ctypes.c_int(m), ctypes.c_int(W), ctypes.c_int(R), self.peers,
               self.state[m], ctypes.c_longlong(self.stage)])

    def bind(self, members):
        """Every member's logits and ind, for a pull."""
        ptrs = lambda k: torch.tensor([x[k].data_ptr() for x in members], dtype=torch.int64, device="cuda")
        self.lists = ptrs(0), ptrs(2)

    def select(self, L, members, outs, R):
        for m, (logits, stride, ind, bt, max_pages) in enumerate(members):
            sel, lens, live = outs[m]
            L(self.fn["k31_dcp_select"], (R, 1, 1), (1024, 1, 1),
              [self.lam[m], self.state[m], self.err, ctypes.c_longlong(self.stage), ctypes.c_longlong(10**10),
               ctypes.c_int(R), ctypes.c_int(m), bt, ctypes.c_int(max_pages), sel, lens, live,
               *([logits, ctypes.c_int(stride), ind, *self.lists] if self.hot else [])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--word", type=int, default=8, help="bytes a candidate in the cubin's build")
    ap.add_argument("--hot", type=int, default=384, help="candidates a member sends in the cubin's build")
    ap.add_argument("--base")
    ap.add_argument("--base-word", type=int, default=8)
    ap.add_argument("--base-hot", type=int, default=384, help="0: main's ABI, every candidate sent")
    ap.add_argument("--rows", default="1,64,128,256")
    ap.add_argument("--keys", default="300,16384", help="each member's local keys a row, at most")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--dist", default="ties", choices=("ties", "smooth", "skew", "dealt"))
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    rows = [int(x) for x in a.rows.split(",")]
    torch.zeros(1, device="cuda")
    L = Launcher()
    builds = {"new": Build(a.cubin, a.word, a.hot, max(rows))}
    if a.base:
        builds["base"] = Build(a.base, a.base_word, a.base_hot, max(rows))
    g = torch.Generator().manual_seed(0)
    bad = 0
    for n_max in (int(x) for x in a.keys.split(",")):
        for R in rows:
            for call in range(a.calls):
                n = torch.randint(1, n_max + 1, (R,), generator=g).tolist()
                n[: min(R, 3)] = [n_max, 1, min(n_max, 2049)][: min(R, 3)]
                stride = -(-n_max // 384) * 384
                members, host = [], []
                dealt = scores(R, stride * W, g, "smooth") if a.dist == "dealt" else None
                for m in range(W):
                    nm = [max(1, x - (m * x) // (3 * W)) for x in n]  # members hold a little fewer keys
                    v = dealt[:, m::W].contiguous() if a.dist == "dealt" else scores(R, stride, g, a.dist, m)
                    vb = v.view(torch.int16).numpy()
                    vf = v.float().numpy()
                    ind = numpy.stack([local_topk(vf[r], nm[r]) for r in range(R)])
                    pages = -(-n_max // PAGE)
                    bt = torch.randperm(R * pages, generator=g).view(R, pages).int() + 1
                    members.append((v.cuda(), stride, torch.from_numpy(ind).cuda(), bt.cuda(), pages))
                    host.append((vb, ind, bt.numpy()))
                got = {}
                for name, b in builds.items():
                    outs = [(torch.full((R, TOPK), 7, dtype=torch.int32, device="cuda"),
                             torch.full((R,), 7, dtype=torch.int32, device="cuda"),
                             torch.full((R,), 7, dtype=torch.int32, device="cuda")) for _ in range(W)]
                    b.bind(members)
                    b.pack(L, members, R)
                    b.select(L, members, outs, R)
                    torch.cuda.synchronize()
                    assert int(b.err) == 0, (name, int(b.err))
                    got[name] = [tuple(t.cpu().numpy() for t in o) for o in outs]
                miss, pulls = 0, 0
                for r in range(R):
                    want = reference([h[0][r] for h in host], [h[1][r] for h in host], [h[2][r] for h in host])
                    pulls += any(lv > a.hot for _row, _lens, lv in want)
                    for m, (row, lens, live) in enumerate(want):
                        for name in got:
                            s, l, lv = got[name][m]
                            miss += not ((s[r] == row).all() and l[r] == lens and lv[r] == live)
                same = all((x == y).all() for name in got for o, p in zip(got[name], got["new"]) for x, y in zip(o, p))
                bad += miss + (not same)
                print(f"keys <= {n_max:6d} rows {R:3d} call {call}: rows off the reference {miss}, "
                      f"builds byte-identical {same}, rows with a member past --hot {pulls}", flush=True)
            times = []
            for name, b in builds.items():
                outs = [(torch.empty(R, TOPK, dtype=torch.int32, device="cuda"),
                         torch.empty(R, dtype=torch.int32, device="cuda"),
                         torch.empty(R, dtype=torch.int32, device="cuda")) for _ in range(W)]
                p = L.time(a.iters, lambda: b.pack(L, members, R)) / W
                ps = L.time(a.iters, lambda: (b.pack(L, members, R), b.select(L, members, outs, R))) / W
                times.append(f"{name} pack {p:6.1f} select {ps - p:6.1f}")
            print(f"keys <= {n_max:6d} rows {R:3d}: us a call  " + "   ".join(times), flush=True)
    print("FAIL" if bad else "PASS", bad)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
