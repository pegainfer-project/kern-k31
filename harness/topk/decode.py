#!/usr/bin/env python3
"""k31_topk_held on D's shape against a reference build and an exact reference, on one GPU. D runs DCP8,
so a row holds every 8th position of its context: 4k / 16k / 32k local keys at 32k / 128k / 256k. Each
case's rows are indexer-like scores (sum over 32 heads of w_h relu(q_h . k), keys smooth along the
position with sinks and a recent window) or an adversarial pattern; every row's output must be the
same bytes as the reference build's and as the exact top 2048 (ties at the cut by smallest
position), and the same on a second run. Prints each case's µs a call, both builds.

    python3 harness/topk/decode.py --base base.cubin [--cubin new.cubin] [--rows 16,64,256] [--n 4096,16384,32768]
"""
import argparse
import ctypes
import sys
from pathlib import Path

import numpy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa"))
from check import Launcher, module, ok, cu  # noqa: E402

TOPK, SPLIT_KV, ENTRY = 2048, 384, b"k31_topk_held"


def reference(v, n):
    """The top TOPK of v[:n] (a numpy row), ties by smallest position, ascending, -1 padded."""
    if n <= TOPK:
        return numpy.concatenate([numpy.arange(n), -numpy.ones(TOPK - n)]).astype(numpy.int32)
    order = numpy.lexsort((-numpy.arange(n), v[:n]))
    return numpy.sort(order[-TOPK:]).astype(numpy.int32)


def indexer_like(rows, n, g, peaky=False):
    """[rows, n] scores sum_h w_h relu(q_h . k_j) / sqrt(128), keys smooth along j (coarse noise
    interpolated plus white), a few sink keys at the head, the last 512 keys pulled toward each row's
    query; peaky adds a few hundred far-above keys a row."""
    d, h = 128, 32
    coarse = torch.randn(n // 64 + 2, d, device="cuda", generator=g)
    smooth = torch.nn.functional.interpolate(coarse.T[None], size=n, mode="linear", align_corners=False)[0].T
    k = 0.8 * smooth + 0.6 * torch.randn(n, d, device="cuda", generator=g)
    k[:4] *= 4
    out = torch.empty(rows, n, device="cuda")
    for r in range(rows):
        q = torch.randn(h, d, device="cuda", generator=g)
        w = torch.randn(h, device="cuda", generator=g).abs() * 0.3 - 0.05
        kr = k.clone()
        kr[-512:] += 0.5 * q.mean(0)
        s = (w[:, None] * torch.relu(q @ kr.T / d ** 0.5)).sum(0)
        if peaky:
            s[torch.randint(0, n, (300,), device="cuda", generator=g)] += 5
        out[r] = s
    return out


def adversarial(kind, rows, n, g):
    """[rows, n] rows that stress the tie rule and the fallbacks."""
    dev = "cuda"
    if kind == "equal":
        return torch.full((rows, n), 0.75, device=dev)
    if kind == "four":
        return torch.randint(0, 4, (rows, n), device=dev, generator=g).float()
    if kind == "cut_ties":
        # 1500 above, a few thousand on one value at the cut, the rest below
        x = torch.rand(rows, n, device=dev, generator=g) - 2
        for r in range(rows):
            p = torch.randperm(n, device=dev, generator=g)
            x[r, p[:1500]] = 3 + torch.rand(1500, device=dev, generator=g)
            x[r, p[1500:1500 + min(4000, n - 1500)]] = 1.0
        return x
    if kind == "cluster":
        return 1 + 1e-3 * torch.randn(rows, n, device=dev, generator=g)
    if kind == "zeros":
        return torch.relu(torch.randn(rows, n, device=dev, generator=g) - 1.5)
    if kind == "neg":
        return -torch.randn(rows, n, device=dev, generator=g).abs()
    if kind == "inf":
        x = torch.randn(rows, n, device=dev, generator=g)
        x[:, 1000:] = -float("inf")
        return x
    if kind == "nan":
        x = torch.randn(rows, n, device=dev, generator=g)
        x[:, ::997] = float("nan")
        return x
    if kind == "bits":
        # every bf16 pattern alike: NaNs of either sign and any payload, -0, subnormals, infinities
        return torch.randint(-32768, 32768, (rows, n), device=dev, generator=g, dtype=torch.int32).short().view(torch.bfloat16)
    if kind == "fp16":
        return torch.randn(rows, n, device=dev, generator=g).half().float()
    raise ValueError(kind)


def lens_of(kind, rows, n, g):
    """A batch's lengths: every row n, a spread below n, or the short and degenerate ones."""
    if kind == "full":
        return torch.full((rows,), n, dtype=torch.int32)
    if kind == "spread":
        return torch.randint(n // 2, n + 1, (rows,), dtype=torch.int32)
    short = torch.tensor([0, 1, 2, 7, 2047, 2048, 2049, 2050, 3000, n - 1, n, n - 7], dtype=torch.int32)
    return short.repeat(rows // len(short) + 1)[:rows].clamp(max=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the reference k31_topk build")
    ap.add_argument("--cubin", default=None, help="the build measured (default: build/k31_topk.cubin)")
    ap.add_argument("--rows", default="16,32,64,128,256")
    ap.add_argument("--n", default="4096,16384,32768", help="local keys a row")
    ap.add_argument("--cases", default="real,peaky,real/spread,real/short,equal,four,cut_ties,cluster,zeros,neg,inf,nan,bits,fp16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--time", default="real", help="cases timed (comma list)")
    ap.add_argument("--no-ref", action="store_true", help="skip the numpy reference (bytes vs the base only)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.init(), torch.zeros(1, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)

    def load(path):
        mod = ok(cu.cuModuleLoadData(Path(path).read_bytes())) if path else module("k31_topk")
        f = ok(cu.cuModuleGetFunction(mod, ENTRY))
        p, _ = ok(cu.cuModuleGetGlobal(mod, b"k31_topk_smem"))
        sm = torch.empty(1, dtype=torch.int32)
        ok(cu.cuMemcpyDtoH(sm.data_ptr(), p, 4))
        ok(cu.cuFuncSetAttribute(f, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(sm)))
        return f, int(sm)

    base, mine = load(a.base), load(a.cubin)
    L = Launcher()
    timed = set(a.time.split(","))
    bad, total = [], 0
    for n in map(int, a.n.split(",")):
        stride = -(-n // SPLIT_KV) * SPLIT_KV
        for case in a.cases.split(","):
            dist, _, lk = case.partition("/")
            for rows in map(int, a.rows.split(",")):
                if dist not in timed and rows != max(map(int, a.rows.split(","))):
                    continue
                x = torch.full((rows, stride), -7.0, device="cuda")
                x = x.bfloat16()
                x[:, :n] = (indexer_like(rows, n, g, dist == "peaky") if dist in ("real", "peaky")
                            else adversarial(dist, rows, n, g))
                lens = lens_of(lk or "full", rows, n, g).cuda()
                outs = [torch.full((rows, TOPK), -2, dtype=torch.int32, device="cuda") for _ in range(3)]
                run = lambda f, o: L(f[0], (rows, 1, 1), (512, 1, 1), [x, ctypes.c_int(stride), lens, o], f[1])
                run(base, outs[0]), run(mine, outs[1]), run(mine, outs[2])
                torch.cuda.synchronize()
                total += rows
                same = torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
                why = [] if same else ["bytes"]
                if not a.no_ref:
                    xf, ln, got = x.float().cpu().numpy(), lens.cpu().tolist(), outs[1].cpu().numpy()
                    wrong = [r for r in range(rows) if not numpy.array_equal(got[r], reference(xf[r], ln[r]))]
                    if wrong:
                        why.append(f"ref rows {wrong[:4]}")
                line = f"n {n:6d} {case:12s} rows {rows:4d}: {'OK' if not why else 'BAD ' + ' '.join(why)}"
                if dist in timed:
                    tb = L.time(a.iters, lambda: run(base, outs[0]))
                    tm = L.time(a.iters, lambda: run(mine, outs[1]))
                    line += f"  base {tb:6.2f} us  new {tm:6.2f} us  x{tb / tm:4.2f}"
                print(line, flush=True)
                if why:
                    bad.append(line)
    print(f"{total} rows, {len(bad)} bad cases")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
