#!/usr/bin/env python3
"""k31_topk against vLLM's glm_topk_prefill and an exact reference, on one GPU: every row's set must be
the reference's top 2048 (ties at the cut taken by smallest position), sorted, and the same bytes on a
second run. Rows cover causal lengths of a prefill chunk, short rows, and degenerate ones (constant,
few distinct values) that exercise the fallback. Prints both kernels' time on the chunk.

    python3 harness/topk/check.py [--ctx 196608] [--rows 8192] [--iters 10]
"""
import argparse
import ctypes
import sys
from pathlib import Path

import numpy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa"))
from check import Launcher, module, ok, cu  # noqa: E402

TOPK = 2048


def reference(x, n):
    """The top TOPK of x[:n] by value, ties by smallest position, ascending."""
    if n <= TOPK:
        return list(range(n)) + [-1] * (TOPK - n)
    v = x[:n].cpu().numpy()
    order = numpy.lexsort((-numpy.arange(n), v))
    return sorted(order[-TOPK:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=196608)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cubin", default=None, help="a k31_topk build other than build/k31_topk.cubin")
    ap.add_argument("--nt", type=int, default=512, help="its block size")
    ap.add_argument("--plain", action="store_true", help="random rows only (no degenerate rows at the head)")
    ap.add_argument("--check", type=int, default=64, help="rows of the chunk checked against the reference")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    R, stride = a.rows, (a.ctx + 255) // 256 * 256
    lens = torch.arange(a.ctx - R + 1, a.ctx + 1, dtype=torch.int32, device=dev)
    logits = torch.randn(R, stride, device=dev) * 3
    # degenerate rows at the head: constant, four values, a heavy tie at the cut, short, just past TOPK
    if not a.plain:
        logits[0].fill_(1.5)
        logits[1] = torch.randint(0, 4, (stride,), device=dev).float()
        logits[2] = torch.where(torch.rand(stride, device=dev) < 0.1, torch.randn(stride, device=dev) + 4,
                                torch.full((stride,), 0.25, device=dev))
        lens[3], lens[4], lens[6] = 1000, TOPK + 1, 20000
        logits[7, int(lens[7]) - 3000: int(lens[7])] += 6  # a recent window the sample sees little of
        logits[8, :100] += 10                               # sinks
        logits[9, :] = -logits[9, :].abs()                  # all negative
        logits[10] = 1 + 1e-3 * torch.randn(stride, device=dev)            # one narrow cluster
        logits[11] = torch.relu(torch.randn(stride, device=dev) - 1.5)     # mostly exact zeros
        logits[12, 1000:] = -float("inf")                                   # fewer finite than 2048
        lens[13] = 5000
        logits[13, :5000] = 1 + 1e-3 * torch.randn(5000, device=dev)
        logits[5, : lens[5]] = torch.randn(int(lens[5]), device=dev).half().float()  # fp16-exact: bins hold ties
    zero = torch.zeros(R, dtype=torch.int32, device=dev)
    mod = ok(cu.cuModuleLoadData(Path(a.cubin).read_bytes())) if a.cubin else module("k31_topk")
    mine = ok(cu.cuModuleGetFunction(mod, b"k31_topk"))
    p, _ = ok(cu.cuModuleGetGlobal(mod, b"k31_topk_smem"))
    sm = torch.empty(1, dtype=torch.int32)
    ok(cu.cuMemcpyDtoH(sm.data_ptr(), p, 4))
    smem = int(sm)
    ok(cu.cuFuncSetAttribute(mine, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
    vllm = ok(cu.cuModuleGetFunction(module("vllm_topk"), b"glm_topk_prefill"))
    ok(cu.cuFuncSetAttribute(vllm, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, TOPK * 4))
    L = Launcher()
    out, again, ref_out = (torch.empty(R, TOPK, dtype=torch.int32, device=dev) for _ in range(3))
    run = lambda o: L(mine, (R, 1, 1), (a.nt, 1, 1), [logits, ctypes.c_int(stride), lens, o], smem)
    runv = lambda: L(vllm, (R, 1, 1), (512, 1, 1),
                     [logits, zero, lens, ref_out, ctypes.c_int(stride), ctypes.c_int(1), ctypes.c_int(TOPK),
                      ctypes.c_int(0)], TOPK * 4)
    run(out), run(again), runv()
    torch.cuda.synchronize()
    rows = sorted(set(range(min(14, R))) | set(range(max(0, R - a.check), R)) | set(torch.randint(0, R, (a.check,)).tolist()))
    bad = []
    for r in rows:
        n = int(lens[r])
        want = reference(logits[r], n)
        got = out[r].tolist()
        if got != want:
            bad.append((r, n, len(set(got) ^ set(want))))
        if (a.plain or r not in (0, 1, 2, 5, 9, 11, 12, 13)) and n > TOPK:
            v = set(ref_out[r][ref_out[r] >= 0].tolist())
            if v != set(want):
                bad.append(("vllm", r, len(v ^ set(want))))
    print(f"rows checked {len(rows)}, mismatches {bad[:8]} ({len(bad)})")
    print(f"deterministic: {torch.equal(out, again)}")
    t_mine = L.time(a.iters, lambda: run(out))
    t_vllm = L.time(a.iters, runv)
    gb = (lens.double().sum() * 4 / 1e9).item()
    print(f"smem {smem}; rows {R} ctx {a.ctx}: k31_topk {t_mine:.1f} us ({gb / t_mine * 1e6:.0f} GB/s of one read), "
          f"vllm {t_vllm:.1f} us ({gb / t_vllm * 1e6:.0f} GB/s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
