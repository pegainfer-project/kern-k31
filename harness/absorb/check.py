#!/usr/bin/env python3
"""The absorbed query of K3.1's sparse attention, D's (k3_mla_absorb_mma's OUT_HEADS=128 build, an f32
partial, at most 64 rows) and P's (k31_absorb, the bf16 query, any rows, grid.x swept), against torch: q_nope through W_UK in f32 then bf16 then e4m3, the rope columns
copied through bf16 to e4m3, heads 96.. left alone.

    python3 harness/absorb/check.py [--rows 1,17,64,65,8192] [--gx 1,2,4,8]
"""
import argparse
import ctypes
import sys
from pathlib import Path

import torch
from cuda.bindings import driver as cu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa"))
from check import Launcher, module, ok  # noqa: E402

HEADS, NOPE, ROPE, LAT, QW, OUT_HEADS = 96, 128, 64, 512, 192, 128
ROW = LAT + ROPE
BUILDS = {"f32": ("k3_mla_absorb_mma+OUT_HEADS=128+Q_FP8=1", "kern_k3_mla_absorb_mma"),
          "bf16": ("k31_absorb", "kern_k31_absorb")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,17,64,65,8192")
    ap.add_argument("--gx", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    L = Launcher()
    w = (torch.randn(HEADS * 256, LAT, device=dev) * 0.05).bfloat16()
    w_uk = w.view(HEADS, 256, LAT)[:, :NOPE].float()
    for kind, (name, entry) in BUILDS.items():
        fn = ok(cu.cuModuleGetFunction(module(name), entry.encode()))
        shapes = [(B, gx) for B in map(int, a.rows.split(",")) for gx in
                  ([1] if kind == "f32" else sorted({min(g, (B + 63) // 64) for g in map(int, a.gx.split(","))}))
                  if kind == "bf16" or B <= 64]
        for B, gx in shapes:
            q = torch.randn(B, HEADS * QW, device=dev)
            q = q.bfloat16() if kind == "bf16" else q
            out = torch.full((B, OUT_HEADS, ROW), 0x5A, dtype=torch.uint8, device=dev)
            run = lambda: L(fn, (gx, HEADS, 4), (256, 1, 1), [q, w, out, ctypes.c_int(B)])
            us = L.time(a.iters, run)
            qh = q.view(B, HEADS, QW).bfloat16().float()
            lat = torch.einsum("bhd,hdj->bhj", qh[..., :NOPE], w_uk).bfloat16().float()
            want = torch.cat([lat, qh[..., NOPE:]], -1).to(torch.float8_e4m3fn)
            got = out[:, :HEADS].view(torch.float8_e4m3fn)
            exact = (got.view(torch.uint8) == want.view(torch.uint8)).float().mean().item()
            ulp = (got.float() - want.float()).abs().max().item()
            pad = bool((out[:, HEADS:] == 0x5A).all())
            print(f"{kind:4s} rows {B:5d} grid.x {gx:2d}: e4m3 bytes equal {exact:.6f}, max abs diff {ulp:.3e}, "
                  f"pad heads untouched {pad}, {us:8.1f} us")


if __name__ == "__main__":
    main()
