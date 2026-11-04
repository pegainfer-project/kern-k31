#!/usr/bin/env python3
"""The absorbed query of K3.1's sparse attention, D's (k3_mla_absorb_mma's OUT_HEADS=128 build, an f32
partial, at most 64 rows) and P's (k31_absorb_tc, the bf16 query, any rows), against torch: q_nope through
W_UK in f32 then bf16 then e4m3, the rope columns copied through bf16 to e4m3, heads 96.. left alone.

    python3 harness/absorb/check.py [--rows 1,17,64,65,8192]
"""
import argparse
import ctypes
import sys
from pathlib import Path

import torch
from cuda.bindings import driver as cu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa"))
from check import Launcher, module, ok, tmap  # noqa: E402

HEADS, NOPE, ROPE, LAT, QW, OUT_HEADS = 96, 128, 64, 512, 192, 128
ROW = LAT + ROPE
BUILDS = {"f32": ("k3_mla_absorb_mma+OUT_HEADS=128+Q_FP8=1", "kern_k3_mla_absorb_mma"),
          "tc": ("k31_absorb_tc", "kern_k31_absorb_tc")}
TC_SMEM, SMS = 1024 + 3 * (16384 + 4 * 8192) + 32768 + 16384 + 8192 + 256, 152


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,17,64,65,8192")
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
        if kind == "tc":
            ok(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     TC_SMEM))
        for B in [B for B in map(int, a.rows.split(",")) if kind == "tc" or B <= 64]:
            q = torch.randn(B, HEADS * QW, device=dev)
            q = q if kind == "f32" else q.bfloat16()
            out = torch.full((B, OUT_HEADS, ROW), 0x5A, dtype=torch.uint8, device=dev)
            if kind == "tc":
                bf16, u8 = (cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8)
                maps = [tmap(q, bf16, [QW, HEADS, B], [QW * 2, HEADS * QW * 2], [64, 1, 128], 128),
                        tmap(w, bf16, [LAT, HEADS * 256], [LAT * 2], [64, 64], 128),
                        tmap(out, u8, [ROW, OUT_HEADS, B], [ROW, OUT_HEADS * ROW], [128, 1, 128], 128),
                        tmap(out, u8, [ROW, OUT_HEADS, B], [ROW, OUT_HEADS * ROW], [64, 1, 128], 0)]
                run = lambda: L(fn, (SMS - 1, 1, 1), (384, 1, 1), [*maps, ctypes.c_int(B)], TC_SMEM)
            else:
                run = lambda: L(fn, (1, HEADS, 4), (256, 1, 1), [q, w, out, ctypes.c_int(B)])
            us = L.time(a.iters, run)
            qh = q.view(B, HEADS, QW).bfloat16().float()
            lat = torch.einsum("bhd,hdj->bhj", qh[..., :NOPE], w_uk).bfloat16().float()
            want = torch.cat([lat, qh[..., NOPE:]], -1).to(torch.float8_e4m3fn)
            got = out[:, :HEADS].view(torch.float8_e4m3fn)
            exact = (got.view(torch.uint8) == want.view(torch.uint8)).float().mean().item()
            ulp = (got.float() - want.float()).abs().max().item()
            pad = bool((out[:, HEADS:] == 0x5A).all())
            print(f"{kind:4s} rows {B:5d}: e4m3 bytes equal {exact:.6f}, max abs diff {ulp:.3e}, "
                  f"pad heads untouched {pad}, {us:8.1f} us")


if __name__ == "__main__":
    main()
