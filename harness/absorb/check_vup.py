#!/usr/bin/env python3
"""k31_vup_gate against torch: the sparse kernel's latent output (heads padded to 128) through W_UV, landed
to bf16, times bf16(sigmoid(gate)), landed to bf16; the gate read from a wider fused row as P's is.

    python3 harness/absorb/check_vup.py [--rows 1,63,64,65,8192]
"""
import argparse
import ctypes
import sys
from pathlib import Path

import torch
from cuda.bindings import driver as cu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa"))
from check import Launcher, module, ok  # noqa: E402

HEADS, IN_HEADS, NOPE, LAT = 96, 128, 128, 512
SMEM = 3 * (64 + 128) * 72 * 2
GATE_STRIDE, GATE_AT = 14400 + 160, 2112


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,63,64,65,8192")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    L = Launcher()
    fn = ok(cu.cuModuleGetFunction(module("k31_vup_gate"), b"kern_k31_vup_gate"))
    ok(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM))
    w = (torch.randn(HEADS * 256, LAT, device=dev) * 0.05).bfloat16()
    w_uv = w.view(HEADS, 256, LAT)[:, NOPE:].float()
    for B in map(int, a.rows.split(",")):
        lat = torch.randn(B, IN_HEADS, LAT, device=dev).bfloat16()
        fused = (torch.randn(B, GATE_STRIDE, device=dev) * 3).bfloat16()
        gate = fused[:, GATE_AT:]
        out = torch.full((B + 1, HEADS * NOPE), 7.0, dtype=torch.bfloat16, device=dev)
        run = lambda: L(fn, ((B + 63) // 64, HEADS, 1), (256, 1, 1),
                        [lat, w, gate, ctypes.c_int(GATE_STRIDE), out, ctypes.c_int(B)], SMEM)
        us = L.time(a.iters, run)
        o = torch.einsum("bhj,hdj->bhd", lat[:, :HEADS].float(), w_uv).bfloat16().float().reshape(B, -1)
        s = torch.sigmoid(gate[:, :HEADS * NOPE].float()).bfloat16().float()
        want = (o * s).bfloat16()
        got = out[:B]
        exact = (got == want).float().mean().item()
        rel = ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
        tail = bool((out[B] == 7.0).all())
        tf = 2 * B * HEADS * NOPE * LAT / us / 1e6
        print(f"rows {B:5d}: bf16 equal {exact:.6f}, max relative diff {rel:.3e}, row B untouched {tail}, "
              f"{us:8.1f} us ({tf:5.0f} TF)")


if __name__ == "__main__":
    main()
