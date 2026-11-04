#!/usr/bin/env python3
"""k3_dcp's SPARSE exchange on NRANKS GPUs of one process, the members' Lamport stages peer-mapped: each
member's sparse partials (o [R, 128, 512] bf16, (max, sum) a head, `live` a row) through fixup, the
exchange of heads, the merge, W_UV and the gate in one launch, against torch. The build's NRANKS must be
the GPUs given (`nvcc -cubin -arch=sm_103a -O3 -DSPARSE -DNRANKS=4 source/k3_dcp.cu`).

    python3 harness/dsa/check_dcp_sparse.py <cubin> [--nranks 4] [--rows 1,7,64]
"""
import argparse
import ctypes
import math

import torch
from cuda.bindings import driver as cu


def ok(r):
    err, *rest = r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(err)
    return rest[0] if len(rest) == 1 else rest


HEADS, HEADS_PAD, LAT, NOPE = 96, 128, 512, 128
SMEM, GRID = 65536, 256
LOG2E = math.log2(math.e)


def launch(fn, args, stream):
    keep = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
    ptrs = (ctypes.c_void_p * len(keep))(*[ctypes.addressof(k) for k in keep])
    ok(cu.cuLaunchKernel(fn, GRID, 1, 1, 256, 1, 1, SMEM, stream, ctypes.addressof(ptrs), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--nranks", type=int, default=4)
    ap.add_argument("--rows", default="1,7,64")
    ap.add_argument("--calls", type=int, default=5, help="calls a shape, so the stages rotate")
    a = ap.parse_args()
    W = a.nranks
    HL = HEADS // W
    torch.manual_seed(0)
    devs = [torch.device(f"cuda:{r}") for r in range(W)]
    fns = []
    for d in devs:
        torch.zeros(1, device=d)
        with torch.cuda.device(d):
            m = ok(cu.cuModuleLoadData(open(a.cubin, "rb").read()))
            f = ok(cu.cuModuleGetFunction(m, b"kern_k3_dcp_exchange"))
            ok(cu.cuFuncSetAttribute(f, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM))
            fns.append(f)
    for i in range(W):
        for j in range(W):
            if i != j:
                with torch.cuda.device(devs[i]):
                    ctx = ok(cu.cuCtxGetCurrent())
                    with torch.cuda.device(devs[j]):
                        peer = ok(cu.cuCtxGetCurrent())
                    ok(cu.cuCtxSetCurrent(ctx))
                    r = cu.cuCtxEnablePeerAccess(peer, 0)[0]
                    assert r in (cu.CUresult.CUDA_SUCCESS,
                                 cu.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED), r
    R_MAX = max(map(int, a.rows.split(",")))
    stage = W * R_MAX * HL * (LAT // 8 + 1) * 16
    lam = [torch.full((3 * stage // 2,), -32768, dtype=torch.int16, device=d) for d in devs]
    peers = [torch.tensor([t.data_ptr() for t in lam], dtype=torch.int64, device=d) for d in devs]
    state = [torch.zeros(8, dtype=torch.int32, device=d) for d in devs]
    err = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    w = (torch.randn(HEADS * 256, LAT) * 0.05).bfloat16()
    w_uv = w.view(HEADS, 256, LAT)[:, NOPE:].float()
    w_l = [w.view(W, HL * 256, LAT)[r].contiguous().to(devs[r]) for r in range(W)]
    worst = 0.0
    for R in map(int, a.rows.split(",")):
        for call in range(a.calls):
            o = [torch.randn(R, HEADS_PAD, LAT).bfloat16() for _ in range(W)]
            md = [torch.stack([torch.randn(R, HEADS_PAD) * 2 + 5, torch.rand(R, HEADS_PAD) * 2000 + 1], -1)
                  for _ in range(W)]
            live = [torch.randint(0, 2, (R,), dtype=torch.int32) * torch.randint(1, 2048, (R,), dtype=torch.int32)
                    for _ in range(W)]
            live[0][0] = 5
            for r in range(W):
                md[r][R // 2, 3, 0] = -math.inf  # a head with nothing there
            gate = (torch.randn(R, HEADS * NOPE) * 3).bfloat16()
            # reference: natural lse a partial, the merge in f32, bf16 landed; W_UV, bf16; the gate, bf16
            lse = torch.stack([torch.where((live[r][:, None] != 0) & (md[r][..., 0] != -math.inf) & (md[r][..., 1] > 0),
                                           (LOG2E * md[r][..., 0] + torch.log2(md[r][..., 1])) / LOG2E,
                                           torch.tensor(-math.inf)) for r in range(W)])[:, :, :HEADS]
            mx = lse.max(0).values
            wq = torch.where(lse == -math.inf, 0.0, torch.exp(lse - torch.where(mx == -math.inf, 0.0, mx)))
            num = sum(wq[r][..., None] * o[r][:, :HEADS].float() for r in range(W))
            den = wq.sum(0)[..., None]
            merged = torch.where(den > 0, num / den.clamp(min=1e-30), 0.0).bfloat16().float()
            up = torch.einsum("bhc,hdc->bhd", merged, w_uv).bfloat16().float().reshape(R, -1)
            want = (up * torch.sigmoid(gate.float()).bfloat16().float()).bfloat16()

            outs = []
            for r, d in enumerate(devs):
                outs.append(torch.full((R, HL * NOPE), 9.0, dtype=torch.bfloat16, device=d))
            ins = [(o[r].to(devs[r]), md[r].to(devs[r]), live[r].to(devs[r]),
                    gate.view(R, W, HL * NOPE)[:, r].contiguous().to(devs[r])) for r in range(W)]
            for d in devs:
                torch.cuda.synchronize(d)
            for r, d in enumerate(devs):
                with torch.cuda.device(d):
                    s = torch.cuda.current_stream(d).cuda_stream
                    launch(fns[r], [ins[r][0], ins[r][1], ins[r][2], w_l[r], ins[r][3], outs[r], lam[r], peers[r],
                                    state[r], err[r], ctypes.c_int(r), ctypes.c_int(R), ctypes.c_longlong(stage),
                                    ctypes.c_longlong(5_000_000_000)], s)
            for d in devs:
                torch.cuda.synchronize(d)
            got = torch.cat([outs[r].cpu() for r in range(W)], -1)
            e = [int(x.item()) for x in err]
            exact = (got == want).float().mean().item()
            rel = ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
            worst = max(worst, rel)
            print(f"rows {R:3d} call {call}: err {e}, gated bf16 equal {exact:.6f}, max relative diff {rel:.3e}")
    print(f"worst max relative diff {worst:.3e}")


if __name__ == "__main__":
    main()
