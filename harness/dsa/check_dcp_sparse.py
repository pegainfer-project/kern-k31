#!/usr/bin/env python3
"""k3_dcp's SPARSE exchange as D runs it, NRANKS = 8 members split over the visible GPUs (every member's
Lamport stages peer-mapped, a member's grid a share of its GPU's SMs so all of them stay resident): each
member's sparse partials (o [R, 128, 512] bf16 behind a TMA map, (max, sum) a head, `live` a row, empty
partials' o NaN) through every head's W_UV, the push of heads to their owners, the merge and the gate in
one launch, against an f64 reference of the whole chain. Prints, per call, the relative RMS of `gated`
against it and the floor bf16 of the reference itself has; calls of a shape rotate the stages.

    nvcc -cubin -arch=sm_103a -DSPARSE=1 -o k3_dcp_sparse.cubin source/k3_dcp.cu
    python3 harness/dsa/check_dcp_sparse.py k3_dcp_sparse.cubin [--rows 1,7,64,100,256] [--calls 3]
"""
import argparse
import ctypes
import math

import torch
from cuda.bindings import driver as cu

HEADS, HEADS_PAD, LAT, NOPE, W = 96, 128, 512, 128, 8
HL = HEADS // W
REC = NOPE // 8 + 1
SMEM = 1024 + 4 * 2 * 128 * 64 * 2 + 128 * REC * 16 + 128  # k3_dcp.cu SPARSE's SMEM_BYTES
SMS = 152
LOG2E = math.log2(math.e)


def ok(r):
    err, *rest = r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(err)
    return rest[0] if len(rest) == 1 else rest


class Map(ctypes.Structure):
    _align_ = 64
    _fields_ = [("w", ctypes.c_uint64 * 16)]


def tmap(t, dims, strides, box):
    """A bf16 tiled map over `t`, 128 B swizzle, out-of-bounds zero."""
    m = ok(cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, len(dims), t.data_ptr(),
        [cu.cuuint64_t(x) for x in dims], [cu.cuuint64_t(x) for x in strides], [cu.cuuint32_t(x) for x in box],
        [cu.cuuint32_t(1)] * len(dims), cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
    out = Map()
    for i, x in enumerate(m.opaque):
        out.w[i] = int(x)
    return out


def reference(o, md, live, gate, w_uv):
    """gated [R, HEADS * NOPE] in f64: the members' partials merged by their lse, through W_UV, gated."""
    lse = torch.stack([torch.where((live[r][:, None] != 0) & (md[r][..., 0] != -math.inf) & (md[r][..., 1] > 0),
                                   (LOG2E * md[r][..., 0].double() + torch.log2(md[r][..., 1].double())) / LOG2E,
                                   torch.tensor(-math.inf, dtype=torch.float64)) for r in range(W)])[:, :, :HEADS]
    mx = lse.max(0).values
    wq = torch.where(lse == -math.inf, 0.0, torch.exp(lse - torch.where(mx == -math.inf, 0.0, mx)))
    num = sum(torch.where(wq[r][..., None] > 0, wq[r][..., None] * o[r][:, :HEADS].double(), 0.0) for r in range(W))
    den = wq.sum(0)[..., None]
    merged = torch.where(den > 0, num / den.clamp(min=1e-300), 0.0)
    return torch.einsum("bhc,hdc->bhd", merged, w_uv).reshape(len(gate), -1) * torch.sigmoid(gate.double())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--rows", default="1,7,64,100,256")
    ap.add_argument("--calls", type=int, default=3)
    a = ap.parse_args()
    ng = torch.cuda.device_count()
    dev = [torch.device(f"cuda:{r * ng // W}") for r in range(W)]
    ds = sorted(set(dev), key=str)
    grid = SMS // (W // ng) - 2
    fns = {}
    for d in ds:
        torch.zeros(1, device=d)
        with torch.cuda.device(d):
            m = ok(cu.cuModuleLoadData(open(a.cubin, "rb").read()))
            fns[d] = ok(cu.cuModuleGetFunction(m, b"kern_k3_dcp_exchange"))
            ok(cu.cuFuncSetAttribute(fns[d], cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     SMEM))
    for x in ds:
        for y in ds:
            if x != y:
                with torch.cuda.device(y):
                    peer = ok(cu.cuCtxGetCurrent())
                with torch.cuda.device(x):
                    r = cu.cuCtxEnablePeerAccess(peer, 0)[0]
                    assert r in (cu.CUresult.CUDA_SUCCESS, cu.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED), r
    rows = list(map(int, a.rows.split(",")))
    stage = W * max(rows) * HL * REC * 16
    lam = [torch.full((3 * stage // 2,), -32768, dtype=torch.int16, device=d) for d in dev]
    peers = [torch.tensor([t.data_ptr() for t in lam], dtype=torch.int64, device=d) for d in dev]
    state = [torch.zeros(8, dtype=torch.int32, device=d) for d in dev]
    err = [torch.zeros(1, dtype=torch.int32, device=d) for d in dev]
    streams = [torch.cuda.Stream(device=d) for d in dev]
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(HEADS * 256, LAT, generator=g) * 0.03).bfloat16()
    w_uv = w.view(HEADS, 256, LAT)[:, NOPE:].double()
    w_dev = {d: w.to(d) for d in ds}
    worst = 0.0
    for R in rows:
        for call in range(a.calls):
            # partials of one head share a direction (they average the same kind of latent rows)
            base = torch.randn(R, HEADS_PAD, LAT, generator=g)
            o = [(base + 0.5 * torch.randn(R, HEADS_PAD, LAT, generator=g)).bfloat16() for _ in range(W)]
            md = [torch.stack([torch.randn(R, HEADS_PAD, generator=g) * 2 + 5,
                               torch.rand(R, HEADS_PAD, generator=g) * 2000 + 1], -1) for _ in range(W)]
            live = [torch.randint(1, 2048, (R,), generator=g, dtype=torch.int32) for _ in range(W)]
            for r in range(1, W):  # rows whose positions have not reached every member yet
                live[r][torch.rand(R, generator=g) < 0.1 * r] = 0
            for r in range(W):
                md[r][R // 2, 3, 0] = -math.inf  # a head with nothing there
                o[r][live[r] == 0] = float("nan")  # an empty partial's o is whatever the dummy left
            gate = (torch.randn(R, HEADS * NOPE, generator=g) * 3).bfloat16()
            ref = reference(o, md, live, gate, w_uv)

            outs = [torch.full((R, HL * NOPE), 9.0, dtype=torch.bfloat16, device=d) for d in dev]
            ins = [(o[r].to(dev[r]), md[r].to(dev[r]), live[r].to(dev[r]),
                    gate.view(R, W, HL * NOPE)[:, r].contiguous().to(dev[r])) for r in range(W)]
            for d in ds:
                torch.cuda.synchronize(d)
            keep = []
            for r in range(W):
                args = [tmap(ins[r][0], [LAT, HEADS_PAD, R], [LAT * 2, HEADS_PAD * LAT * 2], [64, 1, 128]),
                        ins[r][1], ins[r][2], tmap(w_dev[dev[r]], [LAT, HEADS * 256], [LAT * 2], [64, 128]),
                        ins[r][3], outs[r], lam[r], peers[r], state[r], err[r], ctypes.c_int(r), ctypes.c_int(R),
                        ctypes.c_longlong(stage), ctypes.c_longlong(3_000_000_000)]
                k = [ctypes.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else x for x in args]
                p = (ctypes.c_void_p * len(k))(*[ctypes.addressof(x) for x in k])
                keep.append((k, p))
                with torch.cuda.device(dev[r]):
                    ok(cu.cuLaunchKernel(fns[dev[r]], grid, 1, 1, 256, 1, 1, SMEM, streams[r].cuda_stream,
                                         ctypes.addressof(p), 0))
            for d in ds:
                torch.cuda.synchronize(d)
            got = torch.cat([outs[r].cpu() for r in range(W)], -1).double()
            rms = lambda x: x.pow(2).mean().sqrt().item()
            rel = rms(got - ref) / rms(ref)
            worst = max(worst, rel)
            print(f"rows {R:3d} call {call}: err {max(int(x.item()) for x in err)}, finite "
                  f"{bool(torch.isfinite(got).all())}, gated relative RMS {rel:.3e} (bf16 of the reference "
                  f"{rms(ref.bfloat16().double() - ref) / rms(ref):.3e})")
    print(f"worst relative RMS {worst:.3e}")


if __name__ == "__main__":
    main()
