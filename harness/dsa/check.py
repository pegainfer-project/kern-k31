#!/usr/bin/env python3
"""The indexer chain of a K3.1 prefill call against a torch reference, on one GPU, no checkpoint:
k31_idx_key -> k31_idx_q -> k31_idx_plan -> k31_idx_gather -> DeepGEMM MQA logits -> vLLM top-k ->
k31_topk_slots, launched from the cubins scripts/build.py wrote, with the parameter blocks and
tensor maps the generator will use. Prints each stage's agreement and each launch's time.

    python3 harness/dsa/check.py [--ctx 3000,9000] [--rows 512] [--iters 20]

Needs torch with CUDA and cuda-python (cuda.bindings).
"""
import argparse
import ctypes
import math
from pathlib import Path

import torch
from cuda.bindings import driver as cu

ROOT = Path(__file__).resolve().parents[2]
IH, ID, TOPK, PAGE, ROW = 32, 128, 2048, 64, 144
SMS, SMEM = 152, 220160
MQA = ("_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb0ELb1ELb0ELb0ELj4ELj256ELj3ELj5ELj152ELj128ELj256E"
       "N7cutlass12float_e4m3_tEffLj2EEEvjjjPKjS4_S4_PT13_14CUtensorMap_stS7_S7_S7_S7_")


def ok(r):
    err, *rest = r
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else rest


def module(name):
    return ok(cu.cuModuleLoadData((ROOT / "build" / f"{name}.cubin").read_bytes()))


def tmap(t, dtype, dims, strides, box, swizzle):
    sw = {0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, 128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B}
    u32, u64 = (lambda xs: [cu.cuuint32_t(x) for x in xs]), (lambda xs: [cu.cuuint64_t(x) for x in xs])
    m = ok(cu.cuTensorMapEncodeTiled(dtype, len(dims), t.data_ptr(), u64(dims), u64(strides), u32(box),
                                     u32([1] * len(dims)),
                                     cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, sw[swizzle],
                                     cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
    return (ctypes.c_uint64 * 16)(*[int(x) for x in m.opaque])


class Launcher:
    def __init__(self):
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def __call__(self, fn, grid, block, args, smem=0):
        """args: tensors (their pointers) or ctypes scalars / tensor maps."""
        keep = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
        ptrs = (ctypes.c_void_p * len(keep))(*[ctypes.addressof(k) for k in keep])
        stream = torch.cuda.current_stream().cuda_stream
        ok(cu.cuLaunchKernel(fn, *grid, *block, smem, stream, ctypes.addressof(ptrs), 0))

    def time(self, iters, f):
        """Mean µs of one launch, replayed from a CUDA graph of `iters` of them (no host launch cost)."""
        f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                f()
        g.replay()
        self.ev[0].record()
        g.replay()
        self.ev[1].record()
        torch.cuda.synchronize()
        return self.ev[0].elapsed_time(self.ev[1]) * 1e3 / iters


def ue8m0(amax, eps):
    return torch.exp2(torch.ceil(torch.log2(torch.clamp(amax, min=eps) / 448.0)))


def ref_key(k, w, b):
    y = torch.nn.functional.layer_norm(k.float(), (ID,), w.float(), b.float(), 1e-5).bfloat16().float()
    sc = ue8m0(y.abs().amax(-1, keepdim=True), 1e-4)
    return (y / sc).to(torch.float8_e4m3fn), sc.squeeze(-1)


def ref_q(q, iw):
    v = q.float()
    sc = ue8m0(v.abs().amax(-1, keepdim=True), 1e-10)
    return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn), iw.float() * sc.squeeze(-1) * 0.015625


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="3000,9000", help="each sequence's length with this call's rows")
    ap.add_argument("--rows", type=int, default=512, help="this call's rows of each sequence (its tail)")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ctx = [int(c) for c in a.ctx.split(",")]
    nseq, R = len(ctx), a.rows * len(ctx)
    pages = [math.ceil(c / PAGE) for c in ctx]
    max_pages = max(pages)
    perm = torch.randperm(sum(pages)) + 1                         # page 0 stays unused
    bt = torch.zeros(nseq, max_pages, dtype=torch.int32)
    o = 0
    for s, p in enumerate(pages):
        bt[s, :p] = perm[o:o + p]
        o += p
    bt = bt.to(dev)
    cache = torch.zeros((sum(pages) + 1) * PAGE * ROW, dtype=torch.uint8, device=dev)
    L = Launcher()
    g, mqa_mod, topk_mod = module("k31_dsa"), module("k31_mqa_logits"), module("vllm_topk")
    fn = {n: ok(cu.cuModuleGetFunction(g, n.encode())) for n in
          ("k31_idx_key", "k31_idx_q", "k31_idx_plan", "k31_idx_gather", "k31_topk_slots")}
    mqa = ok(cu.cuModuleGetFunction(mqa_mod, MQA.encode()))
    ok(cu.cuFuncSetAttribute(mqa, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              SMEM))
    topk = ok(cu.cuModuleGetFunction(topk_mod, b"glm_topk_prefill"))
    ok(cu.cuFuncSetAttribute(topk, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              TOPK * 4))
    times = {}

    # 1. every position's key into the cache (prefix and this call alike), a slot through the page table
    N = sum(ctx)
    lda = ID + IH + 64                                              # a strided view, as from a fused GEMM
    proj = torch.randn(N, lda, device=dev).bfloat16()
    kw = (1 + 0.1 * torch.randn(ID, device=dev)).bfloat16()
    kb = (0.1 * torch.randn(ID, device=dev)).bfloat16()
    slot = torch.cat([bt[s, torch.arange(c, device=dev) // PAGE].long() * PAGE + torch.arange(c, device=dev) % PAGE
                      for s, c in enumerate(ctx)])
    key = lambda: L(fn["k31_idx_key"], ((N + 3) // 4, 1, 1), (128, 1, 1),
                    [proj, ctypes.c_int(lda), kw, kb, slot, cache, ctypes.c_int(N)])
    times["k31_idx_key"] = L.time(a.iters, key)
    rk, rsc = ref_key(proj[:, :ID], kw, kb)
    rows = cache.view(-1, ROW)[slot]
    kbytes = (rows[:, :ID] == rk.view(torch.uint8)).float().mean().item()
    kscale = (rows[:, ID:ID + 4].contiguous().view(torch.float32).squeeze(-1) == rsc).float().mean().item()
    print(f"key   bytes equal {kbytes:.6f}, scales equal {kscale:.6f}")
    bad = (rows[:, :ID] != rk.view(torch.uint8)).nonzero()
    for r, c in bad[:4].tolist():
        print(f"  key differs at row {r} col {c}: kernel {rows[r, c].view(torch.float8_e4m3fn).float().item()}"
              f" torch {rk[r, c].float().item()} (scale {rsc[r].item()})")

    # 2. this call's queries: the last `rows` positions of each sequence
    cu_rows = torch.tensor([0] + [a.rows * (s + 1) for s in range(nseq)], dtype=torch.int64, device=dev)
    seq_lens = torch.tensor(ctx, dtype=torch.int32, device=dev)
    ldq = IH * ID + 64
    q = torch.randn(R, ldq, device=dev).bfloat16()
    iw = torch.randn(R, IH + 8, device=dev).bfloat16()
    iq = torch.empty(R, IH, ID, dtype=torch.uint8, device=dev)
    wout = torch.empty(R, IH, dtype=torch.float32, device=dev)
    qq = lambda: L(fn["k31_idx_q"], (R, 1, 1), (256, 1, 1),
                   [q, ctypes.c_int(ldq), iw, ctypes.c_int(IH + 8), iq, wout, ctypes.c_int(R)])
    times["k31_idx_q"] = L.time(a.iters, qq)
    rq, rw = ref_q(q[:, :IH * ID].view(R, IH, ID), iw[:, :IH])
    print(f"query bytes equal {(iq == rq.view(torch.uint8)).float().mean().item():.6f}, "
          f"weights equal {(wout == rw).float().mean().item():.6f}")

    # 3. plan, gather, logits, top-k, slots
    ks, ke = (torch.empty(R, dtype=torch.int32, device=dev) for _ in range(2))
    klen, row_seq, tlen = (torch.empty(R, dtype=torch.int32, device=dev) for _ in range(3))
    zero = torch.zeros(R, dtype=torch.int32, device=dev)
    plan = lambda: L(fn["k31_idx_plan"], ((R + 255) // 256, 1, 1), (256, 1, 1),
                     [cu_rows, seq_lens, ctypes.c_int(nseq), ks, ke, klen, row_seq, ctypes.c_int(R)])
    times["k31_idx_plan"] = L.time(a.iters, plan)
    kg = torch.empty(N, ID, dtype=torch.uint8, device=dev)
    ksf = torch.empty(N, dtype=torch.float32, device=dev)
    gblocks = sum(pages) + nseq
    gather = lambda: L(fn["k31_idx_gather"], (gblocks, 1, 1), (256, 1, 1),
                       [cache, bt, ctypes.c_int(max_pages), seq_lens, ctypes.c_int(nseq), kg, ksf])
    times["k31_idx_gather"] = L.time(a.iters, gather)
    print(f"gather keys equal {(kg == rk.view(torch.uint8)).float().mean().item():.6f}, "
          f"scales equal {(ksf == rsc).float().mean().item():.6f}")

    stride = math.ceil(N / 256) * 256
    logits = torch.full((R, stride), float("nan"), dtype=torch.float32, device=dev)
    U8, F32 = cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    sf_map = tmap(ksf, F32, [N, 1], [N * 4], [256, 1], 0)
    maps = [tmap(iq, U8, [ID, R * IH], [ID], [ID, 128], 128), sf_map,
            tmap(kg, U8, [ID, N], [ID], [ID, 256], 128), sf_map,
            tmap(wout, F32, [IH, R], [IH * 4], [IH, 4], 0)]
    mqa_call = lambda: L(mqa, (SMS, 1, 1), (384, 1, 1),
                         [ctypes.c_int(R), ctypes.c_int(N), ctypes.c_int(stride), ks, ke, ctypes.c_int64(0), logits,
                          *maps], SMEM)
    times["mqa_logits"] = L.time(a.iters, mqa_call)
    kd = rk.float() * rsc[:, None]
    ref = torch.empty(R, N, device=dev)
    for r0 in range(0, R, 256):
        s = torch.einsum("rhd,nd->rhn", rq[r0:r0 + 256].float(), kd)
        ref[r0:r0 + 256] = (s.relu() * rw[r0:r0 + 256, :, None]).sum(1)
    err = []
    for r in range(R):
        n, s0 = int(klen[r]), int(ks[r])
        x, y = logits[r, :n], ref[r, s0:s0 + n]
        err.append(((x - y).abs().max() / y.abs().max().clamp(min=1e-6)).item())
    print(f"logits max relative error {max(err):.3e}")

    ind = torch.empty(R, TOPK, dtype=torch.int32, device=dev)
    tk = lambda: L(topk, (R, 1, 1), (512, 1, 1),
                   [logits, zero, klen, ind, ctypes.c_int(stride), ctypes.c_int(1), ctypes.c_int(TOPK),
                    ctypes.c_int(0)], TOPK * 4)
    times["topk"] = L.time(a.iters, tk)
    recall, rrecall = [], []
    for r in range(R):
        n, s0 = int(klen[r]), int(ks[r])
        got = set(ind[r][ind[r] >= 0].tolist())
        want = set(torch.topk(logits[r, :n], min(n, TOPK)).indices.tolist())
        recall.append(len(got & want) / len(want))
        true = set(torch.topk(ref[r, s0:s0 + n], min(n, TOPK)).indices.tolist())
        rrecall.append(len(got & true) / len(true))
    print(f"top-k recall vs torch.topk of the kernel's own logits: min {min(recall):.6f}")
    print(f"top-k recall vs the f32 reference's top-k: min {min(rrecall):.6f}, mean {sum(rrecall) / R:.6f}")

    picked = ind.clone()
    slots = lambda: L(fn["k31_topk_slots"], (R, 1, 1), (256, 1, 1),
                      [ind, bt, ctypes.c_int(max_pages), row_seq, klen, tlen, ctypes.c_int(R)])
    slots()
    torch.cuda.synchronize()
    want = torch.where(picked < 0, -1, bt[row_seq.long()[:, None], (picked // PAGE).clamp(min=0).long()] * PAGE
                       + picked % PAGE)
    print(f"slots equal {(ind == want).float().mean().item():.6f}, "
          f"tlen ok {bool((tlen == klen.clamp(max=TOPK)).all())}")

    print(f"\nrows {R}, keys {N} ({', '.join(map(str, ctx))}), us per launch:")
    for k, v in times.items():
        print(f"  {k:16s} {v:10.1f}")


if __name__ == "__main__":
    main()
