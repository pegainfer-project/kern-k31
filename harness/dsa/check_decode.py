#!/usr/bin/env python3
"""D's indexer under DCP on one GPU, W = 8 members emulated: every member's keys in its own 144-byte
token-major pages (position p on member p % 8), its local logits by the paged MQA kernel
(k31_paged_mqa_logits, with DeepGEMM's metadata kernel), its local top-k (vLLM's), its candidates
(k31_dcp_pack), the all-gather as a concatenation, and every member's share of the group's top 2048
(k31_dcp_select). Checks the local logits against a torch reference and the union of the members'
selections against the stable global top-k (vLLM's merge order) of the same logits.

    python3 harness/dsa/check_decode.py [--ctx-max 200000] [--rows 64]
"""
import argparse
import ctypes
import math

import torch
from cuda.bindings import driver as cu

from check import IH, ID, PAGE, ROW, TOPK, Launcher, module, ok, ref_key, ref_q, tmap

W, SMS = 8, 152
PAGED = ("_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj32ELj128ELj64ELb0ELb1ELb0ELj3ELj4ELj256ELj16ELj128ELj256E"
         "N7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")
META = "_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb1ELb0ELj256ELj152EEEvjjPKjS3_Pj"
PAGED_SMEM = 198656


def stable_topk(scores, gids, k):
    """vLLM's merge: higher score first, smaller global position on a tie."""
    order = sorted(range(len(gids)), key=lambda i: (-scores[i], gids[i]))
    return {gids[i] for i in order[:k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-max", type=int, default=200000, help="the longest row's global context")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    R = a.rows
    ctx = torch.randint(1, a.ctx_max + 1, (R,)).tolist()
    ctx[:4] = [a.ctx_max, 1, 9, 2047 * W][: min(R, 4)]
    torch.zeros(1, device=dev)  # the primary context the cubins load into
    L = Launcher()
    g, pm, topk_mod = module("k31_dsa"), module("k31_paged_mqa_logits"), module("vllm_topk")
    fn = {n: ok(cu.cuModuleGetFunction(g, n.encode())) for n in ("k31_idx_key", "k31_idx_q", "k31_dcp_pack",
                                                                  "k31_dcp_select")}
    paged, meta = ok(cu.cuModuleGetFunction(pm, PAGED.encode())), ok(cu.cuModuleGetFunction(pm, META.encode()))
    topk = ok(cu.cuModuleGetFunction(topk_mod, b"glm_topk_prefill"))
    attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    ok(cu.cuFuncSetAttribute(paged, attr, PAGED_SMEM))
    ok(cu.cuFuncSetAttribute(topk, attr, TOPK * 4))
    U8, F32 = cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32

    # the queries, one a row, shared by every member
    q = torch.randn(R, IH * ID, device=dev).bfloat16()
    iw = torch.randn(R, IH, device=dev).bfloat16()
    iq = torch.empty(R, IH, ID, dtype=torch.uint8, device=dev)
    wout = torch.empty(R, IH, dtype=torch.float32, device=dev)
    L(fn["k31_idx_q"], (R, 1, 1), (256, 1, 1), [q, ctypes.c_int(IH * ID), iw, ctypes.c_int(IH), iq, wout,
                                                ctypes.c_int(R)])
    rq, rw = ref_q(q.view(R, IH, ID), iw)
    kw = (1 + 0.1 * torch.randn(ID, device=dev)).bfloat16()
    kb = (0.1 * torch.randn(ID, device=dev)).bfloat16()

    gathered, members, times, err = [], [], {}, 0.0
    for m in range(W):
        lens = [max(0, (c - m + W - 1) // W) for c in ctx]
        pages = [max(1, math.ceil(n / PAGE)) for n in lens]
        max_pages = max(pages)
        perm = torch.randperm(sum(pages)) + 1
        bt = torch.zeros(R, max_pages, dtype=torch.int32)
        o = 0
        for r, p in enumerate(pages):
            bt[r, :p] = perm[o:o + p]
            o += p
        bt = bt.to(dev)
        npages = sum(pages) + 1
        cache = torch.zeros(npages * PAGE * ROW, dtype=torch.uint8, device=dev)
        N = sum(lens)
        kproj = torch.randn(N, ID, device=dev).bfloat16()
        slot = torch.cat([bt[r, torch.arange(n, device=dev) // PAGE].long() * PAGE + torch.arange(n, device=dev) % PAGE
                          for r, n in enumerate(lens)])
        L(fn["k31_idx_key"], ((N + 3) // 4, 1, 1), (128, 1, 1),
          [kproj, ctypes.c_int(ID), kw, kb, slot, cache, ctypes.c_int(N)])
        rk, rsc = ref_key(kproj, kw, kb)

        lens_t = torch.tensor(lens, dtype=torch.int32, device=dev)
        stride = max(256, math.ceil(max(lens) / 256) * 256)
        logits = torch.full((R, stride), float("nan"), dtype=torch.float32, device=dev)
        sched = torch.empty((SMS + 1) * 2, dtype=torch.int32, device=dev)
        page_bytes = PAGE * ROW
        sf = cache[ID:]
        maps = [tmap(iq, U8, [ID, R * IH], [ID], [ID, 128], 128),
                tmap(sf, F32, [4, PAGE, npages], [ROW, page_bytes], [4, PAGE, 1], 0),
                tmap(cache, U8, [ID, PAGE, npages], [ROW, page_bytes], [ID, PAGE, 1], 128),
                tmap(sf, F32, [4, PAGE, npages], [ROW, page_bytes], [4, PAGE, 1], 0),
                tmap(wout, F32, [IH, R], [IH * 4], [IH, 4], 0)]
        md = lambda: L(meta, (1, 1, 1), (1024, 1, 1), [ctypes.c_uint(R), ctypes.c_uint(R), lens_t, ctypes.c_void_p(0),
                                                      sched], (R + 32 + 1) * 4)
        lg = lambda: L(paged, (SMS, 1, 1), (384, 1, 1),
                       [ctypes.c_uint(R), ctypes.c_uint(stride), ctypes.c_uint(max_pages), lens_t, logits, bt,
                        ctypes.c_void_p(0), sched, *maps], PAGED_SMEM)
        times.setdefault("paged_meta", []).append(L.time(a.iters, md))
        times.setdefault("paged_logits", []).append(L.time(a.iters, lg))
        kd = rk.float() * rsc[:, None]
        o = 0
        for r, n in enumerate(lens):
            if n:
                s = (torch.einsum("hd,nd->hn", rq[r].float(), kd[o:o + n]).relu() * rw[r, :, None]).sum(0)
                err = max(err, ((logits[r, :n] - s).abs().max() / s.abs().max().clamp(min=1e-6)).item())
            o += n

        ind = torch.empty(R, TOPK, dtype=torch.int32, device=dev)
        zero = torch.zeros(R, dtype=torch.int32, device=dev)
        tk = lambda: L(topk, (R, 1, 1), (512, 1, 1), [logits, zero, lens_t, ind, ctypes.c_int(stride), ctypes.c_int(1),
                                                      ctypes.c_int(TOPK), ctypes.c_int(0)], TOPK * 4)
        times.setdefault("local_topk", []).append(L.time(a.iters, tk))
        cand = torch.empty(R, TOPK, 2, dtype=torch.float32, device=dev)
        pk = lambda: L(fn["k31_dcp_pack"], (R, 1, 1), (256, 1, 1), [logits, ctypes.c_int(stride), ind, ctypes.c_int(m),
                                                                     ctypes.c_int(W), cand])
        times.setdefault("pack", []).append(L.time(a.iters, pk))
        gathered.append(cand)
        members.append((bt, max_pages, logits, lens))

    allc = torch.stack(gathered)  # [W, R, 2048, 2], as the all-gather lays it out
    picked = [set() for _ in range(R)]
    for m, (bt, max_pages, logits, lens) in enumerate(members):
        sel, sl, live = (torch.empty(R, TOPK, dtype=torch.int32, device=dev), torch.empty(R, dtype=torch.int32,
                         device=dev), torch.empty(R, dtype=torch.int32, device=dev))
        sc = lambda: L(fn["k31_dcp_select"], (R, 1, 1), (1024, 1, 1),
                       [allc, ctypes.c_int(R), ctypes.c_int(m), bt, ctypes.c_int(max_pages), sel, sl, live])
        times.setdefault("select", []).append(L.time(a.iters, sc))
        # back from slots to local positions to global positions
        inv = {}
        for r in range(R):
            for j, p in enumerate(bt[r].tolist()):
                inv[(r, p)] = j
        for r in range(R):
            n = int(live[r])
            assert int(sl[r]) == max(n, 1)
            for s in sel[r, :n].tolist():
                picked[r].add((inv[(r, s // PAGE)] * PAGE + s % PAGE) * W + m)
    bad = 0
    for r in range(R):
        sc, gid = [], []
        for m, (_, _, logits, lens) in enumerate(members):
            row = logits[r, :lens[r]].tolist()
            sc += row
            gid += [l * W + m for l in range(lens[r])]
        bad += picked[r] != stable_topk(sc, gid, TOPK)
    print(f"rows {R}, global ctx 1..{max(ctx)}: local logits max relative error {err:.3e}; "
          f"rows whose merged selection differs from the stable global top-k: {bad}")
    print("us per launch (mean over members):")
    for k, v in times.items():
        print(f"  {k:14s} {sum(v) / len(v):10.1f}")


if __name__ == "__main__":
    main()
