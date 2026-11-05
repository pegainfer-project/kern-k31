#!/usr/bin/env python3
"""D's indexer under DCP on one GPU, W = 8 members emulated: every member's keys in its own 144-byte
token-major pages (position p on member p % 8), its local logits by the paged MQA kernel
(k31_paged_mqa_logits, bf16 out, with DeepGEMM's metadata kernel), its local top-k (k31_topk_held), its
best 384 candidates pushed into every member's Lamport stage (k31_dcp_pack, the members' stages and states on one GPU),
and every member's share of the group's top 2048 (k31_dcp_select). Checks the local logits against a torch reference and the union of the members' selections
against the global top 2048 of the same logits, by value: bf16 ties may pick any of the equal positions.

    python3 harness/dsa/check_decode.py [--ctx-max 200000] [--rows 64]
"""
import argparse
import ctypes
import math

import torch
from cuda.bindings import driver as cu

from check import (BF16, IH, ID, PAGE, ROW, SPLIT_KV, TOPK, TOPK_SMEM, U8, U32, Launcher, module, ok, ref_key,
                   ref_logits, ref_q, tmap, topk_exact)

W, SMS, HOT = 8, 152, 384
PAGED = ("_ZN9deep_gemm22sm100_paged_mqa_logitsILj32ELj128ELj64ELj4ELj128ELj3ELj3ELj3ELj384ELj8ELj128ELj384E"
         "N7cutlass12float_e4m3_tELj3EEEvjjjPKjP13__nv_bfloat16S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")
META = "_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj384ELj152ELj4ELj256EEEvjPKjS3_Pj"
PAGED_SMEM, META_THREADS = 218112, 256


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
    g, pm, topk_mod = module("k31_dsa"), module("k31_paged_mqa_logits"), module("k31_topk")
    fn = {n: ok(cu.cuModuleGetFunction(g, n.encode())) for n in ("k31_idx_key", "k31_idx_q", "k31_dcp_pack",
                                                                  "k31_dcp_poison", "k31_dcp_select")}
    paged, meta = ok(cu.cuModuleGetFunction(pm, PAGED.encode())), ok(cu.cuModuleGetFunction(pm, META.encode()))
    topk = ok(cu.cuModuleGetFunction(topk_mod, b"k31_topk_held"))
    attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    ok(cu.cuFuncSetAttribute(paged, attr, PAGED_SMEM))
    ok(cu.cuFuncSetAttribute(topk, attr, TOPK_SMEM))

    # the queries, one a row, shared by every member
    q = torch.randn(R, IH * ID, device=dev).bfloat16()
    iw = torch.randn(R, IH, device=dev).bfloat16()
    iq = torch.empty(R, IH, ID, dtype=torch.uint8, device=dev)
    sfq = torch.empty(R, IH, 4, dtype=torch.uint8, device=dev)
    wout = torch.empty(R, IH, dtype=torch.bfloat16, device=dev)
    rid = torch.empty(R, dtype=torch.int32, device=dev)
    L(fn["k31_idx_q"], (R, 1, 1), (256, 1, 1), [q, ctypes.c_int(IH * ID), iw, ctypes.c_int(IH), iq, sfq, wout, rid,
                                                ctypes.c_int(R)])
    rq, rsq, rw = ref_q(q.view(R, IH, ID), iw)
    assert bool((rid == torch.arange(R, device=dev, dtype=torch.int32)).all())
    kw = (1 + 0.1 * torch.randn(ID, device=dev)).bfloat16()
    kb = (0.1 * torch.randn(ID, device=dev)).bfloat16()

    members, times, err = [], {}, (0.0,)
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
        stride = max(SPLIT_KV, math.ceil(max(lens) / SPLIT_KV) * SPLIT_KV)
        logits = torch.full((R, stride), float("nan"), dtype=torch.bfloat16, device=dev)
        sched = torch.empty((SMS + 1) * 2, dtype=torch.int32, device=dev)
        page_bytes = PAGE * ROW
        sf = cache[ID:]
        maps = [tmap(iq, U8, [ID, R * IH], [ID], [ID, 128], 128),
                tmap(sfq, U32, [IH, R], [IH * 4], [IH, 4], 0),
                tmap(cache, U8, [ID, PAGE, npages], [ROW, page_bytes], [ID, PAGE, 1], 128),
                tmap(sf, U32, [4, PAGE, npages], [ROW, page_bytes], [4, PAGE, 1], 0),
                tmap(wout, BF16, [IH, R], [IH * 2], [IH, 4], 0)]
        md = lambda: L(meta, (1, 1, 1), (META_THREADS, 1, 1), [ctypes.c_uint(R), lens_t, rid, sched],
                       (2 * R + META_THREADS // 32 + 1) * 4)
        lg = lambda: L(paged, (SMS, 1, 1), (512, 1, 1),
                       [ctypes.c_uint(R), ctypes.c_uint(stride), ctypes.c_uint(max_pages), lens_t, logits, bt,
                        rid, sched, *maps], PAGED_SMEM)
        times.setdefault("paged_meta", []).append(L.time(a.iters, md))
        times.setdefault("paged_logits", []).append(L.time(a.iters, lg))
        kd = rk.float() * rsc[:, None]
        o = 0
        for r, n in enumerate(lens):
            if n:
                s = ref_logits(rq[r:r + 1], rsq[r:r + 1], rw[r:r + 1], kd[o:o + n])[0]
                mag = ref_logits(rq[r:r + 1], rsq[r:r + 1], rw[r:r + 1].abs(), kd[o:o + n])[0].max().clamp(min=1e-6)
                err = max(err, (((logits[r, :n].float() - s).abs().max() / mag).item(), m, r, n))
            o += n

        ind = torch.empty(R, TOPK, dtype=torch.int32, device=dev)
        tk = lambda: L(topk, (R, 1, 1), (512, 1, 1), [logits, ctypes.c_int(stride), lens_t, ind], TOPK_SMEM)
        times.setdefault("local_topk", []).append(L.time(a.iters, tk))
        members.append((bt, max_pages, logits, lens, stride, ind))

    # every member's three stages of [W][R][HOT] candidates, poisoned once, and its zeroed state; its logits
    # and ind for a pull
    stage_bytes = W * R * HOT * 8
    lam = [torch.empty(3 * stage_bytes, dtype=torch.uint8, device=dev) for _ in range(W)]
    lists = [torch.tensor([x[k].data_ptr() for x in members], dtype=torch.int64, device=dev) for k in (2, 5)]
    state = [torch.zeros(8, dtype=torch.int32, device=dev) for _ in range(W)]
    late = torch.zeros(1, dtype=torch.int32, device=dev)
    peers = torch.tensor([x.data_ptr() for x in lam], dtype=torch.int64, device=dev)
    for x in lam:
        L(fn["k31_dcp_poison"], (SMS, 1, 1), (256, 1, 1), [x, ctypes.c_longlong(3 * stage_bytes)])
    outs = [(torch.empty(R, TOPK, dtype=torch.int32, device=dev), torch.empty(R, dtype=torch.int32, device=dev),
             torch.empty(R, dtype=torch.int32, device=dev)) for _ in range(W)]

    def exchange():
        for m, (bt, max_pages, logits, lens, stride, ind) in enumerate(members):
            L(fn["k31_dcp_pack"], (R, 1, 1), (1024, 1, 1), [logits, ctypes.c_int(stride), ind, ctypes.c_int(m),
              ctypes.c_int(W), ctypes.c_int(R), peers, state[m], ctypes.c_longlong(stage_bytes)])
        for m, (bt, max_pages, *_rest) in enumerate(members):
            sel, sl, live = outs[m]
            L(fn["k31_dcp_select"], (R, 1, 1), (1024, 1, 1),
              [lam[m], state[m], late, ctypes.c_longlong(stage_bytes), ctypes.c_longlong(10**9), ctypes.c_int(R),
               ctypes.c_int(m), bt, ctypes.c_int(max_pages), sel, sl, live, members[m][2], ctypes.c_int(members[m][4]),
               members[m][5], *lists])

    times["pack + select"] = [L.time(a.iters, exchange) / W]
    exchange()
    torch.cuda.synchronize()
    assert int(late) == 0
    picked = [set() for _ in range(R)]
    for m, (bt, max_pages, logits, lens, *_rest) in enumerate(members):
        sel, sl, live = outs[m]
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
        row = torch.empty(ctx[r], device=dev)
        for m, (_, _, logits, lens, *_rest) in enumerate(members):
            row[m::W] = logits[r, :lens[r]].float()
        got = torch.tensor(sorted(picked[r]), dtype=torch.long, device=dev)
        bad += not topk_exact(got, row, TOPK)
    print(f"rows {R}, global ctx 1..{max(ctx)}: local logits max error {err[0]:.3e} of the row's sum of |terms| (member, row, keys: {err[1:]}); "
          f"rows whose merged selection is not the global top 2048: {bad}")
    print("us per launch (mean over members):")
    for k, v in times.items():
        print(f"  {k:14s} {sum(v) / len(v):10.1f}")


if __name__ == "__main__":
    main()
