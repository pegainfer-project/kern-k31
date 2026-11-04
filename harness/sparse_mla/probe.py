#!/usr/bin/env python3
"""K3.1's sparse MLA through FlashInfer's own launch (trtllm-gen, e4m3 q and KV 576 / 512, bf16 o), the
call vLLM's FlashInferMLASparse backend makes on SM100: q [T, 1, 128, 576] with K3's 96 heads padded to 128,
the fp8 cache as [pages, 1, 64, 576], each row's top-k as physical token rows [T, 1, topk] (-1 past the row's
count), seq_lens the count, LSE base 2. Checks it against an f32 torch reference and dumps every buffer, so
the same call run under tools/kernel-capture is both the ABI source and the oracle of the manifest op.

    python3 harness/sparse_mla/probe.py --rows 8192 --pages 2048 --out <dir>
"""
import argparse
import json
import math
from pathlib import Path

import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

HEADS, HEADS_PAD, KV_LORA, ROPE, TOPK, PAGE = 96, 128, 512, 64, 2048, 64
D_QK = KV_LORA + ROPE
SCALE = 192 ** -0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--pages", type=int, default=2048, help="cache pages; the rows pick among their tokens")
    ap.add_argument("--min-len", type=int, default=1, help="shortest top-k count a row gets")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    T, N = a.rows, a.pages * PAGE
    e4m3 = lambda x: x.clamp(-448, 448).to(torch.float8_e4m3fn)
    kv = e4m3(torch.randn(a.pages, 1, PAGE, D_QK, device=dev))
    q = torch.zeros(T, 1, HEADS_PAD, D_QK, device=dev)
    q[:, :, :HEADS] = torch.randn(T, 1, HEADS, D_QK, device=dev)
    q = e4m3(q)
    lens = torch.randint(a.min_len, TOPK + 1, (T,), device=dev, dtype=torch.int32)
    lens[: min(T, 4)] = torch.tensor([TOPK, 1, a.min_len, 777][: min(T, 4)], dtype=torch.int32)
    ind = torch.full((T, 1, TOPK), -1, dtype=torch.int32, device=dev)
    for r in range(T):
        n = int(lens[r])
        ind[r, 0, :n] = torch.randperm(N, device=dev)[:n].int()
    ws = torch.zeros(4 << 30, dtype=torch.uint8, device=dev)
    out = torch.empty(T, 1, HEADS_PAD, KV_LORA, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(T, HEADS_PAD, dtype=torch.float32, device=dev)
    torch.cuda.synchronize()
    trtllm_batch_decode_with_kv_cache_mla(
        query=q, kv_cache=kv, workspace_buffer=ws, qk_nope_head_dim=128, kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE, block_tables=ind, seq_lens=lens, max_seq_len=TOPK, sparse_mla_top_k=TOPK,
        out=out, bmm1_scale=SCALE, bmm2_scale=1.0, lse=lse, return_lse=True, return_lse_base="base2",
        backend="trtllm-gen")
    torch.cuda.synchronize()

    flat = kv.view(N, D_QK).float()
    oerr, lerr = 0.0, 0.0
    for r0 in range(0, T, 64):
        r1 = min(T, r0 + 64)
        for r in range(r0, r1):
            n = int(lens[r])
            k = flat[ind[r, 0, :n].long()]
            s = q[r, 0, :HEADS].float() @ k.T * SCALE
            o = torch.softmax(s, -1) @ k[:, :KV_LORA]
            l2 = torch.logsumexp(s, -1) / math.log(2)
            got = out[r, 0, :HEADS].float()
            oerr = max(oerr, ((got - o).abs().max() / o.abs().max()).item())
            lerr = max(lerr, (lse[r, :HEADS] - l2).abs().max().item())
    print(f"rows {T}, cache tokens {N}, lens {int(lens.min())}..{int(lens.max())}: "
          f"o max relative error {oerr:.3e}, lse max abs error {lerr:.3e}")

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, t in [("q", q), ("kv", kv), ("ind", ind), ("lens", lens), ("out", out), ("lse", lse)]:
            t.contiguous().view(torch.uint8).cpu().numpy().tofile(a.out / f"{name}.bin")
        named = [("q", q), ("kv", kv), ("ind", ind), ("lens", lens), ("out", out), ("lse", lse), ("ws", ws)]
        (a.out / "names.txt").write_text("".join(f"{n} {hex(t.data_ptr())}\n" for n, t in named))
        (a.out / "meta.json").write_text(json.dumps({"rows": T, "pages": a.pages, "topk": TOPK, "heads": HEADS_PAD,
                                                     "scale": SCALE, "o_rel": oerr, "lse_abs": lerr}))


if __name__ == "__main__":
    main()
