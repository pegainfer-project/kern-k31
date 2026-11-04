# cuDNN frontend FROST KDA prefill (vendored)

Upstream: https://github.com/NVIDIA/cudnn-frontend, the `nvidia-cudnn-frontend`
1.32.0 wheel (aarch64, CPython 3.12), Python package `cudnn`.
Compiler: `nvidia-cutlass-dsl` 4.8.0 (CuTe DSL).
License: the KDA kernel and `linear_attention/frost/common` are Apache-2.0
(`LICENSE`), `frost/` and `frost/tile_dsl/` are MIT (`LICENSE-MIT`), both
Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES; `NOTICE` is the
wheel's. Every vendored file keeps its SPDX header.

FROST's KDA prefill is the chunked delta rule (16-row chunks, the WY form with
a blockwise (I + L)^-1 per chunk) as one persistent kernel: one work item per
(sequence, head), longest first, on at most 152 CTAs, with the state carried
in tensor memory between chunks. It reads and writes the state in place in a
pool row named by seed and final indices, at any row stride, which is how a
KDA line holds it. P runs it in place of FlashKDA (see `scripts/cudnn_kda_abi.py`).

## What is vendored

Unmodified, at their paths under the wheel's `cudnn/`:

- `linear_attention/frost/kernel/kda_prefill_f16.py` (the kernel, its
  prologue and their hosts) and `kda_prefill_config.py`;
- `linear_attention/frost/common/{blockwise_inverse,host,split_k,thd}.py`;
- `frost/tile_dsl/{barrier,handles,mma,pointwise,regtile,requirements,swizzle,tma}.py`;
- `frost/{device,buffers}.py` and every package's `__init__.py` on the way.

That is the set of `cudnn.frost` / `cudnn.linear_attention` modules the
compile loads. The wheel's `cudnn` package itself (its compiled bindings)
still answers the device queries `frost/device.py` and `frost/buffers.py`
make; no kernel code comes from it, and `build.py` checks that every module
it loaded under those two packages came from this directory.

## What is ours

`build.py`: one `@cute.jit` host that launches the prologue and the prefill
as cuDNN's uncut forward does (`kda_uncut_forward_f16.uncut_forward_host`
with `prep=False`), compiled at kern-k31's static configuration over
placeholders marked as that build marks them, written out as one cubin:

- bf16 q / k / v / gate / o and state; safe gate, lower bound -5; q and k
  L2-normed and beta's sigmoid in the kernel; i64 `cu_seqlens`;
- the state read from and written to a pool row (seed and final indices),
  no checkpoints;
- 152 SMs, the uncut schedule (H = 96 is never split or chained at our
  shapes).

Every shape, stride, pointer, the scale and the sequence count are runtime
values. The cubin holds two entries:

| Entry | Grid | Block | Dynamic smem |
|---|---|---|---|
| `cudnn_kernel_cutlass_frost_kda_prefill_prologue_…` | 2 | 1024 | 32,776 |
| `cudnn_kernel_cutlass_frost_kda_prefill_KdaPrefillCfg…` | min(152, sequences x 96) | 512 | 208,384 |

Both are launched with programmatic dependent launch and wait on the
previous grid before touching anything it wrote.

## Build

```sh
python3 source/cudnn-frost-kda/build.py prebuilt/cudnn_frost_kda.cubin
```

on a GB300 with the two wheels above and torch. The build is deterministic:
two runs write the same bytes (`2d9667fe…`, the hash `kernels.toml` pins).
