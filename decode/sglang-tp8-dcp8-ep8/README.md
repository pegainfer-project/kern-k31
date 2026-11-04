# SGLang's K3 decode graphs at TP8 / DCP8 / EP8

The CUDA graphs SGLang captures for full Kimi-K3 decode on 8 GB300 GPUs (two 4-GPU nodes on one NVLink
fabric), TP8 with decode context parallelism 8 and expert parallelism 8, at 1, 8 and 48 sequences. They
are the reference a kern decode for this layout is built against: every launch, its geometry and its
parameter bytes, and the modules the launches come from.

Captured 2026-10-09 with kern's CUPTI injection library
([`tools/kernel-capture`](https://github.com/pegainfer-project/kern/tree/master/tools/kernel-capture)):
it records each launch as it is made, so the launches SGLang makes while capturing a graph are the
graph's contents.

## Versions

| Component | Version |
|---|---|
| Image | `lmsysorg/sglang:nightly-dev-cu13-20261004-295a53c9` |
| SGLang | `295a53c90ea4adb2588084dc0e3004cd3145e414` (2026-10-04) |
| sglang-kernel | 0.4.8 |
| FlashInfer | flashinfer-python / -cubin / -jit-cache 0.7.0.post1 (cu130) |
| CUTLASS DSL | nvidia-cutlass-dsl 4.8.0 |
| PyTorch | 2.13.0+cu130 (NCCL 2.29.7 as linked) |
| Triton | 3.7.1 |
| cuBLAS | 13.1.1.3 |

Server flags (`scripts/capture.sh` has the full command): `--tp-size 8 --dcp-size 8 --ep-size 8`,
`--kv-cache-dtype fp8_e4m3 --page-size 64`, `--moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4`,
`--decode-attention-backend cutedsl_mla`, `--cuda-graph-bs-decode 1 8 48`, and
`--disaggregation-mode decode --disaggregation-transfer-backend fake`, so a request skips prefill and
decodes over its prompt's length of uncomputed context.

## The graphs

| Sequences | Launches per rank | Same symbols and geometry on all 8 ranks |
|---|---:|---|
| 48 | 2490 | yes |
| 8 | 2237 | yes |
| 1 | 2211 | yes |

`graphs/graph-bs<B>.rank<r>.jsonl.xz` holds each rank's launches in order, one JSON record per launch:
symbol, grid, block, dynamic shared memory, function attributes, and every parameter's bytes, with
device pointers marked by the allocation they fall in. SGLang runs each batch size three times while
capturing (two warmups, then the captured run); `scripts/extract.py` keeps the third and checks that it
launches the same kernels at the same grids as the second.

One MLA layer and one KDA layer at 48 sequences, rank 0:

| # | MLA layer | grid | | KDA layer | grid |
|---:|---|---|---|---|---|
| 1 | `attn_res_fused_tma_kernel` | 48 | | `attn_res_fused_tma_kernel` | 48 |
| 2 | cuBLASLt `64x32` split-K + reduce | 136 | | cuBLASLt `64x48` split-K + reduce | 288 |
| 3 | `RMSNormKernel` ×2 | 12 | | cuBLASLt `64x16` split-K + reduce | 64 |
| 4 | cuBLASLt `128x48` TNT, then NNT | 144, 384 | | cuBLASLt `64x8` | 144 |
| 5 | ATen elementwise copy | 4608 | | `kda_decode_fusion_many_heads_kernel` | 48×12 |
| 6 | `set_mla_kv_concat_q_fp8_kernel` | 582 | | | |
| 7 | MLA decode `split_kv_kernel` (FlashInfer CuTe DSL, fp8) | 2×48 | | | |
| 8 | `fixup_zero_kv_rows_kernel` | 1×48 | | | |
| 9 | `_dcp_pack_a2a_send_kernel` (Triton) | 48×96 | | | |
| 10 | `helixAllToAllKernel` (TRT-LLM) | 1×76×2 | | | |
| 11 | `_dcp_lse_combine_kernel` (Triton) | 48×12 | | | |
| 12 | cuBLASLt `64x8`, `64x16` split-K + reduce | 144 | | | |
| 13 | `mla_output_gate_kernel` | 36 | | | |
| 14 | cuBLASLt `128x24`, NCCL all-reduce, ATen add | 112, 21, 336 | | the same | |
| 15 | `attn_res_fused_tma_kernel` | 48 | | the same | |
| 16 | cuBLASLt `128x48` split-K (fp32 out) + reduce, `situ_and_mul_kernel`, cuBLASLt `128x24` | | | the same | |
| 17 | `route_quant_fused_kernel`, `routingIndicesClusterKernel` | 96, 8 | | the same | |
| 18 | TRT-LLM gen `bmm` FC1 (mxfp8 × mxfp4), FC2 | 48×153, 28×153 | | the same | |
| 19 | `finalizeKernel`, NCCL all-reduce | 14×48, 28 | | the same | |
| 20 | `RMSNormKernel`, cuBLASLt `64x48`, `add3_kernel` | 24, 112, 168 | | the same | |

That is 35 launches for an MLA layer and 24 for a KDA layer, two TP all-reduces in each. At 1 and 8
sequences the small GEMMs move to FlashInfer's TGV GEMM and SGLang's `tiny_n_gemm` / `tiny_k_gemm`, the
MLA decode adds its split-KV `reduction_kernel`, and the MoE GEMMs switch to 8-row tiles.

## Decode context parallelism

- **KV layout.** SGLang's CUDA DCP interleaves tokens: position `p` of a sequence lives on DCP rank
  `p % 8`, at local row `p // 8` (`sglang/srt/layers/dcp/layout.py`, `interleave_size = 1`). The
  allocator hands out logical slots eight times wider than a rank's pool, in pages of 64 × 8 = 512
  (`sglang/srt/mem_cache/allocation_sizing.py`); a rank keeps the slots with `slot % 8 == rank` at row
  `slot // 8`. A rank's 64-row page therefore covers 512 positions of the sequence, each rank holds an
  eighth of every sequence, and the prefix cache shares in 512-token pages.
- **Attention.** Every rank runs the MLA decode over its own rows for all 48 sequences and all heads,
  producing a partial output and its log-sum-exp; `fixup_zero_kv_rows_kernel` handles a rank that owns
  no rows of a short sequence. `_dcp_pack_a2a_send_kernel`, TRT-LLM's Helix all-to-all and
  `_dcp_lse_combine_kernel` then move the partials from position-sharded to head-sharded and merge
  them, so each rank leaves attention with its TP8 share of the heads for every sequence.
- **Everything else is TP8.** The KDA recurrent state and conv windows are split by head, the dense
  projections by column or row, and the routed experts by EP8.
- **Hand-off from a prefill instance.** A prefill instance that holds a sequence's KV contiguously has
  to deal each page's rows out to eight ranks, every eighth row to each. SGLang's PD transfer packs
  them with `copy_mla_rows_into_pack` (`sglang/srt/disaggregation/common/dcp_pack.py`); a page is not
  a unit of transfer here.

## Kernels and licenses

`kernels.toml` lists all 62 modules the three graphs launch: SHA-256, size, origin, upstream source and
every symbol with its launch count per batch size. 36 of them are in `prebuilt/`; the other 26 are
listed only.

| Origin | In `prebuilt/` | License |
|---|---|---|
| SGLang JIT kernels (`sglang/kernels/jit/csrc`) and Triton kernels (`sglang/kernels/ops`) | yes | Apache-2.0, `licenses/LICENSE.sglang`; Triton runtime MIT, `licenses/LICENSE.triton` |
| FlashInfer CuTe DSL kernels (RMSNorm, MLA decode fp8, TGV GEMM) | yes | Apache-2.0, `licenses/LICENSE.flashinfer` |
| TRT-LLM Helix all-to-all, as FlashInfer ships and builds it | yes | Apache-2.0 (NVIDIA headers in FlashInfer's `csrc/nv_internal`), `licenses/LICENSE.flashinfer` |
| TRT-LLM gen batched GEMM (flashinfer-cubin 0.7.0.post1) | yes | Apache-2.0, `../../prebuilt/LICENSE.trtllm-gen.txt` |
| FlashInfer / TRT-LLM MoE routing and finalize (flashinfer-jit-cache modules of 2.6-36 MB) | no | Apache-2.0; the routing kernels build from `../../source/flashinfer-moe-routing` |
| cuBLASLt (`nvjet_*`, `splitKreduce_kernel`) | no | NVIDIA proprietary; extracted cubins are not redistributed |
| NCCL (one 136 MB module) | no | BSD-3-Clause; install NCCL |
| PyTorch ATen elementwise kernels (fat modules of 3-8 MB) | no | BSD-3-Clause |

The cuBLASLt GEMMs are 976 of the 2490 launches at 48 sequences. A kern decode needs its own GEMMs for
those shapes.

## Building

The prebuilt cubins are the bytes SGLang's process loaded; no rebuild here has been checked against
them. Their recipes:

- **SGLang JIT kernels** compile at first use through `sglang.kernels.jit` (`load_jit`) for the
  detected GPU's arch-specific target, with `-DSGL_CUDA_ARCH=1030 -std=c++20 -O3
  --expt-relaxed-constexpr` plus each kernel's own flags; the template arguments are those in the
  mangled symbol (`kernels.toml` lists every one). `python -m sglang.kernels.jit --cuda-target 10.3`
  prints the toolchain flags.
- **FlashInfer CuTe DSL kernels** compile from the Python sources `kernels.toml` names, with CUTLASS DSL
  4.8.0 for sm_103a.
- **Triton kernels** compile from SGLang's Python sources with Triton 3.7.1.
- **TRT-LLM gen batched GEMMs** are shipped as cubins in flashinfer-cubin; there is no source recipe.

## Scripts

- `scripts/capture.sh`: the server, with the injection library, on one node (run on both).
- `scripts/timeline.py`: a rank's launches in bursts split by idle gaps, to find the capture window.
- `scripts/extract.py <capture dir> <out>`: the three graphs of every rank, checked against the warmup
  and across ranks (`graphs/graphs.json`).
- `scripts/inventory.py <rank dir> <graphs dir> <out json>`: every kernel, its module and origin.
- `scripts/assemble.py`: this directory, from the inventory.
