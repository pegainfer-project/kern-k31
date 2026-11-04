# FlashInfer MoE routing (vendored)

Upstream: https://github.com/flashinfer-ai/flashinfer
Commit: `6870e3ff` (2026-09-21, `nightly-v0.6.18-20260819-342`)
License: Apache-2.0 (Copyright (c) 2022-2026, NVIDIA CORPORATION), `LICENSE` in
this directory; every vendored file keeps its upstream header.

These are TensorRT-LLM gen's MoE routing kernels as FlashInfer ships them. A
D member holds 112 of the 896 experts (every 8th), more than kern's own
routing glue (`k3_moe_prefill.cu`, 64 local experts at most) handles. The kernels turn the router's precomputed
top-k ids into the tables the TRT-LLM gen batched GEMMs read. This is the same
path SGLang takes for K3 (`trtllm_fp4_block_scale_routed_moe` with
precomputed top-k).

## What is vendored

From `include/` of the FlashInfer checkout, unmodified:

- `flashinfer/trtllm/fused_moe/RoutingKernel.cuh`, `RoutingKernel.h`,
  `RoutingKernelTopK.cuh`, `RoutingDevKernel.h`, `DevKernel.h`, `IntFastDiv.h`
- `flashinfer/trtllm/common/cudaUtils.h`, `flashinfer/exception.h`,
  `flashinfer/logging.h`

From `csrc/nv_internal/include/` of the same checkout, unmodified:
`tensorrt_llm/common/{assert,logger,stringUtils,tllmException}.h`.

From the flashinfer-cubin 0.6.18 bundle
`cubins/8ec29a98612c3670f9f28825d1ed19f09496073b/batched_gemm-fa419f4-31ee4e5`,
unmodified:
`include/trtllmGen_bmm_export/trtllm/gen/{DtypeDecl,MmaDecl,SfLayoutDecl,CommonUtils}.h`.
FlashInfer downloads this bundle at JIT time; the two batched-GEMM cubins in
`../../prebuilt/` come from the same bundle. The headers sit under
`include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/`, the path
FlashInfer symlinks them to.

This is exactly the set of FlashInfer headers the translation unit includes.

## What is ours

- `kern_moe_routing.cu` explicitly instantiates the multi-kernel path of
  upstream's `runPostTopKPipeline`, for
  `routingPrecomputed::KernelParams<float, 1024, 16>`:
  - `routingInitExpertCounts`;
  - `routingIndicesHistogramKernel`;
  - `routingIndicesOffsetsKernel`.

  The cubin's entries are their mangled names. The block is 1024 threads. The
  grids are fixed functions of the token count, so the path lowers to a
  static launch list at any token count. The single-cluster and cooperative
  paths need host-side decisions and a cooperative launch, so they are not
  used.
- `abi.cu` prints the parameter struct's layout (`abi.json`): its field
  offsets and the bytes of `IntFastDiv(16)`. kern packs the struct from that
  layout (kern `tools/kernels/abi/flashinfer_moe_routing.py`).
- `build.sh` is the cubin recipe.

## Behaviour to know

- Rows within one expert are ordered by shared-memory atomics. The routing
  tables (`route_map`, `exp2perm`) therefore differ between runs. Every row's
  GEMM result and the top-k combine do not: two runs write byte-identical
  outputs.
- The padding rows of `route_map` are never written; the GEMMs stop at
  `cta_limit`.

## Build

```sh
CUTLASS_INCLUDE=<flashinfer>/3rdparty/cutlass/include \
SPDLOG_INCLUDE=<flashinfer>/3rdparty/spdlog/include \
  bash source/flashinfer-moe-routing/build.sh build/flashinfer_moe_routing.cubin [abi.json]
```

The bundled cubin (`d797b1d2…`) was built with:

- CUDA 13.0 (V13.0.88, `cuda_13.0.r13.0/compiler.36424714_0`);
- CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633`;
- spdlog `c3aed4b68373955e1cc94307683d44dca1515d2b`.

These are FlashInfer's submodule revisions at the commit above.
