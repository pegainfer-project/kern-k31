# FlashInfer CuTe-DSL MLA decode

Upstream: FlashInfer 0.6.x, `flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py`
(NVIDIA, CuTe DSL), BSD-3-Clause (`LICENSE`, the header of that file).
Compiler: `nvidia-cutlass-dsl` 4.6.0, CUDA 13, for `sm_103a`.

`build.py` compiles it once for K3's bf16 D: 96 heads, one query token, a
latent of 512 plus a rope of 64 in bf16, 64-token pages, a variable split of
the KV with the 256-split reducer. The cubin holds two entries: the split-KV
attention (384 threads, cluster 2x1x1, 232448 B of shared memory) and its
reduction (128 threads, 1024 B). Their 28-parameter ABI was lifted from the
DSL's launch (kern-k3-pp-prefill's `docs/k3-kernel-abi.md`, K5);
`scripts/k3_kv_bf16.py` lays the launches. `../../prebuilt/mla_decode_h96_p64.cubin`
is the pinned artifact (built 2026-09-19) and this recipe its origin.
