# DeepGEMM MQA logits, patched for a token-major key cache

Upstream: https://github.com/deepseek-ai/DeepGEMM, commit
`ab69f76be5bb9ea3499bc755002b1a876cb0b3d9` (the `third_party/DeepGEMM`
submodule). License: MIT, `LICENSE` in this directory (upstream's).

`deep_gemm/impls/sm100_mqa_logits.cuh` and `deep_gemm/layout/mqa_logits.cuh`
are upstream's with one knob, `DG_SF_KV_STRIDE` (default 1, upstream's
behavior): the paged kernel's per-key f32 scales sit that many floats apart in
shared memory. K3.1's key cache is token-major, 144 bytes a token (128 e4m3,
the scale, 12 pad), so that kern can push and deal it row by row; TMA moves at
least 16 bytes in its inner dimension, so a page's 64 scales land as 64 x 16
bytes (`DG_SF_KV_STRIDE 4`) and the math warps read every fourth float. The
contiguous-KV kernel asserts the stride is 1. Every other file comes from the
submodule unchanged; `source/k31_paged_mqa_logits.cu` puts this directory
first on the include path.
