# DeepGEMM MQA logits, patched for a token-major key cache

Upstream: https://github.com/deepseek-ai/DeepGEMM, commit
`5081f8513af57e61696de4f4f04b0e707ffb503e` (2.8, the `third_party/DeepGEMM`
submodule; vLLM's SM100 indexer runs the same kernels). License: MIT,
`LICENSE` in this directory (upstream's).

`deep_gemm/impls/sm100_mqa_logits.cuh` and `deep_gemm/layout/mqa_logits.cuh`
are upstream's with one knob, `DG_SF_KV_STRIDE` (default 1, upstream's
behavior): the paged kernel's per-key UE8M0 scale words sit that many words
apart in shared memory. K3.1's key cache is token-major, 144 bytes a token
(128 e4m3, the scale word, 12 pad), so that kern can push and deal it row by
row; TMA moves at least 16 bytes in its inner dimension, so a page's 64 scale
words land as one 3-D box of 64 x 16 bytes (`DG_SF_KV_STRIDE 4`), and the warp
that transposes them for UTCCP reads every fourth word and writes them
compact, each chunk at or below where it was read. The contiguous-KV kernel
keeps stride 1. Every other file comes from the submodule unchanged;
`kernels.toml` puts this directory first on the include path.
