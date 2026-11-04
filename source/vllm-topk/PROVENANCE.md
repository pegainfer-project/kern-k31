# vLLM prefill top-k (vendored)

Upstream: https://github.com/vllm-project/vllm, commit
`f4edf9f0f078ae67392510aafd1871288948a3c3` (2026-10-23).
License: Apache-2.0, `LICENSE` in this directory (upstream's).

`sampler_topk.cuh` is lines 48-594 of upstream's `csrc/sampler.cu`, unmodified:
the bin extraction, the histogram steps, `topKPerRowJob` and
`topKPerRowPrefill`. It is the device code behind `ops.top_k_per_row_prefill`,
the top-k vLLM's sparse indexer (`sparse_attn_indexer.py`) runs over DeepGEMM's
compressed MQA logits on a prefill chunk.

`topk.cu` is ours: it wraps the two launches upstream's host launcher makes
(`topKPerRowPrefill<512, false>` for the first 12288 rows, `<512, true>` past
them) as `extern "C"` entries so a manifest can name them. Same block (512),
same dynamic shared memory (`topK * sizeof(int)`), same arguments.
