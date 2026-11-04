# Independent planner source

`cake_kimi_k3_mla.py` is an unmodified copy of FlashInfer's
[`flashinfer/mla/cake_kimi_k3_mla.py` at 3637083a](https://github.com/flashinfer-ai/flashinfer/blob/3637083a/flashinfer/mla/cake_kimi_k3_mla.py),
SHA-256 `b5f612c81011566511979f6ef7de98526f8e12f8b2c37995c8fd7b38b32d386f`.
The Apache-2.0 license is retained beside it. No CUDA routine is invoked: the
test temporarily stubs `torch` and calls only the upstream CPU planner.

Applicable attribution from FlashInfer's `NOTICE` at the same source pin:

```text
FlashInfer
Copyright 2025-2026 NVIDIA
Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
```

Keeping the complete, hash-verified source makes the oracle independent of
the specialized port and allows offline CI. Update the source pin and
review the resulting plans together when changing the planner contract.
