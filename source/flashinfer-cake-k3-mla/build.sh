#!/usr/bin/env bash
# Build the vendored Cake Kimi-K3 MLA wide decode into one cubin.
#   source/flashinfer-cake-k3-mla/build.sh <out.cubin>
# The flags are FlashInfer's JIT flags for this module (`--use_fast_math` from its
# exporter record), less the host-only ones.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?out.cubin}"
nvcc="${NVCC:-nvcc}"
arch="${KERN_SM:-sm_103a}"
"$nvcc" -cubin -O3 -std=c++17 -arch="$arch" --expt-relaxed-constexpr --use_fast_math \
  -o "$out" "$here/cake_kimi_k3_mla_fp8_paged_attention_ea17187e76fd47ab2653_kernel.cu"
echo "built $out ($arch)" >&2
