#!/usr/bin/env bash
# Build the routing cubin and print its parameter layout.
#   CUTLASS_INCLUDE=<cutlass/include> SPDLOG_INCLUDE=<spdlog/include> \
#     source/flashinfer-moe-routing/build.sh <out.cubin> [abi.json]
# The headers need CUTLASS/CuTe and spdlog; FlashInfer pins both as submodules
# (PROVENANCE.md names the revisions the bundled cubin was built against).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?out.cubin}"
abi="${2:-}"
nvcc="${NVCC:-nvcc}"
arch="${KERN_SM:-sm_103a}"
: "${CUTLASS_INCLUDE:?set CUTLASS_INCLUDE to a CUTLASS include directory}"
: "${SPDLOG_INCLUDE:?set SPDLOG_INCLUDE to a spdlog include directory}"
flags=(-std=c++17 -O3 --expt-relaxed-constexpr
       -DTLLM_GEN_EXPORT_INTERFACE -DTLLM_ENABLE_CUDA -DENABLE_BF16 -DENABLE_FP8
       -I"$here/include" -I"$CUTLASS_INCLUDE" -I"$SPDLOG_INCLUDE")
"$nvcc" -cubin -arch="$arch" "${flags[@]}" -o "$out" "$here/kern_moe_routing.cu"
echo "built $out ($arch)" >&2
if [ -n "$abi" ]; then
  tmp="$(mktemp -d)"
  "$nvcc" "${flags[@]}" -diag-suppress 1427 -Xcompiler -Wno-invalid-offsetof -o "$tmp/abi" "$here/abi.cu"
  "$tmp/abi" 16 > "$abi"
  echo "wrote $abi" >&2
fi
