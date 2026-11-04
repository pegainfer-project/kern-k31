#!/usr/bin/env bash
# harness/p_glue/mk.sh KIND CAND OUT: build KIND's harness (gate: k3_mla_glue.cu's kern_k3g_mla_gate,
# situ: k3_prefill_glue.cu's kern_k3g_situ, residual: its K1b / K1d) of CAND, a candidate copy of the
# kernel's source, against the committed source (git HEAD); run OUT under loop/withgpu -n 1.
# residual: NVCCFLAGS=-DTWO=1 times K1d with its second partial (nb 1, 2, 4 either way).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../../loop/env.sh"
kind=$1 cand=$(realpath "$2") out=$(realpath -m "$3")
case $kind in
  gate) src=source/k3_mla_glue.cu ;;
  situ | residual) src=source/k3_prefill_glue.cu ;;
  *) echo "kind: gate | situ | residual" >&2; exit 2 ;;
esac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$REPO" show "HEAD:$src" | grep -v '#include "k3_fmha_plan.cuh"' | sed 's/kern_k3g_/base_k3g_/g' > "$tmp/base_g.cu"
grep -v '#include "k3_fmha_plan.cuh"' "$cand" | sed 's/kern_k3g_/cand_k3g_/g' > "$tmp/cand_g.cu"
"$K3_NVCC" -arch=sm_103a -O3 -std=c++17 -I"$REPO/source" -I"$tmp" ${NVCCFLAGS:-} -o "$out" "$here/$kind.cu"
