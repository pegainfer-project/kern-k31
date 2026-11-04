#!/usr/bin/env bash
# harness/xchg/build.sh: xbench. This tree's kernels (a translation unit each, they share helper
# names), main's k3_ar_fused.cu / k3_dcp.cu with their entries renamed (byte-for-byte checks against
# main), FlashInfer's TRT-LLM allreduce fusion as written ($K3_REF/flashinfer), the harness; -lcuda.
set -euo pipefail
cd "$(dirname "$0")"
source ../../loop/env.sh
NVCC=$K3_NVCC
F="-O3 -std=c++20 -arch=sm_103a -lineinfo ${XF:-}"
mkdir -p obj
git show main:source/k3_ar_fused.cu | sed 's/kern_k3_/kern_main_/g; s|#include "k3_|#include "../../source/k3_|' > obj/main_fused.cu
git show main:source/k3_dcp.cu | sed 's/kern_k3_dcp_/kern_maindcp_/g' > obj/main_dcp.cu
for d in AR FUSED DCP HEAD; do $NVCC $F -DOURS_$d -c ours.cu -o obj/ours_$d.o & done
$NVCC $F -I. -c obj/main_fused.cu -o obj/main_fused.o &
$NVCC $F -I. -c obj/main_dcp.cu -o obj/main_dcp.o &
$NVCC $F -c new.cu -o obj/new.o &
$NVCC -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a --expt-relaxed-constexpr -I"$K3_REF/flashinfer/include" \
  -c fi_ar.cu -o obj/fi_ar.o &
$NVCC $F -c main.cu -o obj/main.o &
wait
$NVCC $F obj/*.o -o xbench -lcuda
