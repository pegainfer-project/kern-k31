#!/usr/bin/env bash
# harness/attn/run.sh kda|absorb OUT ARGS...: build the harness and the L2 flush kernel with $K3_NVCC, run it
# under nsys on one GPU, print each cubin's per-launch time and whether its outputs match the first's bit for bit.
#   loop/withgpu -n 1 harness/attn/run.sh kda work/r --rows 8,24 build/<main>.cubin "build/<variant>.cubin#65536"
# kda: k3_kda_core's TP8 build (12 heads), state and weights cold, the partial in L2 as its GEMM leaves it.
# absorb: k3_mla_absorb_mma (or k3_mla_absorb_tc, CUBIN!tc), W_UK cold, q_partial in L2.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../loop/env.sh"
what=$1 out=$(realpath -m "$2"); shift 2
mkdir -p "$(dirname "$out")"
"$K3_NVCC" -cubin -arch=sm_103a -O3 "$here/flush.cu" -o "$out.flush.cubin"
"$K3_NVCC" -O2 -std=c++17 -arch=sm_103a "$here/$what.cu" -o "$out.bin" -lcuda
kernel=$([[ $what == kda ]] && echo kern_k3_kda_core || echo kern_k3_mla_absorb)
FLUSH_CUBIN="$out.flush.cubin" nsys profile -t cuda --cuda-graph-trace=node -f true -o "$out" "$out.bin" "$@" > "$out.txt"
nsys export -t sqlite -f true -o "$out.sqlite" "$out.nsys-rep" > /dev/null
python3 "$here/tab.py" "$out.sqlite" "$out.txt" "$kernel" "$@"
