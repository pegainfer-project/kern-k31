#!/usr/bin/env bash
# scripts/deploy.sh <kern checkout> <kernel index> <out>: the manifests of a full-K3 P/D
# deployment, each verified by the checkout's kern.
#   <out>/p/stage{0..7}.json  P: the packed 93-layer prefill (16 sequences a chunk of 8192 rows)
#                             cut into PP8 stages every 12 layers
#   <out>/d.json              D: the 93-layer TP8 / EP8 / DCP8 decode step (64 sequences)
# Both at the generator's defaults: fp8 pages, a bf16 KDA state, 1M context.
# The kern checkout is the commit step/Cargo.toml pins, built (`cargo build --release`); the kernel
# index is kern master's kernel_index/.
set -euo pipefail
kern=$(realpath "$1") index=$(realpath "$2") out=$(realpath -m "$3")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=$kern/target/release/kern
export KERN=$kern KERN_INDEX_DIR=$index
FC1=bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_siTuGlu_dynB_sm100f
FC2=bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f
gen() { python3 "$repo/scripts/gen_stage.py" "$@"; }

mkdir -p "$out"
gen --layers 93 --chunk 8192 --pack 16 > "$out/p-93.json"
at=$(python3 -c "
import json, sys
calls = json.load(open(sys.argv[1]))['programs']['prefill']['calls']
print(','.join(str(next(i for i, c in enumerate(calls) if c['label'] == f'l{l}.res_in')) for l in range(12, 93, 12)))
" "$out/p-93.json")
rm -rf "$out/p"
"$bin" cut "$out/p-93.json" --at "$at" --out "$out/p"
rm "$out/p-93.json"

gen --layers 93 --tp 8 --seqs 64 \
  --moe-variants "$FC1,$FC2" --peer-ar > "$out/d.json"
"$bin" verify "$out/d.json"
