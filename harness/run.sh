#!/usr/bin/env bash
# The D1 acceptance run: every kernel a full-K3 decode at TP8 / EP8 / DCP8
# adds, against CPU references, at the decode batches, plus the SASS check.
#
#   CUDA_VISIBLE_DEVICES=<one free GPU> harness/run.sh [sass|tp8|d1|conv ...]
#
# Needs build/ from scripts/build.py (the pinned cubins, hashes checked) and
# nvcc/cuobjdump (NVCC_BIN, default /usr/local/cuda-13.1/bin). Four parts:
#   sass  every D1 cubin: no stack, no local memory, no .MULTICAST in its SASS
#   tp8   harness/k3 (kern's K3 kernel harness, copied from its tools/k3-harness)
#         built with the TP8 rank's model constants (HEADS 12; the build copy's ref.h
#         patched) over the HEADS=12 / INNER=1536 variants and the router
#   d1    harness/d1.cu: vup_gate at 12 heads, the DCP fixup / pack / combine
#         over 8 simulated members, FlashInfer routing of a 112-expert slice (contiguous and every 8th)
#         with kern's finalize
#   conv  harness/kda_conv_layout.cu: kda_core's CONV_TAP_MAJOR + LINE_GUARD build (an engine's conv
#         pool, padded rows at slot -1) against the TP8 D build it varies, bit for bit
# Prints one RESULT line per run and a summary; exit 1 if anything fails.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
bin="${NVCC_BIN:-/usr/local/cuda-13.1/bin}"
out="${OUT:-$here/out}"
build="$root/build"
reps="${REPS:-20}"
parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=(sass tp8 d1 conv)
mkdir -p "$out"
log="$out/run.log"
: > "$log"
fail=0

# One run: its RESULT line lands in the log; a harness or driver error (exit 2)
# leaves none, so it gets one.
r() {
  "$@" >> "$log" 2>&1
  [ $? -eq 2 ] && printf 'RESULT\terror\t%s\tFAIL\n' "$*" >> "$log"
  return 0
}

# A host without nvcc runs the binaries a host with one built into $out.
build() {
  if [ -x "$bin/nvcc" ]; then
    "$bin/nvcc" -O2 -std=c++17 -arch=sm_103a "$1" -o "$2" -lcuda || exit 2
  else
    [ -x "$2" ] || { echo "no nvcc here and no prebuilt $2" >&2; exit 2; }
  fi
}

cubin() {
  local f="$build/$1.cubin"
  [ -f "$f" ] || { echo "missing $f: run scripts/build.py" >&2; exit 2; }
  echo "$f"
}

D1=(k3_conv_silu+HEADS=12 k3_kda_core+HEADS=12 k3_kda_out_gate+HEADS=12 k3_mla_vup_gate+HEADS=12
    k3_mla_prep+INNER=1536+MLA_FUSED=3648 k3_dcp k3_router_argmax)

sass() {
  for k in "${D1[@]}"; do
    local f; f="$(cubin "$k")"
    local res; res="$("$bin/cuobjdump" -res-usage "$f" | grep -o 'Function [^:]*:\|STACK:[0-9]*\|LOCAL:[0-9]*' | paste -sd' ')"
    local mc; mc="$("$bin/cuobjdump" -sass "$f" | grep -c MULTICAST)"
    local ok=PASS
    echo "$res" | grep -q 'STACK:[1-9]\|LOCAL:[1-9]' && ok=FAIL
    [ "$mc" -ne 0 ] && ok=FAIL
    printf 'RESULT\tsass\t%s\t%s\tmulticast=%s\t%s\n' "$k" "$res" "$mc" "$ok" >> "$log"
    [ "$ok" = PASS ] || fail=1
  done
}

tp8() {
  local h="$out/k3-harness-tp8"
  mkdir -p "$h"
  cp "$here/k3/harness.cu" "$h/"
  sed -e 's/^  HEADS = 96,/  HEADS = 12,/' \
      -e 's/^static const long long REC_BYTES = 6291456LL;/static const long long REC_BYTES = (long long)HEADS * 128 * 128 * 4;/' \
      "$here/k3/ref.h" > "$h/ref.h"
  [ "$(grep -c '^  HEADS = 12,\|REC_BYTES = (long long)HEADS' "$h/ref.h")" -eq 2 ] \
    || { echo "ref.h no longer has the constants this patch rewrites" >&2; exit 2; }
  build "$h/harness.cu" "$h/harness"
  local run=(r "$h/harness" --reps "$reps")
  for B in 1 2 8 64; do
    "${run[@]}" --kernel conv_silu --cubin "$(cubin k3_conv_silu+HEADS=12)" --B "$B" --grid "$B,3,3" --block 128,1,1
    "${run[@]}" --kernel kda_core --cubin "$(cubin k3_kda_core+HEADS=12)" --B "$B"
    "${run[@]}" --kernel kda_out_gate --cubin "$(cubin k3_kda_out_gate+HEADS=12)" --B "$B" --grid "$B,3,1"
    "${run[@]}" --kernel mla_prep --cubin "$(cubin k3_mla_prep+INNER=1536+MLA_FUSED=3648)" --B "$B"
    "${run[@]}" --kernel mla_prep --cubin "$(cubin k3_mla_prep+INNER=1536+MLA_FUSED=3648)" --B "$B" \
      --grid "$B,3,1" --block 512,1,1 --nmla 2 --layer 1
    "${run[@]}" --kernel router_topk --cubin "$(cubin k3_router_argmax)" --B "$B" --block 1024,1,1
  done
}

d1() {
  build "$here/d1.cu" "$out/d1"
  local run=(r "$out/d1" --reps "$reps")
  for B in 1 2 8 48 64; do
    "${run[@]}" --kernel vup_gate --cubin "$(cubin k3_mla_vup_gate+HEADS=12)" --heads 12 --B "$B"
    for ctx in 1 7 64 515 2048; do
      "${run[@]}" --kernel dcp --cubin "$(cubin k3_dcp)" --B "$B" --ctx "$ctx"
    done
  done
  for T in 1 8 48 64; do
    for rank in 0 3 7; do
      for tile in 8 16; do
        for stride in 1 8; do
          "${run[@]}" --kernel routing --cubin "$(cubin flashinfer_moe_routing)" --cubin2 "$(cubin k3_moe_prefill)" \
            --B "$T" --rank "$rank" --tile "$tile" --stride "$stride"
        done
      done
    done
  done
}

conv() {
  build "$here/kda_conv_layout.cu" "$out/kda_conv_layout"
  local d=HEADS=12+KDA_CONV=1+LDS=6400
  for B in 2 8 64; do
    r "$out/kda_conv_layout" --B "$B" --base "$(cubin k3_kda_core+$d+WSM_FA=12+WSM_LDS=6400)" \
      --variant "$(cubin k3_kda_core+CONV_TAP_MAJOR=1+$d+LINE_GUARD=1+WSM_FA=12+WSM_LDS=6400)"
    # the D step's two builds (gen_stage.py routes up to 25 rows to the first)
    r "$out/kda_conv_layout" --B "$B" --smem 65536 \
      --base "$(cubin k3_kda_core+HEADS=12+KDA_CONV=1+KDA_SMEM_STATE=1+LDS=6400+ROWS_PER_ITER=1+WSM_FA=12+WSM_LDS=6400)" \
      --variant "$(cubin k3_kda_core+CONV_TAP_MAJOR=1+HEADS=12+KDA_CONV=1+KDA_SMEM_STATE=1+LDS=6400+LINE_GUARD=1+ROWS_PER_ITER=1+WSM_FA=12+WSM_LDS=6400)"
    r "$out/kda_conv_layout" --B "$B" --smem 65536 \
      --base "$(cubin k3_kda_core+HEADS=12+KDA_CONV=1+KDA_PDL=1+KDA_SMEM_STATE=1+LDS=6400+ROWS_PER_ITER=1+WSM_FA=12+WSM_LDS=6400)" \
      --variant "$(cubin k3_kda_core+CONV_TAP_MAJOR=1+HEADS=12+KDA_CONV=1+KDA_PDL=1+KDA_SMEM_STATE=1+LDS=6400+LINE_GUARD=1+ROWS_PER_ITER=1+WSM_FA=12+WSM_LDS=6400)"
    r "$out/kda_conv_layout" --B "$B" \
      --base "$(cubin k3_kda_core+HEADS=12+KDA_CONV=1+KDA_MINB=4+LDS=6400+ROWS_PER_ITER=2+WSM_FA=12+WSM_LDS=6400)" \
      --variant "$(cubin k3_kda_core+CONV_TAP_MAJOR=1+HEADS=12+KDA_CONV=1+KDA_MINB=4+LDS=6400+LINE_GUARD=1+ROWS_PER_ITER=2+WSM_FA=12+WSM_LDS=6400)"
  done
}

for p in "${parts[@]}"; do "$p"; done
grep -h '^RESULT' "$log" | sed 's/^RESULT\t//' | column -t -s $'\t'
grep -q '^RESULT.*FAIL\|CUDA error\|has no entry' "$log" && fail=1
echo
[ $fail -eq 0 ] && echo "all runs PASS ($log)" || echo "FAILURES, see $log"
exit $fail
