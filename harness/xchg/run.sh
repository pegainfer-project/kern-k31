#!/usr/bin/env bash
# run.sh OUT TEST...: xbench on every GPU of the lease (WITHGPU_LEASE, as loop/d/ranks), ranks in lease order
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../loop/env.sh"
out=$(realpath -m "$1"); shift
rm -rf "$out" && mkdir -p "$out"
world=0
for h in $WITHGPU_LEASE; do IFS=, read -ra g <<<"${h#*:}"; world=$((world + ${#g[@]})); done
host_name() { printf "${WITHGPU_HOST_FORMAT:-%s}\n" "$1"; }
r=0 pids=()
for h in $WITHGPU_LEASE; do
  IFS=, read -ra g <<<"${h#*:}"
  for gpu in "${g[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$(host_name "${h%%:*}")" \
      "X_DEADLINE_NS=${X_DEADLINE_NS:-30000000000} timeout ${X_TIMEOUT:-600} $here/xbench --rank $r --world $world --gpu $gpu --dir $out $*" > "$out/rank$r.log" 2>&1 < /dev/null &
    pids+=($!) r=$((r + 1))
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "lease: $WITHGPU_LEASE"
grep -h -E "RES|STAMP|ISSUE|READROW|PROBE|VERIFY|STATTN|STDCP|STLAND|CKATTN|CKDCP" "$out/rank0.log"; grep -h -E "STAMP|PROBE" "$out/rank4.log"
exit $rc
