#!/usr/bin/env bash
# harness/xchg/abn.sh OUT ROUNDS M1.json M2.json ...: the D bench of several manifests interleaved in one
# lease (loop/d/bench's run of each, round by round), then loop/score of each run
set -uo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
source "$here/loop/env.sh"
out=$(realpath -m "$1") n=$2; shift 2
ms=(); for m in "$@"; do ms+=("$(realpath "$m")"); done
mkdir -p "$out"
for i in $(seq 1 "$n"); do
  for k in "${!ms[@]}"; do
    "$here/loop/d/ranks" "$out/m$k.$i" --manifest "${ms[$k]}" --tokens /dev/null --ctx 65536,131072 --bench 8,16,24,36,48 --warmup 10 --iters 50
  done
done
for k in "${!ms[@]}"; do for i in $(seq 1 "$n"); do echo "m$k.$i $(basename "${ms[$k]}") $("$here/loop/score" "$out/m$k.$i")"; done; done
