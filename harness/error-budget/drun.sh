#!/usr/bin/env bash
# work/eb/drun.sh MANIFEST OUT TOKENS ROWS [EXTRA k3_step args]: k3_step over TOKENS, every step's logits kept;
# a TP8 manifest (`tp8` in its name) on two hosts of the D pool, else one GPU.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../loop/env.sh"
m=$(realpath "$1") out=$(realpath -m "$2") tok=$(realpath "$3") rows=$4; shift 4
mkdir -p "$out"; cd "$LOOP_DIR"
if [[ $m == *tp8* || $m == *v[123]-* ]]; then
  ./withgpu -n 4 -H 2 ./d/ranks "$out" --manifest "$m" --tokens "$tok" --rows "$rows" --every 1 --graph "$@"
else
  ./withgpu -n 1 env KERN_CACHE_DIR="$KERN_CACHE_DIR" "$K3_STEP_BIN" --manifest "$m" --weights "$K3_WEIGHTS" --gpu 0 \
    --tokens "$tok" --rows "$rows" --every 1 --out "$out" "$@" > "$out/rank0.log" 2>&1
fi
echo "== $out rc=$?"
