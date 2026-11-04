#!/usr/bin/env bash
# work/eb/dt.sh MANIFEST OUT: loop/d/check's TP8 run of MANIFEST, compared (dcmp.py) with the oracle and main's run.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../loop/env.sh"
m=$(realpath "$1") out=$(realpath -m "$2") base=$K3_LOOP_DATA/base
"$LOOP_DIR/withgpu" -n 4 -H 2 "$LOOP_DIR/d/ranks" "$out" --manifest "$m" --tokens "$K3_LOOP_DATA/ids-r4.i64" \
  --rows 4 --every 64 --graph || { echo "== $out FAIL run"; exit 1; }
echo "== $out"
echo "vs oracle: $(python3 "$here/dcmp.py" "$base/g4-tp1" "$out")"
echo "vs main:   $(python3 "$here/dcmp.py" "$base/g4-tp8" "$out")"
grep -h "ms per step" "$out"/rank0.log
