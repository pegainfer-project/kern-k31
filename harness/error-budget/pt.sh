#!/usr/bin/env bash
# work/eb/pt.sh REF CAND OUT [kern test args]: loop/p/check's kern test with free workload args
# (defaults: --prefill 6000 --chunk 2048, the check's), on one P GPU.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../loop/env.sh"
base=$(realpath "$1") cand=$(realpath "$2") out=$(realpath -m "$3"); shift 3
mkdir -p "$(dirname "$out")"
python3 - "$base" "$cand" "$out.cand.json" <<'PY'
import json, sys
base, cand = (json.load(open(p)) for p in sys.argv[1:3])
changed = {k for k, m in cand["modules"].items() if k in base["modules"] and base["modules"][k]["sha256"] != m["sha256"]}
cand["modules"] = {(k + "+candidate" if k in changed else k): m for k, m in cand["modules"].items()}
for op in cand["ops"].values():
    for l in op["impl"]["launches"]:
        if l.get("module") in changed:
            l["module"] += "+candidate"
json.dump(cand, open(sys.argv[3], "w"))
PY
args=("$@"); [[ " $* " == *" --prefill "* ]] || args+=(--prefill 6000); [[ " $* " == *" --chunk "* ]] || args+=(--chunk 2048)
cd "$LOOP_DIR"
WITHGPU_HOSTS=${PHOSTS:?PHOSTS: the P pool} "$LOOP_DIR/withgpu" -n 1 env KERN_CACHE_DIR="$KERN_CACHE_DIR" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "$K3_KERN_BIN" test --reference "$base" --manifest "$out.cand.json" --kernels "$REPO/build" \
  --weights "$K3_WEIGHTS" --tokenizer "$K3_TOKENIZER" --gpu 0 --capacity 16384 --no-perf --no-sweep "${args[@]}" \
  --out "$out.json" > "$out.log" 2>&1
rc=$?
echo "== $out rc=$rc"; grep -E "^(logits|noise|PASS|FAIL|INCONCLUSIVE)" "$out.log" | cut -c1-300
