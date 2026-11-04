# Sourced by every loop/ script: machine paths from loop/local.env, the repo root.
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$LOOP_DIR/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "${K3_LOOP_ENV:-$LOOP_DIR/local.env}"
set +a
export LD_LIBRARY_PATH="$K3_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="/dev/shm/k3loop-$(id -un)"
mkdir -p "$TMPDIR"
