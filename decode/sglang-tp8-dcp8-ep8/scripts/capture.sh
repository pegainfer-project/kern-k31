#!/usr/bin/env bash
# Capture SGLang's full-K3 TP8 / DCP8 / EP8 decode graphs on one of two 4-GPU nodes (run once per node).
#
#   MODEL=<kimi-k3 checkpoint> CAPTURE_LIB=<libkernelcapture.so> OUT=<capture dir> \
#   NODE_RANK=<0|1> DIST_INIT=<node 0 address>:<port> scripts/capture.sh
#
# CAPTURE_LIB is kern's CUPTI injection library (tools/kernel-capture in pegainfer-project/kern); it
# records every launch, including the ones made while SGLang captures its decode CUDA graphs, into
# $OUT/pid<N>/launches.jsonl and every module into $OUT/pid<N>/module_<id>.cubin. The decode server
# runs SGLang's fake PD transfer, so a request skips prefill and decodes over its prompt's length of
# uncomputed context; nothing beyond the graph capture at startup is needed for the graphs. Set the
# NCCL / Gloo interface variables for your network; NCCL_MNNVL_ENABLE assumes an NVL72 fabric.
set -euo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20261004-295a53c9}
: "${MODEL:?}" "${CAPTURE_LIB:?}" "${OUT:?}" "${NODE_RANK:?}" "${DIST_INIT:?}"
mkdir -p "$OUT"
exec docker run --rm --network host --ipc host --privileged --gpus all --ulimit memlock=-1:-1 --shm-size 64g \
  -v "$MODEL:$MODEL:ro" -v "$(dirname "$CAPTURE_LIB"):$(dirname "$CAPTURE_LIB"):ro" -v "$OUT:$OUT" \
  -e CUDA_INJECTION64_PATH="$CAPTURE_LIB" -e KERNEL_CAPTURE_DIR="$OUT" \
  -e NCCL_MNNVL_ENABLE=1 -e NCCL_CUMEM_ENABLE=1 -e NCCL_NVLS_ENABLE=1 -e OMP_NUM_THREADS=4 \
  ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME} ${GLOO_SOCKET_IFNAME:+-e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME} \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name k3 --trust-remote-code \
  --page-size 64 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.88 --mamba-full-memory-ratio 0.5 \
  --context-length 1048576 --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --cuda-graph-bs-decode 1 8 48 --cuda-graph-backend-prefill disabled --watchdog-timeout 3600 \
  --disaggregation-mode decode --disaggregation-transfer-backend fake \
  --tp-size 8 --dcp-size 8 --ep-size 8 --chunked-prefill-size 16384 \
  --prefill-attention-backend trtllm_mla --decode-attention-backend cutedsl_mla \
  --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT" --host 0.0.0.0 --port 20000
