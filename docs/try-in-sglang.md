# Try K3.1 in SGLang

kern runs Kimi-K3.1 Pro's decode step as the model forward of an SGLang
**decode instance**. Everything else is SGLang's: the prefill instance, the
router, the scheduler, the KV / index-K / KDA-state pools, P/D transfer,
CUDA graphs and sampling. kern reads and writes SGLang's pools as SGLang
lays them out, so the SGLang image runs as published, with no patch.

The decode instance is all kern provides. It has no prefill program, so it
only runs behind an SGLang prefill instance in a P/D deployment.

## Requirements

| | |
|---|---|
| GPUs | GB300 (sm_103a), 4 per node, NVLink-connected nodes (MNNVL, IMEX channel available). Decode takes 2 nodes (TP8 / DCP8); SGLang's prefill takes 2 more. |
| SGLang | `lmsysorg/sglang:dev-kimi-k31-pro-cu134` (tested at `sha256:a42079a8d1299b94824eab821de5f261c016797434cf69dc1fb3b15bdfddfc0e`, arm64) |
| kern | `pip install "kern-llm>=0.4.0"` inside that image, decode nodes only |
| Manifest | [`Pegainfer/kern-k31-sm103`](https://huggingface.co/Pegainfer/kern-k31-sm103) on Hugging Face (kernels are fetched from `Pegainfer/kern-kernels` on first start) |
| Weights | [`moonshotai/Kimi-K3.1-Pro`](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), unmodified (tested at revision `6d0a020`), the same path on every node |

Limits of this manifest: at most 256 sequences a decode step, context up to
131,072 tokens, exactly two GB300 nodes at TP8 / DCP8.

## Deploy

Run every server in the image above with host networking, the IMEX channel
and unlimited locked memory, for example:

```bash
docker run --gpus all --network host --ipc host --ulimit memlock=-1:-1 --shm-size 64g \
  --device /dev/nvidia-caps-imex-channels/channel0 -v <weights>:<weights> \
  lmsysorg/sglang:dev-kimi-k31-pro-cu134 ...
```

GB300 has no GPU-direct RDMA, so P/D transfer goes over NVLink. Every P and
D process takes this environment, plus your cluster's NCCL / Gloo interface
settings (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`):

```bash
export MOONCAKE_PROTOCOL=nvlink MC_FORCE_MNNVL=1 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
```

**Prefill**: SGLang's own K3.1, on two nodes, `--node-rank 0` and `1`:

```bash
python3 -m sglang.launch_server --model-path moonshotai/Kimi-K3.1-Pro --trust-remote-code \
  --tp-size 8 --nnodes 2 --node-rank <0|1> --dist-init-addr <P node 0>:<port> \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --page-size 64 --mamba-ssm-dtype bfloat16 --dsa-decode-backend trtllm --context-length 131072 \
  --host 0.0.0.0 --port 30000
```

**Decode**: kern, on two other nodes:

```bash
pip install "kern-llm>=0.4.0"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Pegainfer/kern-k31-sm103', local_dir='kern-k31')"

KERN_MANIFEST=kern-k31/manifests/kimi-k31-d-tp8-sglang.json \
python3 -m sglang.launch_server --model-path moonshotai/Kimi-K3.1-Pro --trust-remote-code \
  --tp-size 8 --dcp-size 8 --nnodes 2 --node-rank <0|1> --dist-init-addr <D node 0>:<port> \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --page-size 64 --mamba-ssm-dtype bfloat16 --dsa-decode-backend trtllm --context-length 131072 \
  --max-running-requests 256 --host 0.0.0.0 --port 30000
```

**Router**:

```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<P node 0>:30000 8998 --decode http://<D node 0>:30000 --host 0.0.0.0 --port 8000
```

Requests go to the router (`/v1/chat/completions`, `/generate`, ...).

What each flag is for:

- `KERN_MANIFEST` turns the plugin on; `kern-llm` registers it with SGLang, and without the variable
  SGLang runs its own model. kern loads its own weight slices from `--model-path` and the kernels from
  the registry (cached in `~/.cache/kern`, or `$KERN_CACHE_DIR`); SGLang loads no weights for the
  decode instance.
- `--page-size 64 --mamba-ssm-dtype bfloat16 --dsa-decode-backend trtllm` lay SGLang's pools out the way
  the manifest reads them (64-token pages, raw fp8 MLA cache, bf16 KDA state). P must use the same
  three, since it writes the bytes D reads. A different value makes kern refuse the pool at startup,
  naming the layer state that does not match.
- `--context-length 131072 --max-running-requests 256` keep SGLang within the manifest's bounds; left
  out, SGLang takes the checkpoint's 1M context and sizes over a thousand slots.

## Results

GB300, P (SGLang, TP8) on two nodes and D on two others, the same P for both
decode instances, 2026-11-06.

**Accuracy.** SGLang is not run-to-run deterministic, so a decode is judged
against SGLang's own run-to-run spread; where two runs part, the table gives
the largest top-1 / top-2 logprob margin at the parting token.

| | SGLang decode | kern decode |
|---|---|---|
| GSM8K, 5-shot, greedy, all 1319, two runs | 0.969 / 0.967 | 0.970 / 0.966 |
| 16 prompts of 3k-31k tokens (past DSA's top-2048), 48 greedy tokens: same as an SGLang run | 7 / 16, margin <= 0.5 | 7 / 16 and 6 / 16, margin <= 0.375 |

**Speed.** Synthetic multi-turn chat through the router (history
accumulates, `ignore_eos`), 64 conversations at a time; P's prefix-cache hit
rate is within 1 point of an infinite cache in both:

| workload | decode | TPOT mean / p99 | output tok/s |
|---|---|---|---|
| 8 turns, 3072-token opening, 768 new + 256 out a turn | SGLang | 23.3 / 25.3 ms | 2083 |
| | kern | 20.5 / 22.3 ms | 2300 |
| 6 turns, 2048-token opening, 768 new + 256 out a turn | SGLang | 24.0 / 25.5 ms | 2129 |
| | kern | 20.7 / 22.3 ms | 2381 |

Decode step time by batch over all multi-turn runs (median, from the decode
instance's own logs):

| running requests | SGLang | kern |
|---|---|---|
| 1-32 | 15.0 ms | 14.9 ms |
| 33-64 | 24.4 ms | 21.7 ms |
| 65-96 | 32.0 ms | 26.4 ms |
| 97-112 | 32.1 ms | 28.7 ms |
| 113-128 | 34.3 ms | 30.4 ms |
