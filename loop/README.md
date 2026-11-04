# The K3 kernel loop

Agents optimize the P and D kernels of Kimi-K3 overnight against a fixed
workload, on sub-models small enough to measure in minutes. The orchestrator
merges what passes into `main`.

**Objective, in order:** fewer kernels per layer (fuse; communication
kernels may be hand-written), then less time. GEMMs, quantization and the
parallel layout are fixed. Every numerical difference from `main` is
explained, not just tolerated. The agents' brief is [`TASK.md`](TASK.md),
each run's target is under [`rounds/`](rounds/), and each run keeps its
memory in `notes/<run>.md`.

## Benches and checks

The workload is the recorded 1P2D AgentX run (c64): P item shapes by rows,
D steps by batch rows. Weights are in [`score`](score).

| | Sub-model (`gen`) | Bench | Check |
|---|---|---|---|
| P | stage 0 of PP8 (embedding, layers 0-11: 9 KDA, 3 MLA), cut from the packed 93-layer prefill; stage 0 because it builds the MLA attention tables a later stage would get zeroed | `p/bench`: `kern bench` over [`p/workload.toml`](p/workload.toml), 1 GPU | `p/check`: `kern test` of layers 0-11 (embedding to head) against `main`, 1 GPU |
| D | 16 layers of the TP8 / EP8 / DCP8 step | `d/bench`: `k3_step --bench` on a real TP8 group over two hosts, 64k and 128k contexts, 8-48 rows | `d/check`: 4 layers teacher-forced over real text against the one-GPU oracle, judged against `main`'s band |

D is measured on real TP8 rather than a single rank with stubbed
collectives, so that the all-reduce and the DCP exchange can be fused and
measured too.

`main` on 2026-10-10 (GB300):

| | Launches | Weighted cost |
|---|---|---|
| P stage 0 | 316 (26.3/layer) | 132.00 ms/item |
| D 16 layers | 395 (24.7/layer) | 4.562 ms/step |

D check of `main` against its oracle: top-1 0.9176, relRMS median 1.77% /
max 8.3%, one confident flip; that band comes from TP8 summing bf16 partials
across ranks and merging DCP attention by LSE. Two runs of `main` are
bit-identical.

## GPUs

[`withgpu`](withgpu) leases GPUs from a pool of hosts with one flock per
GPU and skips GPUs that run anything outside a lease. `-H 2` leases on two
hosts at once for the TP8 runs. The P runs and the D runs use separate pools,
so that single-GPU P leases never starve the 8-GPU D leases.

## Running it

Copy [`local.env.example`](local.env.example) to `local.env` and fill in
the paths. Then:

```sh
loop/build && loop/gen                       # kernels, manifests, launch counts
loop/agents up d-mlp "<hostA> <hostB>" 10 loop/rounds/r1/d-mlp.md
loop/agents status
```
