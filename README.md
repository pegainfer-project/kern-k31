# kern-k31

Kimi-K3.1 Pro (896 experts, 93 layers) served disaggregated on the
[kern runtime](https://github.com/pegainfer-project/kern) and its
`kern-serve`:

- **P**: the packed prefill (up to 16 sequences in a chunk of 8192 rows,
  1M context) as eight pipeline stages, one GPU each, cut every 12
  layers.
- **D**: the decode step at TP8 / EP8 / DCP8, eight GPUs.

kern contains no model. This repository is what makes it K3.1: the manifest
generator, the kernels with their sources and pinned hashes, and the
tools that check them.

K3.1 is K3 with DeepSeek-V3.2-style sparse attention (DSA) on its 24 MLA
layers (0-based 3, 7, ..., 91, 92): an indexer of 32 heads x 128 scores
every earlier token and the layer attends only the top 2048. Every other
tensor keeps K3's name, shape and dtype; each MLA layer adds
`self_attn.indexer.{wq_b, wk, k_norm, weights_proj}`. The repository starts
from [kern-k3-pp-prefill](https://github.com/pegainfer-project/kern-k3-pp-prefill)
at `76b0bdb`, whose manifests are K3's dense attention: they are exact for
K3.1 only while a sequence stays within 2048 tokens. The MLA cache stays
fp8 and the KDA state bf16.

| Path | What it is |
|---|---|
| [`scripts/gen_stage.py`](scripts/gen_stage.py) | the generator of every K3 manifest here, on kern's `tools/`; its MLA cache by `--kv` from [`k3_kv_fp8.py`](scripts/k3_kv_fp8.py) (the default) or [`k3_kv_bf16.py`](scripts/k3_kv_bf16.py) |
| [`scripts/deploy.sh`](scripts/deploy.sh) | the P stages and the D manifest of a deployment, verified |
| [`scripts/k3_vllm.py`](scripts/k3_vllm.py) | a D manifest as vLLM's model forward, on [`k3_hosted.py`](scripts/k3_hosted.py)'s program boundary |
| [`scripts/k3_sglang.py`](scripts/k3_sglang.py) | the same D as SGLang's model forward |
| [`kernels.toml`](kernels.toml) | every module the manifests pin: SHA-256, source and defines, or prebuilt path and upstream |
| [`source/`](source/) | the handwritten CUDA (Apache-2.0, [`source/LICENSE`](source/LICENSE)), vendored FlashKDA (MIT), FlashInfer MoE routing and FlashInfer's Cake Kimi-K3 MLA decode (Apache-2.0), each with its `LICENSE` and `PROVENANCE.md` |
| [`prebuilt/`](prebuilt/) | the cubins with no source recipe here; origins and licenses in [`prebuilt/README.md`](prebuilt/README.md) |
| [`step/`](step/) | `k3_step`: a decode manifest teacher-forced, one process per rank |
| [`harness/`](harness/) | kernel acceptance runs and the error-budget study |
| [`decode/sglang-tp8-dcp8-ep8/`](decode/sglang-tp8-dcp8-ep8/) | SGLang's full-K3 decode CUDA graphs at TP8 / DCP8 / EP8, captured launch by launch: the reference D's layout follows |
| [`loop/`](loop/) | the overnight kernel loop that optimizes P and D |

## Deploy

Everything is pinned to kern
[`5d0910c`](https://github.com/pegainfer-project/kern/tree/5d0910c5e5beef5b1b204853ad65f3e1fe994777)
(`step/Cargo.toml` pins the same commit). On GB300 (sm_103a):

1. **kern.** Check out that commit and build `kern` and `kern-serve`
   (`cargo build --release -p kern-run -p kern-serve`; kern-serve needs
   protoc and libssl-dev).
2. **Manifests.**

   ```sh
   scripts/deploy.sh <kern checkout> <kern master>/kernel_index <out>
   ```

   writes `<out>/p/stage0..7.json` and `<out>/d.json`, each verified by
   `kern`. The generator reads modules this repository pins from
   `kernels.toml`, everything else from kern's kernel index (`kernel_index/`).
3. **Kernels.** CUDA 13.0 nvcc (V13.0.88) reproduces the pinned cubins:

   ```sh
   NVCC=<cuda-13.0>/bin/nvcc python3 scripts/build.py
   ```

   `build.py` copies the prebuilt cubins into `build/`, compiles the rest
   with `nvcc -cubin -arch=sm_103a` and the `defines` in `kernels.toml`,
   and checks every SHA-256. P loads them with `--kernels build`; modules
   from the index come through kern's cache (`KERN_CACHE_DIR`).
4. **D** first, eight processes over two 4-GPU hosts, rank 0 on host A:

   ```sh
   kern-serve --manifest d.json --weights <kimi-k3> --capacity 2000000 --max-seqs 64 --gpu 0 \
       --decode-listen 0.0.0.0:<pd port> --listen 0.0.0.0:<ranks port>
   kern-serve --manifest d.json --weights <kimi-k3> --capacity 2000000 --max-seqs 64 --gpu <g> \
       --head <A>:<ranks port> --rank <r>          # r = 1..7, g = r mod 4
   ```

5. **P**, eight processes, the head (stage 0) serving HTTP and holding the
   host tier:

   ```sh
   kern-serve --manifest p/stage0.json --kernels build --weights <kimi-k3> --chunk 8192 --max-seqs 128 \
       --gpu 0 --port <http port> --listen 0.0.0.0:<pipe port> --decode <A>:<pd port> \
       --host-gib 128 --host-turn 8192
   kern-serve --manifest p/stage<s>.json --kernels build --weights <kimi-k3> --chunk 8192 --max-seqs 128 \
       --gpu <g> --head <P head>:<pipe port>      # s = 1..7
   ```

   More decode tiers are more `--decode` endpoints on the head.

Every process needs NCCL 2.30.7 (with its unversioned `libnccl.so`) on
`LD_LIBRARY_PATH`, and `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1` across
hosts. The head serves the OpenAI completions and chat API.

The same manifests run on HGX B300 (148 SMs): every launch that needs all
its blocks resident is sized for 148 (the router 2 x 144, the fused TP
sums 144, the DCP exchange 2 x 144, two of its blocks an SM), and the fp8
Cake attention's splits are FlashInfer's plan at 148 SMs for a member's
share of the capture context ([`scripts/cake_plan.py`](scripts/cake_plan.py),
checked against FlashInfer's own planner by
[`harness/cake_plan/check.py`](harness/cake_plan/check.py)).

## Results (GB300)

None for K3.1 yet. K3's numbers on this stack (1P2D on AgentX, 2026-10-11)
are in kern-k3-pp-prefill's
[`loop/rounds/r1/E2E.md`](https://github.com/pegainfer-project/kern-k3-pp-prefill/blob/main/loop/rounds/r1/E2E.md).

## P: the packed prefill

- `--pack N` (16) makes the prefill ragged: up to N sequences a call, back to
  back, split by `cu_seqlens`. The KDA layers run FlashKDA's varlen build
  with per-sequence conv windows and states; the MLA layers run TRT-LLM
  gen's context FMHA batched over the sequences; the head reads each
  sequence's last row.
- A long call's MLA layer attends its own rows through the causal FMHA and
  its sequences' cached context in rounds of `--window` keys (65536) through
  the dense sibling of the same kernel, each round gathered from the pages
  and expanded to k | v while still in L2, every pass folded by its softmax
  stats into an f32 accumulator the gate reads (`source/k3_fmha_plan.cuh`,
  `source/k3_mla_glue.cu`). The expansion workspace is the window's, not the
  context's: 4.3 GiB a stage at 64k against 60 GiB for a whole 1M context.
  `harness/p_fmha_window.cu` checks the passes against the plain causal
  call and an f32 reference and times them (8192 rows over 64k: +6%; over
  256k in 64k rounds: +0.8%; an empty round 17 µs).
- The MLA cache is SGLang's fp8 one (`--kv fp8`, the default: e4m3 at
  scale 1, cast as k3_mla_prep's append casts) on P and D alike, so a
  prefill's pages hand to a decode tier as they are; the P lifts a round's
  rows to bf16 for the expansion and keeps its own rows in bf16. Over fp8
  pages a short call expands its context like any other (the absorbed
  path is the bf16 DSL kernel's; `--kv bf16` keeps it). The KDA recurrent
  state is stored bf16 (`--kda-state bf16`, the default) and computed f32
  in both recurrence kernels; `--kda-state f32` is the f32 line.
- Under `--kv fp8` the P follows vLLM's InferenceX K3 prefill recipe: its
  attention runs on e4m3 (q and the expanded k | v cast at scale 1 into the
  FMHA's e4m3 builds, o bf16; `use_prefill_query_quantization`), and four
  projections are MXFP8 (vLLM's online mxfp8 of `in_proj_qkvgfab`,
  `fused_qkv_a_g_proj`, `routed_expert_down_proj`, `routed_expert_up_proj`):
  KDA's q | k | v | g | wsm, MLA's q_a | kv_a | g, lat_down and lat_up. Their
  bf16 weights are `source` tensors quantized while the weights load, the
  activation is quantized per call (`source/k3_mxfp8.cu`), and cuBLASLt
  multiplies over both block scales (`extern:cublaslt_mxfp8_tn`). Everything
  else, the shared expert and o_proj included, stays bf16.
- Every layer has its own KV or KDA state, so `kern cut` splits the
  whole-model manifest into stages that carry only their own layers'
  states. A stage hands the next one the hidden stream and the
  attention-residual bank (`blocks`, one snapshot every 12 layers).
- At EP1 a stage holds all of its layers' experts. One launch
  (`k3_moe_route`) takes a chunk from the router to the routing tables of
  TRT-LLM gen's batched mxfp8 × mxfp4 GEMMs (FC1 with the fused SiTU gate,
  then FC2). A stage's raw experts are bound as `source` tensors and
  shuffled by the `derive` program while the weights load, so the
  checkpoint and its shuffled copy never share a GPU.

## D: TP8 / EP8 / DCP8

SGLang's layout: the hidden replicated on 8 GPUs, KDA and the MLA o side
split by head (12 a rank), the MLA KV dealt by position (`p % 8`), 112 of
the 896 experts a rank (`rank, rank + 8, …`).

| Module | What it is |
|---|---|
| `k3_kda_core+HEADS=12+KDA_CONV=1+…` | the KDA step over a rank's 12 heads, the conv fused in |
| `k3_mla_prep+INNER=1536+MLA_FUSED=3648`, `k3_mla_absorb_tc` | the MLA q side and the absorb on tcgen05 (a block a head; `k3_mla_absorb_mma` under `--kv fp8`), every head over the rank's own positions |
| `k3_dcp` | the DCP exchange around one all-to-all: fixup, pack, and the LSE merge of 8 partials into the rank's 12 heads |
| `k3_ar_fused`, `peer_allreduce_bf16+NRANKS=8` | the step's all-reduces as a Lamport one-shot over peer memory (`--peer-ar`; NCCL without), with the work on either side fused |
| `k3_moe_front+EXPERTS=896+…` | everything between the front GEMM and the batched GEMMs: router top-k, the latent in mxfp8, the rank's routing tables, the shared expert's activation |
| `k3_head+NRANKS=8` | the vocab-parallel head: every rank's slice of the logits to one token a row |

The routed MoE runs the same TRT-LLM gen batched-GEMM families at decode
tiles, the variants SGLang picks; `scripts/check_decode_bmm.py` replays
kern's port of their parameter block against every `bmm_*` launch of the
captured graphs.

The same flags at `--tp 1` give the one-GPU oracle with every
expert and head. `scripts/check_binds.py <manifest> <checkpoint>` checks
every rank's weight slices without a GPU, `scripts/rank_memory.py` prints
weights, slots and KV tokens a rank, and `scripts/compare_steps.py` judges
a `k3_step` group against the oracle (top-1 flips by the oracle's margin,
logits' relative RMS, every rank's tokens identical).
[`harness/run.sh`](harness/run.sh) is the kernels' acceptance run: the SASS
check (no stack, no local memory), the K3 harness over the 12-head and
896-expert variants, and [`harness/d1.cu`](harness/d1.cu) over the DCP
kernels with 8 members simulated on one GPU.

## D inside vLLM

kern's vLLM plugin (`python/kern_vllm`) can run the D step as the model
forward of an unmodified vLLM, one kern rank per tensor-parallel rank;
vLLM keeps scheduling, the KV pages, CUDA graphs and sampling.

```sh
python3 scripts/k3_vllm.py --input d.json --output d-vllm.json
kern verify d-vllm.json
```

- [`k3_hosted.py`](scripts/k3_hosted.py) is the engine-neutral program
  boundary: `decode_batch` stops at the final norm's `hidden`, and `head`
  turns `head_in` into this rank's contiguous `[seqs, 20480]` f32 slice
  of the vocab, which the engine gathers. The in-graph argmax is gone.
- [`k3_vllm.py`](scripts/k3_vllm.py) puts every state on vLLM's hybrid
  pages: 768-token blocks, a KDA line at the start of a 884,736 B page,
  an MLA page as 12 kernel blocks of 64 tokens (`kda_core`'s line
  stride is the only launch it changes).
- Run it with `-tp 8 -dcp 8`, bf16 KV and `--max-num-seqs` /
  `--max-num-batched-tokens` at most the manifest's 64. The manifest has
  no prefill, so prompts longer than one token need a KV connector.

kern's `docs/vllm.md` has the 4-layer TP8 / DCP8 run on two 4-GPU GB300
nodes.

## D inside SGLang

kern's SGLang plugin (`python/kern_sglang`) runs the same D as the model
forward of an unmodified SGLang, again one kern rank per tensor-parallel
rank; SGLang keeps scheduling, its KV and mamba pools, CUDA graphs and
sampling.

```sh
python3 scripts/k3_sglang.py --input d.json --output d-sglang.json
kern verify d-sglang.json
```

- [`k3_sglang.py`](scripts/k3_sglang.py) takes `k3_hosted.py`'s boundary
  and puts every state on SGLang's pools: an MLA layer's latent rows as
  the 64-row pages `k3_vllm.py` binds, a KDA layer's recurrent state as
  SGLang's `[slot][head][v][k]` f32 (`ssm.l<L>`, the line's rec byte for
  byte) and its conv window as SGLang's `[slot][tap][q | k | v]` bf16
  (`conv.l<L>`), each indexed by its own table of the same slot ids.
- `kda_core` is the only kernel that changes: its `CONV_TAP_MAJOR`
  build reads the window from a state of its own in that layout, and
  `LINE_GUARD` skips SGLang's padded rows (slot -1). Without the two
  defines it builds to the pinned bytes; `harness/run.sh conv` checks the
  variant against the D build bit for bit.
- Generate the D for it with `--kv bf16 --kda-state f32` (SGLang's pools
  as it runs them) and run it as SGLang's decode instance with
  `--tp-size 8 --dcp-size 8`, `--kv-cache-dtype bf16`, `--mamba-ssm-dtype float32`, and
  `--max-running-requests` / `--cuda-graph-max-bs-decode` at most the
  manifest's 64. Prompts reach the cache through SGLang's P/D (the
  manifest has no prefill); a unified server takes one-token prompts only.

## Check

```sh
python3 scripts/check.py [build]
```

checks that every module in `kernels.toml` has a well-formed hash and a
source or upstream, that its source and build files exist, that the
bundled cubins (and, given a directory, the built ones) match their
hashes, and the same for the captured decode graphs' modules. CI runs it on
every push.

## Provenance

The handwritten kernels started as kern's `tools/kernels-src/` and the
generator as kern's `tools/gen_k3.py`, at kern
[`b5c7d14`](https://github.com/pegainfer-project/kern/tree/b5c7d14), the
last commit before kern moved the K3 line to kern-k3-pp-prefill. Also
from that commit: `scripts/flash_kda_abi.py` (verbatim), `harness/k3/`
(kern's `tools/k3-harness`), and the load-time ops `scripts/k3_once.py`
registers back into kern's `once` (`k3_weight_prep`, `k3_moe_prefill`).
Everything up to
[`76b0bdb`](https://github.com/pegainfer-project/kern-k3-pp-prefill/tree/76b0bdb)
is kern-k3-pp-prefill's history; everything since is this repository's.

kern's sources keep their [Apache-2.0 license](source/LICENSE), FlashKDA
its [MIT license](source/flash-kda/LICENSE), FlashInfer its
[Apache-2.0 license](source/flashinfer-moe-routing/LICENSE). The cubins not
in kern's kernel index are all built from this repository by
`scripts/build.py`.
