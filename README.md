# kern-k31

Kimi-K3.1 Pro served disaggregated on the
[kern runtime](https://github.com/pegainfer-project/kern): a packed prefill
(P) as eight pipeline stages, one GPU each, handing its pages to a decode
step (D) at TP8 / EP8 / DCP8 on eight GPUs. kern contains no model; this
repository is what makes it K3.1: the manifest generator, the kernels with
their sources and pinned hashes, and the tools that check them.

## K3.1 against K3

K3.1 is K3 (93 layers: 69 KDA, 24 gated MLA; 896 experts, top 16, MXFP4)
with DeepSeek-V3.2-style sparse attention (DSA) on the 24 MLA layers,
0-based 3, 7, ..., 91, 92. Each of them gains an indexer:

| Tensor (`self_attn.indexer.*`) | Shape | Role |
|---|---|---|
| `wq_b` | [4096, 1536] | 32 query heads x 128 from the normed q latent |
| `wk` | [128, 7168] | one shared key from the hidden |
| `k_norm.{weight,bias}` | [128] | LayerNorm of the key, in f32 |
| `weights_proj` | [32, 7168] | a weight per head from the hidden |

A row scores every earlier token as the weighted sum over heads of
`relu(q_h . k)`, keeps the top 2048 and attends only those. There is no
position encoding in the indexer, as in the MLA. While a sequence stays
within 2048 tokens the top 2048 is everything, so dense attention is exact.

Every K3 tensor keeps its name, shape and dtype, and the tokenizer is
byte-identical. `config.json` adds `dsa_config` and names the model
`kimi_k31`. The chat template adds a `medium` reasoning effort and keeps a
tool argument's JSON literal as written.

The reference is vLLM's K3.1 support
([vllm#65685](https://github.com/vllm-project/vllm/pull/65685), image
`vllm/vllm-openai:kimi-k31`).

## Status

The generator, kernels and tools are kern-k3-pp-prefill's at
[`76b0bdb`](https://github.com/pegainfer-project/kern-k3-pp-prefill/tree/76b0bdb):
K3's dense attention. They load K3.1's checkpoint and are exact for it only
within 2048 tokens. Nothing K3.1-specific has landed yet.

## Design

Unchanged from K3: the MLA cache is fp8 (e4m3 at scale 1, 576 B a token,
64-token pages) on P and D alike, and the KDA state is bf16.

**The indexer key cache** is one more paged state per MLA layer, 144 B a
token laid out token-major: 128 B of e4m3 key, its 4 B scale, 12 B of
padding. The padding is there because kern pushes P's rows to D and maps
them through TMA only in multiples of 16 B. The state shares the MLA cache's
slot mapping and block table, and on D it is dealt by position like it.

**Projections.** `wk | weights_proj` read the same normed hidden as the
fused `q_a | kv_a | g` projection and append to it as 160 more columns.
`wq_b` is a GEMM of its own off the q latent, because appending it to `q_b`
would move strides the attention kernels hard-code.

**P.** Rows whose context is within 2048 keep the dense path. A longer
row's MLA layer scores its whole context, takes its top 2048 and runs
sparse attention over the fp8 pages. That replaces the windowed rounds of
dense attention K3 needs past 64k.

**D.** Position p lives on member p % 8. Each member scores its own
positions and takes a local top 2048. The global top 2048 is the top of the
8 x 2048 candidates, with ties broken by position. Each member then attends
the selected positions it owns, and the existing LSE merge across members
is unchanged.

**Kernels.** Sparse attention is TRT-LLM-gen's sparse MLA (e4m3 q and KV,
576 / 512), vLLM's default for K3.1 on SM100: it reads the fp8 cache as it
is, from the FlashInfer cubin bundle this repository already pins. It has
no 96-head tile, so the heads are padded to 128. Scoring is DeepGEMM's MQA
logits (32 heads x 128) built ahead of time, the top 2048 is vLLM's top-k,
and the key insert follows vLLM's LayerNorm, UE8M0 scale and e4m3 cast.
kern-glm53-pp-prefill already runs the same indexer for GLM-5.3, whose
shapes match.

## Layout

| Path | What it is |
|---|---|
| [`scripts/gen_stage.py`](scripts/gen_stage.py) | the generator of every manifest, on kern's `tools/`; MLA cache flavors in [`k3_kv_fp8.py`](scripts/k3_kv_fp8.py) and [`k3_kv_bf16.py`](scripts/k3_kv_bf16.py) |
| [`scripts/deploy.sh`](scripts/deploy.sh) | P's stages and D's manifest of a deployment, verified |
| [`scripts/k3_vllm.py`](scripts/k3_vllm.py), [`scripts/k3_sglang.py`](scripts/k3_sglang.py) | the D manifest as vLLM's or SGLang's model forward |
| [`kernels.toml`](kernels.toml) | every module a manifest pins: SHA-256, source and defines, or prebuilt path and upstream |
| [`source/`](source/) | handwritten CUDA and vendored FlashKDA, cuDNN's FROST KDA prefill (CuTe DSL), FlashInfer MoE routing and Cake MLA decode, each with its license and provenance |
| [`prebuilt/`](prebuilt/) | cubins with no source recipe here; origins in [`prebuilt/README.md`](prebuilt/README.md) |
| [`step/`](step/) | `k3_step`: a D manifest teacher-forced, one process per rank |
| [`harness/`](harness/) | kernel acceptance runs |
| [`decode/sglang-tp8-dcp8-ep8/`](decode/sglang-tp8-dcp8-ep8/) | SGLang's K3 decode graphs captured launch by launch, which D's layout follows |
| [`loop/`](loop/) | the kernel optimization loop and its GPU lease tools |

## Build and deploy

Everything is pinned to kern
[`5d0910c`](https://github.com/pegainfer-project/kern/tree/5d0910c5e5beef5b1b204853ad65f3e1fe994777),
the commit `step/Cargo.toml` names, on GB300 (sm_103a).

```sh
NVCC=<cuda-13.0>/bin/nvcc python3 scripts/build.py       # cubins into build/, every SHA-256 checked
scripts/deploy.sh <kern checkout> <kern master>/kernel_index <out>   # <out>/p/stage0..7.json, <out>/d.json
python3 scripts/check_binds.py <manifest> <checkpoint>   # every rank's weight slices, no GPU
```

CUDA 13.0 nvcc (V13.0.88) reproduces the pinned cubins. Every process needs
NCCL 2.30.7 with its unversioned `libnccl.so` on `LD_LIBRARY_PATH`, and
`NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1` across hosts. Start D's eight
ranks first, then P's eight stages with the head (stage 0) serving HTTP:

```sh
kern-serve --manifest d.json --weights <ckpt> --capacity 2000000 --max-seqs 64 --gpu 0 \
    --decode-listen 0.0.0.0:<pd port> --listen 0.0.0.0:<ranks port>
kern-serve --manifest d.json --weights <ckpt> --capacity 2000000 --max-seqs 64 --gpu <g> \
    --head <A>:<ranks port> --rank <r>                        # r = 1..7
kern-serve --manifest p/stage0.json --kernels build --weights <ckpt> --chunk 8192 --max-seqs 128 \
    --gpu 0 --port <http port> --listen 0.0.0.0:<pipe port> --decode <A>:<pd port> \
    --host-gib 128 --host-turn 8192
kern-serve --manifest p/stage<s>.json --kernels build --weights <ckpt> --chunk 8192 --max-seqs 128 \
    --gpu <g> --head <P head>:<pipe port>                     # s = 1..7
```

## Check

`python3 scripts/check.py [build]` checks every module in `kernels.toml`
(hash, source or upstream, the bundled and built cubins) and the captured
decode graphs' modules. CI runs it on every push.

## Provenance

This repository starts from
[kern-k3-pp-prefill](https://github.com/pegainfer-project/kern-k3-pp-prefill)
at `76b0bdb`. Its handwritten kernels and generator began in kern's `tools/`
at [`b5c7d14`](https://github.com/pegainfer-project/kern/tree/b5c7d14). K3's
measurements and optimization records stay there.

kern's sources keep their [Apache-2.0 license](source/LICENSE), FlashKDA its
[MIT license](source/flash-kda/LICENSE), FlashInfer its
[Apache-2.0 license](source/flashinfer-moe-routing/LICENSE), cuDNN frontend's FROST its
[Apache-2.0](source/cudnn-frost-kda/LICENSE) and [MIT](source/cudnn-frost-kda/LICENSE-MIT) licenses.
