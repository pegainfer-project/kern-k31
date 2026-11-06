# kern-k31

Kimi-K3.1 Pro served disaggregated on the
[kern runtime](https://github.com/pegainfer-project/kern): a packed prefill
(P) as eight pipeline stages, one GPU each, handing its pages to a decode
step (D) on eight GPUs at TP8 / DCP8, its experts EP2 x TP4. kern contains
no model; this repository is what makes it K3.1: the manifest generator, the
kernels with their sources and pinned hashes, and the tools that check them.

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
position encoding in the indexer, as in the MLA.

Every K3 tensor keeps its name, shape and dtype, and the tokenizer is
byte-identical. `config.json` adds `dsa_config` and names the model
`kimi_k31`. The chat template adds a `medium` reasoning effort and keeps a
tool argument's JSON literal as written.

The reference is vLLM's K3.1 support
([vllm#65685](https://github.com/vllm-project/vllm/pull/65685), image
`vllm/vllm-openai:kimi-k31`).

## Status

The generator, kernels and tools began as kern-k3-pp-prefill's at
[`76b0bdb`](https://github.com/pegainfer-project/kern-k3-pp-prefill/tree/76b0bdb).
K3.1's DSA runs on P and D alike (the default `--kv fp8`; `--kv bf16` keeps
K3's dense attention). D also runs as the model forward of an unmodified
SGLang decode instance, over SGLang's own pools:
[`docs/try-in-sglang.md`](docs/try-in-sglang.md) says how, and what it
measured.

## Design

Unchanged from K3: the MLA cache is fp8 (e4m3 at scale 1, 576 B a token,
64-token pages) on P and D alike, and the KDA state is bf16.

**The indexer key cache** is one more paged state per MLA layer, 144 B a
token laid out token-major: 128 B of e4m3 key, its UE8M0 scale as a 4 B
word, 12 B of padding. The padding is there because kern pushes P's rows to
D and maps them through TMA only in multiples of 16 B. The state shares the
MLA cache's slot mapping and block table, and on D it is dealt by position
like it. Hosted by SGLang, D reads SGLang's index-K page instead (64 keys,
then their 64 f32 scales; `scripts/k3_sglang.py`).

**Projections.** `wk | weights_proj` read the same normed hidden as the
fused `q_a | kv_a | g` projection. On D at TP8 they append to it as 160 more
columns, f32 out; P and the one-GPU oracle run them as a bf16 GEMM of their
own. `wq_b` is a GEMM of its own off the q latent, because appending it to
`q_b` would move strides the attention kernels hard-code.

**P.** Every row's MLA layer scores its sequence's keys up to its own
position, takes its top 2048 (all of them when there are fewer) and runs
sparse attention over the fp8 pages. That replaces the windowed rounds of
dense attention K3 needs past 64k.

**D.** Position p lives on member p % 8. Each member scores its own
positions, takes a local top 2048 and pushes its best 384 to every member.
The global top 2048 (higher score first, the smaller position among equal
scores) is cut from the 8 x 384; when a member that sent all 384 may hold
more of it, its whole local list is pulled and the cut taken again
(`source/k31_dsa.cu`). Each member attends the selected positions it owns,
and k3_dcp's SPARSE exchange moves the v-up'd partials from positions to
heads and merges them by LSE.

**Kernels.** Sparse attention is TRT-LLM gen's sparse MLA (e4m3 q and KV,
576 / 512) from FlashInfer 0.7.2.post1's cubin bundle, vLLM's default for
K3.1 on SM100: it reads the fp8 cache as it is. It has no 96-head tile, so
the heads are padded to 128. Scoring is DeepGEMM's MQA logits (32 heads x
128) built ahead of time, the paged kernel patched for the token-major key
cache (`source/deepgemm-k31`). The top 2048 is `k31_topk`: the set vLLM's
`top_k_per_row_prefill` selects, the same bytes on every run. The key insert
follows vLLM's LayerNorm, UE8M0 scale and e4m3 cast.

## Layout

| Path | What it is |
|---|---|
| [`scripts/gen_stage.py`](scripts/gen_stage.py) | the generator of every manifest, on kern's `tools/`; the DSA layer in [`k31_dsa.py`](scripts/k31_dsa.py), MLA cache flavors in [`k3_kv_fp8.py`](scripts/k3_kv_fp8.py) (DSA) and [`k3_kv_bf16.py`](scripts/k3_kv_bf16.py) (K3's dense attention) |
| [`scripts/deploy.sh`](scripts/deploy.sh) | P's stages and D's manifest of a deployment, verified |
| [`scripts/k3_sglang.py`](scripts/k3_sglang.py) | D's manifest over SGLang's pools, as SGLang's model forward |
| [`scripts/k3_vllm.py`](scripts/k3_vllm.py) | a `--kv bf16` D over vLLM's pools, as vLLM's model forward |
| [`kernels.toml`](kernels.toml) | every module a manifest pins: SHA-256, source and defines, or prebuilt path and upstream |
| [`source/`](source/) | handwritten CUDA (the `k31_*` DSA kernels among it), DeepGEMM's MQA logits patched for the key cache, vLLM's prefill top-k, cuDNN's FROST KDA prefill (CuTe DSL), FlashInfer MoE routing and K3's Cake MLA decode, each with its license and provenance |
| [`third_party/DeepGEMM`](third_party/) | DeepGEMM at the commit `kernels.toml` names, a submodule |
| [`prebuilt/`](prebuilt/) | cubins with no source recipe here; origins in [`prebuilt/README.md`](prebuilt/README.md) |
| [`step/`](step/) | `k3_step`: a D manifest teacher-forced, one process per rank |
| [`harness/`](harness/) | kernel acceptance runs |
| [`decode/sglang-tp8-dcp8-ep8/`](decode/sglang-tp8-dcp8-ep8/) | SGLang's K3 decode graphs captured launch by launch, whose attention and DCP layout D follows |
| [`docs/`](docs/) | [`try-in-sglang.md`](docs/try-in-sglang.md), K3.1's decode under SGLang; [`flashkda_backend.md`](docs/flashkda_backend.md), the KDA prefill backends compared |
| [`loop/`](loop/) | K3's archived kernel optimization loop and its GPU lease tools |

## Build and deploy

Everything is pinned to kern
[`1b5172b`](https://github.com/pegainfer-project/kern/tree/1b5172b8de11f6fa3daa1f718615519428b28821),
the commit `step/Cargo.toml` names, on GB300 (sm_103a).

```sh
NVCC=<cuda-13.0>/bin/nvcc python3 scripts/build.py       # cubins into build/, every SHA-256 checked
scripts/deploy.sh <kern checkout> <kern master>/kernel_index <out>   # <out>/p/stage0..7.json, <out>/d.json
python3 scripts/check_binds.py <manifest> <checkpoint>   # every rank's weight slices, no GPU
```

CUDA 13.0 nvcc (V13.0.88) reproduces the pinned cubins. A manifest names
its kernels by hash in the kernel registry (`Pegainfer/kern-kernels`), which
holds every one of them; kern-serve fetches them on first start, and
`--kernels build` runs a local build instead. Every process needs NCCL
2.30.7 with its unversioned `libnccl.so` on `LD_LIBRARY_PATH`, and
`NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1` across hosts. Start D's eight
ranks first, then P's eight stages with the head (stage 0) serving HTTP:

```sh
kern-serve --manifest d.json --weights <ckpt> --capacity 2000000 --max-seqs 64 --gpu 0 \
    --decode-listen 0.0.0.0:<pd port> --listen 0.0.0.0:<ranks port>
kern-serve --manifest d.json --weights <ckpt> --capacity 2000000 --max-seqs 64 --gpu <g> \
    --head <A>:<ranks port> --rank <r>                        # r = 1..7
kern-serve --manifest p/stage0.json --weights <ckpt> --chunk 8192 --max-seqs 128 \
    --gpu 0 --port <http port> --listen 0.0.0.0:<pipe port> --decode <A>:<pd port>
kern-serve --manifest p/stage<s>.json --weights <ckpt> --chunk 8192 --max-seqs 128 \
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

kern's sources keep their [Apache-2.0 license](source/LICENSE); FlashInfer's
MoE routing and Cake MLA its [Apache-2.0 license](source/flashinfer-moe-routing/LICENSE);
vLLM's top-k its [Apache-2.0 license](source/vllm-topk/LICENSE); DeepGEMM its
[MIT license](source/deepgemm-k31/LICENSE); cuDNN frontend's FROST its
[Apache-2.0](source/cudnn-frost-kda/LICENSE) and [MIT](source/cudnn-frost-kda/LICENSE-MIT) licenses;
the TRT-LLM gen cubins theirs in [`prebuilt/`](prebuilt/LICENSE.trtllm-gen.txt).
