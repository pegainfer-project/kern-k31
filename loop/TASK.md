# Task: fewer, faster kernels for Kimi-K3 on kern

Kimi-K3 (93 layers: 3 of every 4 KDA, every 4th MLA; 896-expert MoE on
every layer; attention residuals) is served on GB300 (`sm_103a`) by the kern
runtime, split into prefill (P) and decode (D):

- **P**: eight pipeline stages of 12 layers, one GPU each, TP1 / EP1,
  packed 8192-token chunks of up to 16 sequences.
- **D**: a TP8 / EP8 / DCP8 group: the batch replicated on 8 GPUs, KDA and
  MLA heads 12 per rank, MLA KV dealt by position over the ranks (LSE merge
  after an all-to-all), 112 experts per rank.

The forward pass is a **manifest**: a JSON file of buffers, ops (kernel
launches) and programs. kern runs it blindly. This repository holds the
kernels (`source/`, `kernels.toml`) and the generator of the manifests
(`scripts/gen_stage.py`). `ROUND.md` names your target; this file covers
what applies to every round.

## The objective

**First, fewer kernels.** A layer today launches ~25 kernels on D and ~27 on
P; about 16 of those are not GEMMs. DeepSeek V4.1 Flash runs a layer in
about ten. Fuse. Merge neighbouring elementwise / norm / routing / glue
kernels into one launch, fold them into the prologue or epilogue of the
kernel next to them, and hand-write the communication kernels where that
lets a collective absorb the work around it (an all-reduce that also adds
the residual and normalizes; an all-to-all that packs and merges). Stop
fusing a pair only when it fails or the result would be contrived (two
unrelated loops glued in one grid with nothing shared); say which in
`notes/<run>.md`.

**Then, time.** Fusion is a means: a fused kernel must not make the step
slower. The score is the traffic-weighted step time.

Out of scope:
- the GEMMs, unless your round opens them: cuBLASLt (`gemm_bf16`,
  `gemm_f32`) and TRT-LLM gen batched GEMMs (`moe_fc1`, `moe_fc2`) keep their
  kernels; fusing into their inputs or outputs from the neighbouring kernel
  is in scope, replacing them is not. Round 2 opens D's GEMMs
  (`loop/rounds/r2/GEMM.md`, with a stricter numerics rule)
- quantization: no new low-precision weights, activations or KV beyond
  what the manifests already use
- the parallel layout (TP / EP / DCP widths, PP split, MegaMoE): fixed
- the kern runtime (`$K3_KERN`): read it, never change it. If an idea needs
  a runtime feature, write it in `notes/<run>.md` as a proposal and move on

## What you edit

- `source/`: kernel CUDA. A new kernel or variant gets an entry in
  `kernels.toml` (`source`, `defines`); `loop/build` compiles everything
  with the pinned nvcc and repins `sha256` for whatever changed.
- `scripts/gen_stage.py` (and its helpers in `scripts/`): the ops and the
  calls. A fusion is usually a new kernel plus a generator change that
  emits one call where there were several.
- `notes/<run>.md`: your memory (below).
- Your own scratch harnesses under `work/` (kept out of `main` unless they
  are worth keeping).

Never edit `loop/`. Never commit generated manifests or cubins.

## Tools

All take paths relative to the repository root and lease GPUs themselves
through `loop/withgpu`; never run GPU work outside it.

| Command | What it does | GPUs | Time |
|---|---|---|---|
| `loop/build` | compile kernels, repin changed ones | none | ~1 min |
| `loop/gen [OUT]` | regenerate the bench/check manifests into `loop/out/`; prints launches per layer | none | 5 s |
| `loop/count M...` | launches per program and per layer | none | 0 s |
| `loop/p/bench loop/out/p-stage0.json OUT [--ablate]` | P stage 0 (embedding, layers 0-11) over the recorded item shapes; weighted cost | 1 | ~3 min (+2 with `--ablate`) |
| `loop/p/check loop/out/p-l12.json OUT` | `kern test` of 12 P layers against main: every span, logits, noise floor | 1 | ~1 min |
| `loop/d/bench loop/out/d-l16-tp8.json OUT` | D step, 16 layers, real TP8 over two hosts, 64k/128k contexts, 8-48 rows; weighted cost | 8 | ~2 min |
| `loop/d/check loop/out/d-l4-tp8.json OUT` | 4 TP8 layers teacher-forced over real text against the one-GPU oracle, judged against main's band | 8 | ~1.5 min |

Main's numbers (measured on this pool; do not re-measure; `$K3_LOOP_DATA/base/` is current):

| | main |
|---|---|
| P stage 0, weighted (`loop/score`) | 132.00 ms/item at the loop's start (3ad4ec9); 8192 rows: 148.4 ms over an empty prefix, 350.6 ms over 196k. MLA attention (`mla_fmha`) is ~40% of it |
| P launches | 16.7/layer after round 1 (27.1 at its start); ~122-124 ms/item |
| D 16 layers, weighted | 3.486 ms/step after round 1 (4.562 at its start) |
| D launches | 207 per step, 12.9/layer after round 1 (395 at its start) |
| D check | top-1 0.9176 · relRMS median 0.0177 max 0.0829 · 1 confident flip, vs the oracle; bit-identical run to run |

`$K3_LOOP_DATA/base/` holds main's manifests and reports (`p-bench.json`,
`d-bench/`), for `loop/score` or per-op shares (`p-bench.json` has
`cost.by_op`; run `loop/p/bench ... --ablate` for what each op would save
if it were free).

## Numerics

Not bit-exactness, but every difference explained. A change passes its
check (`loop/p/check` PASS; `loop/d/check` PASS) and its commit message
states where its numerical difference from main comes from: summation order
of a reduction, an fp32 intermediate now rounded to bf16 (or the reverse), a
different exp/rsqrt, Lamport's -0.0 handling, ... and the measured size
(the check's KL / relRMS / flips vs main). "Bit-identical" is a fine
explanation when the check says so. A FAIL means the change is wrong, never
the tolerance. A near-tie that flips is noise; a confident token that flips
is a bug.

## Commits and wins

- Commit each change that passes its check and does not slow its target
  down (`git commit -s`), with launches per layer and cost before → after
  in the message, and the numerics line. One idea per commit.
- Subject: `k3: <what is true after the commit, lowercase, no period>`.
- A change goes to `main` when its check passes and its target's weighted
  cost does not regress beyond noise (P ±0.2%, D ±0.5%); fewer launches at
  equal time is a win. The orchestrator merges, re-measures on main and
  tells you in `ROUND.md` when `main` moved: rebase onto it then.
- This is a public repository. No machine names, user names, absolute
  paths or internal hostnames in anything you commit (use `$VARS` or
  `<placeholders>` in docs).

## References (read-only)

- kern runtime and manifest format: `$K3_KERN` (`docs/manifest.md`,
  `docs/runtime.md`, `crates/kern-manifest/src/types.rs`). A launch can carry
  `when: {var, min, max}` to route one op to different kernels by row
  count.
- `$K3_REF/insights.md`: fusion ideas the orchestrator collected from vLLM,
  SGLang, TRT-LLM and FlashInfer, with source paths.
- Sources under `$K3_REF/`: `sglang`, `vllm`, `flashinfer`, `trtllm`
  (TensorRT-LLM kernels and torch modules), `DeepGEMM`, `nvkda` (NVlabs
  KDA², fast KDA prefill: kda-cake-cute / kda-tirx / kda-cake-ptx).
- Skills: `KernelWiki` (Blackwell kernel techniques and PRs) and
  `ncu-report-skill` (Nsight Compute profiling; `ncu` is installed). The
  GPU is GB300 (sm_103a, 152 SMs, ~8 TB/s HBM), not B200: adjust.
- `decode/sglang-tp8-dcp8-ep8/`: SGLang's captured D graphs, launch by
  launch: what SGLang runs per layer at the same layout.
- `harness/`: kernel-level harnesses against references.

## Boundaries

No network (no curl, pip, git fetch). Read only your worktree and the
paths above. Never search home directories or `find /`. Never touch
processes or containers you did not start.

## Time and sessions

`ROUND.md` ends with your deadline; the session is killed then, so commit
each win as soon as it passes. Keep working until the deadline: a fusion is
an edit → build → gen → check → bench cycle of 5-10 minutes. If your target
is closed (fused as far as it sensibly goes, at its roofline), write why in
`notes/<run>.md`, then take the next most expensive non-GEMM op in your half
(P or D) that no other run is on, and keep going.

This runs as a loop of fresh sessions. Your only memory is the repository:

- **`notes/<run>.md`**: read it first, update it before you stop. The current
  launches/layer and cost, what dominates, every idea tried with its
  measured result and why it was kept or reverted.
- **git**: commit each passing improvement on its own. Leave the tree
  clean.
