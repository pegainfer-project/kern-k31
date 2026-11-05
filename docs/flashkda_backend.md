# KDA chunked-prefill backends for K3.1 on GB300

This study covers what P's `span_kda` should run. Every claim comes from reading the backends' source and from measurements on one GB300 (sm_103a, 152 SMs). All timings are graph-replayed and given in µs.

## Shape and terms

**The K3.1 shape.**
- H = 96 heads, d_k = d_v = 128.
- q, k, v and the gate g are bf16. β arrives as logits. The gate is a "safe gate" with lower bound −5: `decay = exp(LB · sigmoid(exp(A_log) · (g + dt_bias)))`.
- q and k are l2-normalized inside the kernel.
- Variable-length sequences are packed and described by `cu_seqlens`.
- The recurrent state is bf16 `[H, V, K]`. It is read from and written back to a row of the state pool. kern calls that row the KDA **line**.

**Terms used below.**
- **Chunk (BT).** The sequence is cut into chunks of BT tokens. Work inside a chunk is a few small dense matrix products, and they are independent across chunks.
- **Chain step.** Carrying the state from one chunk to the next is strictly sequential. Each (sequence, head) pair is one **chain** of `T/BT` steps. In most backends one chain lives on one CTA; FLA Triton splits d_v four ways, and ptx / cute-dsl-persistent split long chains between CTAs.
- **Intra-chunk solve.** The delta rule needs `(I + L)⁻¹`, where L is the strictly lower-triangular, β-scaled matrix of `k_decay · k_invᵀ` inside the chunk. This is the "UT/WY form": once the inverse is known, the chunk's corrections become matrix products.
- **Prepare / chain split.** Some backends compute everything that does not depend on the state (gate prefix sums, l2norm, decayed q/k, the inverse) in a separate *prepare* kernel and write it to global memory. A second *chain* kernel then reads it back. **Fused** backends do both in one kernel, with prepare warps running a few chunks ahead of the chain.
- **mma.sync.** The legacy warp-level tensor-core instruction (`HMMA m16n8k16`).
- **UMMA.** Blackwell's tcgen05 MMA. One thread issues it, it reads operands from shared memory or **TMEM** (tensor memory, an on-SM accumulator store) and accumulates into TMEM. Its minimum useful M is 64 or 128, and N can be as small as 8 or 16.

## Conclusion

1. **cake is the fastest bf16-state backend for P's `span_kda`.**
   - Speed: it takes **0.46× FlashKDA's time** at T = 8192 (386 vs 837 µs) and 0.48× at T = 512. It is 14–17% faster than cuDNN and FlashInfer's cute-dsl.
   - State pool: it reads and writes a bf16 pool line in place at kern's line stride. That path is bit-identical to the plain path and runs at the same speed.
   - Accuracy: its output is the closest of the bf16-state backends to an fp32 token-by-token reference.
   - Shipping: it ships as **generated CUDA C source** plus a prebuilt NVRTC cubin, with an `extern "C"` entry point. It is the easiest of the fast backends to vendor and modify.
   - **But it is not graph-safe over changing lengths, so kern runs cuDNN FROST for now.** Its packed path publishes per-sequence tensormaps that the host builds from `cu_seqlens` (`PublishTensorMaps` after a pageable host-to-device copy). A graph captured at 8 × 256 and replayed at another length mix left the state wrong (relative error 0.76; `cake_probe.py`). kern replays one graph over every call with no host step. Adopting cake means first moving that descriptor publication onto the device, as FROST's prologue kernel does.
2. **Every fused backend is limited by the serial chain, not by FLOPs or bandwidth.** The prepare/chain and Triton backends also pay for an HBM round trip.
   - At T = 8192 the best kernel reaches about 14% of the bf16 peak and 41% of the HBM floor (§Roofline).
   - With one sequence, only 96 of the 152 SMs have a chain to run.
   - Speed is therefore decided by two numbers: how many chain steps there are (BT = 32 halves them relative to BT = 16), and how short each step is.
3. **cake beats cuDNN and cute-dsl in three ways** (§Why cake beats cuDNN and cute-dsl):
   - **BT = 32 instead of 16.** This is made safe by re-centering the decay exponents on the chunk midpoint.
   - **The whole intra-chunk solve runs in 20 prepare warps, five chunks ahead.**
   - **The per-chunk state decay runs on the tensor core.** The intra-chunk output is also folded into the state-update MMA. Together these cut TMEM traffic 3.4×.
4. **Two fp32-state backends are faster still.**
   - FlashInfer `ptx` is 9% faster than cake at T = 8192 and 15% faster on 8 × 2048. `cute-dsl-persistent` is 5% and 15% faster. Both are BT = 32 designs; the packed batches gain from splitting chains over all 152 SMs.
   - They only accept an fp32 state, have no pool support, and plan every call on the host. `ptx` also ships without source.
   - They are a later step, not the first one.
5. **Neither serving engine runs the fastest backend by default** (§What vLLM and SGLang run by default).
   - **vLLM's default (`auto`) resolves to FlashKDA on SM103.** That is 837 µs at T = 8192, the same as kern today.
   - **vLLM's `flashinfer` option** calls FlashInfer's `auto`, which is cute-dsl here (459 µs).
   - **SGLang's default is FLA's Triton `chunk_kda`.** It takes 1942 µs at T = 8192, 5.0× cake, and SGLang never switches away from it automatically.
   - **SGLang's `nvidia_kda` option** only engages for packed multi-sequence batches, padded to 2k/4k/8k/16k buckets. 8 × 256 and 8 × 2048 both cost about 1265 µs that way; a single sequence stays on Triton.
6. **Moving to cake changes what `span_kda` does in three ways.**
   - **Output gate and conv-window advance.** `k3_kda_rec` runs these on the 56 SMs the heads leave idle. With cake they need their own kernel, or an edit to cake's epilogue warps.
   - **State precision.** The carried state is rounded to bf16 at every 32-token chunk boundary. kern's `k3_kda_rec` keeps fp32 in registers.
   - **What we gain:** about 680 MB of workspace at T = 8192, one launch instead of two, and no 8192-row limit.

### Speed (graph-replayed, µs per call, nonzero bf16 state in and final state out)

| Backend | Chunk | Kernels | T = 512 | T = 1536 | T = 8192 | 8 × 256 | 8 × 2048 |
|---|---|---|---|---|---|---|---|
| FLA Triton `chunk_kda`, vLLM's K3 copy (`kda_prefill_backend=triton`) | 64 | 9 | 168.2 | 439.0 | 2169.8 | 585.8 | 4143.7 |
| FLA Triton `chunk_kda`, SGLang's copy (SGLang default), pool in place | 64 | 9 | 146.3 | 394.6 | 1942.2 | 530.7 | 3758.4 |
| SGLang `nvidia_kda` core, equal-length group padded to its bucket | 64 | 3 + copy | 88.3 | 160.0 | 675.0 | 1265.1 (padded to 8 × 2048) | 1267.3 |
| FlashKDA (vLLM op; kern's `span_kda` runs at the same speed) | 16 | 2 + copy | 66.4 | 169.5 | 837.2 | 172.5 | 1156.6 |
| FlashInfer small-bh | 16 | 2 | 43.2 | 107.4 | 530.6 | 138.4 | 903.1 |
| FlashInfer cute-dsl (= `auto`) | 16 | 1 + metadata | 35.3 | 93.2 | 459.4 | 121.0 | 710.9 |
| cuDNN FROST, via FlashInfer | 16 | 1 + prologue + copy | 39.5 | 96.0 | 450.0 | 123.0 | 697.0 |
| cuDNN FROST, pool in place, called directly | 16 | 1 + prologue | 36.7 | 94.1 | 450.3 | — | 693.5 |
| **FlashInfer cake** | **32** | **1** | **31.8** | **80.5** | **385.6** | **121.5** | **585.7** |
| **cake, pool in place at a kern line stride** | 32 | 1 | **31.8** | **80.7** | **386.8** | — | **586.5** |
| FlashInfer cute-dsl-persistent (fp32 state) | 32 | 1 | 31.8 | 79.1 | 368.1 | — | 496.3 |
| FlashInfer ptx (fp32 state) | 32 | 1 (+ host plan) | 30.9 | 76.4 | 349.9 | — | 497.5 |
| FlashInfer tirx | 64 | 1 | does not run: no CUDA-enabled TVM | | | | |

Notes on the table:
- 8 × n means 8 sequences of n tokens packed into one call.
- Sources: `bench_graph_all.py`, `bench_pool.py`, `bench_fp32_backends.py`, `bench_defaults.py`.
- The `nvidia_kda` row times only the core call that SGLang's wrapper makes per group. In serving, a single sequence never reaches it (it stays on Triton), so its T = 512 / 1536 / 8192 cells show what the kernel costs, not what SGLang runs. The wrapper's l2norm, staging copies and state transposes are not timed.
- The ptx and cute-dsl-persistent rows were run with an fp32 state, because that is their contract. Converting a bf16 line to fp32 and back would add a few µs per call; that cost was not measured.

### Agreement with an fp32 token-by-token recurrence

The reference is a token-by-token fp32 recurrence with no chunking (`agree_ref.py`). The table gives the relative L2 error of the output and of the final state.

| | FLA Triton (vLLM) | FLA Triton (SGLang) | SGLang nvidia_kda | FlashKDA | small-bh | cute-dsl | cuDNN | cake | cute-dsl-persistent | ptx |
|---|---|---|---|---|---|---|---|---|---|---|
| Output, T = 8192 | 4.12e-3 | 4.12e-3 | 4.63e-3 | 5.80e-3 | 4.10e-3 | 4.38e-3 | 4.38e-3 | **4.04e-3** | 4.37e-3 | 4.03e-3 |
| Final state, T = 8192 | 2.89e-3 | 3.34e-3 | 2.80e-3 | 4.85e-3 | 3.21e-3 | 3.61e-3 | 3.61e-3 | 3.69e-3 | 4.00e-3 | 3.30e-3 |

- T = 1536 gives the same picture.
- FlashKDA with an "f32 state" is bit-identical to FlashKDA with a bf16 state, because its recurrence is bf16 either way (`agree_all.py`). It is therefore not a valid reference, and earlier agreement numbers measured against it were relative to a bf16 recurrence.
- kern's own `k3_kda_rec` keeps an fp32 state in registers and was not measured here.

### Contract and shipping

| | FlashKDA (+ kern `k3_kda_rec`) | small-bh | cute-dsl | cuDNN FROST | cake | cute-dsl-persistent | ptx | tirx |
|---|---|---|---|---|---|---|---|---|
| bf16 state in and out | yes | yes | yes | yes | yes | no, fp32 only | no, fp32 only | no, fp32 only |
| Carried-state precision | bf16 smem, rounded every 16 tokens; `k3_kda_rec` keeps fp32 in registers | fp32 registers, bf16 copy per chunk | fp32 TMEM | fp32 TMEM | bf16 at every 32-token boundary (decay MMA reads the bf16 copy) | fp32 TMEM | fp32 TMEM (inferred) | fp32 TMEM, scaled frame |
| State pool, in place | no (`k3_kda_rec` adds `line_index`) | no | `ssm_state_indices` | `state_indices`, free row stride | `ssm_state_indices`, free row stride, measured bit-exact | no | no | no |
| Checkpoints | one per sequence in the vLLM binding (inferred) | no | yes, every multiple of 32 | yes, at kernel level, relative to the call | yes, relative to the call; switches module | no | no | no |
| int64 `cu_seqlens` under graph capture | yes | yes | required | yes (compile key) | required | yes | yes | yes |
| Length limit | H·T < 2³¹; `k3_kda_rec` ≤ 8192 rows | none found | H·T·128 < 2³¹ | none found | none found | B·T·H·128 < 2³¹ | ≤ 16384 per sequence | total < 2²¹ |
| Ships as | AOT cubin (already in kern) | CuTe DSL JIT | CuTe DSL JIT, one build for all shapes | CuTe DSL JIT; AOT dump proven | generated CUDA C + NVRTC cubin, prebuilt | CuTe DSL JIT | static PTX, ptxas at first use | TVM TIRx JIT |
| Host work per call | none | caches max length | metadata kernel | none | per-sequence tensormaps built on the host from `cu_seqlens` and copied in (captured: a replay keeps the capture's) | `cu_seqlens.tolist()` sync + plan | plan + `abs().max()` sync | plan |

## Roofline

**Work per token.**
- Useful work is 6·d² FLOP per token per head: the products S·k, S·q and the rank-1 update. Across 96 heads that is **9.44 MFLOP per token**.
- Chunking adds about 10·BT·d per head for the intra-chunk products. That gives 11.4 MFLOP per token at BT = 16 and 13.4 MFLOP at BT = 32.

**Minimum memory traffic per token.**
- Read q, k, v and g (4 × 96 × 128 × 2 B) plus β.
- Write o (96 × 128 × 2 B).
- That totals **123 KB per token**.
- The state is 3 MB per sequence, read and written once, which is negligible.

**Arithmetic intensity.** Useful intensity is 77 FLOP/B. GB300's balance point is about 1.6 PF / 7 TB/s ≈ 230 FLOP/B. The effective bf16 peak of about 1.6 PF is the value sustained by cuBLAS at about 1.39 GHz under the power cap. So the roofline floor of chunked KDA is the memory floor.

**Floors at T = 8192.**

| Floor | Value |
|---|---|
| HBM floor (1.008 GB at 7 TB/s) | 144 µs |
| Compute floor (77.3 GFLOP at 1.6 PF) | 48 µs |

**Where each backend sits at T = 8192, one sequence.**

| Backend | µs | Useful TF/s | % of bf16 peak | % of HBM floor | Serial steps × µs per step | Tensor pipe busy (busiest SM) |
|---|---|---|---|---|---|---|
| FLA Triton (SGLang default) | 1942 | 40 | 2.5 | 7 | 128 × 2.8 (state kernel only, 357 µs on 4 × 96 CTAs) | — |
| SGLang nvidia_kda core | 675 | 115 | 7.2 | 21 | 128 × 2.2 (chain kernel only, 285 µs) | — |
| FlashKDA | 837 | 92 | 5.8 | 17 | 512 × 1.16 (chain kernel only) | 37% (mma.sync) |
| small-bh | 531 | 146 | 9.1 | 27 | 512 × 0.64 (chain kernel only) | 67% (mma.sync) |
| cute-dsl | 459 | 168 | 10.5 | 31 | 512 × 0.90 | 17% |
| cuDNN | 450 | 172 | 10.7 | 32 | 512 × 0.88 | 17% |
| cake | 386 | 200 | 12.5 | 37 | 256 × 1.51 | 25% |
| cute-dsl-persistent | 368 | 210 | 13.1 | 39 | 256 × 1.44 | 21% |
| ptx | 350 | 221 | 13.8 | 41 | 256 × 1.37 | 24% |

**Why nothing comes close to either floor.**
- **Idle SMs.** One sequence has only 96 chains, so 56 SMs sit idle. ncu reports about 62% of SMs active for every backend.
- **Short, dependent steps.** On the active SMs, each step is a chain of dependent MMAs, TMEM loads and stores, and mbarrier handoffs. The tensor pipe is busy at most 25% of the time.
- **Two levers to go further.**
  - Shorten the step. To hit the HBM floor with 96 chains, a BT = 32 step would have to take 0.56 µs instead of about 1.4 µs.
  - Run one sequence on more than 96 CTAs. That requires splitting chains and handing the state between CTAs (ptx, cute-dsl-persistent and cuDNN's `piece_chain` can do this), or using a d_v split. Today these only pay off when a pack has more chains than SMs.
- **Packs.** At 8 × 2048 (768 chains, 5.05 waves), one CTA per chain pays for 6 full waves. That is why ptx and cute-dsl-persistent, which split chains over exactly 152 CTAs, win there by 15%.

ncu counters for the main kernel at T = 8192 (`ncu_all.sh`, `ncu_tc.sh`, `ncu_fp32.sh` and the `ncu_*table.py` condensers):

| Kernel | Grid × block | Dynamic smem | Regs | mma.sync | UMMA | TMEM ld/st instructions | DRAM read / write |
|---|---|---|---|---|---|---|---|
| FlashKDA prepare | 513×96 × 128 | 21.5 KB | 55 | 1.77e6 | 0 | 0 | 606 / 627 MB |
| FlashKDA chain | 96 × 256 | 98 KB | 128 | 2.05e7 | 0 | 0 | 886 / 188 MB |
| small-bh K1 (prepare, persistent) | 152 × 384 | 204 KB | 168 | 2.16e6 | 0 | 0 | 608 / 629 MB |
| small-bh K2 (chain) | 96 × 384 | 177 KB | 162 | 2.05e7 | 0 | 0 | 884 / 187 MB |
| cute-dsl | 96 × 512 | 214 KB | 128 | 2.06e6 | 9.83e5 | 6.69e6 | 810 / 184 MB |
| cuDNN FROST | 96 × 512 | 208 KB | 128 | 2.06e6 | 9.83e5 | 6.10e6 | 811 / 185 MB |
| cake | 96 × 1024 | 232 KB | 64 | 2.95e6 | 6.88e5 | 1.77e6 | 810 / 183 MB |
| cute-dsl-persistent | 96 × 1024 | 231 KB | 64 | 1.97e5 | 7.37e5 | 3.15e6 | 813 / 187 MB |
| ptx | 96 × 1024 | 232 KB | 64 | 2.56e6 | 4.92e5 | 2.56e6 | 813 / 186 MB |

**What the counters show.**
- **Fused backends move the minimum.** Every fused backend reads 0.81 GB and writes 0.19 GB, which is exactly q/k/v/g in and o out.
- **Prepare/chain backends move 2.3 GB.** FlashKDA and small-bh write an 81 KiB-per-token workspace, about 680 MB, and read it back.
- **cuDNN FROST and FlashInfer cute-dsl are the same kernel design.** Their instruction counts are identical. The source confirms it: same warp roles, TMEM map and MMA list (§cute-dsl).
- **small-bh is a CuTe DSL re-implementation of FlashKDA's two kernels.** Its chain issues exactly FlashKDA's 2.045e7 HMMA and has the same DRAM traffic.

## Why cake beats cuDNN and cute-dsl

FlashInfer paths below are relative to `flashinfer/` in site-packages. The cake kernel that runs for one sequence is `data/csrc/kda/flashkda_generated_bf16_fused_m128_e1fe3b56a3.cu` (**cake.cu** below). cuDNN FROST's kernel is `cudnn/linear_attention/frost/kernel/kda_prefill_f16.py` (**frost** below).

1. **It takes half as many chain steps.**
   - cake walks 32-token chunks: `num_chunks = (len + 31) / 32` (cake.cu:849). FROST is fixed at `B_T = 16` (`kda_prefill_config.py:33`).
   - At T = 8192 that is 256 steps against 512. One cake step takes 1.51 µs per 32 tokens; FROST takes 2 × 0.88 = 1.76 µs for the same 32 tokens.
   - **Why BT = 32 is safe.** With LB = −5, the log2 decay of a 32-token chunk spans [−231, 0]. If the factors are taken relative to the chunk start, as FROST does with `K_inv = k · 2^(−g_t)`, they leave the fp32/bf16 normal range below 2⁻¹²⁶. That is why FROST and FlashKDA stop at 16.
   - cake re-centers on the chunk midpoint instead: `restore = exp2(total − 16·LB·log2e)` (cake.cu:1893-1898; common term at :2000). Every exponent then stays within ±115.
   - The same trick bounds LB. Below about −5.46 it would underflow, and FlashInfer's SM120 backend caps LB at −5 for this reason. K3.1's −5 is safe.
2. **The whole intra-chunk solve is off the chain and runs deep.**
   - cake spends 20 of its 32 warps (12-31) on prepare, as five instances of four warps. Instance i prepares chunk `5·iter + i` (cake.cu:1680-1708).
   - Prepare covers the gate prefix, l2norm, β, the Gram `Kd·K_invᵀ`, `Mqk`, and the 32×32 inverse. The inverse is built from two 16×16 diagonal blocks inverted with fp16 mma.sync, then a 2×2 block assembly (cake.cu:2296-2480, 3383-3455).
   - FROST does the Gram and the inverse in one warp (warp 12, frost:547-583; `blockwise_inverse.py:153-225`) with a ring only two chunks deep.
3. **The serial part does less work per token and has fewer handoffs.** cake's single MMA warp (warp 9, cake.cu:1461-1660) issues 28 UMMA per 32 tokens. All of them have M = 128 (the state's value rows in TMEM lanes) with the A operand in TMEM:

   | Product | UMMA | Shape (N) |
   |---|---|---|
   | S·Kd | 8 | 32 |
   | S·Qd | 8 | 32 |
   | Decay: S · diag(2^g_total), one 16×16 diagonal block of B in smem each | 8 | 16 (cake.cu:1590-1604) |
   | u2 = resid · Inv | 2 | 32 |
   | [S \| o] += u2 · [Kr \| Mqk] | 2 | 160 |

   - The last product works because the `Kr` and `Mqk` tiles are adjacent in smem. One MMA updates the state and adds the intra-chunk output.
   - FROST issues 40 UMMA per 32 tokens, all N = 16 except the update (frost:742-812). It then scales the 128×128 fp32 state on CUDA cores every chunk: `tcgen05.ld`, multiply by 2^g, `tcgen05.st` (frost:1366-1410).
   - The counters show the result: cake issues 1.77e6 TMEM instructions to FROST's 6.10e6.

**The price of this design.**
- The decay MMA reads the bf16 copy of the state and overwrites the fp32 accumulator. So the state carried between chunks is bf16 times bf16 decay factors, and only the update within a chunk is fp32.
- In the measurements this costs nothing visible. cake's output error against the fp32 reference (4.04e-3) is lower than FROST's (4.38e-3), probably because it rounds once per 32 tokens instead of once per 16 for the operands.

## What vLLM and SGLang run by default

| Engine, setting | What runs for K3 KDA prefill | T = 8192 | 8 × 2048 |
|---|---|---|---|
| vLLM, `additional_config.kda_prefill_backend` = `auto` (the default) | FlashKDA on SM90/SM10x/SM12x when its contract holds, else Triton (`vllm/models/kimi_k3/nvidia/kda.py:489-528`, default read at :823) | 837 | 1157 |
| vLLM, `flashinfer` | `recurrent_kda(backend="auto")` (kda.py:~484), which is cute-dsl at H = 96 | 459 | 711 |
| vLLM, `triton` | FLA `chunk_kda_with_fused_gate` (`vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py:681`) | 2170 | 4144 |
| SGLang, `--linear-attn-backend` = `triton` (the default; no automatic switch for KDA prefill) | FLA `chunk_kda` (`sglang/kernels/ops/attention/fla/kda.py:1274`) | 1942 | 3758 |
| SGLang, `flashinfer` | FlashInfer `recurrent_kda` with `backend="cute-dsl"` hard-coded (`kda_flashinfer_prefill.py:149`) | 459 | 711 |
| SGLang, `nvidia_kda` | packed multi-sequence batches only, as groups of ≤ 8 padded to a 2k/4k/8k/16k bucket; a single sequence stays on Triton (`kda_nvidia.py:1-30, 266-273`) | 1942 (Triton) | 1267 |
| SGLang, `ptx_kda` / `cutedsl` / `helion` | other vendored kernels (a hand-CUDA chunk-64 tcgen05 kernel, a CuTe DSL Blackwell kernel, Helion); not measured here | — | — |

SGLang forces K3 decode to Triton (`arg_groups/kimi_k3_hook.py`), whatever the prefill backend.

**FLA Triton `chunk_kda`** (both engines carry a copy; `chunk.py:490-556` in vLLM's).
- BT = 64, written as nine launches: l2norm of q, l2norm of k, the gate cumsum, a torch elementwise, the intra-chunk sub-chunk products, the inverse solve, `recompute_w_u` (W, U and the gated k), `chunk_gated_delta_rule_fwd_h` (the state pass), and the output kernel.
- **Only the state pass is serial.** It writes the state of every chunk to global memory (`h`, `[NT, H, K, V]`) and splits d_v four ways, so it runs on 4 × 96 CTAs. It alone takes 357 µs at T = 8192, about what all of cake takes.
- The other eight kernels round-trip q, k, g, A, W, U and h through HBM and take the remaining 1.6 ms.
- It is the slowest path here, at 5.0× cake.
- Its state is `[V, K]`, matching cake and cuDNN, and its accuracy is good (4.1e-3).

**SGLang `nvidia_kda`** (`sglang/kernels/ops/attention/linear/kda_nvidia_prefill/`, vendored from NVIDIA's KDA_prefill package, Apache-2.0).
- BT = 64, three CuTe DSL kernels, with the state as fp32 `[N, H, K, V]`.
- **First kernel: a fused persistent K1+K2+K3.** It runs on 148 CTAs (`NUM_SMS = 148` is baked in, so 4 of GB300's 152 SMs stay idle) with 1024 threads. It computes the gate, cumsum and scaling plus the intra-chunk `Aqk` and `Akk`, and writes k_scaled, kg and q_scaled (bf16), `A_qk` and `A_kk` (bf16 `[T, H, 64]`) and the chunk-last gate (fp32). That is about 1 KB per head-token, or 97 KiB per token.
- **Second kernel:** the `Akk` inverse.
- **Third kernel: a persistent K4 chain.** 384 threads, six UMMA per 64-token chunk, and the state in TMEM.
- **Timings:** at T = 8192 the kernels take 345 + 44 + 285 µs. The chain alone is 128 steps × 2.2 µs on 96 CTAs.
- **Where it sits:** the BT = 64 chain is the cheapest per token measured (2.2 µs per 64 tokens, against cake's 3.0), but the prepare round trip through HBM costs more than the chain saves.
- **Serving cost:** padding every group to its bucket makes 8 × 256 cost as much as 8 × 2048.

## Backends in detail

### FlashKDA (what kern runs today)

**Where the source is.**
- Vendored upstream at `source/flash-kda/` (commit 7afb9f4).
- vLLM's `_flashkda_C` is the same design. Its fat binary has only sm_90a, sm_100 and sm_120 code, so GB300 runs the sm_100 code: HMMA plus TMA, with no tcgen05 or TMEM.
- kern's P runs FlashKDA's prepare kernel followed by its own `source/k3_kda_rec.cu`.

**Algorithm.** BT = 16, fixed (`fwd_launch.cu:31`).

*Prepare kernel* (`fwd_kernel1.cuh`):
- l2norm in f32, rounded to bf16 (:265-303).
- Gate cumulative sum in log2 within the chunk (:306-330).
- Decayed operands in bf16: `qd = q·2^G·scale`, `kd = k·2^G`, `k_inv = k·2^−G`, `kr = k·2^(Gtot−G)` (:449-471).
- `L = β·strict_tril(kd·k_invᵀ)` and `Mqk = tril(qd·k_invᵀ)` (:481-507).
- The inverse (`utils.cuh:194-369`): the two 8×8 diagonal blocks by f32 forward substitution with shuffles, and the off-diagonal block as `−(B⁻¹C)A⁻¹` with two HMMAs. The result is stored bf16.

*Chain kernel*, per tile (`fwd_kernel2.cuh`):
- `u = kd·Sᵀ` and `o = qd·Sᵀ` (533-568).
- `U = INV·β(v − u)` (589-623).
- `out = o + Mqk·U` (625-650).
- `Sᵀ = Sᵀ·diag(2^Gtot) + krᵀ·U` (659-731).

**Kernels and traffic.**
- Prepare runs on grid `[ceil(T/16) + N, H]` (513 × 96 at T = 8192), with 128 threads in vLLM's build and 256 in the vendored one.
- It writes 13,824 B per (head, tile): kd, qd and kr as bf16 16×128 tiles; g_total f32; INV and Mqk bf16 16×16 (`utils.cuh:70-76`). That is **81 KiB per token**, about 681 MB at T = 8192.
- It runs at about 5.3 TB/s and issue rate is 74%, so it is bound by memory bandwidth and issue together.
- The chain kernel reads that workspace back together with v. The two kernels do not overlap.

**Parallelism.**
- One chain CTA per (sequence, head). It has four MMA warps (32 value columns each), one TMA warp and one store warp, with 3 load stages and 2 store stages (`fwd_kernel2.cuh:186-213`).
- The state lives as a bf16 tile in shared memory and is rounded every 16 tokens (:718-719). `StateFP32` only changes the input/output format.
- One step is 104 dependent HMMAs per warp plus a 32 KB shared-memory read-modify-write of the state, about 1.16 µs.

**What `k3_kda_rec` changes.**
- It reads and writes the state through `line_index`, which is the state pool.
- It keeps the state in fp32 registers and rounds it only as an MMA operand; the line is stored bf16 with `KDA_STATE_BF16=1`.
- Warp 9 advances the conv windows.
- 56 extra gate CTAs sit on the SMs the heads leave idle. They apply RMS·γ·sigmoid(gate) as tiles are published through release/acquire progress counters.
- Its grid is `nseq·96 + 56`, block 512, with 182 KB of shared memory.
- It is still mma.sync over 16-token tiles, and it is capped at 8192 rows because the βs are held in shared memory.

**Why it is slow on GB300.**
- The chain is 71% of the time: 512 steps × 1.16 µs on 96 SMs.
- It uses legacy mma.sync, which is 37% busy on the active SMs.
- The prepare round trip moves 2.3 GB.

### FlashInfer cake

**What it is.**
- A sealed portfolio of code-generated CUDA kernels. There are 69 generated `.cu` files in `data/csrc/kda/`, each with a metadata entry giving its architecture, threads, smem, ABI family and selector key.
- They are compiled with NVRTC (`jit/flash_kda_nvrtc.py`, `--use_fast_math` on sm103), and the cubins are shipped prebuilt inside TVM-FFI `.so` files in `flashinfer_jit_cache`.
- The generator itself is not shipped.

**How a route is chosen.** Hand-written heuristics in `kda_prefill.py:2177-2407` pick a family from the compute capability, SM count, whether the layout is fixed or varlen, N, H and the lengths. An exact-key lookup then finds the module (`kda_prefill.py:3679-3750`).

| Case | Route | Module | Launch |
|---|---|---|---|
| One sequence | `direct_m128` with tensor-core state decay (kda_prefill.py:3405-3428) | `e1fe3b56a3`; with `ssm_state_indices`, the serving ABI module `f4112648ad` | grid N·H, 1024 threads, 232,064 B smem |
| Uniform packs with 768 % 128 == 0 (e.g. 8 × 2048) | `vtile_m128`, persistent (kda_prefill.py:1427-1447) | `b784466ef6` | 128 CTAs × 6 tasks each, so the pipeline never drains between tasks |

- The vtile module **bakes in** NUM_HEADS 96, LB −5 and the scale, and reads a single `uniform_seq_len` instead of the `cu_seqlens` offsets (vtile .cu:114-118, 796). It is only safe for packs with truly uniform lengths.
- The docstring's two-stage BT = 16 prepare/chain family (with S7/S8/S9 = 7, 8 or 9 chain pipeline stages, `kda_prefill.py:3315-3330`) needs 32 or fewer tasks. It is never reached at H = 96.

**Algorithm, kernel, tensor cores and precision.** See §Why cake beats cuDNN and cute-dsl. In addition:
- **Warp roles.** Warps 0-3 compute (state rows); 4-7 are the epilogue, which drains O from TMEM to bf16 with `stmatrix` and TMA-stores it; 9 issues the MMAs; 12-31 prepare; 8, 10 and 11 are idle in this variant.
- **Registers.** Assigned with `setmaxnreg`: 168 for compute and 48 for the rest.
- **TMEM.** 256 columns: fp32 state, bf16 state operand, u_acc, u2 and out.
- **No 2-CTA mode.** Every MMA is `cta_group::1`.
- **What it writes.** Only o (TMA store) and the final state (bf16 vector stores, cake.cu:1283-1297). The backward tapes are compiled out.

**Contract.**
- d = 128. q, k, v and g are bf16 and contiguous.
- β is bf16 with a free token stride, read in place by TMA. **It must be token-major** (a head stride of 1).
- `A_log` is fp32 `[H]`, `dt_bias` fp32 `[H·128]`.
- l2norm, the in-kernel gate and β as logits are all required (`kda_prefill.py:681-760`).
- `cu_seqlens` is int32 or int64, and must be int64 under graph capture.
- **State pool.** With `ssm_state_indices` (int32 `[N]`), the kernel computes `state_base = slot · state_slot_stride + (head·128 + row)·128`. It uses that base for both the read and the write (`f4112648ad.cu:857-862, 1368`), so the update is in place.
- **Pool shape.** The pool must be `[rows, H, 128, 128]` with the inner three dimensions dense. The row stride may be anything at least H·128² elements and 16 B-aligned (`kda_prefill.py:630-652`).
- **Pool measurement** (`bench_pool.py`). The pool was a line with 64 KiB of other data after the state. The pool path was bit-identical to the plain path, other rows and the line tails were untouched, and the time was unchanged.
- **Checkpoints.** Any checkpoint request switches module. An interval that is not a multiple of 32 runs on a BT = 16 module (`b956ea8571`). A multiple of 32 runs on a BT = 32 module without the tensor-core decay (`b21b675208`). Positions are relative to the sequence start in the call, and the output is packed bf16 `[C, H, 128, 128]`.
- **Not accepted:** `initial_state_indices` / `initial_state_source`; an fp32 state without indices (`kda_prefill.py:776-779`, the "does not support this contract" error seen earlier).

**Shipping and lifting.**
- The entry point is `extern "C" kernel_flashkda_bf16_fused_m128`, with about 55 by-value arguments (cake.cu:591):
  - pointers;
  - six **device-resident** 128 B `CUtensorMap`s for q, k, v, g, β and o, passed as pointers, plus one for checkpoints;
  - `seq_order` int[N];
  - `num_heads`, `scale` and `lower_bound`, all as runtime values in the direct module;
  - `state_indices_addr` and `state_slot_stride`;
  - `beta_token_stride`;
  - backward-tape and checkpoint pointers that can be null.
- Host work per call is only descriptor and metadata preparation when pointers change. The 768 B descriptor table is what eager calls pay for.
- What kern needs to run it:
  - (a) the cubin;
  - (b) a static buffer that holds the six encoded tensormaps, written once at load time over the static span buffers, with the token extent set to capacity;
  - (c) `seq_order`, which can be the identity or longest-first;
  - (d) grid `seqs·96`, block 1024, and an opt-in to 232,064 B of shared memory.
- **No per-shape schedule selection is needed if we lift only the direct module.** At 8 × 2048 the direct module should land close to vtile: 6 waves × 64 steps × 1.5 µs ≈ 580 µs (inferred).
- **License.** The generated sources carry no header; they ship in FlashInfer, which is Apache-2.0.

### FlashInfer cute-dsl (`backend="auto"` picks it here)

**Source.** `kda_prefill_cute.py` plus `kda_kernels/kda_chunked_bt16.py` (**bt16** below).

**Which route runs.**
- `auto` would pick small-bh only if `2·nseq·H ≤ SMs`, which fails at H = 96.
- Within cute-dsl, the two-kernel split route (decomp, dv2) runs only if `nseq·H·2 ≤ SMs` (bt16:6320-6338).
- So at H = 96 the backend always runs the fused "engine" kernel.

**Algorithm.**
- BT = 16 (bt16:256).
- The gate is computed through tanh in fp32. The 16-row prefix is a scalar fp32 scan (bt16:1648-1657), followed by `exp2(±g)`.
- l2norm is `rsqrt(max(Σx², 1e-24))`.
- β's sigmoid is computed in the TMA warp.
- The inverse uses block-diagonal 8×8 blocks: `(I−D)(I+D²)(I+D⁴)`, which is exact because each block is nilpotent at 8, then one 8→16 combine. It runs on mma.sync with fp16 operands (bt16:877-925).
- Scale is applied when O is drained.

**Kernel.**
- A metadata kernel builds a longest-first `seq_order` on the device. Then one fused kernel runs on grid `(H, N)` with 512 threads, 214 KB of shared memory and **no workspace**.
- The warp roles match FROST's exactly (bt16:726-737), and so do the TMEM map (272 of 512 columns), the 8-stage load ring and 2-stage decay ring, and the MMA list:
  - 0-7: two ping-pong groups for gate, decay operands and the left state half;
  - 8-11: seed, right state half, Y/U staging, O drain;
  - 12: Gram and inverse;
  - 13: tcgen05;
  - 14: TMA and β;
  - 15: QK and the O store.
- Visible differences from FROST: the 8×8-block inverse, and no persistent ticket scheduler.

**Why it is slower than cake.**
- Each 16-token step makes about four MMA commit/wait round trips and three full TMEM round trips. It also rescales the 128×128 fp32 state on CUDA cores (`tcgen05_pack_rescale_state_half_tmem`, bt16:3584).
- At 8 × 2048, 768 CTAs is 6 waves.

**Contract.**
- `ssm_state_indices` is supported; `initial_state_indices` is not.
- Checkpoints: the interval must be a multiple of 32, with optional `checkpoint_state_indices`.
- `cu_seqlens` must be int64 under capture.

**Shipping.**
- `cute.compile(host_unified, "--enable-tvm-ffi")` behind an `lru_cache`. All shapes are symbolic, so **one build serves every T, H and N**.
- The compile key is the dtypes, the flags and LB as a constant.
- `export_to_c` can emit an object file with the cubins embedded (bt16:9535).
- Lifting would mean extracting the two cubins and reproducing six host-encoded by-value tensormaps plus the CuTe tensor structs. The parameter layout must be read from the generated signature, as was done for cuDNN.

### cuDNN FROST

The ABI, AOT cubin export, pool and checkpoints are covered in `cudnn-vendor.md` (go/no-go: GO). This section covers only the kernel internals.

**Shape of the run.**
- BT = 16.
- One persistent prologue launch (2 CTAs) sorts the work items and patches per-sequence tensormaps.
- The main kernel runs on `min(152, N·96)` CTAs with 512 threads and 208 KB of shared memory.
- At H = 96 the engine always picks the "uncut" schedule. A chain split needs `piece_budget ≥ 3` and a d_v split needs a budget of exactly 2, but `piece_budget = min(152 // (N·96), …) = 1` (`piece_chain.py:55, 85-142`).

**Per 16-token chunk and head: 42 mma.sync plus 20 UMMA.** This matches ncu exactly.
- mma.sync work:
  - KK = K_decay·K_invᵀ, 16 HMMA (frost:547-564);
  - the inverse: 4×4 diagonal blocks by the series I − L + L² − L³, then 4→8 and 8→16 corrections, 10 HMMA, with bf16 operands (`blockwise_inverse.py:153-225`);
  - A = tril(Q_decay·K_invᵀ), 16 HMMA (frost:370-388).
- UMMA work, all with M = d_v = 128:
  - K·S and Q·S, 8 each at N = 16;
  - U = Y·T⁻¹, 1;
  - S += Uᵀ·K_restore, 2 at N = 64;
  - O += U·A, 1 (frost:742-812).
- About 124 TMEM loads and stores per chunk on CUDA cores: the bf16 state repack, the fp32 decay, the Y and U staging, and the O drain.

**The serial loop.** update → bf16 repack → KS → `Y = β(V − KS)` in bf16 arithmetic → U → U repack → update. That is about six mbarrier handoffs per chunk, at about 0.88 µs.

**Traffic and the state.**
- Global traffic is the minimum: T⁻¹, A, W and U never leave the SM.
- The state is fp32 in TMEM for the whole call, with a bf16 copy as the MMA operand. The final state is written with 16 B `st.global`.

**Waves.** Throughput scales with the number of tiles: 4 × 2048 on 152 CTAs takes 352 µs against 451 µs for 1 × 8192. 8 × 2048 pays for 6 full waves.

### FlashInfer small-bh

**Source.** `kda_prefill_cute_small_bh.py` plus `kda_kernels/kda_chunked_bt16_small_bh.py` (**sbh** below). The source has no provenance note. From the matching workspace layout and the identical HMMA count and traffic, it is a CuTe DSL version of FlashKDA's prepare plus chain (inferred).

**Prepare kernel K1.**
- Persistent over (64-token tile, H, N), with `max_active_clusters = 152` baked in.
- 384 threads: a TMA warp, a β warp, a gate warpgroup and an mma.sync warpgroup (sbh:476-528, 636-957).
- The inverse is `(I+L)(I+L²)(I+L⁴)(I+L⁸)` with bf16 operands (sbh:1147-1255).
- It writes Qd, Kd and Kr as bf16 `[T + 64N, H, 128]`, Mqk and Mkk as bf16 16×16, and Gk as fp32: 81 KiB per token. It runs at about 75% of DRAM peak.

**Chain kernel K2.**
- Grid `(D/BV, H, N)` with BV = 128 at H = 96 (sbh:3051).
- The state is fp32 in mma.sync accumulator registers and is repacked to bf16 each chunk.
- Each chunk computes `dV = S·K`, `P = Mkk(V − dV)`, `O = Q·S + Mqk·P` and `S = Gk∘S + Krᵀ·P` (sbh:2463-2760).
- At 0.64 µs per step, its chain is the fastest 16-token step measured. The tensor pipe is 67% busy, but on legacy mma.sync.

**Overall.** It loses because K1 does not overlap K2 and the workspace round trip costs about 200 µs. "Small-bh" means few (sequence × head) pairs, where BV = 64 gives two CTAs per head; H = 96 is outside that regime.

**Contract and shipping.** No pool and no checkpoints. Two `cute.compile`s; K1's key includes H and the SM count.

### FlashInfer cute-dsl-persistent

**Source.** `kda_prefill_persistent.py` plus `kda_kernels/persistent/`.

**Design.**
- BT = 32, M = 128.
- 1024 threads with the same role split as cake: compute 0-3, epilogue 4-7, MMA/load 8-11, and five four-warp prepare instances 12-31 (`persistent/kernel.py:13-22`).
- The Gram products run on tcgen05 (hence only 2e5 mma.sync). The solve is 8×8 unit-lower inverses by shuffles, then a block-diagonal combine.
- The state is fp32 in TMEM.

**Scheduling.**
- A host planner packs chains onto `min(N·96, 152)` CTAs longest-first.
- It splits chains with a wrap-around split and an exact fp32 state handoff when that helps (`persistent/forward.py:510-597`). That is how it reaches 152 CTAs and 496 µs at 8 × 2048.

**Contract.**
- fp32 state only.
- No pool, no checkpoints.
- The plan depends on the `cu_seqlens` contents (`tolist()`, a host sync).
- A captured graph needs a workspace warmed on the same stream.

### FlashInfer ptx

**What it is.**
- Four static `.ptx` files in `data/csrc/kda/ptx/`: `fixed_h64`, `fixed_h96`, `varlen_uniform_h64` and `varlen_mixed_h64`.
- They are NVVM output for sm_103a (PTX 9.2), generated from NVlabs/kda revision `ea37eba`. The kernel source is not shipped.
- At first use, ptxas turns them into cubins, and `shim.cc` is built into a TVM-FFI extension that launches them with `cuLaunchKernelEx`.

**Why it "failed" before.**
- The earlier `FileNotFoundError(2, …)` was `tvm_ffi.cpp.load_inline` running `ninja`. The binary is only in the venv's `bin/`, which is not on PATH when the venv's python is called by absolute path.
- With that directory on PATH it runs: `kernel_kda_fwd_stream_m128`, 1024 threads, 232,448 B of shared memory, 64 registers (`bench_fp32_backends.py`).

**Algorithm and kernel, as far as the PTX and the host shim show.**
- BT = 32 (`kda_kernels/ptx/scheduler.py:33`).
- mma.sync for the 32-wide solve, and tcgen05 `kind::f16` with A from TMEM for the state products.
- 20 UMMA per 32 tokens, the fewest measured, and 2.56e6 TMEM instructions.
- It is the fastest at T = 8192 (350 µs, 1.37 µs per step). How it shortens the step cannot be read without the source.

**Scheduling.** `scheduler.py` (1.2k lines) is a host-side planner:
- longest-first bin-packing of chains in 32-token slots;
- searches over inline and tail chain splits with flag handoff;
- several shape-specific special cases (`offsets == [0, 8192]`, a hard-coded 173-slot packing, a GPC swap only on 148 SMs).

**Contract.**
- fp32 state `[N, H, 128, 128]`. No pool, no checkpoints.
- LB must be exactly −5. H must be 64 or 96.
- Each sequence ≤ 16384 tokens.
- |h0| ≤ 4096. Anything larger goes to a per-token Triton fallback. The check is an `abs().max()` host sync whenever pointers change.
- General varlen at H = 96 runs the `*_h64` programs, which take `num_heads` at run time.

**Lifting.**
- The cubins and a 25-argument by-value ABI are easy to lift. That ABI has 18 pointers, 6 of them into a device tensormap table, plus heads, scale, LB and the flags.
- What makes it hard: a planner to port as a pure host function, per-step plan tables, flags to zero, a 2-CTA cluster for the mixed program, a bf16↔fp32 state conversion, and no source if anything needs changing.

### FlashInfer tirx

**What it is.**
- The kernels are written with TVM's TIRx builder (`kda_kernels/tirx/fused.py`, `split.py`) and JIT-compiled through `tvm.compile(..., tir_pipeline="tirx")`.
- This venv has only `apache_tvm_ffi`; there is no `tvm` and no `tirx_kernels`. `kda_prefill_tirx.py:170-177` turns that ImportError into the RuntimeError seen.

**Design.**
- Varlen always takes the fused route: BT = 64, one persistent kernel with 20 warps.
- The state is fp32 in TMEM, kept in a 2⁻⁶⁰-scaled frame with per-half exponent references so that 64-token chunks do not overflow. T is held as a bf16 hi+lo pair.
- A host-built per-chunk item list allows equal-cost cuts with state handoff between CTAs.
- The split route (two kernels on two streams, the chain spinning on flags) is fixed-length only.

**Contract.** fp32 state, no pool, no checkpoints, and LB as a compile-time constant.

**Lifting.** It would need TVM to generate the kernels, and the item-table stride is baked in at compile time. It is not a candidate.

## Recommendation for P's `span_kda`

**Adopt cake's direct serving module (`flashkda_bf16_fused_m128_f4112648ad`).** It reads and writes the KDA line in place through `ssm_state_indices` / `state_slot_stride`. Steps:

1. **Vendor the source.** Copy the generated `.cu` with its binding header and NVRTC flags (sm_103a, fast-math), and add a PROVENANCE entry: FlashInfer 0.7.2.post1, module hash, Apache-2.0. Build the cubin with kern's build, and check that it is byte-equal to the prebuilt one in `flashinfer_jit_cache`, or at least that its SASS is identical.
2. **ABI port.**
   - One launch: grid `seqs·96`, block 1024, 232,064 B of dynamic shared memory.
   - A load-time static buffer holding the six encoded tensormaps for q, k, v, g, β and o over capacity-sized span buffers.
   - `seq_order` as the identity.
   - Null tape and checkpoint pointers.
   - `state_slot_stride = line_bytes / 2`.
   - Capture the Python launch's parameter bytes and diff them against kern's packed bytes, using the kernel-capture lift flow.
3. **Glue.**
   - `span_gather` writes β token-major. This is the same change the cuDNN plan needed.
   - The output gate (RMS·γ·sigmoid) and the conv-window advance that `k3_kda_rec` does today become one small kernel after the KDA kernel. `cudnn-vendor.md` estimates it at about 7 / 20 / 90 µs for T = 512 / 1536 / 8192.
   - Because the cake source is plain CUDA C, the follow-up is to fuse the gate into the epilogue warps (4-7). Each CTA owns one (sequence, head) with all 128 d_v, so the RMS is local to the CTA.
4. **Gates.**
   - `kern test` against the current P manifest.
   - The P e2e KL check (≤ 1e-2).
   - `kern bench` of `span_kda` at 512 / 1536 / 8192 and at the packed N the service actually sees.
   - A run of K3.1-shaped random LB = −5 gates at maximum magnitude, to exercise the midpoint re-centering.

**What we gain.**
- KDA kernel time drops about 2.2×: 837 → 387 µs at T = 8192, and 66 → 32 µs at T = 512. Some of that goes back to the standalone gate until it is fused.
- One launch, no 81 KiB/token workspace (about 680 MB at T = 8192), and no 8192-row limit.
- Better agreement with an fp32 recurrence than vLLM's FlashKDA (4.0e-3 against 5.8e-3).

**What we lose or must watch.**
- **The idle-SM gate overlap** of `k3_kda_rec`, until the gate is fused into cake's epilogue.
- **Precision of the carried state.** The state between 32-token chunks is bf16, where `k3_kda_rec` keeps fp32 in registers. The measurement says this costs nothing visible against fp32; the e2e KL gate decides.
- **The persistent vtile route for packs.** It bakes in H, LB and the scale and ignores `cu_seqlens`, so it is unsafe for mixed lengths. Packs with more than 152 chains will pay for full waves, as every one-CTA-per-chain backend does.
- **In-kernel checkpoints** are relative to the call and switch module. Keep snapshots at grid points by splitting the span, as the scheduler already does.

**What comes after.**
- If packed P batches dominate, the next 15% is in splitting chains over all 152 SMs. ptx and cute-dsl-persistent show this at 8 × 2048. It would mean a host planner emitting a per-step piece table, which fits kern's pool-decides / runtime-executes split, plus an fp32 handoff between CTAs.
- Neither of those two backends can be taken as is: they need an fp32 state and have no pool, and ptx has no source.
- cuDNN FROST remains the fallback with the best-documented AOT path and in-kernel checkpoints. It costs 14–17% more than cake.

## Measurement files

These sit next to this study in the experiment directory, under `results/kda-backends/`:

| Purpose | Scripts | Logs |
|---|---|---|
| Graph timing, all backends | `bench_graph_all.py` | `bench-graph-all.log` |
| Pool in place | `bench_pool.py` | `bench-pool.log` |
| fp32-state backends | `bench_fp32_backends.py` | `bench-fp32-backends.log` |
| vLLM / SGLang defaults (FLA Triton, nvidia_kda) | `bench_defaults.py`, `debug_nvidia.py` | `bench-defaults.log`, `kernel-inventory-defaults.log`, `debug-nvidia.log` |
| Earlier cuDNN measurements | `bench_fair.py` | `bench-fair.log` |
| Kernel inventory | `kernel_inventory.py` | `kernel-inventory.log` |
| ncu counters | `ncu_one.py`, `ncu_one_fp32.py`, `ncu_all.sh`, `ncu_tc.sh`, `ncu_fp32.sh`, `ncu_table.py`, `ncu_tc_table.py` | `ncu-*.log` |
| Agreement | `agree_ref.py`, `agree_all.py` | `agree-ref.log`, `agree-all.log` |

Software versions: FlashInfer 0.7.2.post1, cudnn-frontend 1.32.0, nvidia-cutlass-dsl 4.8.0, and the vLLM nightly build of `_flashkda_C`.
