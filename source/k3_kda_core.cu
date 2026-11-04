// K3 `k3_kda_core` — Kimi-K3 KDA (Kimi Delta Attention) decode step: the
// delta-rule recurrence for one row `b` and one head `h` per block, with the
// small per-head pieces (beta, the f_b projection of the forget path, the
// bf16 l2norm chain, the output rms + gate) fused in.  Contract:
// docs/k3-kernel-abi.md section K3; certified math source: pegainfer's
// TileLang `kda_core_batched` (kernels/tilelang_defs.py).
//
//   extern "C" __global__ void kern_k3_kda_core(
//       const bf16* conv_q, const bf16* conv_k, const bf16* conv_v,  // [B, INNER]
//       const f32*  wsm_partial,   // [B, WSM=256]   col h = b_proj, cols 96..223 = f_a
//       const f32*  gate_partial,  // [B, KDA_FUSED=49152], band 3 only
//       const bf16* w_f_b,         // [INNER, 128]
//       const f32*  dt_bias,       // [INNER]
//       const f32*  a_log,         // [HEADS]
//       const f32*  gamma_o,       // [128]
//       void* kda_base, const int* line_index, long long line_bytes,  // rec at offset 0
//       bf16* out,                 // [B, INNER]
//       int B, const int* span_at, int span);  // rows [*span_at, +span) are a span (K8 + K11 do them): the block returns
//
//   grid  (B, HEADS=96, 1)      (grid.x == B exactly, per the ABI; the `B`
//                               argument itself is therefore never read)
//   block 128            (one thread per output v-dim in the prologue/epilogue)
//   smem  static ~37 KB/block; + 65,536 B dynamic with -DKDA_SMEM_STATE=1
//
// Math (per block, b = blockIdx.x, h = blockIdx.y, base = h*128):
//   q,k,v   = conv_q/k/v[b, base + d]                                   (bf16)
//   qtot    = sum_d f32(bf16(q[d]*q[d]))          -- square lands in bf16, sum f32
//   qr      = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))                       (same for k)
//   qs[d]   = f32(bf16(q[d]*qr)) * 128^-0.5   ;   kn[d] = f32(bf16(k[d]*kr))
//   beta    = sigmoid(f32(bf16(wsm_partial[b, h])))
//   flow[j] = bf16(wsm_partial[b, 96 + j])                              (128 values)
//   ga[d]   = sum_j f32(flow[j]) * f32(w_f_b[base + d, j])              (f32 accum)
//   raw[d]  = f32(bf16(ga[d])) + dt_bias[base + d]
//   dec[d]  = exp(LB * sigmoid(exp(a_log[h]) * raw[d])),  LB = -5
//   m[dv]   = sum_k S[dv,k]*dec[k]*kn[k]  ;  dlt[dv] = (f32(v[dv]) - m[dv]) * beta
//   S'[dv,k]= S[dv,k]*dec[k] + dlt[dv]*kn[k]                            (in place)
//   attn[dv]= bf16(sum_k S'[dv,k]*qs[k])                                (single landing)
//   out[dv] = bf16(f32(attn[dv]) * rsqrt(mean(attn^2) + 1e-5) * gamma_o[dv])
//             * bf16(sigmoid(f32(bf16(gate_partial[b, 3*INNER + base + dv]))))
//
// Landing points: q/k squares -> bf16 per term; the l2 sum -> bf16 before
// +1e-6; rsqrt -> bf16; q*qr and k*kr -> bf16; ga -> bf16 before dt_bias;
// attn -> bf16 once; the rms product -> bf16 once; the gate -> bf16 twice
// (partial and sigmoid); the state stays f32 throughout.
//
// State traffic is the whole point: rec is 96*128*128 f32 = 6.29 MB per row
// per layer, read once and written once.  The kernel therefore
//   * splits k across 8 lanes (kl = tid & 7) and dv across 16 lane groups
//     (dvl = tid >> 3), so one warp instruction covers 4 dv rows x 128 B of
//     fully contiguous, 16-B-aligned float4 -- 100% sector utilisation on
//     both the load and the store;
//   * holds one dv row's 16 k-values per lane in registers between the
//     m-reduction and the write-back, so rec is touched exactly once for
//     reading and once for writing (no second pass, no L2 re-read);
//   * hoists dec/kn/qs for the lane's fixed 16 k-values into registers ahead
//     of the dv loop (they are dv-invariant), so the inner loop issues no
//     shared-memory traffic at all;
//   * uses the identity  attn[dv] = sum_k (S*dec)*qs + dlt[dv] * (kn.qs)
//     so the second reduction rides along with the first and the row does not
//     have to be re-read after dlt is known.
//   * ROWS_PER_ITER dv rows are in flight per lane (16 B * 4 * RPI bytes of
//     outstanding loads per thread) to keep enough memory parallelism;
//   * issues every load that depends only on (b, h) at entry, in one round
//     trip: the head's 32 KB w_f_b tile into shared memory by cp.async (no
//     registers), the row's scalars into registers.  The prologue walked ~7
//     dependent round trips before (7 us of a 9 us call at 8 rows);
//   * -DKDA_SMEM_STATE=1 also bulk-copies the head's 64 KB state slice into
//     dynamic shared memory (16 bulk copies of 4 KB, one a thread) right after the line index, beside the conv
//     window's loads, so the state read overlaps the whole prologue and the
//     delta rule runs from shared memory (65,536 B of dynamic smem, ~100 KB a
//     block: 2 blocks an SM, the build for batches that fit one wave of them).
//     The register build stays for larger batches (-DKDA_MINB=4: 4 blocks an SM);
//   * streams rec both ways (__ldcs/__stcs): nothing re-reads it, so it must
//     not evict the w_f_b tile from L2  (+9% at B=64);
//   * remaps (blockIdx.x, blockIdx.y) to (b, h) head-fastest, so blocks issued
//     close in time walk contiguous rec instead of B lines 6.5 MB apart
//     (+8% at B=64).
// The w_f_b tile (swizzled in shared memory) is summed by thread d over its row
// d as a pairwise tree -- see the comment on the ga block.
//
// -DKDA_CONV=1 folds K2 (k3_conv_silu.cu) into the prologue: the block's
// thread d runs the short conv + SiLU of column base + d of the q, k and v
// streams itself (the same f32 sums and bf16 landings, the window shifted in
// place in the row's line), so conv_q / conv_k / conv_v never touch memory.
// Its signature drops them and reads q | k | v | g from one partial:
//
//   extern "C" __global__ void kern_k3_kda_core(
//       const f32* partial,        // [B, LDS]  q | k | v | g, 4 * INNER columns
//       const f32* cw,             // [3 stream][4 tap][INNER]
//       const f32* wsm_partial, const bf16* w_f_b, const f32* dt_bias, const f32* a_log, const f32* gamma_o,
//       void* kda_base, const int* line_index, long long line_bytes,
//       bf16* out, int B, const int* span_at, int span);
//
// -DCONV_TAP_MAJOR=1 (with KDA_CONV) keeps the conv window in a state of its
// own, laid out the way an engine's per-slot conv pool is: the row's three taps
// oldest first, each tap q | k | v (element (s, t, c) at t * 3 * INNER +
// s * INNER + c), at `conv_base + conv_index[b] * conv_bytes`.  The recurrent
// state stays at `kda_base + line_index[b] * line_bytes`, so the line holds rec
// only.  The signature gains the three after `line_bytes`:
//
//       void* kda_base, const int* line_index, long long line_bytes,
//       void* conv_base, const int* conv_index, long long conv_bytes,
//
// -DLINE_GUARD=1: a row whose line is negative (an engine's padded row, which
// owns no state) writes a zero output and touches no state.
//
//   nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
//        -o target/cubins/k3_kda_core.cubin tools/kernels-src/k3_kda_core.cu
#include <cuda_bf16.h>

// HEADS is the heads this rank holds: 96 whole, or a tray-group shard
// (`-DHEADS=24` for TP4, docs/multi-gpu.md "最终形态"); the per-head weights
// and the state line are then that rank's slice, the wsm partial keeps its
// whole-model layout (beta at column h, f_a at WSM_FA).
#ifndef HEADS
#define HEADS 96
#endif
#define KD 128
#define INNER (HEADS * KD)
#define KDA_FUSED (4 * INNER)
// Row strides of the gate partial and the wsm partial, and where f_a starts in
// the latter: the whole-model layout by default; one fused GEMM's row (qkvg,
// then beta's HEADS columns and f_a's 128, padded to 256) with LDS = WSM_LDS =
// 4 * INNER + 256 and WSM_FA = HEADS.
#ifndef LDS
#define LDS KDA_FUSED
#endif
#ifndef WSM_LDS
#define WSM_LDS 256
#endif
#ifndef WSM_FA
#define WSM_FA 96
#endif
#define LB (-5.0f)
#define RMS_EPS 1e-5f
#define L2_EPS 1e-6f
#define QSCALE 0.088388347648318447f /* 128^-0.5 */

#ifndef ROWS_PER_ITER
#define ROWS_PER_ITER 4
#endif
#ifndef KDA_SWIZZLE
#define KDA_SWIZZLE 1
#endif
#ifndef KDA_CONV
#define KDA_CONV 0
#endif
#ifndef KDA_STREAM
#define KDA_STREAM 1
#endif
#ifndef CONV_TAP_MAJOR
#define CONV_TAP_MAJOR 0
#endif
#ifndef LINE_GUARD
#define LINE_GUARD 0
#endif
#if CONV_TAP_MAJOR && !KDA_CONV
#error "CONV_TAP_MAJOR addresses the conv window KDA_CONV keeps"
#endif
// The conv window's element (stream s, tap t, column c) is at s * CONV_STREAM + t * CONV_TAP + c.
#if CONV_TAP_MAJOR
#define CONV_STREAM INNER
#define CONV_TAP (3 * INNER)
#else
#define CONV_STREAM (3 * INNER)
#define CONV_TAP INNER
#endif

typedef __nv_bfloat16 bf16;

#define BFLO(u) __bfloat162float(__ushort_as_bfloat16((unsigned short)((u) & 0xffffu)))
#define BFHI(u) __bfloat162float(__ushort_as_bfloat16((unsigned short)((u) >> 16)))

// 8 k-lanes per dv row, 16 k-values per lane, 16 dv rows per pass.
#define KLANES 8
#define KPER (KD / KLANES) /* 16 */
#define KVEC (KPER / 4)    /* 4 float4 per lane */
#define DVGRP (128 / KLANES) /* 16 dv rows covered per pass */

#if KDA_STREAM
// rec has no reuse inside the kernel: stream both directions so the lines do
// not sit in L2 evicting the w_f_b tile.
#define LDST(p) __ldcs(p)
#define STST(p, v) __stcs(p, v)
#else
#define LDST(p) (*(p))
#define STST(p, v) (*(p) = (v))
#endif
// -DKDA_STATE_BF16=1: rec is stored bf16 (vLLM's mamba_ssm_cache_dtype bfloat16) and computed f32, four k
// values a lane's load either way (uint2 of bf16, float4 of f32); the same lane walk, half the bytes.
#ifndef KDA_STATE_BF16
#define KDA_STATE_BF16 0
#endif
#if KDA_STATE_BF16
typedef uint2 rec_t;
#define REC_SLICE (KD * KD * 2)
__device__ __forceinline__ float4 rec_unpack(uint2 v) {
  const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ uint2 rec_pack(float4 f) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(f.x, f.y), hi = __floats2bfloat162_rn(f.z, f.w);
  return make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
}
#else
typedef float4 rec_t;
#define REC_SLICE (KD * KD * 4)
__device__ __forceinline__ float4 rec_unpack(float4 v) { return v; }
__device__ __forceinline__ float4 rec_pack(float4 f) { return f; }
#endif

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 128-thread (4 warp) sum.  `sm` is a 4-float scratch owned by the caller.
__device__ __forceinline__ float block_sum(float v, float* sm) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
  __syncthreads();
  return (sm[0] + sm[1]) + (sm[2] + sm[3]);
}

// block_sum of two values with one barrier: the same sums.
__device__ __forceinline__ void block_sum2(float a, float b, float* sm, float& sa, float& sb) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m), b += __shfl_xor_sync(0xffffffffu, b, m);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = a, sm[4 + (threadIdx.x >> 5)] = b;
  __syncthreads();
  sa = (sm[0] + sm[1]) + (sm[2] + sm[3]);
  sb = (sm[4] + sm[5]) + (sm[6] + sm[7]);
}

#ifndef KDA_SMEM_STATE
#define KDA_SMEM_STATE 0
#endif
#define KDA_BULKN 16  // the state slice's bulk copies, 4 KB each, one a thread
#ifndef KDA_MINB
#define KDA_MINB 1
#endif
extern "C" __global__ __launch_bounds__(128, KDA_MINB) void kern_k3_kda_core(
#if KDA_CONV
    const float* __restrict__ gate_partial,
    const float* __restrict__ cw,
    const float* __restrict__ wsm_partial,
#else
    const bf16* __restrict__ conv_q, const bf16* __restrict__ conv_k,
    const bf16* __restrict__ conv_v,
    const float* __restrict__ wsm_partial,
    const float* __restrict__ gate_partial,
#endif
    const bf16* __restrict__ w_f_b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ gamma_o,
    void* kda_base, const int* __restrict__ line_index, long long line_bytes,
#if CONV_TAP_MAJOR
    void* conv_base, const int* __restrict__ conv_index, long long conv_bytes,
#endif
    bf16* __restrict__ out,
    int B, const int* __restrict__ span_at, int span) {
#if KDA_SWIZZLE
  // Linearise (blockIdx.x, blockIdx.y) -> (b, h) head-fastest so that blocks
  // issued close in time walk contiguous rec, instead of 64 lines 6.5 MB apart.
  const int lin = blockIdx.x + gridDim.x * blockIdx.y;
  const int b = lin / HEADS;
  const int h = lin - b * HEADS;
#else
  const int b = blockIdx.x;
  const int h = blockIdx.y;
#endif
#if KDA_CONV && !KDA_PDL
  // cuBLAS's GEMM behind this kernel launches as a programmatic dependent and waits on its own:
  // let it get its blocks onto the SMs now.
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  if ((unsigned)(b - span_at[0]) < (unsigned)span) return;
  const int d = threadIdx.x;
  const int base = h * KD;
#if LINE_GUARD
  if (line_index[b] < 0) {
    out[(size_t)b * INNER + base + d] = __float2bfloat16(0.0f);
    return;
  }
#endif

  __shared__ __align__(16) float sh_qs[KD];
  __shared__ __align__(16) float sh_kn[KD];
  __shared__ __align__(16) float sh_dec[KD];
  __shared__ float sh_v[KD];
  __shared__ float sh_attn[KD];
  __shared__ __align__(16) float sh_flow[192];   // 8-float groups padded to 12: conflict-free 16-B LDS
  __shared__ float sh_red[8];
  __shared__ __align__(16) uint4 sh_tile[KD * 16];  // the head's w_f_b tile, 32 KB
  __shared__ float sh_gam[KD], sh_gate[KD];

  // Every load that depends on nothing but the indices goes out now, in one round trip: the head's
  // w_f_b tile into shared memory (cp.async, no registers), the row's scalars into registers.
  {
    const uint4* tile = reinterpret_cast<const uint4*>(w_f_b + (size_t)base * KD);
    const unsigned dst = (unsigned)__cvta_generic_to_shared(sh_tile);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = i * KD + d, r = q >> 4, c = q & 15;  // chunk c of row r lands at c ^ (r & 15)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + (r * 16 + (c ^ (r & 15))) * 16), "l"(tile + q));
    }
    asm volatile("cp.async.commit_group;");
  }
#if KDA_SMEM_STATE
  // The head's 64 KB state slice into shared memory by bulk copies, issued with the window's loads:
  // their round trip overlaps the whole prologue.
  extern __shared__ __align__(128) rec_t sh_rec[];
  __shared__ __align__(8) unsigned long long rec_bar;
  const char* rec_src = reinterpret_cast<const char*>(kda_base) + (long long)line_index[b] * line_bytes + (size_t)h * REC_SLICE;
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&rec_bar);
  if (d == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(REC_SLICE));
  }
  __syncthreads();
  if (d < KDA_BULKN)  // a piece a thread: the copies issue side by side
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"((unsigned)__cvta_generic_to_shared(sh_rec) + d * (REC_SLICE / KDA_BULKN)),
                 "l"(rec_src + d * (REC_SLICE / KDA_BULKN)), "r"(REC_SLICE / KDA_BULKN), "r"(bar) : "memory");
#endif
#if KDA_CONV
  // the conv window (this row's cache line) and taps
  const int c = base + d;
#if CONV_TAP_MAJOR
  bf16* win = reinterpret_cast<bf16*>(reinterpret_cast<char*>(conv_base) + (long long)conv_index[b] * conv_bytes) + c;
#else
  bf16* win = reinterpret_cast<bf16*>(reinterpret_cast<char*>(kda_base) + (long long)line_index[b] * line_bytes +
                                      (long long)HEADS * REC_SLICE) + c;
#endif
  bf16 w[3][3];
  float k[3][4];
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    w[s][0] = win[s * CONV_STREAM], w[s][1] = win[s * CONV_STREAM + CONV_TAP], w[s][2] = win[s * CONV_STREAM + 2 * CONV_TAP];
#pragma unroll
    for (int t = 0; t < 4; ++t) k[s][t] = cw[(size_t)s * 4 * INNER + t * INNER + c];
  }
#endif
  const float dtb = dt_bias[base + d], alog = a_log[h], gam = gamma_o[d];
#if KDA_PDL
  // Launched as a programmatic dependent of the projection: everything above reads the cache and the weights,
  // everything below its output.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  const float wsm_beta = wsm_partial[(size_t)b * WSM_LDS + h];
  const float wsm_flow = wsm_partial[(size_t)b * WSM_LDS + WSM_FA + d];
  const float gate_raw = gate_partial[(size_t)b * LDS + 3 * INNER + base + d];

  // ---- l2norm chain for q and k (all-bf16 chain, f32 sums) ----
#if KDA_CONV
  bf16 qkv[3];
  {
    float x[3];
#pragma unroll
    for (int s = 0; s < 3; ++s) x[s] = gate_partial[(size_t)b * LDS + s * INNER + c];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      const bf16 xb = __float2bfloat16_rn(x[s]);
      {
        win[s * CONV_STREAM] = w[s][1];
        win[s * CONV_STREAM + CONV_TAP] = w[s][2];
        win[s * CONV_STREAM + 2 * CONV_TAP] = xb;
      }
      float y = __bfloat162float(w[s][0]) * k[s][0];
      y = fmaf(__bfloat162float(w[s][1]), k[s][1], y);
      y = fmaf(__bfloat162float(w[s][2]), k[s][2], y);
      y = fmaf(__bfloat162float(xb), k[s][3], y);
      const float sb = __bfloat162float(__float2bfloat16_rn(y));
      qkv[s] = __float2bfloat16_rn(sb * (1.0f / (1.0f + expf(-sb))));
    }
  }
  const bf16 qv = qkv[0];
  const bf16 kv = qkv[1];
  sh_v[d] = __bfloat162float(qkv[2]);
#else
  const bf16 qv = conv_q[(size_t)b * INNER + base + d];
  const bf16 kv = conv_k[(size_t)b * INNER + base + d];
  sh_v[d] = __bfloat162float(conv_v[(size_t)b * INNER + base + d]);
#endif
  sh_gam[d] = gam;  // the epilogue's two, parked: no register lives across the delta rule for them
  sh_gate[d] = gate_raw;

  float qsq = __bfloat162float(__hmul(qv, qv));
  float ksq = __bfloat162float(__hmul(kv, kv));
  float qtot, ktot;
  block_sum2(qsq, ksq, sh_red, qtot, ktot);

  const bf16 qr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(qtot)) + L2_EPS));
  const bf16 kr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(ktot)) + L2_EPS));
  const float qsd = __bfloat162float(__hmul(qv, qr)) * QSCALE;
  const float knd = __bfloat162float(__hmul(kv, kr));
  sh_qs[d] = qsd;
  sh_kn[d] = knd;

  // ---- beta, f_a -> f_b projection, the decay gate ----
  const float beta = sigmoidf_(__bfloat162float(__float2bfloat16(wsm_beta)));
  sh_flow[d + ((d >> 3) << 2)] = __bfloat162float(__float2bfloat16(wsm_flow));
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // ga[d] = sum_j flow[j] * w_f_b[base+d, j], thread d over its own row of the tile. Every
  // flow[j]*w[j] is exact in f32 (8+8 mantissa bits), so the only error is the summation: a full pairwise
  // tree, the 16 columns of each of 8 groups (8 rl .. +8 and 64 + 8 rl .. +8) over fma-fused pairs in 4
  // levels, then the 8 groups in 3. It matters because ga lands in bf16 before dt_bias, and a flipped
  // landing moves dec, hence a whole dv row of the state.
  float ga;
  {
    const uint4* row = sh_tile + d * 16;
    float g[KLANES];
#pragma unroll
    for (int rl = 0; rl < KLANES; ++rl) {
      const uint4 a = row[rl ^ (d & 15)], c = row[(8 + rl) ^ (d & 15)];
      const float4 fa = *reinterpret_cast<const float4*>(sh_flow + 12 * rl);
      const float4 fb = *reinterpret_cast<const float4*>(sh_flow + 12 * rl + 4);
      const float4 fc = *reinterpret_cast<const float4*>(sh_flow + 96 + 12 * rl);
      const float4 fd = *reinterpret_cast<const float4*>(sh_flow + 96 + 12 * rl + 4);
      float p[8];
      p[0] = fmaf(fa.x, BFLO(a.x), fa.y * BFHI(a.x));
      p[1] = fmaf(fa.z, BFLO(a.y), fa.w * BFHI(a.y));
      p[2] = fmaf(fb.x, BFLO(a.z), fb.y * BFHI(a.z));
      p[3] = fmaf(fb.z, BFLO(a.w), fb.w * BFHI(a.w));
      p[4] = fmaf(fc.x, BFLO(c.x), fc.y * BFHI(c.x));
      p[5] = fmaf(fc.z, BFLO(c.y), fc.w * BFHI(c.y));
      p[6] = fmaf(fd.x, BFLO(c.z), fd.y * BFHI(c.z));
      p[7] = fmaf(fd.z, BFLO(c.w), fd.w * BFHI(c.w));
#pragma unroll
      for (int st = 1; st < 8; st <<= 1)
#pragma unroll
        for (int i = 0; i < 8; i += 2 * st) p[i] += p[i + st];
      g[rl] = p[0];
    }
#pragma unroll
    for (int st = 1; st < KLANES; st <<= 1)
#pragma unroll
      for (int i = 0; i < KLANES; i += 2 * st) g[i] += g[i + st];
    ga = g[0];
  }

  const float raw = __bfloat162float(__float2bfloat16(ga)) + dtb;
  sh_dec[d] = __expf(LB * sigmoidf_(__expf(alog) * raw));

  // c = sum_k kn[k]*qs[k]; lets attn ride along with the m reduction.  Its
  // internal barrier is also what publishes sh_dec/sh_qs/sh_kn to the block.
  const float ckq = block_sum(knd * qsd, sh_red);

  // ---- delta rule over rec ----
  const int kl = d & (KLANES - 1);   // 8 lanes over k
  const int dvl = d >> 3;            // 16 dv rows per pass
  rec_t* rec = reinterpret_cast<rec_t*>(reinterpret_cast<char*>(kda_base) + (long long)line_index[b] * line_bytes +
                                        (size_t)h * REC_SLICE);

  // dv-invariant coefficients for this lane's 16 k values: k = 32*j + 4*kl + i
  float cdec[KPER], ckn[KPER], cqs[KPER];
#pragma unroll
  for (int j = 0; j < KVEC; ++j) {
    const float4 a = *reinterpret_cast<const float4*>(sh_dec + 32 * j + 4 * kl);
    const float4 c = *reinterpret_cast<const float4*>(sh_kn + 32 * j + 4 * kl);
    const float4 e = *reinterpret_cast<const float4*>(sh_qs + 32 * j + 4 * kl);
    cdec[4 * j + 0] = a.x; cdec[4 * j + 1] = a.y; cdec[4 * j + 2] = a.z; cdec[4 * j + 3] = a.w;
    ckn[4 * j + 0] = c.x;  ckn[4 * j + 1] = c.y;  ckn[4 * j + 2] = c.z;  ckn[4 * j + 3] = c.w;
    cqs[4 * j + 0] = e.x;  cqs[4 * j + 1] = e.y;  cqs[4 * j + 2] = e.z;  cqs[4 * j + 3] = e.w;
  }

  rec_t* recv = rec + kl;   // + dv*32 + 8*j
#if KDA_SMEM_STATE
  const rec_t* srcv = sh_rec + kl;
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra W;\n}" ::"r"(bar) : "memory");
#else
  const rec_t* srcv = recv;
#endif

#pragma unroll 1
  for (int it = 0; it < KD / (DVGRP * ROWS_PER_ITER); ++it) {
    float4 s[ROWS_PER_ITER][KVEC];
    int dv[ROWS_PER_ITER];
#pragma unroll
    for (int r = 0; r < ROWS_PER_ITER; ++r) {
      dv[r] = (it * ROWS_PER_ITER + r) * DVGRP + dvl;
      const rec_t* p = srcv + (size_t)dv[r] * 32;
#pragma unroll
#if KDA_SMEM_STATE
      for (int j = 0; j < KVEC; ++j) s[r][j] = rec_unpack(p[8 * j]);
#else
      for (int j = 0; j < KVEC; ++j) s[r][j] = rec_unpack(LDST(p + 8 * j));
#endif
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_ITER; ++r) {
      float m = 0.0f, aq = 0.0f;
#pragma unroll
      for (int j = 0; j < KVEC; ++j) {
        float4 t = s[r][j];
        t.x *= cdec[4 * j + 0];  t.y *= cdec[4 * j + 1];   // S*dec, kept for write-back
        t.z *= cdec[4 * j + 2];  t.w *= cdec[4 * j + 3];
        m = fmaf(t.x, ckn[4 * j + 0], m);   aq = fmaf(t.x, cqs[4 * j + 0], aq);
        m = fmaf(t.y, ckn[4 * j + 1], m);   aq = fmaf(t.y, cqs[4 * j + 1], aq);
        m = fmaf(t.z, ckn[4 * j + 2], m);   aq = fmaf(t.z, cqs[4 * j + 2], aq);
        m = fmaf(t.w, ckn[4 * j + 3], m);   aq = fmaf(t.w, cqs[4 * j + 3], aq);
        s[r][j] = t;
      }
#pragma unroll
      for (int msk = 1; msk < KLANES; msk <<= 1) {
        m += __shfl_xor_sync(0xffffffffu, m, msk);
        aq += __shfl_xor_sync(0xffffffffu, aq, msk);
      }
      const float dlt = (sh_v[dv[r]] - m) * beta;
      if (kl == 0) sh_attn[dv[r]] = fmaf(dlt, ckq, aq);
      rec_t* p = recv + (size_t)dv[r] * 32;
#pragma unroll
      for (int j = 0; j < KVEC; ++j) {
        float4 o;
        o.x = fmaf(dlt, ckn[4 * j + 0], s[r][j].x);
        o.y = fmaf(dlt, ckn[4 * j + 1], s[r][j].y);
        o.z = fmaf(dlt, ckn[4 * j + 2], s[r][j].z);
        o.w = fmaf(dlt, ckn[4 * j + 3], s[r][j].w);
        STST(p + 8 * j, rec_pack(o));
      }
    }
  }
  __syncthreads();

  // ---- attn landing, rms(gamma_o), sigmoid gate ----
  const bf16 attnb = __float2bfloat16(sh_attn[d]);
  const float af = __bfloat162float(attnb);
  const float atot = block_sum(af * af, sh_red);
  const bf16 o = __float2bfloat16(af * rsqrtf(atot / (float)KD + RMS_EPS) * sh_gam[d]);
  const bf16 g = __float2bfloat16(
      sigmoidf_(__bfloat162float(__float2bfloat16(sh_gate[d]))));
  out[(size_t)b * INNER + base + d] = __hmul(o, g);
}
