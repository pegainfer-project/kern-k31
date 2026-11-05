// K3 `k3_kda_core` — Kimi-K3 KDA (Kimi Delta Attention) decode step: the
// short conv + SiLU of q, k and v, and the delta-rule recurrence, for every row
// and head, with the small per-head pieces (beta, the f_b projection of the
// forget path, the bf16 l2norm chain, the output rms + gate) fused in.
// Contract: docs/k3-kernel-abi.md section K3; certified math source:
// pegainfer's TileLang `kda_core_batched` (kernels/tilelang_defs.py).
//
//   extern "C" __global__ void kern_k3_kda_core(
//       const f32*  partial,       // [B, LDS]  q | k | v | g, 4 * INNER columns
//       const f32*  cw,            // [3 stream][4 tap][INNER]
//       const f32*  wsm_partial,   // [B, WSM_LDS]  col h = b_proj, cols WSM_FA.. = f_a
//       const bf16* w_f_b,         // [INNER, 128]
//       const f32*  dt_bias,       // [INNER]
//       const f32*  a_log,         // [HEADS]
//       const f32*  gamma_o,       // [128]
//       void* kda_base, const int* line_index, long long line_bytes,  // rec at offset 0
//       bf16* out,                 // [B, INNER]
//       int B, const int* span_at, int span);  // rows [*span_at, +span) are a span (K8 + K11 do them): skipped
//
//   grid  (min(B, G), HEADS, 1)   G row slots: a block (r, h) does head h of rows r, r + G, ...
//   block 128                    (one thread per output v-dim in the prologue/epilogue)
//   smem  65,536 B dynamic (the head's w_f_b rows, then one row's state) + ~5 KB static
//
// Math (per row b and head h, base = h*128, thread d):
//   x_s     = bf16(partial[b, s*INNER + base + d]), s = q, k, v
//   y_s     = sum_t f32(win_s[t]) * cw[s][t][base + d]  (t = the window's 3 taps, then x_s;
//             fma in that order) ; q,k,v = bf16(sb * 1/(1 + exp(-sb))), sb = f32(bf16(y_s))
//             -- the window then drops its oldest tap and appends x_s
//   qtot    = sum_d f32(bf16(q[d]*q[d]))          -- square lands in bf16, sum f32
//   qr      = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))                       (same for k)
//   qs[d]   = f32(bf16(q[d]*qr)) * 128^-0.5   ;   kn[d] = f32(bf16(k[d]*kr))
//   beta    = sigmoid(f32(bf16(wsm_partial[b, h])))
//   flow[j] = bf16(wsm_partial[b, WSM_FA + j])                          (128 values)
//   ga[d]   = sum_j f32(flow[j]) * f32(w_f_b[base + d, j])              (f32 accum)
//   raw[d]  = f32(bf16(ga[d])) + dt_bias[base + d]
//   dec[d]  = exp(LB * sigmoid(exp(a_log[h]) * raw[d])),  LB = -5
//   m[dv]   = sum_k S[dv,k]*dec[k]*kn[k]  ;  dlt[dv] = (f32(v[dv]) - m[dv]) * beta
//   S'[dv,k]= S[dv,k]*dec[k] + dlt[dv]*kn[k]                            (in place)
//   attn[dv]= bf16(sum_k S'[dv,k]*qs[k])                                (single landing)
//   out[dv] = bf16(f32(attn[dv]) * rsqrt(mean(attn^2) + 1e-5) * gamma_o[dv])
//             * bf16(sigmoid(f32(bf16(partial[b, 3*INNER + base + dv]))))
//
// Landing points: q/k squares -> bf16 per term; the l2 sum -> bf16 before
// +1e-6; rsqrt -> bf16; q*qr and k*kr -> bf16; ga -> bf16 before dt_bias;
// attn -> bf16 once; the rms product -> bf16 once; the gate -> bf16 twice
// (partial and sigmoid); the state is stored bf16 (vLLM's mamba_ssm_cache_dtype
// bfloat16) and computed f32.  Every sum's order is part of the contract (the
// bits of every build, every grid and every batch are the same): the f32
// reductions are the trees written below, whichever thread runs which branch.
//
// State traffic is the whole point: rec is HEADS*128*128 bf16 per row per
// layer (393 KB at TP8), read once and written once, and at the batches decode
// runs the kernel's time is that traffic plus whatever of a row's serial chain
// (window -> conv -> l2 -> f_b -> decay -> update -> rms) the memory does not
// hide.  The kernel therefore
//   * keeps G * HEADS blocks resident (G = 3 * SMs / HEADS: 69 KB of shared
//     memory, 3 blocks an SM) and has each walk the rows of one head, so a
//     head's w_f_b rows (32 KB) and conv taps load once a block, and a batch
//     past one row a slot pipelines instead of running in waves: a row's
//     window and partial go out a row ahead (into registers), its state into
//     the one 32 KB buffer half by half as the row before frees each half
//     (bulk copies, one mbarrier a half), so a row's whole prologue and the
//     row before's second half and epilogue cover its state's round trip;
//   * splits k across 8 lanes (kl = tid & 7) and dv across 16 lane groups
//     (dvl = tid >> 3), 16 k values a lane, so one warp instruction covers 4
//     dv rows x 64 B of fully contiguous uint2 and the write-back streams
//     (__stcs) with 100% sector use; the f_b projection runs in the same
//     layout (lane kl sums column group kl of rows dvl + 16 i), conflict-free
//     on the linear tile;
//   * holds a dv row's 16 k values per lane in registers between the
//     m-reduction and the write-back, and uses
//       attn[dv] = sum_k (S*dec)*qs + dlt[dv] * (kn.qs)
//     so attn rides along with m and the state is read once;
//   * runs two dv rows of a lane in the two halves of every paired f32
//     instruction (FMUL2 / FFMA2 / FADD2) and the f_b projection on mixed
//     bf16 FMAs (FHFMA.BF16) -- per lane the same IEEE operations in the same
//     order as the scalar forms, at about half the instructions.
//
// -DCONV_TAP_MAJOR=1 keeps the conv window in a state of its own, laid out the
// way an engine's per-slot conv pool is: the row's three taps oldest first,
// each tap q | k | v (element (s, t, c) at t * 3 * INNER + s * INNER + c), at
// `conv_base + conv_index[b] * conv_bytes`.  The recurrent state stays at
// `kda_base + line_index[b] * line_bytes`, so the line holds rec only.  The
// signature gains the three after `line_bytes`:
//
//       void* kda_base, const int* line_index, long long line_bytes,
//       void* conv_base, const int* conv_index, long long conv_bytes,
//
// -DLINE_GUARD=1: a row whose line is negative (an engine's padded row, which
// owns no state) writes a zero output and touches no state.
//
// The kernel launches as a programmatic dependent of the projection GEMM: a
// block's tile, taps and first row's line, window and state go out before
// griddepcontrol.wait (a no-op in a plain launch).
//
//   nvcc -cubin -arch=sm_103a -DHEADS=12 -DLDS=6400 -DWSM_LDS=6400 -DWSM_FA=12 k3_kda_core.cu
#include <cuda_bf16.h>

// HEADS is the heads this rank holds: 96 whole, or a group member's shard
// (`-DHEADS=12` for TP8, docs/multi-gpu.md "最终形态"); the per-head weights
// and the state line are then that rank's slice, the wsm partial keeps its
// whole-model layout (beta at column h, f_a at WSM_FA).
#ifndef HEADS
#define HEADS 96
#endif
#define KD 128
#define INNER (HEADS * KD)
// Row strides of the partial and the wsm partial, and where f_a starts in
// the latter: the whole-model layout by default; one fused GEMM's row (qkvg,
// then beta's HEADS columns and f_a's 128, padded to 256) with LDS = WSM_LDS =
// 4 * INNER + 256 and WSM_FA = HEADS.
#ifndef LDS
#define LDS (4 * INNER)
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

// dv rows a lane runs an update pass, two a pair: 2 or 4 (both halves of the
// state buffer need a pass of their own)
#ifndef ROWS_PER_ITER
#define ROWS_PER_ITER 4
#endif
#ifndef CONV_TAP_MAJOR
#define CONV_TAP_MAJOR 0
#endif
#ifndef LINE_GUARD
#define LINE_GUARD 0
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

// 8 k-lanes per dv row, 16 k-values per lane, 16 dv rows per pass.
#define KLANES 8
#define KPER (KD / KLANES) /* 16 */
#define KVEC (KPER / 4)    /* 4 uint2 per lane */
#define DVGRP (128 / KLANES) /* 16 dv rows covered per pass */

// rec: four k values a lane's load, a uint2 of bf16.
typedef uint2 rec_t;
#define REC_SLICE (KD * KD * 2)  // a head's state, bf16
#define TILE (KD * KD * 2)       // a head's w_f_b rows, bf16
#define BULK 4096                // a bulk copy's bytes
// bf16 -> f32 is exact, a bf16's bits in the high half of a float's: one instruction a value.
__device__ __forceinline__ float bf_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ unsigned rec_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}

// Two lanes' IEEE f32 operations in one instruction (sm_100 FMUL2 / FFMA2 / FADD2): each lane rounds
// exactly as its scalar form, so pairing two independent chains changes no bit. A float operand is
// broadcast to both lanes (the instructions take a scalar register there).
__device__ __forceinline__ unsigned long long pk(float2 a) {
  unsigned long long r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a.x), "f"(a.y));
  return r;
}
__device__ __forceinline__ float2 upk(unsigned long long r) {
  float2 a;
  asm("mov.b64 {%0, %1}, %2;" : "=f"(a.x), "=f"(a.y) : "l"(r));
  return a;
}
__device__ __forceinline__ float2 mul2(float2 a, float b) {
  unsigned long long d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(pk(a)), "l"(pk(make_float2(b, b))));
  return upk(d);
}
__device__ __forceinline__ float2 fma2(float2 a, float b, float2 c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(pk(a)), "l"(pk(make_float2(b, b))), "l"(pk(c)));
  return upk(d);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  unsigned long long d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(pk(a)), "l"(pk(b)));
  return upk(d);
}
// fma(f32(w.lo), f32(f.lo), c) on packed bf16 operands (sm_100 FHFMA.BF16): the bf16 product is exact in
// f32, so this is fmaf of the unpacked values, rounded once, with no unpacking; `_hi` the high halves.
__device__ __forceinline__ float bfma_lo(unsigned w, unsigned f, float c) {
  float d;
  asm("{.reg .b16 wl, wh, fl, fh;\n mov.b32 {wl, wh}, %1;\n mov.b32 {fl, fh}, %2;\n"
      " fma.rn.f32.bf16 %0, wl, fl, %3;\n}" : "=f"(d) : "r"(w), "r"(f), "f"(c));
  return d;
}
__device__ __forceinline__ float bfma_hi(unsigned w, unsigned f, float c) {
  float d;
  asm("{.reg .b16 wl, wh, fl, fh;\n mov.b32 {wl, wh}, %1;\n mov.b32 {fl, fh}, %2;\n"
      " fma.rn.f32.bf16 %0, wh, fh, %3;\n}" : "=f"(d) : "r"(w), "r"(f), "f"(c));
  return d;
}

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


// Bulk copies into shared memory, completing on an mbarrier.
__device__ __forceinline__ void bulk(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void expect(unsigned bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(unsigned bar, unsigned parity) {
  unsigned done;
  do
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  while (!done);
}

// What a row needs ahead of its update, loaded a row early: the conv window's three taps of q, k, v,
// the partial's pre-conv q, k, v, its gate, beta and the flow column.
struct Row {
  int b;
  bool live;  // a row of this call's to do: in the batch, not the span's (LINE_GUARD: and owning a line)
  long long line;
  bf16* win;
  bf16 w[3][3];
  float x[3], gate, beta, flow;
};

#ifndef KDA_MINB
#define KDA_MINB 3
#endif
extern "C" __global__ __launch_bounds__(128, KDA_MINB) void kern_k3_kda_core(
    const float* __restrict__ partial,
    const float* __restrict__ cw,
    const float* __restrict__ wsm_partial,
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
  static_assert(ROWS_PER_ITER % 2 == 0, "the update runs dv rows in pairs");
  constexpr int PASSES = KD / (DVGRP * ROWS_PER_ITER);
  static_assert(PASSES % 2 == 0, "each half of the state buffer takes passes of its own");
  // Block (r, h) of the G * HEADS, head-fastest (blocks issued together walk neighbouring state).
  const int G = gridDim.x;
  const int lin = blockIdx.x + G * blockIdx.y;
  const int h = lin % HEADS, r = lin / HEADS;
  const int d = threadIdx.x, base = h * KD, c = base + d;
  const int kl = d & (KLANES - 1);   // 8 lanes over k
  const int dvl = d >> 3;            // 16 dv rows per pass

  extern __shared__ __align__(128) uint4 dyn[];
  const uint4* tile = dyn;                                            // w_f_b rows base .. base + 127
  const rec_t* srec = reinterpret_cast<const rec_t*>(dyn + TILE / 16);  // one row's state
  __shared__ __align__(16) float sh_qs[KD];
  __shared__ __align__(16) float sh_kn[KD];
  __shared__ __align__(16) float sh_dec[KD];
  __shared__ float sh_v[KD];
  __shared__ float sh_attn[KD];
  __shared__ __align__(16) bf16 sh_flow[KD];
  __shared__ float sh_red[16];  // block_sum2's 8, then ckq's 4 and the rms's 4: no barrier orders their reuse
  __shared__ float sh_gam[KD], sh_gate[KD];
  __shared__ __align__(8) unsigned long long bars[3];  // the tile's, the state buffer's two halves'
  const unsigned bar_tile = (unsigned)__cvta_generic_to_shared(bars);
  const unsigned dst_rec = (unsigned)__cvta_generic_to_shared(srec);

  // A row's state half q into the buffer, from thread 0; the half's mbarrier counts the bytes.
  auto fetch = [&](const Row& w, int q) {
    const char* src = reinterpret_cast<const char*>(kda_base) + w.line * line_bytes + (size_t)h * REC_SLICE + q * (REC_SLICE / 2);
    expect(bar_tile + 8 * (1 + q), REC_SLICE / 2);
#pragma unroll
    for (int i = 0; i < REC_SLICE / 2 / BULK; ++i)
      bulk(dst_rec + q * (REC_SLICE / 2) + i * BULK, src + i * BULK, BULK, bar_tile + 8 * (1 + q));
  };
  // A row's line and window pointer; then its window, then (past the projection) its partial.
  const int at = span_at[0];
  auto locate = [&](int b) {
    Row w;
    w.b = b;
    w.live = b < B && (unsigned)(b - at) >= (unsigned)span;
    w.line = w.live ? line_index[b] : 0;
#if LINE_GUARD
    w.live = w.live && w.line >= 0;
#endif
#if CONV_TAP_MAJOR
    w.win = reinterpret_cast<bf16*>(reinterpret_cast<char*>(conv_base) + (w.live ? (long long)conv_index[b] : 0) * conv_bytes) + c;
#else
    w.win = reinterpret_cast<bf16*>(reinterpret_cast<char*>(kda_base) + w.line * line_bytes + (long long)HEADS * REC_SLICE) + c;
#endif
    return w;
  };
  auto window = [&](Row& w) {
    if (!w.live) return;
#pragma unroll
    for (int s = 0; s < 3; ++s)
#pragma unroll
      for (int t = 0; t < 3; ++t) w.w[s][t] = w.win[s * CONV_STREAM + t * CONV_TAP];
  };
  auto inputs = [&](Row& w) {
    if (!w.live) return;
#pragma unroll
    for (int s = 0; s < 3; ++s) w.x[s] = partial[(size_t)w.b * LDS + s * INNER + c];
    w.gate = partial[(size_t)w.b * LDS + 3 * INNER + c];
    w.beta = wsm_partial[(size_t)w.b * WSM_LDS + h];
    w.flow = wsm_partial[(size_t)w.b * WSM_LDS + WSM_FA + d];
  };

  // ---- once a block: the head's tile (bulk copies), its conv taps, dt_bias, a_log, gamma_o ----
  // The first row's window goes out first: a block's requests queue in order, and the window gates the
  // prologue while the tile is due at the f_b projection and the state at the update.
  Row cur = locate(r), nxt = locate(r + G);
  window(cur);
  if (d == 0) {
#pragma unroll
    for (int i = 0; i < 3; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_tile + 8 * i));
    asm volatile("fence.mbarrier_init.release.cluster;");
    expect(bar_tile, TILE);
#pragma unroll
    for (int i = 0; i < TILE / BULK; ++i)
      bulk((unsigned)__cvta_generic_to_shared(tile) + i * BULK, reinterpret_cast<const char*>(w_f_b + (size_t)base * KD) + i * BULK,
           BULK, bar_tile);
  }
  if (d == 0 && cur.live) fetch(cur, 0), fetch(cur, 1);
  float k[3][4];
#pragma unroll
  for (int s = 0; s < 3; ++s)
#pragma unroll
    for (int t = 0; t < 4; ++t) k[s][t] = cw[(size_t)s * 4 * INNER + t * INNER + c];
  const float dtb = dt_bias[base + dvl + DVGRP * kl], alog = a_log[h];  // dtb: the row this thread lands
  sh_gam[d] = gamma_o[d];
  // A programmatic dependent of the projection: everything above reads the cache and the weights,
  // everything below its output.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  inputs(cur);

  for (unsigned uses = 0; cur.b < B; ) {
    // the next row's window and partial go out now, a row ahead, on a line loaded a row before
    const Row nn = locate(nxt.b + G);
    window(nxt);
    inputs(nxt);
    if (!cur.live) {
#if LINE_GUARD
      if ((unsigned)(cur.b - at) >= (unsigned)span) out[(size_t)cur.b * INNER + c] = __float2bfloat16(0.0f);
#endif
      // the buffer's last reader was a live row's update, behind its barriers
      if (d == 0 && nxt.live) fetch(nxt, 0), fetch(nxt, 1);
      cur = nxt, nxt = nn;
      continue;
    }

    // ---- the short conv + SiLU of q, k, v; the window shifts in place ----
    bf16 qkv[3];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      const bf16 xb = __float2bfloat16_rn(cur.x[s]);
      cur.win[s * CONV_STREAM] = cur.w[s][1];
      cur.win[s * CONV_STREAM + CONV_TAP] = cur.w[s][2];
      cur.win[s * CONV_STREAM + 2 * CONV_TAP] = xb;
      float y = __bfloat162float(cur.w[s][0]) * k[s][0];
      y = fmaf(__bfloat162float(cur.w[s][1]), k[s][1], y);
      y = fmaf(__bfloat162float(cur.w[s][2]), k[s][2], y);
      y = fmaf(__bfloat162float(xb), k[s][3], y);
      const float sb = __bfloat162float(__float2bfloat16_rn(y));
      qkv[s] = __float2bfloat16_rn(sb * (1.0f / (1.0f + expf(-sb))));
    }
    const bf16 qv = qkv[0], kv = qkv[1];
    sh_v[d] = __bfloat162float(qkv[2]);
    sh_gate[d] = cur.gate;  // the epilogue's, parked: no register lives across the update for it
    sh_flow[d] = __float2bfloat16(cur.flow);  // published by block_sum2's barrier

    // ---- l2norm chain for q and k (all-bf16 chain, f32 sums) ----
    float qtot, ktot;
    block_sum2(__bfloat162float(__hmul(qv, qv)), __bfloat162float(__hmul(kv, kv)), sh_red, qtot, ktot);
    const bf16 qr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(qtot)) + L2_EPS));
    const bf16 kr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(ktot)) + L2_EPS));
    const float qsd = __bfloat162float(__hmul(qv, qr)) * QSCALE;
    const float knd = __bfloat162float(__hmul(kv, kr));
    sh_qs[d] = qsd;
    sh_kn[d] = knd;
    const float beta = sigmoidf_(__bfloat162float(__float2bfloat16(cur.beta)));

    // ga[r] = sum_j flow[j] * w_f_b[base+r, j]. Every flow[j]*w[j] is exact in f32 (8+8 mantissa bits), so
    // the only error is the summation: a full pairwise tree, the 16 columns of each of 8 groups (8 g .. +8
    // and 64 + 8 g .. +8) over fma-fused pairs in 4 levels, then the 8 groups in 3. Lane kl sums group kl
    // of its 8 rows; the xor butterfly over the 8 lanes is the groups' tree (a + b == b + a bit for bit),
    // and thread (kl, dvl) lands row dvl + 16 kl. It matters because ga lands in bf16 before dt_bias, and a
    // flipped landing moves dec, hence a whole dv row of the state.
    wait(bar_tile, 0);
    float ga = 0.0f;
    {
      const uint4* flow = reinterpret_cast<const uint4*>(sh_flow);
      const uint4 fa = flow[kl], fc = flow[8 + kl];
#pragma unroll
      for (int i = 0; i < KLANES; ++i) {
        const uint4 a = tile[(dvl + DVGRP * i) * 16 + kl], e = tile[(dvl + DVGRP * i) * 16 + 8 + kl];
        float p[8];
        p[0] = bfma_lo(a.x, fa.x, bfma_hi(a.x, fa.x, -0.0f));
        p[1] = bfma_lo(a.y, fa.y, bfma_hi(a.y, fa.y, -0.0f));
        p[2] = bfma_lo(a.z, fa.z, bfma_hi(a.z, fa.z, -0.0f));
        p[3] = bfma_lo(a.w, fa.w, bfma_hi(a.w, fa.w, -0.0f));
        p[4] = bfma_lo(e.x, fc.x, bfma_hi(e.x, fc.x, -0.0f));
        p[5] = bfma_lo(e.y, fc.y, bfma_hi(e.y, fc.y, -0.0f));
        p[6] = bfma_lo(e.z, fc.z, bfma_hi(e.z, fc.z, -0.0f));
        p[7] = bfma_lo(e.w, fc.w, bfma_hi(e.w, fc.w, -0.0f));
#pragma unroll
        for (int st = 1; st < 8; st <<= 1)
#pragma unroll
          for (int q = 0; q < 8; q += 2 * st) p[q] += p[q + st];
        float g = p[0];
#pragma unroll
        for (int msk = 1; msk < KLANES; msk <<= 1) g += __shfl_xor_sync(0xffffffffu, g, msk);
        if (kl == i) ga = g;
      }
    }
    const float raw = __bfloat162float(__float2bfloat16(ga)) + dtb;
    sh_dec[dvl + DVGRP * kl] = __expf(LB * sigmoidf_(__expf(alog) * raw));

    // c = sum_k kn[k]*qs[k]; lets attn ride along with the m reduction.  Its
    // internal barrier is also what publishes sh_dec/sh_qs/sh_kn to the block.
    const float ckq = block_sum(knd * qsd, sh_red + 8);

    // ---- delta rule over rec ----
    // dv-invariant coefficients for this lane's 16 k values: k = 32*j + 4*kl + i
    float cdec[KPER], ckn[KPER], cqs[KPER];
#pragma unroll
    for (int j = 0; j < KVEC; ++j) {
      const float4 a = *reinterpret_cast<const float4*>(sh_dec + 32 * j + 4 * kl);
      const float4 e = *reinterpret_cast<const float4*>(sh_kn + 32 * j + 4 * kl);
      const float4 f = *reinterpret_cast<const float4*>(sh_qs + 32 * j + 4 * kl);
      cdec[4 * j + 0] = a.x; cdec[4 * j + 1] = a.y; cdec[4 * j + 2] = a.z; cdec[4 * j + 3] = a.w;
      ckn[4 * j + 0] = e.x;  ckn[4 * j + 1] = e.y;  ckn[4 * j + 2] = e.z;  ckn[4 * j + 3] = e.w;
      cqs[4 * j + 0] = f.x;  cqs[4 * j + 1] = f.y;  cqs[4 * j + 2] = f.z;  cqs[4 * j + 3] = f.w;
    }
    rec_t* recv = reinterpret_cast<rec_t*>(reinterpret_cast<char*>(kda_base) + cur.line * line_bytes + (size_t)h * REC_SLICE) + kl;

    // Two dv rows a step, row q of the pair in lane q of every float2: each multiply-add of the update runs
    // both rows in one instruction, every chain in the order of the scalar form (k ascending within a
    // lane, then the xor tree over the 8 k lanes). The buffer's first half is dv rows 0..63, the passes
    // before PASSES / 2; once every thread is past them, the next row's first half takes its place.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      wait(bar_tile + 8 * (1 + half), uses & 1);
#pragma unroll 1
      for (int it = half * PASSES / 2; it < (half + 1) * PASSES / 2; ++it) {
        rec_t u[ROWS_PER_ITER][KVEC];
#pragma unroll
        for (int q = 0; q < ROWS_PER_ITER; ++q)
#pragma unroll
          for (int j = 0; j < KVEC; ++j) u[q][j] = srec[(size_t)((it * ROWS_PER_ITER + q) * DVGRP + dvl) * 32 + kl + 8 * j];
#pragma unroll
        for (int q = 0; q < ROWS_PER_ITER; q += 2) {
          const int dv0 = (it * ROWS_PER_ITER + q) * DVGRP + dvl, dv1 = dv0 + DVGRP;
          float2 t[KPER];   // S*dec, kept for the write-back
          float2 m = make_float2(0.0f, 0.0f), aq = make_float2(0.0f, 0.0f);
#pragma unroll
          for (int j = 0; j < KVEC; ++j) {
            const rec_t u0 = u[q][j], u1 = u[q + 1][j];
            t[4 * j + 0] = mul2(make_float2(bf_lo(u0.x), bf_lo(u1.x)), cdec[4 * j + 0]);
            t[4 * j + 1] = mul2(make_float2(bf_hi(u0.x), bf_hi(u1.x)), cdec[4 * j + 1]);
            t[4 * j + 2] = mul2(make_float2(bf_lo(u0.y), bf_lo(u1.y)), cdec[4 * j + 2]);
            t[4 * j + 3] = mul2(make_float2(bf_hi(u0.y), bf_hi(u1.y)), cdec[4 * j + 3]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              m = fma2(t[4 * j + i], ckn[4 * j + i], m);
              aq = fma2(t[4 * j + i], cqs[4 * j + i], aq);
            }
          }
#pragma unroll
          for (int msk = 1; msk < KLANES; msk <<= 1) {
            m = add2(m, make_float2(__shfl_xor_sync(0xffffffffu, m.x, msk), __shfl_xor_sync(0xffffffffu, m.y, msk)));
            aq = add2(aq, make_float2(__shfl_xor_sync(0xffffffffu, aq.x, msk), __shfl_xor_sync(0xffffffffu, aq.y, msk)));
          }
          const float2 dlt = make_float2((sh_v[dv0] - m.x) * beta, (sh_v[dv1] - m.y) * beta);
          if (kl == 0) sh_attn[dv0] = fmaf(dlt.x, ckq, aq.x), sh_attn[dv1] = fmaf(dlt.y, ckq, aq.y);
#pragma unroll
          for (int j = 0; j < KVEC; ++j) {
            float2 o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = fma2(dlt, ckn[4 * j + i], t[4 * j + i]);
            __stcs(recv + (size_t)dv0 * 32 + 8 * j, make_uint2(rec_pack(o[0].x, o[1].x), rec_pack(o[2].x, o[3].x)));
            __stcs(recv + (size_t)dv1 * 32 + 8 * j, make_uint2(rec_pack(o[0].y, o[1].y), rec_pack(o[2].y, o[3].y)));
          }
        }
      }
      __syncthreads();
      if (d == 0 && nxt.live) fetch(nxt, half);
    }
    ++uses;

    // ---- attn landing, rms(gamma_o), sigmoid gate ----
    const bf16 attnb = __float2bfloat16(sh_attn[d]);
    const float af = __bfloat162float(attnb);
    const float atot = block_sum(af * af, sh_red + 12);
    const bf16 o = __float2bfloat16(af * rsqrtf(atot / (float)KD + RMS_EPS) * sh_gam[d]);
    const bf16 g = __float2bfloat16(sigmoidf_(__bfloat162float(__float2bfloat16(sh_gate[d]))));
    out[(size_t)cur.b * INNER + c] = __hmul(o, g);
    cur = nxt, nxt = nn;
  }
}
