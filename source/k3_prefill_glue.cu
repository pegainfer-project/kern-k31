// The residual stream and the router top-k of a prefill chunk: K1 and K6 of
// docs/k3-kernel-abi.md laid out for thousands of rows instead of a decode
// batch, with the landing add that ends one layer fused into the mix that
// starts the next.
//
// The decode kernels (k3_residual.cu, k3_router_argmax.cu) give a row a whole
// 1024-thread block so that one row finishes fast; over a chunk that block is
// one row per SM in flight, a chain of barriers waiting on one DRAM round trip
// at a time. Here a row is a block of 14 warps (residual) or one warp (top-k),
// so an SM holds many rows. Every value is the decode kernels' to the bit: the
// reductions keep their trees (below), the landings are the same.
//
//   [K1a] kern_k3g_attnres_rms(prefix, blocks, sw, gamma, normed, nb, snapshot, B)
//   [K1b] kern_k3g_land_add_attnres_rms(partial bf16, prefix, blocks, sw, gamma, prefix2, normed,
//                                       nb, snapshot, B)
//         the ABIs and math of k3_residual.cu's K1a and K1b (-DLAND_BF16).
//   [K1d] kern_k3g_land_add2_attnres_rms(p1, p2, prefix2, hidden, two, blocks, sw, gamma, normed, nb, B)
//         K1c then K1a without a snapshot: hidden = bf16(prefix2 + p1 + (two ? p2 : 0)), written,
//         then normed = rms(attnres(blocks, hidden, nb), gamma). The next layer's residual mix
//         reads the hidden it just made from registers.
//     grid (B, 1, 1)   block (448, 1, 1)   smem 0 dynamic
//
//   kern_k3g_embed_rms(ids, table, hidden, blocks, gamma, normed, seq_lens, cu_q, lens, nseq, ns, short_max,
//                      sync, sync_words, B)
//         layer 0's K1a with the embedding gather: the row from the table, hidden and snapshot 0;
//         and the stage's setup (zero_share below): the MoE routing state zeroed, the FMHA's tables.
//   kern_k3g_attnres_rms_setup(prefix, blocks, sw, gamma, normed, nb, snapshot, sync, sync_words, B)
//         K1a where a later stage starts, the routing state zeroed alongside.
//
//   [K1c] kern_k3g_land_add2(p1, p2, prefix2, hidden, two, B)
//         K1c alone, before a snapshot layer and at a stage's end.
//     grid (B, 4, 1)   block (224, 1, 1)
//
// A chunk's GEMMs land their own outputs (cuBLASLt with a bf16 D), so p1, p2
// and the activations' gate | up arrive as the bf16 values K1c and K7 round
// their f32 partials to first; those roundings are gone, the rest is K1c's.
//
//   [K7]  kern_k3g_situ(p, act, n, B)
//         k3_land.cu's kern_k3_land_situ on a landed [B, 2n] gate | up.
//     grid (B, ceil(n / 2048), 1)   block (256, 1, 1): 8 columns a thread.
//
//   [K6]  kern_k3g_router_topk(S, bias, rs, idx, wts, B)
//         the ABI and math of k3_router_argmax.cu's router top-k.
//     grid (ceil(B / 4), 1, 1)   block (128, 1, 1): one warp per row.
//
//   kern_k3g_finalize_rms(fc2, exp2perm, wts, gamma, out, T, cols)
//         k3_moe_prefill.cu's kern_k3_moe_finalize, its bf16 row then
//         k3_land.cu's kern_k3_rms by gamma, the row never leaving registers.
//     grid (rows, 1, 1)   block (cols / 8, 1, 1)
//
// Reduction trees. K1 sums a row of H = 7168 as 28 groups of 32 sixteen-byte
// vectors: each thread's 8 elements serially, the group by a 32-lane xor
// butterfly, then the 28 group sums plus 4 zero slots by one more butterfly.
// The decode kernel's group is a warp (warp w owns vectors 32w..32w+31); here
// warp w owns groups w, w + 14, i.e. thread t vectors t + 448k, and runs the
// same butterfly per group, so every sum is the same
// tree over the same values. K6's picks are exact comparisons, so its layout
// (expert j * 32 + lane) is free; its weight sum is the same 16-lane butterfly.
// The finalize's rms is kern_k3_rms's tree with thread h owning columns
// 8h..8h+8: its squares serially, a warp butterfly, then the 32 warp slots
// (the ones past the row zero) by another.
#include <cuda_bf16.h>

#include "k3_fmha_plan.cuh"

typedef __nv_bfloat16 bf16_t;
typedef __nv_bfloat162 bf162_t;
typedef unsigned int u32;

#define KH 7168
#define KNB_MAX 8
#define KEPS 1e-5f
#define RWARPS 14  // two rows of 448 threads an SM at 60 registers: half the 7-warp row's latency chain
#define RTHREADS (RWARPS * 32)
#define RGROUPS 2  // vectors per thread: 896 / 448
#define RSLOTS 32  // 28 groups + 4 zero slots, the decode kernel's 32 warps

struct V8 {
  unsigned w[4];
};

__device__ __forceinline__ V8 ldv(const void* p) {
  const uint4 t = *(const uint4*)p;
  V8 v;
  v.w[0] = t.x;
  v.w[1] = t.y;
  v.w[2] = t.z;
  v.w[3] = t.w;
  return v;
}
__device__ __forceinline__ void stv(void* p, const V8& v) { *(uint4*)p = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]); }
// bf16 pairs as plain bits: a bf16 is the top half of its f32, and cvt.rn.bf16x2.f32 packs two
// (what __float22bfloat162_rn emits) without the half-word moves of going through bf162_t
__device__ __forceinline__ bf162_t as_bf162(unsigned u) { return *reinterpret_cast<const bf162_t*>(&u); }
__device__ __forceinline__ unsigned from_bf162(bf162_t h) { return *reinterpret_cast<const unsigned*>(&h); }
__device__ __forceinline__ float2 bf2f(unsigned u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
__device__ __forceinline__ unsigned f2bf(float2 f) {
  unsigned r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(f.y), "f"(f.x));
  return r;
}

// bf16(f32(a) + f32(b)) for two bf16 pairs: their f32 sum is exact unless the exponents are more than 16
// apart, and then the smaller cannot reach the bf16 half ulp, so one rounding of the exact sum is the same
__device__ __forceinline__ unsigned add_bf2(unsigned a, unsigned b) {
  unsigned r;
  asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ unsigned long long b64(float2 a) { return *reinterpret_cast<unsigned long long*>(&a); }
__device__ __forceinline__ float2 f2(unsigned long long d) { return *reinterpret_cast<float2*>(&d); }
// two IEEE ops in one f32x2 instruction, each lane rounded as the scalar op
__device__ __forceinline__ float2 add2f(float2 a, float2 b) {
  unsigned long long d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(b64(a)), "l"(b64(b)));
  return f2(d);
}
__device__ __forceinline__ float2 mul2f(float2 a, float2 b) {
  unsigned long long d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(b64(a)), "l"(b64(b)));
  return f2(d);
}
__device__ __forceinline__ float2 fma2f(float2 a, float2 b, float2 c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(b64(a)), "l"(b64(b)), "l"(b64(c)));
  return f2(d);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Four (two) values reduced over the warp at once, each by the 32-lane xor butterfly's own tree: a lane
// trades the values it drops for its partner's copies of the ones it keeps, so a level's sums are the
// butterfly's (fp addition commutes) at half the shuffles. Lane l ends with the sum of value
// 2 * bit4(l) + bit3(l) (of value bit4(l)).
__device__ __forceinline__ float warp_sum4(float a, float b, float c, float d, int lane) {
  const bool h = lane & 16, e = lane & 8;
  float k0 = h ? c : a, k1 = h ? d : b;
  k0 += __shfl_xor_sync(0xffffffffu, h ? a : c, 16);
  k1 += __shfl_xor_sync(0xffffffffu, h ? b : d, 16);
  float v = e ? k1 : k0;
  v += __shfl_xor_sync(0xffffffffu, e ? k0 : k1, 8);
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_sum2(float a, float b, int lane) {
  const bool h = lane & 16;
  float v = h ? b : a;
  v += __shfl_xor_sync(0xffffffffu, h ? a : b, 16);
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct RowSmem {
  float red[(KNB_MAX + 1) * 2][RSLOTS];  // [value][group] partials
  float val[(KNB_MAX + 1) * 2];          // [value] row sums
  float red2[RSLOTS];
};

// Group g's slot is written by lane 0 of the warp that owns it; the 4 pad
// slots stay zero, as the decode kernel's idle warps 28..31 leave them.
__device__ __forceinline__ void zero_pads(RowSmem& s, int t) {
  if (t < (KNB_MAX + 1) * 2 * 4) s.red[t >> 2][28 + (t & 3)] = 0.f;
  if (t < 4) s.red2[28 + t] = 0.f;
}

__device__ __forceinline__ void sq_dot(const V8& x, const float* swv, float& sq, float& dp) {
  sq = 0.f;
  dp = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = bf2f(x.w[j]);
    sq += f.x * f.x;
    sq += f.y * f.y;
    dp += f.x * swv[2 * j];
    dp += f.y * swv[2 * j + 1];
  }
}

// A row's DRAM reads go out before its first use: candidate 0 into registers, the others' lines
// (one prefetch a 128-byte line) into L2, so pass 1 waits on one round trip instead of one a candidate.
__device__ __forceinline__ void load_cand0(const bf16_t* __restrict__ blk_row, int nb, int t, V8 (&x0)[RGROUPS]) {
  if (nb > 0)
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) x0[k] = ldv(blk_row + (t + k * RTHREADS) * 8);
  if ((t & 7) == 0)
    for (int c = 1; c < nb; ++c)
#pragma unroll
      for (int k = 0; k < RGROUPS; ++k)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(blk_row + (size_t)c * KH + (t + k * RTHREADS) * 8));
}

// mixed = attnres(candidates 0..nb-1 of blk_row, candidate 0 = x0, candidate nb = pv), then
// normed_row = rms(mixed, gamma). Called by the whole block after zero_pads.
__device__ __forceinline__ void attnres_rms_row(const bf16_t* __restrict__ blk_row, const V8 (&pv)[RGROUPS],
                                                const V8 (&x0)[RGROUPS], const float* __restrict__ sw, const bf16_t* __restrict__ gamma,
                                                bf16_t* __restrict__ normed_row, int nb, int t, RowSmem& s) {
  const int lane = t & 31, warp = t >> 5;
  const int ncand = nb + 1;
  V8 mixed[RGROUPS];

  if (nb == 0) {
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) mixed[k] = pv[k];  // bf16(1.0f * f32(prefix)) == prefix
  } else {
    // sw from L1 at each use: its 16 registers would cost the SM a row
    auto swk = [&](int k, float (&w)[8]) {
      const float4* sp4 = (const float4*)(sw + (t + k * RTHREADS) * 8);
      const float4 a = sp4[0], b = sp4[1];
      w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
      w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    };
    // value 2k of a lane's four is group k's sum of squares, 2k + 1 its dot product with sw
    auto score = [&](int c, const V8 (&x)[RGROUPS]) {
      float w[8], sq[RGROUPS], dp[RGROUPS];
#pragma unroll
      for (int k = 0; k < RGROUPS; ++k) {
        swk(k, w);
        sq_dot(x[k], w, sq[k], dp[k]);
      }
      const float v = warp_sum4(sq[0], dp[0], sq[1], dp[1], lane);
      if ((lane & 7) == 0) s.red[c * 2 + (lane >> 3 & 1)][warp + (lane >> 4) * RWARPS] = v;
    };
    score(0, x0);
    for (int c = 1; c < nb; ++c) {
      V8 x[RGROUPS];
#pragma unroll
      for (int k = 0; k < RGROUPS; ++k) x[k] = ldv(blk_row + (size_t)c * KH + (t + k * RTHREADS) * 8);
      score(c, x);
    }
    score(nb, pv);
    __syncthreads();

    for (int v = warp; v < 2 * ncand; v += RWARPS) {
      const float x = warp_sum(s.red[v][lane]);
      if (lane == 0) s.val[v] = x;
    }
    __syncthreads();

    float sc = -3.0e38f;
    if (lane < ncand) {
      const float sqv = s.val[2 * lane], dpv = s.val[2 * lane + 1];
      sc = dpv * rsqrtf(sqv * (1.0f / (float)KH) + KEPS);
    }
    float mx = sc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float ex = (lane < ncand) ? __expf(sc - mx) : 0.f;
    float den = ex;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    const float pmine = ex / den;

    float2 acc[RGROUPS][4];
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[k][j] = make_float2(0.f, 0.f);
#pragma unroll
    for (int c = 0; c < KNB_MAX; ++c) {
      if (c >= nb) break;
      const float p = __shfl_sync(0xffffffffu, pmine, c);
#pragma unroll
      for (int k = 0; k < RGROUPS; ++k) {
        const V8 x = ldv(blk_row + (size_t)c * KH + (t + k * RTHREADS) * 8);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[k][j] = fma2f(make_float2(p, p), bf2f(x.w[j]), acc[k][j]);
      }
    }
    const float p = __shfl_sync(0xffffffffu, pmine, nb);
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) {
#pragma unroll
      for (int j = 0; j < 4; ++j) mixed[k].w[j] = f2bf(fma2f(make_float2(p, p), bf2f(pv[k].w[j]), acc[k][j]));
    }
  }

  float sq[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    sq[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf2f(mixed[k].w[j]);
      sq[k] += f.x * f.x;
      sq[k] += f.y * f.y;
    }
  }
  const float sqs = warp_sum2(sq[0], sq[1], lane);
  if ((lane & 15) == 0) s.red2[warp + (lane >> 4) * RWARPS] = sqs;
  __syncthreads();
  const float tot = warp_sum(s.red2[lane]);
  const float r = rsqrtf(tot * (1.0f / (float)KH) + KEPS);

#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    const int v = t + k * RTHREADS;
    const V8 g = ldv(gamma + v * 8);
    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o.w[j] = from_bf162(__hmul2(as_bf162(f2bf(mul2f(bf2f(mixed[k].w[j]), make_float2(r, r)))), as_bf162(g.w[j])));
    stv(normed_row + v * 8, o);
  }
}

// ---------------------------------------------------------------- K1a
extern "C" __global__ void __launch_bounds__(RTHREADS, 3) kern_k3g_attnres_rms(
    const bf16_t* __restrict__ prefix, bf16_t* __restrict__ blocks, const float* __restrict__ sw,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int nb, int snapshot, int B) {
  __shared__ RowSmem s;
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  zero_pads(s, t);

  V8 pv[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) pv[k] = ldv(prefix + (size_t)b * KH + (t + k * RTHREADS) * 8);
  if (snapshot && nb < KNB_MAX)
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) stv(blocks + ((size_t)b * KNB_MAX + nb) * KH + (t + k * RTHREADS) * 8, pv[k]);
  V8 x0[RGROUPS];
  load_cand0(blocks + (size_t)b * KNB_MAX * KH, nb, t, x0);

  attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, x0, sw, gamma, normed + (size_t)b * KH, nb, t, s);
}

// ---------------------------------------------------------------- K1a, first layer
// The chunk's embedding gather and layer 0's mix and norm (nb == 0: the mix is the row): hidden and
// snapshot 0 are the embedding row, normed its rms by gamma.
// A stage's setup rides along in its first launch: the MoE routing state (k3_moe_route.cu's barrier and
// tables) zeroed, each block its share of the `words`, and for the packed prefill's stage 0 the FMHA's
// length tables (k3_fmha_plan.cuh, the own set and nw round sets) by block 0.
__device__ __forceinline__ void zero_share(unsigned* __restrict__ p, int words, int B, int t) {
  const int per = (words + B - 1) / B, lo = blockIdx.x * per, hi = min(lo + per, words);
  for (int i = lo + t; i < hi; i += blockDim.x) p[i] = 0u;
}

extern "C" __global__ void __launch_bounds__(RTHREADS, 3) kern_k3g_embed_rms(
    const long long* __restrict__ ids, const bf16_t* __restrict__ table, bf16_t* __restrict__ hidden,
    bf16_t* __restrict__ blocks, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed,
    const int* __restrict__ seq_lens, const long long* __restrict__ cu_q, int* __restrict__ lens, int nseq, int ns,
    int short_max, int window, int nw, unsigned* __restrict__ sync, int sync_words, int B) {
  __shared__ RowSmem s;
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  zero_pads(s, t);
  zero_share(sync, sync_words, B, t);
  if (b == 0)
    for (int k = t; k <= nw; k += blockDim.x) fmha_plan_set(seq_lens, cu_q, lens, nseq, ns, k, window, short_max);

  const bf16_t* __restrict__ row = table + ids[b] * KH;
  V8 pv[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) pv[k] = ldv(row + (t + k * RTHREADS) * 8);
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    stv(hidden + (size_t)b * KH + (t + k * RTHREADS) * 8, pv[k]);
    stv(blocks + (size_t)b * KNB_MAX * KH + (t + k * RTHREADS) * 8, pv[k]);
  }

  V8 x0[RGROUPS];
  attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, x0, nullptr, gamma, normed + (size_t)b * KH, 0, t, s);
}

extern "C" __global__ void __launch_bounds__(RTHREADS, 3) kern_k3g_attnres_rms_setup(
    const bf16_t* __restrict__ prefix, bf16_t* __restrict__ blocks, const float* __restrict__ sw,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int nb, int snapshot, unsigned* __restrict__ sync,
    int sync_words, int B) {
  __shared__ RowSmem s;
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  zero_pads(s, t);
  zero_share(sync, sync_words, B, t);

  V8 pv[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) pv[k] = ldv(prefix + (size_t)b * KH + (t + k * RTHREADS) * 8);
  if (snapshot && nb < KNB_MAX)
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) stv(blocks + ((size_t)b * KNB_MAX + nb) * KH + (t + k * RTHREADS) * 8, pv[k]);
  V8 x0[RGROUPS];
  load_cand0(blocks + (size_t)b * KNB_MAX * KH, nb, t, x0);

  attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, x0, sw, gamma, normed + (size_t)b * KH, nb, t, s);
}

// ---------------------------------------------------------------- K1b
extern "C" __global__ void __launch_bounds__(RTHREADS, 3) kern_k3g_land_add_attnres_rms(
    const bf16_t* __restrict__ partial, const bf16_t* __restrict__ prefix, const bf16_t* __restrict__ blocks,
    const float* __restrict__ sw, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ prefix2,
    bf16_t* __restrict__ normed, int nb, int snapshot, int B) {
  __shared__ RowSmem s;
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  zero_pads(s, t);

  V8 pv[RGROUPS], lp[RGROUPS], x0[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) lp[k] = ldv(partial + (size_t)b * KH + (t + k * RTHREADS) * 8);
  if (!snapshot)
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) pv[k] = ldv(prefix + (size_t)b * KH + (t + k * RTHREADS) * 8);
  load_cand0(blocks + (size_t)b * KNB_MAX * KH, nb, t, x0);
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    if (snapshot) {
#pragma unroll
      for (int j = 0; j < 4; ++j) pv[k].w[j] = f2bf(bf2f(lp[k].w[j]));
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        pv[k].w[j] = add_bf2(pv[k].w[j], lp[k].w[j]);
      }
    }
    stv(prefix2 + (size_t)b * KH + (t + k * RTHREADS) * 8, pv[k]);
  }

  attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, x0, sw, gamma, normed + (size_t)b * KH, nb, t, s);
}

// ---------------------------------------------------------------- K1d
// hidden = bf16(prefix2 + p1 (+ p2)) for one vector, K1c's sum order.
__device__ __forceinline__ V8 add2v(const V8& a, const V8& c, const V8& pr, int two) {
  V8 o;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    o.w[j] = two ? f2bf(add2f(add2f(bf2f(pr.w[j]), bf2f(a.w[j])), bf2f(c.w[j]))) : add_bf2(pr.w[j], a.w[j]);
  }
  return o;
}

__device__ __forceinline__ V8 add2(const bf16_t* __restrict__ p1, const bf16_t* __restrict__ p2,
                                   const bf16_t* __restrict__ prefix2, int two, size_t off) {
  V8 c;
  if (two) c = ldv(p2 + off);
  return add2v(ldv(p1 + off), c, ldv(prefix2 + off), two);
}

extern "C" __global__ void __launch_bounds__(224) kern_k3g_land_add2(
    const bf16_t* __restrict__ p1, const bf16_t* __restrict__ p2, const bf16_t* __restrict__ prefix2,
    bf16_t* __restrict__ hidden, int two, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const size_t off = (size_t)b * KH + (blockIdx.y * 224 + threadIdx.x) * 8;
  stv(hidden + off, add2(p1, p2, prefix2, two, off));
}

extern "C" __global__ void __launch_bounds__(RTHREADS, 3) kern_k3g_land_add2_attnres_rms(
    const bf16_t* __restrict__ p1, const bf16_t* __restrict__ p2, const bf16_t* __restrict__ prefix2,
    bf16_t* __restrict__ hidden, int two, const bf16_t* __restrict__ blocks, const float* __restrict__ sw,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int nb, int B) {
  __shared__ RowSmem s;
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  zero_pads(s, t);

  V8 pv[RGROUPS], a[RGROUPS], c[RGROUPS], x0[RGROUPS];
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    const size_t off = (size_t)b * KH + (t + k * RTHREADS) * 8;
    a[k] = ldv(p1 + off);
    pv[k] = ldv(prefix2 + off);
  }
  if (two)
#pragma unroll
    for (int k = 0; k < RGROUPS; ++k) c[k] = ldv(p2 + (size_t)b * KH + (t + k * RTHREADS) * 8);
  load_cand0(blocks + (size_t)b * KNB_MAX * KH, nb, t, x0);
#pragma unroll
  for (int k = 0; k < RGROUPS; ++k) {
    pv[k] = add2v(a[k], c[k], pv[k], two);
    stv(hidden + (size_t)b * KH + (t + k * RTHREADS) * 8, pv[k]);
  }

  attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, x0, sw, gamma, normed + (size_t)b * KH, nb, t, s);
}

// ---------------------------------------------------------------- K6
#ifndef EXPERTS
#define EXPERTS 896
#endif
#define TOPK 16
#define TK_PER (EXPERTS / 32)
static_assert(EXPERTS % 32 == 0, "experts per lane");

__device__ __forceinline__ u32 ord_f32(float f) {
  const u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Lane l holds experts j * 32 + l. A round takes the largest remaining key
// (REDUX.MAX), then the smallest expert holding it (REDUX.MIN), blanks it, and
// lane r keeps round r's pick.
extern "C" __global__ void __launch_bounds__(128) kern_k3g_router_topk(
    const float* __restrict__ S, const float* __restrict__ bias, const bf16_t* __restrict__ rs,
    int* __restrict__ idx, float* __restrict__ wts, int B) {
  const int lane = threadIdx.x & 31;
  const int b = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (b >= B) return;
  const float* __restrict__ row = S + (long long)b * EXPERTS;

  u32 o[TK_PER];
  float g[TK_PER];
#pragma unroll
  for (int j = 0; j < TK_PER; ++j) {
    const int e = j * 32 + lane;
    const float sg = 1.0f / (1.0f + expf(-row[e]));
    g[j] = sg;
    o[j] = ord_f32(sg + bias[e]);
  }

  float pick_w = 0.f;
  int pick_e = 0;
  for (int r = 0; r < TOPK; ++r) {
    u32 m = o[0];
#pragma unroll
    for (int j = 1; j < TK_PER; ++j) m = max(m, o[j]);
    m = __reduce_max_sync(0xffffffffu, m);
    u32 my = 0xffffffffu;
#pragma unroll
    for (int j = TK_PER - 1; j >= 0; --j)
      if (o[j] == m) my = (u32)(j * 32 + lane);
    const u32 ew = __reduce_min_sync(0xffffffffu, my);
    float gw = 0.f;
    if (my == ew) {
      const int jw = (int)(ew >> 5);
#pragma unroll
      for (int j = 0; j < TK_PER; ++j)
        if (j == jw) {
          gw = g[j];
          o[j] = 0u;
        }
    }
    gw = __shfl_sync(0xffffffffu, gw, (int)(ew & 31u));
    if (lane == r) {
      pick_w = gw;
      pick_e = (int)ew;
    }
  }

  float den = pick_w;
#pragma unroll
  for (int off = TOPK / 2; off > 0; off >>= 1) den += __shfl_xor_sync(0xffffffffu, den, off);
  if (lane < TOPK) {
    wts[b * TOPK + lane] = (pick_w / (den + 1e-20f)) * __bfloat162float(rs[0]);
    idx[b * TOPK + lane] = pick_e;
  }
}

// ---------------------------------------------------------------- MoE
// The row's 16 (permuted row, weight) pairs staged in shared memory, read back
// by every thread (a broadcast) instead of 32 global loads a thread; each pair
// of columns accumulated by one packed fp32 FMA (FFMA2: two IEEE FMAs).
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;"
      : "=l"(d)
      : "l"(*reinterpret_cast<unsigned long long*>(&a)), "l"(*reinterpret_cast<unsigned long long*>(&b)),
        "l"(*reinterpret_cast<unsigned long long*>(&c)));
  return *reinterpret_cast<float2*>(&d);
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3g_finalize_rms(
    const bf16_t* __restrict__ fc2, const int* __restrict__ exp2perm, const float* __restrict__ wts,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ out, int T, int cols) {
  __shared__ float sm[33];
  __shared__ int sp[TOPK];
  __shared__ float sw[TOPK];
  const int t = blockIdx.x, h = threadIdx.x;
  const int lane = h & 31, warp = h >> 5;
  if (h < TOPK && t < T) {
    sp[h] = exp2perm[t * TOPK + h];
    sw[h] = wts[t * TOPK + h];
  }
  __syncthreads();
  float2 acc[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
  if (t < T) {
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int p = sp[k];
      if (p < 0) continue;
      const float w = sw[k];
      const uint4 u = reinterpret_cast<const uint4*>(fc2 + (long long)p * cols)[h];
      const bf162_t* v = reinterpret_cast<const bf162_t*>(&u);
#pragma unroll
      for (int l = 0; l < 4; ++l) acc[l] = fma2(make_float2(w, w), __bfloat1622float2(v[l]), acc[l]);
    }
  }
  bf162_t x[4];
  float sum = 0.0f;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    x[l] = __floats2bfloat162_rn(acc[l].x, acc[l].y);
    const float2 f = __bfloat1622float2(x[l]);
    sum += f.x * f.x;
    sum += f.y * f.y;
  }
  sum = warp_sum(sum);
  if (lane == 0) sm[warp] = sum;
  if (h < 32 && h >= (int)(blockDim.x >> 5)) sm[h] = 0.f;
  __syncthreads();
  if (warp == 0) {
    const float v = warp_sum(sm[lane]);
    if (lane == 0) sm[32] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(sm[32] / (float)cols + 1e-5f);
  const uint4 gw = reinterpret_cast<const uint4*>(gamma)[h];
  const bf162_t* g = reinterpret_cast<const bf162_t*>(&gw);
  uint4 o;
  bf162_t* ov = reinterpret_cast<bf162_t*>(&o);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const float2 f = __bfloat1622float2(x[l]);
    ov[l] = __hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), g[l]);
  }
  reinterpret_cast<uint4*>(out + (long long)t * cols)[h] = o;
}

// ---------------------------------------------------------------- K7
__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// 1 / d without __frcp_rn's slow-path branch, equal to it for every d in [2^-126, 2^126) (k3_moe_route.cu)
__device__ __forceinline__ float rcp_fast(float d) {
  float q;
  asm("{\n .reg .f32 r, e;\n rcp.approx.ftz.f32 r, %1;\n fma.rn.f32 e, %1, r, 0fBF800000;\n neg.f32 e, e;\n"
      " fma.rn.f32 %0, r, e, r;\n}"
      : "=f"(q)
      : "f"(d));
  return q;
}

// k3_land.cu's situ_f on operands already landed; FAST: 1 + e^-g in rcp_fast's range, every g >= -87
template <bool FAST>
__device__ __forceinline__ float situ(float g, float u) {
  const float a = 4.0f * tanh_approx(g * 0.25f);
  const float d = 1.0f + __expf(-g);
  const float s = FAST ? rcp_fast(d) : __frcp_rn(d);
  const float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

template <bool FAST>
__device__ __forceinline__ V8 situ8(const V8& g, const V8& u) {
  V8 o;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 gf = bf2f(g.w[k]), uf = bf2f(u.w[k]);
    o.w[k] = f2bf(make_float2(situ<FAST>(gf.x, uf.x), situ<FAST>(gf.y, uf.y)));
  }
  return o;
}

extern "C" __global__ void __launch_bounds__(256) kern_k3g_situ(const bf16_t* __restrict__ p, bf16_t* __restrict__ act,
                                                                 int n, int B) {
  const int b = blockIdx.x;
  const int j = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  const bool in = b < B && j < n;
  V8 g, u;
  bool ok = true;
  if (in) {
    g = ldv(p + (size_t)b * 2 * n + j);
    u = ldv(p + (size_t)b * 2 * n + n + j);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 gf = bf2f(g.w[k]);
      ok = ok && gf.x >= -87.0f && gf.y >= -87.0f;
    }
  }
  const bool fast = __all_sync(0xffffffffu, ok);
  if (in) stv(act + (size_t)b * n + j, fast ? situ8<true>(g, u) : situ8<false>(g, u));
}
