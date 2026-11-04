// K6 -- router top-k + f32 argmax (kern K3 decode kernel set).
//
//   nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
//        -o target/cubins/k3_router_argmax.cubin tools/kernels-src/k3_router_argmax.cu
//
// EXPERTS (896) is a multiple of 32; the router block is EXPERTS rounded up to
// 256 threads.
//
// Entries (docs/k3-kernel-abi.md section K6):
//
//   extern "C" __global__ void kern_k3_router_topk(
//       const f32*  S,      // [B, LDS]  router GEMM f32 partial in columns 0 .. EXPERTS (LDS = EXPERTS)
//       const f32*  bias,   // [EXPERTS]
//       const bf16* rs,     // [1]  routed_scaling
//       int*        idx,    // [B, TOPK=16]
//       f32*        wts,    // [B, TOPK]
//       int B);
//   grid (B, 1, 1)   block ROUTER_THREADS   smem 8 * EXPERTS + 128 B static, 0 dynamic
//
//   extern "C" __global__ void kern_k3_argmax_f32_partial(
//       const f32* logits, f32* pmax, int* pidx, int n);
//   grid (B, PARTS=64)  block 1024  smem 256 B static, 0 dynamic
//
//   extern "C" __global__ void kern_k3_argmax_f32_final(
//       const f32* pmax, const int* pidx, i64* out, int parts);
//   grid (B, 1, 1)   block 64   smem 256 B static, 0 dynamic
//
// Math / landing points
// ---------------------
// router:  sig[e]    = 1 / (1 + expf(-S[b,e]))                      (f32, no landing)
//          biased[e] = sig[e] + bias[e]                             (f32)
//          16 sequential max picks over `biased`, ties -> smallest e; each pick
//          removes its entry (pegainfer writes -1e30, we drop its key to 0 --
//          identical because sig in [0,1] and bias is finite, so biased > -1e30).
//          wts[t]    = sig[idx[t]] / (sum_t sig[idx[t]] + 1e-20) * f32(rs[0])
//          The only landing is f32(rs[0]); idx/wts stay int/f32. expf (not the
//          fast __expf) is deliberate: idx must match a CPU reference exactly,
//          and a 1e-7 wobble in sig can flip a near-tie.
//
// argmax:  pure f32 compare over the row, tie -> smallest index. No landing;
//          `out` is the i64 token id. Two stage: PARTS partial argmaxes over
//          contiguous chunks (any partition is fine -- stage 2 re-applies the
//          global "largest value, then smallest index" rule over the partials).
//
// Implementation notes
// --------------------
// * Order-preserving keys. Both kernels reduce (value, index) pairs. ord() is
//   the standard monotone f32 -> u32 map, so "larger value, then smaller index"
//   is a plain integer max, with the tie rule structural rather than branchy.
// * router: the 16 picks are strictly sequential, so the round latency is the
//   whole kernel. A 64-bit key would cost a 5-step __shfl_xor chain = 10 SHFL
//   per round; instead each round is two *single-instruction* warp reductions
//   (REDUX.MAX on the 32-bit ord, then REDUX.MIN on the expert id of whoever
//   matched). Measured 2.9 us vs 3.7 us for the 64-bit-key shuffle version at
//   B=1, bit-identical idx. Experts are laid out P=7 per lane (e = lane*7 + j)
//   so "smallest lane, then smallest slot" *is* "smallest e", and the smem read
//   at stride 7 is bank-conflict free (gcd(7,32) = 1).
// * argmax stage 1: 2 float4 (32 B) per thread and __launch_bounds__(1024, 2).
//   The occupancy hint is load-bearing: at 36 registers only one 1024-thread
//   block fits per SM (2048 threads/SM is the hardware cap) and the kernel runs
//   40 us at B=64; at <=32 registers two fit and it runs 20.5 us.
#include <cuda_bf16.h>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#define TOPK 16
#define ROUTER_LANES 32
#define ROUTER_PER_LANE (EXPERTS / ROUTER_LANES) /* 28 */
#define ROUTER_THREADS ((EXPERTS + 255) / 256 * 256)
// S's row stride: its own width, or a fused GEMM's whose first columns it is.
#ifndef LDS
#define LDS EXPERTS
#endif
static_assert(EXPERTS % ROUTER_LANES == 0, "experts per lane");

typedef unsigned int u32;
typedef unsigned long long u64;

// monotone f32 -> u32 (total order on non-NaN floats)
__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float unord_f32(u32 o) {
  return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}

// max over o[L, L + N) as a tree split at the largest power of two below N:
// for N = 7 the ((0 1)(2 3))((4 5) 6) order.
template <int L, int N>
__device__ __forceinline__ u32 tree_max(const u32 (&o)[ROUTER_PER_LANE]) {
  if constexpr (N == 1) {
    return o[L];
  } else {
    constexpr int h = N > 4 ? (N > 8 ? (N > 16 ? 16 : 8) : 4) : (N > 2 ? 2 : 1);
    return max(tree_max<L, h>(o), tree_max<L + h, N - h>(o));
  }
}

// ---------------------------------------------------------------- router top-k
//
// Phase 1 (ROUTER_THREADS threads, EXPERTS active): one expert per thread -- coalesced
// 896 B loads of S and bias, sigmoid, sig + ord(biased) to smem.
// Phase 2 (warp 0 only): each lane owns experts t*7 .. t*7+6 in registers.
// Per round: 6 register maxes, REDUX.MAX over the warp, an unrolled scan for
// this lane's smallest matching slot, REDUX.MIN over the warp, and the winner
// lane blanks that slot and stages (e, sig) in smem. No dynamic indexing into
// the register arrays -- that would spill to local memory.
extern "C" __global__ void __launch_bounds__(ROUTER_THREADS, 1) kern_k3_router_topk(
    const float* __restrict__ S, const float* __restrict__ bias,
    const __nv_bfloat16* __restrict__ rs, int* __restrict__ idx,
    float* __restrict__ wts, int B) {
  __shared__ float s_sig[EXPERTS];
  __shared__ u32 s_ord[EXPERTS];
  __shared__ int s_e[TOPK];
  __shared__ float s_w[TOPK];

  const int b = blockIdx.x;
  const int t = threadIdx.x;
  if (b >= B) return;

  if (t < EXPERTS) {
    float sg = 1.0f / (1.0f + expf(-S[(long long)b * LDS + t]));
    s_sig[t] = sg;
    s_ord[t] = ord_f32(sg + bias[t]);
  }
  __syncthreads();

  if (t < ROUTER_LANES) {
    u32 o[ROUTER_PER_LANE];
    float g[ROUTER_PER_LANE];
#pragma unroll
    for (int j = 0; j < ROUTER_PER_LANE; ++j) {
      o[j] = s_ord[t * ROUTER_PER_LANE + j];
      g[j] = s_sig[t * ROUTER_PER_LANE + j];
    }

    for (int r = 0; r < TOPK; ++r) {
      u32 m = tree_max<0, ROUTER_PER_LANE>(o);
      m = __reduce_max_sync(0xffffffffu, m);  // largest remaining biased value

      u32 my = 0xffffffffu;  // this lane's smallest expert holding that value
#pragma unroll
      for (int j = ROUTER_PER_LANE - 1; j >= 0; --j)
        if (o[j] == m) my = (u32)(t * ROUTER_PER_LANE + j);
      u32 ew = __reduce_min_sync(0xffffffffu, my);  // tie -> smallest e

      if (my == ew) {  // exactly one lane
        int jw = (int)ew - t * ROUTER_PER_LANE;
        float gw = 0.0f;
#pragma unroll
        for (int j = 0; j < ROUTER_PER_LANE; ++j)
          if (j == jw) {
            gw = g[j];
            o[j] = 0u;  // strictly below ord() of any real float
          }
        s_e[r] = (int)ew;
        s_w[r] = gw;
      }
    }
    __syncwarp();

    if (t < TOPK) {
      float sg = s_w[t];
      float den = sg;
#pragma unroll
      for (int off = TOPK / 2; off > 0; off >>= 1)
        den += __shfl_xor_sync(0x0000ffffu, den, off);
      wts[b * TOPK + t] = (sg / (den + 1e-20f)) * __bfloat162float(rs[0]);
      idx[b * TOPK + t] = s_e[t];
    }
  }
}

// ------------------------------------------------------------- argmax stage 1
#define ARGMAX_IDX_TOP 0x7fffffff

__device__ __forceinline__ u64 amax_key(float v, int i) {
  return ((u64)ord_f32(v) << 32) | (u32)(ARGMAX_IDX_TOP - i);
}
__device__ __forceinline__ u64 amax_key4(float4 x, int base) {
  u64 a = max(amax_key(x.x, base), amax_key(x.y, base + 1));
  u64 c = max(amax_key(x.z, base + 2), amax_key(x.w, base + 3));
  return max(a, c);
}

template <int BLOCK>
__device__ __forceinline__ u64 block_max_u64(u64 v, u64* sm) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = BLOCK / 32;
  if (nwarps == 1) return v;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < nwarps) ? sm[lane] : 0ull;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

extern "C" __global__ void __launch_bounds__(1024, 2) kern_k3_argmax_f32_partial(
    const float* __restrict__ logits, float* __restrict__ pmax,
    int* __restrict__ pidx, int n) {
  __shared__ u64 sm[32];

  const int parts = gridDim.y;
  const int chunk = (n + parts - 1) / parts;
  const int lo = blockIdx.y * chunk;
  const int hi = min(lo + chunk, n);
  const float* __restrict__ row = logits + (long long)blockIdx.x * n;

  // n % 4 == 0 and chunk % 4 == 0 make lo and hi multiples of 4, so the
  // vector path covers the chunk exactly -- no ragged tail. lo >= hi (a chunk
  // past the end of a short row) makes nv negative and both loops empty.
  u64 best = 0ull;
  if (((n | chunk) & 3) == 0) {  // 16 B aligned chunk start and row stride
    const float4* v4 = (const float4*)(row + lo);
    const int nv = (hi - lo) >> 2;
    for (int u = threadIdx.x * 2; u < nv; u += blockDim.x * 2) {
      best = max(best, amax_key4(v4[u], lo + (u << 2)));
      if (u + 1 < nv) best = max(best, amax_key4(v4[u + 1], lo + ((u + 1) << 2)));
    }
  } else {
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x)
      best = max(best, amax_key(row[j], j));
  }

  best = block_max_u64<1024>(best, sm);
  if (threadIdx.x == 0) {
    const int o = blockIdx.x * parts + blockIdx.y;
    if (best == 0ull) {  // empty chunk: neutral element for stage 2
      pmax[o] = -__builtin_inff();
      pidx[o] = ARGMAX_IDX_TOP;
    } else {
      pmax[o] = unord_f32((u32)(best >> 32));
      pidx[o] = ARGMAX_IDX_TOP - (int)(u32)(best & 0xffffffffull);
    }
  }
}

// ------------------------------------------------------------- argmax stage 2
extern "C" __global__ void __launch_bounds__(64, 1) kern_k3_argmax_f32_final(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    long long* __restrict__ out, int parts) {
  __shared__ u64 sm[32];

  const long long base = (long long)blockIdx.x * parts;
  u64 best = 0ull;
  for (int t = threadIdx.x; t < parts; t += blockDim.x) {
    int i = pidx[base + t];
    if (i != ARGMAX_IDX_TOP) best = max(best, amax_key(pmax[base + t], i));
  }
  best = block_max_u64<64>(best, sm);
  if (threadIdx.x == 0)
    out[blockIdx.x] =
        (best == 0ull) ? 0ll
                       : (long long)(ARGMAX_IDX_TOP - (int)(u32)(best & 0xffffffffull));
}

// ------------------------------------------------------------- argmax, one launch
//   extern "C" __global__ void kern_k3_argmax_f32_fused(
//       const f32* logits, i64* out, int* done, u64* pkey, int n);
//   grid (B, ARGMAX_PARTS)  block ARGMAX_THREADS  smem 256 B static
// Stage 1 and 2 above in one grid: block (b, p) takes the p-th of ARGMAX_PARTS contiguous
// chunks of row b (every thread busy: 10 float4 at n = 163840), writes its (value, index)
// key, and the row's last block to count itself in `done[b]` (a zeroed carry it re-zeroes)
// takes the max of the row's keys. Keys order "larger value, then smaller index", so any
// partition gives the stage-1/2 result.
#define ARGMAX_PARTS 16
#define ARGMAX_THREADS 256

extern "C" __global__ void __launch_bounds__(ARGMAX_THREADS) kern_k3_argmax_f32_fused(
    const float* __restrict__ logits, long long* __restrict__ out, int* __restrict__ done,
    u64* __restrict__ pkey, int n) {
  __shared__ u64 sm[32];
  __shared__ int s_last;
  const int b = blockIdx.x, p = blockIdx.y;
  const int chunk = (n + ARGMAX_PARTS - 1) / ARGMAX_PARTS;
  const int lo = p * chunk, hi = min(lo + chunk, n);
  const float* __restrict__ row = logits + (long long)b * n;
  u64 best = 0ull;
  if (((n | chunk) & 3) == 0) {
    const float4* v4 = (const float4*)(row + lo);
    const int nv = (hi - lo) >> 2;
    for (int u = threadIdx.x; u < nv; u += ARGMAX_THREADS) best = max(best, amax_key4(v4[u], lo + (u << 2)));
  } else {
    for (int j = lo + threadIdx.x; j < hi; j += ARGMAX_THREADS) best = max(best, amax_key(row[j], j));
  }
  best = block_max_u64<ARGMAX_THREADS>(best, sm);
  if (threadIdx.x == 0) {
    pkey[b * ARGMAX_PARTS + p] = best;
    __threadfence();
    s_last = atomicAdd(done + b, 1) == ARGMAX_PARTS - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (threadIdx.x < 32) {
    __threadfence();
    u64 k = threadIdx.x < ARGMAX_PARTS ? __ldcg(pkey + b * ARGMAX_PARTS + threadIdx.x) : 0ull;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) k = max(k, __shfl_xor_sync(0xffffffffu, k, off));
    if (threadIdx.x == 0) {
      out[b] = k == 0ull ? 0ll : (long long)(ARGMAX_IDX_TOP - (int)(u32)(k & 0xffffffffull));
      done[b] = 0;
    }
  }
}
