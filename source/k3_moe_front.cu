// A DCP decode step's MoE front in one launch: everything between the front
// GEMM (router | this rank's LATL lat_down rows | its shared gate | up, one f32 row of LDS
// columns) and the batched GEMMs, and the shared expert's down projection.
//
//   kern_k3_moe_front(const f32* S, const f32* bias, const bf16* rs, i32* idx, f32* wts,
//       u8* q, u8* sf, bf16* act, i32* ready, i32* ids, i32* done, i32* cta_batch, i32* cta_limit,
//       i32* num_non_exiting, i32* total_padded, i32* route_map, i32* exp2perm,
//       const bf16* wsh, bf16* sh_out, const u8* w13, const u8* w13_sf, int first, int T,
//       u8* stage, const u64* peers, i32* lstate, i32* err, i64 timeout_ns)
//   grid (T + NSH / 64, 1, 1)   block (1024, 1, 1)   shared SMEM_SH bytes   T <= ROWS_MAX
//   or grid (T, 1, 1), shared 0: the row blocks alone, the shared down projection left to a GEMM after FC2
//
// Launched as a programmatic dependent of the front GEMM: every block waits for it at entry
// (griddepcontrol.wait), then lets FC1 launch. FC1 waits for the whole grid: past a few dozen rows the
// shared blocks finish well after the routing tables (gen_stage.py launches the row blocks alone there).
//
// Row b's block (b < T):
//   * the router top-k of columns [0, EXPERTS) (k3_router_argmax.cu's picks and
//     order), exact: warp w < EXPERTS / 32 holds 32 biased scores; tau, the 16th
//     largest warp maximum, bounds the top 16 from below; the scores >= tau
//     (16-30 of them on real routing) are compacted and warp 0 sorts them 32 at a
//     time into a running top 16 (below);
//   * meanwhile warps 28..31 run k3_land.cu's land_situ on the shared expert's
//     gate | up (the SH columns after the latent) and publish the row to each
//     shared-expert block T + c (ready[c * ROWS_MAX + b] = 1, st.release.gpu); once the
//     candidates are in, warps 26..27 land this rank's slice of the latent (columns
//     EXPERTS + [0, LATL), the latent's [first * LATL, (first + 1) * LATL); `first` is
//     the rank) to bf16 and quantise it to mxfp8 as k3_moe_prefill.cu's kern_k3_moe_quant
//     (four threads per 32-element group), and warps 1..8 trade the slices of row b
//     with the group (gather, below) into the whole row's q | sf;
// then the last row block to finish (`done`, a zeroed carry it re-zeroes) builds
// the routing tables of this rank's experts `first + j * STRIDE` (j < LOCAL)
// over all T rows: what FlashInfer's routing writes for the batched GEMMs'
// TILE-row CTAs, every expert's rows in the order of shared-memory atomics
// (as FlashInfer's are; the GEMMs' rows do not depend on it), padding rows
// routing token 0, exp2perm -1 for another rank's expert.
// Blocks T + c, when the grid has them: the shared expert's down projection of 64 output columns
// (shared_down, below); the row blocks publish to them only then.
//
//   nvcc -cubin -arch=sm_103a -DEXPERTS=896 -DLDS=2880 -DNRANKS=8 -DSH=768 -DSTRIDE=8 k3_moe_front.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#ifndef LDS
#define LDS 6016
#endif
#ifndef SH
#define SH 768
#endif
#ifndef STRIDE
#define STRIDE 8
#endif
#define LATENT 3584
#ifndef NRANKS
#define NRANKS 8
#endif
#define LATL (LATENT / NRANKS)  // the latent columns of this rank's front
#ifndef NSH
#define NSH 7168  // the shared expert's output width (the hidden size)
#endif
#define ROWS_MAX 256         // the rows a step may take (gen_stage.py's D_SEQS_MAX): the flag tables' stride
#define READERS (NSH / 64)  // the shared-expert blocks: ready[c * ROWS_MAX + b] is row b's flag for block T + c
#define MOE_I 3072           // an expert's intermediate width: FC1 is [2 MOE_I, LATENT] mxfp4 + UE8M0 per 32
#define FC1_W (2 * MOE_I * (LATENT / 2))
#define FC1_SF (2 * MOE_I * (LATENT / 32))
static_assert(FC1_W % (READERS * 16) == 0 && FC1_SF % (READERS * 16) == 0, "16-byte shares");
#define TOPK 16
#define TILE 16
#define LOCAL (EXPERTS / STRIDE)
#define THREADS 1024
#define QUANT_THREADS (LATL / 8)
#define SITU_THREADS (SH / 4)
static_assert(QUANT_THREADS % 4 == 0 && SH % 4 == 0, "whole 32-column groups");
#define IDS_PER ((ROWS_MAX * TOPK + THREADS - 1) / THREADS)  // the expanded ids a thread routes at most
static_assert(LOCAL <= THREADS, "an expert a thread");

typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float situ_f(float pg, float pu) {
  float g = __bfloat162float(__float2bfloat16(pg));
  float u = __bfloat162float(__float2bfloat16(pu));
  float a = 4.0f * tanh_approx(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

// Eight latent columns of a 32-column group (its four threads are adjacent lanes):
// landed to bf16, the group's UE8M0 scale, e4m3 at nearest.
__device__ __forceinline__ void quant8(const float* __restrict__ x, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                       int u) {
  const float4 a = reinterpret_cast<const float4*>(x)[2 * u], c = reinterpret_cast<const float4*>(x)[2 * u + 1];
  const float raw[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
  float v[8];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = __bfloat162float(__float2bfloat16(raw[j]));
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  if ((u & 3) == 0) sf[u >> 2] = (uint8_t)(e + 127);
  uint8_t o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __nv_cvt_float_to_fp8(v[j] * s, __NV_SATFINITE, __NV_E4M3);
  reinterpret_cast<uint2*>(q)[u] = *reinterpret_cast<const uint2*>(o);
}

// The last block's routing tables over T rows' ids, thread t the expanded ids t + k THREADS: each one's
// slot within its expert claimed by a shared-memory atomic (an expert's rows in arrival order, as
// FlashInfer's atomics leave them; the GEMMs compute every row on its own and the combine gathers by exp2perm).
__device__ __forceinline__ void route(const int* __restrict__ idx, int first, int T, int* count,
                                      int* __restrict__ cta_batch, int* __restrict__ cta_limit,
                                      int* __restrict__ num_non_exiting, int* __restrict__ total_padded,
                                      int* __restrict__ route_map, int* __restrict__ exp2perm) {
  __shared__ int off[LOCAL + 1];
  const int i = threadIdx.x, lane = i % 32;
  int js[IDS_PER], slots[IDS_PER];
#pragma unroll
  for (int k = 0; k < IDS_PER; ++k) {
    const int e = i + k * THREADS;
    int j = -1;
    if (e < T * TOPK) {
      const int d = __ldcg(idx + e) - first;
      if (d >= 0 && d % STRIDE == 0) j = d / STRIDE;
    }
    js[k] = j;
    slots[k] = j >= 0 ? atomicAdd_block(count + j, 1) : 0;
  }
  __syncthreads();
  if (i < 32) {  // off = exclusive scan of the experts' CTA counts
    constexpr int EPL = (LOCAL + 31) / 32;
    int c[EPL], sum = 0;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
      c[k] = e < LOCAL ? (count[e] + TILE - 1) / TILE : 0;
      sum += c[k];
    }
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += n;
    }
    int run = incl - sum;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
      if (e < LOCAL) off[e] = run;
      run += c[k];
    }
    if (lane == 31) {
      off[LOCAL] = incl;
      num_non_exiting[0] = incl;
      total_padded[0] = incl * TILE;
    }
  }
  __syncthreads();
  if (i < LOCAL) {
    for (int c = off[i]; c < off[i + 1]; ++c) {
      cta_batch[c] = i;
      cta_limit[c] = min((c + 1) * TILE, off[i] * TILE + count[i]);
    }
    for (int r = off[i] * TILE + count[i]; r < off[i + 1] * TILE; ++r) route_map[r] = 0;
  }
#pragma unroll
  for (int k = 0; k < IDS_PER; ++k) {
    const int e = i + k * THREADS;
    if (e < T * TOPK) {
      int row = -1;
      if (js[k] >= 0) {
        row = off[js[k]] * TILE + slots[k];
        route_map[row] = e / TOPK;
      }
      exp2perm[e] = row;
    }
  }
}

// A warp's 32 keys sorted descending by a bitonic network. The scores' keys are 64 bits: the biased
// score's order bits, then the expert complemented (larger key = larger score, ties to the smaller
// expert: k3_router_argmax.cu's order).
template <typename K>
__device__ __forceinline__ K sort32(K v, int lane) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const K o = __shfl_xor_sync(0xffffffffu, v, j);
      v = (((lane & j) == 0) == ((lane & k) == 0)) ? max(v, o) : min(v, o);
    }
  return v;
}

// Two descending lists of 16 (a in lanes 0-15, b reversed in 16-31: a bitonic sequence) merged
// descending: lanes 0-15 hold the top 16 of both.
__device__ __forceinline__ unsigned long long merge32(unsigned long long v, int lane) {
#pragma unroll
  for (int j = 16; j > 0; j >>= 1) {
    const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, j);
    v = (lane & j) == 0 ? max(v, o) : min(v, o);
  }
  return v;
}

#define SORT_WARPS (EXPERTS / 32)
#define QUANT_WARP0 (SORT_WARPS - (QUANT_THREADS + 31) / 32)
static_assert(EXPERTS % 32 == 0 && TOPK <= SORT_WARPS && SORT_WARPS <= 31 && QUANT_WARP0 > 0,
              "a warp of experts each, at least TOPK of them; a warp left for the SiTU, the latent's after warp 0");

// ---- blocks T .. T + NSH / 64 - 1: the shared expert's down projection ----
// out[t, n] = bf16(sum_k act[t, k] * Wsh[n, k]) for t < T, n < H, k < SH, act the row blocks' SiTU rows.
// Block T + c takes Wsh's rows [64 c, 64 c + 64): they land in shared memory by cp.async at entry
// (warps 8-30), then the T activation rows SROWS at a time once published (ready[c ROWS_MAX + b] = 1 by row
// block b, polled by warps 0-1; each block clears its own flags for the next layer), both K-major in 128-byte
// swizzle atoms. The weight stays put while a chunk of rows after another goes through the same projection.
// One thread issues tcgen05.mma M = 64, N = the rows rounded to 8, K = 16, 48 in a row into TMEM (~25 ns
// each whatever N), every warp moves tiles out. Each output is one chain over k = 0, 16, 32, ... in
// order, f32 accumulators: the order and instruction cuBLASLt's kernels for this shape sum in (nvjet tst
// 64 x 8..48 tiles, no split-K), so the result is cuBLASLt's bit for bit (a harness compares every output
// at 1-64 rows on random and real weights). The row blocks never wait on these and have lower indices, so
// they are all dispatched first: the polls cannot hang.
#define SROWS 64
#define WORKERS (THREADS - 32)  // warp 31 prefetches FC1's first expert (prefetch_first), the rest project
#define work_sync() asm volatile("bar.sync 1, %0;" ::"n"(WORKERS) : "memory")
#define ACOL (SROWS * 128)  // a 64-column swizzle atom column of 64 rows
#define UMMA_OPERAND (SH / 64 * ACOL)
#define SMEM_SH (1024 + 2 * UMMA_OPERAND + 64 * SROWS * 2)
static_assert(SH % 64 == 0, "whole swizzle atoms");

__device__ __forceinline__ uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// row r's 16-byte chunk k in the K-major 128-byte swizzle layout
__device__ __forceinline__ uint32_t swizzled(int r, int k) { return (k / 8) * ACOL + r * 128 + ((k % 8) ^ (r % 8)) * 16; }

// tcgen05 shared-memory descriptor: K-major, 128-byte swizzle, 8-row groups 1 KB apart
__device__ __forceinline__ uint64_t umma_desc(uint32_t at) {
  return (uint64_t)((at & 0x3FFFF) >> 4) | 1ull << 16 | (uint64_t)(1024 >> 4) << 32 | 1ull << 46 | 2ull << 61;
}

// out tile [t][64] = W (64 x SH, at sw) x act (rows x SH, at sa)^T: one thread issues the K steps into
// TMEM columns [0, 8 tiles), then warp w < 31 moves tiles w / 4, w / 4 + 8 (7 in lane quarter 3, whose
// warp 31 prefetches) .. of its lane quarter (rows 16 (w % 4) + [0, 16)).
__device__ __forceinline__ void umma_rows(uint32_t sw, uint32_t sa, uint32_t tm, uint64_t* bar, int tiles,
                                          int parity, __nv_bfloat16* tile) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
    // kind::f16: D f32, A and B bf16, both K-major, N = 8 tiles, M = 64
    const uint32_t idesc = 1u << 4 | 1u << 7 | 1u << 10 | (uint32_t)tiles << 17 | (64u >> 4) << 24;
    const uint64_t da = umma_desc(sw), db = umma_desc(sa);
#pragma unroll
    for (int s = 0; s < SH / 16; ++s) {
      const uint32_t off = ((s / 4) * ACOL + (s % 4) * 32) >> 4;
      asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                   "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                   ::"r"(tm), "l"(da + off), "l"(db + off), "r"(idesc), "r"(s) : "memory");
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_at(bar))
                 : "memory");
  }
  const int q = warp % 4;
  for (int j = warp / 4; j < tiles; j += q == 3 ? WORKERS / 128 : WORKERS / 128 + 1) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}"
                 ::"r"(smem_at(bar)), "r"(parity) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t v[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "r"(tm + ((32 * q) << 16) + 8 * j));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (lane < 16)  // M = 64: D row 16 q + l sits in TMEM lane 32 q + l
#pragma unroll
      for (int i = 0; i < 8; ++i) tile[(8 * j + i) * SROWS + 16 * q + lane] = __float2bfloat16(__uint_as_float(v[i]));
  }
}

// FC1's first CTAs take the touched experts of the lowest local indices: row b publishes its two lowest
// local picks j_b < j2_b (ids[c * ROWS_MAX + b] = 1 + j_b + 256 j2_b, LOCAL for none; the value is the message,
// relaxed); warp 31 of block T + c, apart from the projection from the start, takes the two least over the
// rows (clearing the flags) and pulls its share of those experts' FC1 weights and scales into L2 while HBM
// is otherwise idle.
__device__ __forceinline__ void prefetch_first(int* __restrict__ ids, int T, int c, const uint8_t* __restrict__ w13,
                                               const uint8_t* __restrict__ w13_sf) {
  const int lane = threadIdx.x % 32;
  int a = LOCAL, b2 = LOCAL;  // the lane's two least
  for (int r = lane; r < T; r += 32) {
    int v;
    int* flag = ids + c * ROWS_MAX + r;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    while (v == 0) asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    *flag = 0;
    const int j1 = (v - 1) % 256, j2 = (v - 1) / 256;
    if (j1 < a) { b2 = min(a, j2); a = j1; } else b2 = min(b2, j1 == a ? j2 : j1);
  }
  const int j = __reduce_min_sync(0xffffffffu, a);
  const int k = __reduce_min_sync(0xffffffffu, min(a == j ? b2 : a, b2));
  if (lane < 2 && (lane ? k : j) < LOCAL) {
    const int e = lane ? k : j;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 ::"l"(w13 + (size_t)e * FC1_W + (size_t)c * (FC1_W / READERS)), "r"(FC1_W / READERS) : "memory");
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 ::"l"(w13_sf + (size_t)e * FC1_SF + (size_t)c * (FC1_SF / READERS)), "r"(FC1_SF / READERS) : "memory");
  }
}

__device__ __forceinline__ void shared_down(const __nv_bfloat16* __restrict__ act, int* __restrict__ ready,
                                            const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ out,
                                            int T, int c, int* __restrict__ ids, const uint8_t* __restrict__ w13,
                                            const uint8_t* __restrict__ w13_sf) {
  extern __shared__ __align__(128) uint8_t sm[];  // the weight rows, SROWS activation rows, the bf16 tile [SROWS][64]
  __shared__ __align__(8) uint64_t s_bar;
  __shared__ uint32_t s_tmem;
  const int t = threadIdx.x, warp = t / 32, n0 = c * SROWS;
  if (t >= WORKERS) {
    prefetch_first(ids, T, c, w13, w13_sf);
    return;
  }
  uint8_t* sw = sm + ((1024 - (smem_at(sm) & 1023)) & 1023);  // swizzle atoms 1 KB aligned
  uint8_t* sa = sw + UMMA_OPERAND;
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(sa + UMMA_OPERAND);
  if (t >= 256)
    for (int i = t - 256; i < SROWS * (SH / 8); i += WORKERS - 256)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                   ::"r"(smem_at(sw + swizzled(i / (SH / 8), i % (SH / 8)))), "l"(W + (size_t)n0 * SH + i * 8) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(smem_at(&s_tmem)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (t == 64) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_at(&s_bar)) : "memory");
  // a chunk's operand, tile and accumulators are reused by the next only past the barriers after its stores
  for (int r0 = 0, k = 0; r0 < T; r0 += SROWS, ++k) {
    const int n = min(SROWS, T - r0);
    if (t < n) {
      int v;
      int* flag = ready + c * ROWS_MAX + r0 + t;
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
      while (v != 1) {
        __nanosleep(64);
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
      }
      *flag = 0;
    }
    work_sync();
    const __nv_bfloat16* a = act + (size_t)r0 * SH;
    for (int i = t; i < n * (SH / 8); i += WORKERS)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                   ::"r"(smem_at(sa + swizzled(i / (SH / 8), i % (SH / 8)))), "l"(a + i * 8) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the operands, to the tensor cores' reads
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    work_sync();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    umma_rows(smem_at(sw), smem_at(sa), s_tmem, &s_bar, (n + 7) / 8, k & 1, tile);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    work_sync();
    for (int i = t; i < n * (SROWS / 8); i += WORKERS) {
      const int r = i / (SROWS / 8), col = i % (SROWS / 8);
      reinterpret_cast<uint4*>(out + (size_t)(r0 + r) * NSH + n0)[col] =
          reinterpret_cast<const uint4*>(tile + r * SROWS)[col];
    }
  }
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(s_tmem) : "memory");
}

// ---- a row's mxfp8 latent over the group ----
// Row b's block quantises its rank's LATL columns (LATL / 32 whole groups: the bytes and scales a quantisation
// of the whole row gives them) and pushes them as GVEC 16-byte vectors into slot (rank, b) of every member's
// Lamport stage: GVEC_Q of e4m3 bytes, then the scales one a 16-bit lane. A slot is poisoned when every
// 16-bit lane of it reads 0x8000: a scale lane is at most 0x00fe, and an e4m3 -0 (0x80) is sent as +0, so
// data never reads as poison. (A zero product's sign is all that can differ; FC1 sums 3584 of them.)
// Two stages alternate by call (`lstate[0]`, flipped by the tables block); a member re-poisons each slot as
// it reads it, which is before any peer can come round to that stage again (two calls on, past this
// member's next pushes). A peer's first push waits on nothing: the layer's attention all-reduce before
// this call has seen every member through its once-program, which poisons the stages.
#define GVEC_Q (LATL / 16)
#define GVEC (GVEC_Q + (LATL / 32 + 7) / 8)
#define GSTAGE (NRANKS * ROWS_MAX * GVEC)  // vectors
#define GTHREADS ((SORT_WARPS - 1) * 32)   // the sort warps past warp 0 push and gather
static_assert(LATL % 32 == 0 && NRANKS * GVEC <= GTHREADS, "whole groups, a vector a thread");

__device__ __forceinline__ uint32_t pos_zero(uint32_t w) {
#pragma unroll
  for (int k = 0; k < 32; k += 8)
    if (((w >> k) & 0xffu) == 0x80u) w &= ~(0xffu << k);
  return w;
}

__device__ __forceinline__ bool lane_poisoned(uint32_t w) { return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u; }

__device__ __forceinline__ uint4 ld_volatile4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// Thread k < NRANKS GVEC of the gather: pushes vector k % GVEC of the row's slice to member k / GVEC, then waits
// for vector k % GVEC of member k / GVEC's slice in its own stage, lands it in q / sf and re-poisons it. A
// deadline that passes is a broken group: the member records the late rank in `err` and traps.
__device__ __forceinline__ void gather(const uint8_t* s_q, const uint8_t* s_sf, uint4* stage,
                                       const unsigned long long* peers, long long phase_at, int rank, int b, int k,
                                       uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int* err,
                                       long long timeout_ns) {
  const int r = k / GVEC, v = k % GVEC;
  uint4 o;
  if (v < GVEC_Q) {
    const uint4 x = reinterpret_cast<const uint4*>(s_q)[v];
    o = make_uint4(pos_zero(x.x), pos_zero(x.y), pos_zero(x.z), pos_zero(x.w));
  } else {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int g = (v - GVEC_Q) * 8 + 2 * j;
      w[j] = (g < LATL / 32 ? s_sf[g] : 0u) | (g + 1 < LATL / 32 ? (uint32_t)s_sf[g + 1] << 16 : 0u);
    }
    o = make_uint4(w[0], w[1], w[2], w[3]);
  }
  const long long slot = (long long)(rank * ROWS_MAX + b) * GVEC + v;
  reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[r]) + phase_at)[slot] = o;
  uint4* mine = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(stage) + phase_at) + (long long)(r * ROWS_MAX + b) * GVEC + v;
  uint4 x = ld_volatile4(mine);
  const long long t0 = clock64();
  while (lane_poisoned(x.x) || lane_poisoned(x.y) || lane_poisoned(x.z) || lane_poisoned(x.w)) {
    if (clock64() - t0 > 2 * timeout_ns) {
      atomicMax(err, 1 + r);
      __threadfence_system();
      __trap();
    }
    x = ld_volatile4(mine);
  }
  *mine = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  if (v < GVEC_Q) {
    reinterpret_cast<uint4*>(q + (long long)b * LATENT + r * LATL)[v] = x;
  } else {
    const uint32_t w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (v - GVEC_Q) * 8 + j;
      if (g < LATL / 32) sf[(long long)b * (LATENT / 32) + r * (LATL / 32) + g] = (uint8_t)(w[j / 2] >> (16 * (j % 2)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(THREADS, 1) kern_k3_moe_front(
    const float* __restrict__ S, const float* __restrict__ bias, const __nv_bfloat16* __restrict__ rs,
    int* __restrict__ idx, float* __restrict__ wts, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
    __nv_bfloat16* __restrict__ act, int* __restrict__ ready, int* __restrict__ ids, int* __restrict__ done,
    int* __restrict__ cta_batch,
    int* __restrict__ cta_limit, int* __restrict__ num_non_exiting, int* __restrict__ total_padded,
    int* __restrict__ route_map, int* __restrict__ exp2perm, const __nv_bfloat16* __restrict__ wsh,
    __nv_bfloat16* __restrict__ sh_out, const uint8_t* __restrict__ w13, const uint8_t* __restrict__ w13_sf, int first,
    int T, uint4* stage, const unsigned long long* __restrict__ peers, int* __restrict__ lstate, int* err,
    long long timeout_ns) {
  __shared__ float s_sig[EXPERTS];
  __shared__ unsigned long long s_cand[EXPERTS];
  __shared__ u32 s_max[32], s_tau;
  __shared__ int s_n, s_last, s_jmin, s_count[LOCAL];

  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  if (b >= T) {
    shared_down(act, ready, wsh, sh_out, T, b - T, ids, w13, w13_sf);
    return;
  }
  const float* row = S + (long long)b * LDS;
  if (warp < SORT_WARPS) {
    const float sg = 1.0f / (1.0f + expf(-row[t]));
    s_sig[t] = sg;
    const unsigned long long key = (unsigned long long)ord_f32(sg + bias[t]) << 32 | (u32)~t;
    const u32 m = __reduce_max_sync(0xffffffffu, (u32)(key >> 32));
    if (lane == 0) s_max[warp] = m;
    if (t == 0) s_n = 0;
    if (t < LOCAL) s_count[t] = 0;  // the tables' per-expert counts, should this block be the last
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    // tau, the 16th largest warp maximum of the score bits: at least 16 keys have score bits >= tau, and
    // the keys order by those bits first, so the top 16 are among the keys whose bits are >= tau
    if (warp == 0) {
      const u32 v = sort32(lane < SORT_WARPS ? s_max[lane] : 0u, lane);
      if (lane == TOPK - 1) s_tau = v;
    }
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    const bool in = (u32)(key >> 32) >= s_tau;
    const unsigned ballot = __ballot_sync(0xffffffffu, in);
    int at = 0;
    if (lane == 0 && ballot) at = atomicAdd(&s_n, __popc(ballot));
    at = __shfl_sync(0xffffffffu, at, 0);
    if (in) s_cand[at + __popc(ballot & ((1u << lane) - 1))] = key;
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    if (warp == 0) {
      // the candidates (in no order) 32 at a time: each chunk sorted, its top 16 merged into the running top 16
      const int n = s_n;
      unsigned long long top = 0;
      for (int c = 0; c < n; c += 32) {
        const unsigned long long v = sort32(c + lane < n ? s_cand[c + lane] : 0ull, lane);
        const unsigned long long r = __shfl_sync(0xffffffffu, v, 31 - lane);
        top = c == 0 ? v : merge32(lane < TOPK ? top : r, lane);
      }
      if (lane < TOPK) {
        const int e = (int)~(u32)top;
        const float sg = s_sig[e];
        float den = sg;
#pragma unroll
        for (int off = TOPK / 2; off > 0; off >>= 1) den += __shfl_xor_sync(0x0000ffffu, den, off);
        wts[b * TOPK + lane] = (sg / (den + 1e-20f)) * __bfloat162float(rs[0]);
        idx[b * TOPK + lane] = e;
      }
      const int d = (int)~(u32)top - first, mine = lane < TOPK && d >= 0 && d % STRIDE == 0 ? d / STRIDE : LOCAL;
      const int j = __reduce_min_sync(0xffffffffu, mine), j2 = __reduce_min_sync(0xffffffffu, mine == j ? LOCAL : mine);
      if (lane == 0) s_jmin = j + 256 * j2;
      asm volatile("bar.arrive 4, %0;" ::"n"(32 + THREADS - SORT_WARPS * 32) : "memory");  // to the SiTU warps
    } else {
      // this rank's latent slice to shared memory (the lanes past it quantise columns nobody reads), then to
      // the group
      __shared__ __align__(16) uint8_t s_q[(QUANT_THREADS + 31) / 32 * 32 * 8];
      __shared__ uint8_t s_sf[(QUANT_THREADS + 31) / 32 * 8];
      const long long phase_at = (long long)lstate[0] * GSTAGE * 16;
      if (warp >= QUANT_WARP0) quant8(row + EXPERTS, s_q, s_sf, t - QUANT_WARP0 * 32);
      asm volatile("bar.sync 5, %0;" ::"n"(GTHREADS) : "memory");
      if (t - 32 < NRANKS * GVEC)
        gather(s_q, s_sf, stage, peers, phase_at, first, b, t - 32, q, sf, err, timeout_ns);
    }
  } else {
    // every pass's gate | up loads in flight before the first is used
    constexpr int SITU_LANES = THREADS - SORT_WARPS * 32, PASSES = (SITU_THREADS + SITU_LANES - 1) / SITU_LANES;
    float4 g[PASSES], u[PASSES];
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      const int k = t - SORT_WARPS * 32 + p * SITU_LANES;
      if (k < SITU_THREADS) {
        g[p] = reinterpret_cast<const float4*>(row + EXPERTS + LATL)[k];
        u[p] = reinterpret_cast<const float4*>(row + EXPERTS + LATL + SH)[k];
      }
    }
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      const int k = t - SORT_WARPS * 32 + p * SITU_LANES;
      if (k < SITU_THREADS) {
        __nv_bfloat162 o[2];
        o[0] = __floats2bfloat162_rn(situ_f(g[p].x, u[p].x), situ_f(g[p].y, u[p].y));
        o[1] = __floats2bfloat162_rn(situ_f(g[p].z, u[p].z), situ_f(g[p].w, u[p].w));
        reinterpret_cast<uint2*>(act + (long long)b * SH)[k] = *reinterpret_cast<const uint2*>(o);
      }
    }
    // the row's activation published to the shared-expert blocks (release: cumulative
    // over the SiTU warps' stores, ordered before it by the barrier)
    asm volatile("bar.sync 3, %0;" ::"n"(THREADS - SORT_WARPS * 32));
    const int readers = gridDim.x > T ? READERS : 0;
    for (int c = t - SORT_WARPS * 32; c < readers; c += THREADS - SORT_WARPS * 32)  // a flag per reader CTA
      asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(ready + c * ROWS_MAX + b), "r"(1) : "memory");
    // then the row's lowest local pick to each shared-expert block's prefetcher
    asm volatile("bar.sync 4, %0;" ::"n"(32 + THREADS - SORT_WARPS * 32) : "memory");
    for (int c = t - SORT_WARPS * 32; c < readers; c += THREADS - SORT_WARPS * 32)
      asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(ids + c * ROWS_MAX + b), "r"(s_jmin + 1) : "memory");
  }

  __syncthreads();
  if (t == 0) {  // releases the block's ids (ordered before by the barrier), acquires the others'
    int n;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(n) : "l"(done) : "memory");
    s_last = n == T - 1;
  }
  __syncthreads();
  if (!s_last) return;
  route(idx, first, T, s_count, cta_batch, cta_limit, num_non_exiting, total_padded, route_map, exp2perm);
  if (t == 0) {
    *done = 0;
    lstate[0] ^= 1;
  }
}
