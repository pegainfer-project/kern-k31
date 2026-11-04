// D's K = 7168 f32-out projections (qkvg, front) over 1-48 rows in cuBLAS's own order, the split-K reduce in the
// same launch: row-major C[m, n] = A[m, K] @ W[N, K]^T, bf16 operands, f32 accumulation and result, m <= BN.
//
// cuBLAS (cublasGemmEx f32: nvjet split-K + splitKreduce_kernel) splits K at fixed 64-column blocks (b1, b2),
// accumulates each split in one tcgen05.mma chain (kind::f16, K 16 a step) and adds the splits in order,
// ((p0 + p1) + p2). This kernel does the same, bit for bit: CTA (tile, split) runs split `split`'s chain for BM
// weight rows (W the M operand, the rows of A the N = BN one), and the SPLITS CTAs of a tile are one cluster:
// each pushes row r of its partial into the shared memory of CTA r % SPLITS, one cluster barrier, and each adds
// its rows' partials in split order. BM = 64: 8 KB weight stages, two or three CTAs an SM (CTA_KB), so every
// cluster of three is resident at once.
//
// W and A come by TMA (128-byte swizzle, 64-column boxes); the first STAGES weight boxes are requested before
// griddepcontrol.wait (only the weight is read before it), so under PDL they stream while the kernel before
// this one finishes. `unpin` (a weight an earlier call pinned) goes back to evict_normal from warps 2-3 before the
// wait; `pin` (a weight read a few kernels on) goes into L2 at evict_last, one bulk prefetch a CTA, at the end.
//
//   kern_k3_gemm_f32(tmap W [N, K] box 64 x BM sw128, tmap A [rows, K] box 64 x BN sw128, f32* C, int ldc,
//       int m, int N, int K, int b1, int b2, const u8* pin, i64 pin_bytes, const u8* unpin, i64 unpin_bytes)
//     grid (ceil(N / BM) * SPLITS, 1, 1)   cluster (SPLITS, 1, 1)   block (128, 1, 1)   PDL
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef BN
#define BN 16
#endif
#ifndef SPLITS
#define SPLITS 3
#endif
#ifndef BM
#define BM 64
#endif
#define BK 64
#define W_BYTES (BM * BK * 2)
#define A_BYTES (BN * BK * 2)
#define RPC ((BN + SPLITS - 1) / SPLITS)  // rows of C a CTA of the cluster sums: r % SPLITS == its split
#define RQ ((RPC + 3) / 4 * 4)
// three CTAs an SM (CTA_KB 73; two at 100): a cluster of SPLITS at one CTA an SM leaves too few clusters resident
#ifndef CTA_KB
#define CTA_KB (BM == 64 ? 73 : 110)
#endif
#define CTA_SMEM (CTA_KB * 1024)
#ifndef STAGES
#define STAGES ((CTA_SMEM - SPLITS * BM * RQ * 4) / (W_BYTES + A_BYTES))
#endif
#define TMEM_COLS (BN <= 32 ? 32 : 64)
static_assert(BN % (BM / 8) == 0 && BN <= 64, "UMMA N: a multiple of 8 (M = 64) or 16 (M = 128)");
static_assert(BM == 64 || BM == 128, "UMMA M");
static_assert(SPLITS >= 1 && SPLITS <= 3, "bounds b1, b2");


struct __align__(1024) Smem {
  uint8_t w[STAGES][W_BYTES];
  uint8_t a[STAGES][A_BYTES];
  float red[SPLITS][BM][RQ];  // split p's partial of this CTA's rows: [p][weight row][j], row of C split + j SPLITS
  uint64_t full[STAGES], empty[STAGES], done;
  uint32_t tmem;
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(b)), "r"(n));
}

__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(b)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W%=;\n}" ::"r"(su32(b)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          su32(dst)),
      "l"(map), "r"(su32(bar)), "r"(x), "r"(y)
      : "memory");
}

// K-major, 128-byte swizzle: rows of 128 B, 8-row atoms 1024 B apart.
__device__ __forceinline__ uint64_t sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
}

// kind::f16: f32 D, bf16 A and B, both K-major, M = BM, N = BN.
constexpr uint32_t IDESC = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);

__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(IDESC), "r"(acc));
}

__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(su32(bar)) : "memory");
}

__device__ __forceinline__ void tmem_ld8(uint32_t addr, float* v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(addr));
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  return remote;
}

__device__ __forceinline__ void st_dsmem4(uint32_t addr, const float* v) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3])
               : "memory");
}

extern "C" __global__ void __launch_bounds__(128, 1)
    kern_k3_gemm_f32(const __grid_constant__ CUtensorMap tw, const __grid_constant__ CUtensorMap ta, float* __restrict__ C,
                     int ldc, int m, int N, int K, int b1, int b2, const uint8_t* __restrict__ pin,
                     long long pin_bytes, const uint8_t* __restrict__ unpin, long long unpin_bytes) {
  asm volatile("griddepcontrol.launch_dependents;");
#ifdef GF_TIMES
  unsigned long long t0, t1 = 0, t2;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
#endif
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tile = blockIdx.x / SPLITS, split = (int)cluster_rank();
  const int nkb_all = K / BK;
  const int bounds[4] = {0, SPLITS == 1 ? nkb_all : b1, SPLITS == 2 ? nkb_all : b2, nkb_all};
  const int kb0 = bounds[split], nkb = bounds[split + 1] - kb0;
#define LOAD_W(dst, i, bar) tma_2d(dst, &tw, bar, (kb0 + (i)) * BK, tile * BM)

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) mbar_init(&s.full[i], 1), mbar_init(&s.empty[i], 1);
    mbar_init(&s.done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&ta));
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tw));
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&s.tmem)), "n"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = s.tmem;

  if (warp == 0) {
    if (lane == 0) {
      const int pre = nkb < STAGES ? nkb : STAGES;
      for (int i = 0; i < pre; ++i) {
        mbar_expect(&s.full[i], W_BYTES + A_BYTES);
        LOAD_W(s.w[i], i, &s.full[i]);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < pre; ++i) tma_2d(s.a[i], &ta, &s.full[i], (kb0 + i) * BK, 0);
      for (int i = pre; i < nkb; ++i) {
        const int st = i % STAGES;
        mbar_wait(&s.empty[st], ((i / STAGES) - 1) & 1);
        mbar_expect(&s.full[st], W_BYTES + A_BYTES);
        LOAD_W(s.w[st], i, &s.full[st]);
        tma_2d(s.a[st], &ta, &s.full[st], (kb0 + i) * BK, 0);
      }
    }
  } else if (warp >= 2) {
    // warps 2-3: put `unpin` (a weight an earlier call pinned) back to evict_normal before the wait
    const long long t = (long long)blockIdx.x * 64 + threadIdx.x - 64, step = (long long)gridDim.x * 64 * 128;
    for (long long o = t * 128; o < unpin_bytes; o += step)
      asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(unpin + o));
  } else if (warp == 1) {
    // the whole warp walks the loop (uniform descriptors), one elected lane issues every MMA and commit
    uint32_t leader;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(leader));
    for (int i = 0; i < nkb; ++i) {
      const int st = i % STAGES;
      mbar_wait(&s.full[st], (i / STAGES) & 1);
#ifdef GF_TIMES
      if (i == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
#endif
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t wa = su32(s.w[st]), aa = su32(s.a[st]);
      if (leader) {
#ifdef GF_NOMMA
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(&s.empty[st])) : "memory");
        (void)wa, (void)aa;
#else
#pragma unroll
        for (int k = 0; k < BK / 16; ++k) umma(tmem, sw128(wa + k * 32), sw128(aa + k * 32), i | k);
        umma_commit(&s.empty[st]);
#endif
      }
      __syncwarp();
    }
    if (leader) umma_commit(&s.done);
    __syncwarp();
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  mbar_wait(&s.done, 0);
#ifdef GF_TIMES
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t2));
  if (threadIdx.x == 32) {
    unsigned long long* tt = reinterpret_cast<unsigned long long*>(C + 56 * (size_t)ldc) + blockIdx.x * 4;
    tt[0] = t0, tt[1] = t1, tt[2] = t2;
  }
#endif
  asm volatile("tcgen05.fence::after_thread_sync;");
  // D row i: TMEM lane i (M = 128), lane 32 (i / 16) + i % 16 (M = 64)
  float v[BN];
#pragma unroll
  for (int c = 0; c < BN; c += 8) tmem_ld8(tmem + ((warp * 32) << 16) + c, v + c);
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  const int row = BM == 128 ? warp * 32 + lane : warp * 16 + lane;
  const bool mine = BM == 128 || lane < 16;

  // row r of this split's partial goes to the red[split] of the cluster's CTA r % SPLITS; one barrier, then each
  // CTA adds its rows' partials in split order
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "n"(TMEM_COLS));
  const int n = tile * BM + row;
  if (mine) {
#pragma unroll
    for (int c = 0; c < SPLITS; ++c) {
      float t[RQ];
#pragma unroll
      for (int j = 0; j < RQ; ++j) t[j] = c + j * SPLITS < BN ? v[c + j * SPLITS] : 0.f;
      const uint32_t dst = SPLITS > 1 ? mapa(su32(&s.red[split][row][0]), c) : su32(&s.red[split][row][0]);
#pragma unroll
      for (int q = 0; q < RQ; q += 4) st_dsmem4(dst + q * 4, t + q);
    }
  }
  if (SPLITS > 1) cluster_sync();
  else __syncthreads();
  if (mine) {
    float pv[SPLITS][RQ];
#pragma unroll
    for (int p = 0; p < SPLITS; ++p)
#pragma unroll
      for (int q = 0; q < RQ; q += 4) *reinterpret_cast<float4*>(&pv[p][q]) = *reinterpret_cast<const float4*>(&s.red[p][row][q]);
#pragma unroll
    for (int j = 0; j < RPC; ++j) {
      const int r = split + j * SPLITS;
      float acc = pv[0][j];
#pragma unroll
      for (int p = 1; p < SPLITS; ++p) acc += pv[p][j];
      if (r < m && n < N) C[(size_t)r * ldc + n] = acc;
    }
  }
#ifdef GF_TIMES
  if (threadIdx.x == 32) {
    unsigned long long t3;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t3));
    reinterpret_cast<unsigned long long*>(C + 56 * (size_t)ldc)[blockIdx.x * 4 + 3] = t3;
  }
#endif
  // `pin`: a weight read a few kernels on (past the MoE's stream), pulled into L2 at evict_last, a CTA's share by one
  // bulk prefetch once its stores are out
  if (pin_bytes && threadIdx.x == 0) {
    const long long per = (pin_bytes / gridDim.x + 15) & ~15ll, o = (long long)blockIdx.x * per;
    if (o < pin_bytes) {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(pin + o),
                   "r"((unsigned)(per < pin_bytes - o ? per : pin_bytes - o)), "l"(pol)
                   : "memory");
    }
  }
}
