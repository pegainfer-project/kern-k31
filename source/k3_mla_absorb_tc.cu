// K3 MLA decode, absorb on tcgen05: k3_mla_absorb_mma.cu's math (docs/k3-kernel-abi.md K5a) for a decode batch of
// at most 64 rows, one block a head.
//
//   extern "C" __global__ void kern_k3_mla_absorb_tc(
//       CUtensorMap w_map,       // w_kv_b [HEADS*256, 512] bf16: box 64 columns x 128 rows, 128 B swizzle
//       const f32*  q_partial,   // [B, HEADS*192]  f32 partial, nope 128 | rope 64
//       bf16*       q_abs,       // [B, HEADS, 576]  latent 512 | rope 64
//       int B);                  // B <= 64
//
//   grid  (HEADS, 1, 1)   block (256, 1, 1)   smem SMEM_BYTES dynamic   one block an SM (TMEM)
//
// Block h computes q_abs^T for its head: D[j][b] = sum_d W_UK_h[d][j] * q_h[b][d] as four M = 128 column groups,
// N = the rows rounded up to 16 (TMEM columns), K = 128 in eight k16 MMAs each, issued by one thread as the
// group's two TMA boxes land. A is W_UK as the boxes leave it: 64 columns x 128 d, columns contiguous, 128 B
// swizzle (MN-major: 8 d a 1 KB atom, atoms 1 KB apart, the group's two boxes 16 KB apart); B is the bf16 query
// (K-major, 128 B swizzle, two 64-d chunks of 64 rows), loaded once for the head. Rows past B are zero and never
// stored. Each thread then reads one column's rows from TMEM. The rope columns go straight through registers.
//
//   q_h     = bf16(q_partial[b, h*192 .. +192])
//   q_abs   = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) for j<512 | q_h[128..192] ]   f32 acc
//
// The products and their k16 steps are mma.sync's in k3_mla_absorb_mma.cu (which the fp8 query still runs),
// and so are the bits.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#define HEADS 96
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define BMAX 64
#define THREADS 256
#define GROUPS (LAT / 128)
#define BOX (64 * NOPE * 2)    // 16 KB: 64 columns x 128 d
#define Q_CHUNK (BMAX * 128)   // 8 KB: 64 rows x 64 d
#define SMEM_BYTES (1024 + 2 * GROUPS * BOX + 2 * Q_CHUNK + 128)

__device__ __forceinline__ uint64_t umma_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  // start >> 4 | LBO >> 4 << 16 | SBO >> 4 << 32 | version 1 << 46 | 128 B swizzle (2) << 61
  return (uint64_t)((addr >> 4) & 0x3FFF) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 | (uint64_t)((sbo >> 4) & 0x3FFF) << 32 |
         (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void wait(uint32_t bar) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}" ::"r"(bar) : "memory");
}

extern "C" __global__ void __launch_bounds__(THREADS, 1) kern_k3_mla_absorb_tc(const __grid_constant__ CUtensorMap w_map,
                                                                       const float* __restrict__ q_partial,
                                                                       __nv_bfloat16* __restrict__ q_abs, int B) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  const uint32_t ws = base, qs = ws + 2 * GROUPS * BOX, full = qs + 2 * Q_CHUNK, done = full + 8 * GROUPS,
                 hold = done + 8 * GROUPS;
  uint8_t* const gen = smem_raw + (base - raw);
  const int h = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int n = (B + 15) & ~15;
  if (t == 0) {
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(full + 8 * g));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(done + 8 * g));
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(full + 8 * g), "r"(2 * BOX));
#pragma unroll
      for (int i = 2 * g; i < 2 * g + 2; ++i)
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     ::"r"(ws + i * BOX), "l"(&w_map), "r"(64 * i), "r"(h * 256), "r"(full + 8 * g) : "memory");
    }
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(hold), "r"(GROUPS * 64));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  // the query: 64 rows x 128 f32, 8 float4 a thread, and the rope: 64 rows x 64, 4 float4 a thread
  float4 qv[BMAX * NOPE / 4 / THREADS], rv[BMAX * ROPE / 4 / THREADS];
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d = i % (NOPE / 4) * 4;
    qv[k] = r < B ? *reinterpret_cast<const float4*>(q_partial + (size_t)r * (HEADS * QW) + h * QW + d) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
#pragma unroll
  for (int k = 0; k < BMAX * ROPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (ROPE / 4), d = i % (ROPE / 4) * 4;
    if (r < B) rv[k] = *reinterpret_cast<const float4*>(q_partial + (size_t)r * (HEADS * QW) + h * QW + NOPE + d);
  }
  // q: row r, d .. +4 is half of chunk (d % 64) / 8 ^ (r % 8) of row r in d chunk d / 64
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d = i % (NOPE / 4) * 4;
    if (r < n) {
      const __nv_bfloat162 lo = __floats2bfloat162_rn(qv[k].x, qv[k].y), hi = __floats2bfloat162_rn(qv[k].z, qv[k].w);
      *reinterpret_cast<uint2*>(gen + 2 * GROUPS * BOX + (d / 64) * Q_CHUNK + r * 128 + (((d % 64) / 8) ^ (r % 8)) * 16 + (d % 8) * 2) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
    }
  }
#pragma unroll
  for (int k = 0; k < BMAX * ROPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (ROPE / 4), d = i % (ROPE / 4) * 4;
    if (r < B) {
      __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(q_abs + ((size_t)r * HEADS + h) * ROW + LAT + d);
      o[0] = __floats2bfloat162_rn(rv[k].x, rv[k].y);
      o[1] = __floats2bfloat162_rn(rv[k].z, rv[k].w);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(gen + (hold - base));
  if (warp == 0) {
    // bf16 A (MN-major) and B (K-major), f32 D, M 128, N n
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 15) | ((uint32_t)(n >> 3) << 17) | (8u << 24);
#pragma unroll 1
    for (int g = 0; g < GROUPS; ++g) {
      wait(full + 8 * g);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int k = 0; k < NOPE / 16; ++k)
        asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n"
                     "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                     ::"r"(tmem + 64 * g), "l"(umma_desc(ws + 2 * g * BOX + k * 2048, BOX, 1024)),
                     "l"(umma_desc(qs + (k / 4) * Q_CHUNK + (k % 4) * 32, 16, 1024)), "r"(idesc), "r"(k));
      asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
                   "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(done + 8 * g));
    }
  }
  // warp w: lanes 32 (w % 4) .. +32 of groups 2 (w / 4) and + 1, 8 rows a load
#pragma unroll 1
  for (int g = 2 * (warp / 4); g < 2 * (warp / 4) + 2; ++g) {
    wait(done + 8 * g);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    __nv_bfloat16* o = q_abs + (size_t)h * ROW + 128 * g + 32 * (warp % 4) + lane;
    for (int b0 = 0; b0 < B; b0 += 8) {
      uint32_t v[8];
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                   : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                   : "r"(tmem + ((32 * (warp % 4)) << 16) + 64 * g + b0));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (b0 + i < B) o[(size_t)(b0 + i) * HEADS * ROW] = __float2bfloat16_rn(__uint_as_float(v[i]));
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(GROUPS * 64));
}
