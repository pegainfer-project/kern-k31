// K3.1 MLA absorb for a prefill call on tcgen05: P's bf16 query folded into the latent space for the sparse
// attention, e4m3 rows of HEADS_PAD heads; k3_mla_absorb_mma's products (D's, at most 64 rows) as a GEMM a
// head (rows x 128 by 128 x 512) on the 5th-generation tensor cores.
//
//   extern "C" __global__ void kern_k31_absorb_tc(
//       CUtensorMap q_map,     // q [rows, HEADS*192] bf16: dims (192, HEADS, rows), box (64, 1, 128), 128 B swizzle
//       CUtensorMap w_map,     // w_kv_b [HEADS*256, 512] bf16: box (64, 64), 128 B swizzle; W_UK = rows h*256+0..128
//       CUtensorMap y_map,     // q_abs [rows, HEADS_PAD, 576] u8: dims (576, HEADS_PAD, rows), box (128, 1, 128), 128 B swizzle
//       CUtensorMap r_map,     // q_abs again: box (64, 1, 128), no swizzle, for the rope
//       int B);
//
//   grid (odd, at most SMs)   block (384, 1, 1)   smem SMEM_BYTES dynamic   one block an SM (TMEM)
//
// A block walks the items (128-row tile, head, half of the 512 latent columns) from blockIdx.x by
// gridDim.x, half fastest; an odd grid alternates a block's halves, so the rope's work spreads evenly. Warp 0 streams an item's two 64-d chunks of the tile's query (K-major) and of
// W_UK's 64 d x 256 columns (MN-major, four 64-column boxes) through STAGES stages; warp 1 issues
// M 128 x N 256 x K 16 MMAs (bf16, f32 accumulate) into one of two TMEM accumulators; warps 4..11 each take
// 32 rows x 128 columns of the other, one row a thread, land them bf16 then e4m3 into a swizzled staging
// tile that one TMA store writes (rows past the map clipped). Warps 2 and 3 take the rope of each first
// half's item alongside, TMA in, e4m3 in shared memory, TMA out.
//
//   q_abs = e4m3([ bf16(sum_d q_h[d] W_UK_h[d, j]) for j < 512 | q_h[128..192] ])   by cvt.rn, no saturation
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "k3_e4m3.cuh"

#define HEADS 96
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define BM 128
#define BN 256
#define KC 64
#define STAGES 3
#define A_BYTES (BM * KC * 2)    // 16 KB: 128 rows x 64 d
#define WBOX (KC * 64 * 2)        // 8 KB: 64 d x 64 columns
#define STAGE_BYTES (A_BYTES + 4 * WBOX)
#define YBOX (BM * 128)           // 16 KB: 128 rows x 128 e4m3 columns
#define Y_BYTES (2 * YBOX)
#define RIN (BM * ROPE * 2)       // 16 KB: the tile's rope, bf16, 128 B swizzle
#define ROUT (BM * ROPE)          // 8 KB: as e4m3
#define SMEM_BYTES (1024 + STAGES * STAGE_BYTES + Y_BYTES + RIN + ROUT + 256)

#ifndef HEADS_PAD
#define HEADS_PAD 128
#endif

__device__ __forceinline__ uint64_t umma_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  // start >> 4 | LBO >> 4 << 16 | SBO >> 4 << 32 | version 1 << 46 | 128 B swizzle (2) << 61
  return (uint64_t)((addr >> 4) & 0x3FFF) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 |
         (uint64_t)((sbo >> 4) & 0x3FFF) << 32 | (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(bar),
               "r"(parity)
               : "memory");
}

__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
               "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(bar));
}

__device__ __forceinline__ unsigned e4m3x4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  const float2 x = __bfloat1622float2(__floats2bfloat162_rn(__uint_as_float(a), __uint_as_float(b)));
  const float2 y = __bfloat1622float2(__floats2bfloat162_rn(__uint_as_float(c), __uint_as_float(d)));
  return e4m3x2_nosat(x.x, x.y) | (unsigned)e4m3x2_nosat(y.x, y.y) << 16;
}

__device__ __forceinline__ unsigned e4m3_bf16x4(uint2 w) {
  const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
  const float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
  return e4m3x2_nosat(x.x, x.y) | (unsigned)e4m3x2_nosat(y.x, y.y) << 16;
}

extern "C" __global__ void __launch_bounds__(384, 1) kern_k31_absorb_tc(const __grid_constant__ CUtensorMap q_map,
                                                                       const __grid_constant__ CUtensorMap w_map,
                                                                       const __grid_constant__ CUtensorMap y_map,
                                                                       const __grid_constant__ CUtensorMap r_map,
                                                                       int B) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  const uint32_t ybuf = base + STAGES * STAGE_BYTES, rin = ybuf + Y_BYTES, rout = rin + RIN, bars = rout + ROUT;
  const uint32_t full = bars, empty = full + 8 * STAGES, tfull = empty + 8 * STAGES, tempty = tfull + 16,
                 yempty = tempty + 16, rfull = yempty + 8, hold = rfull + 8;
  uint8_t* const gen = smem_raw + (base - raw);
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int items = (B + BM - 1) / BM * HEADS * 2;
  if (t == 0) {
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(full + 8 * s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(empty + 8 * s));
    }
    for (int a = 0; a < 2; ++a) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tfull + 8 * a));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(tempty + 8 * a));
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(yempty));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(rfull));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(hold));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(smem_raw + (hold - raw));

  if (warp == 0) {
    if (lane == 0) {
      uint32_t s = 0, ph = 0;
      for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int half = item % 2, h = item / 2 % HEADS, tile = item / 2 / HEADS;
        for (int kc = 0; kc < NOPE / KC; ++kc) {
          wait(empty + 8 * s, ph ^ 1);
          const uint32_t a = base + s * STAGE_BYTES, f = full + 8 * s;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(f), "r"(STAGE_BYTES));
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(a),
              "l"(&q_map), "r"(kc * KC), "r"(h), "r"(tile * BM), "r"(f)
              : "memory");
          for (int i = 0; i < 4; ++i)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
                    a + A_BYTES + i * WBOX),
                "l"(&w_map), "r"(half * BN + 64 * i), "r"(h * 256 + kc * KC), "r"(f)
                : "memory");
          if (++s == STAGES) s = 0, ph ^= 1;
        }
      }
    }
  } else if (warp == 1) {
    // bf16 A (K-major) and B (MN-major), f32 D, M 128, N 256
    const uint32_t idesc =
        (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);
    uint32_t s = 0, ph = 0, it = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, ++it) {
      const uint32_t acc = it & 1, tph = (it >> 1) & 1;
      wait(tempty + 8 * acc, tph ^ 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for (int kc = 0; kc < NOPE / KC; ++kc) {
        wait(full + 8 * s, ph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t a = base + s * STAGE_BYTES;
#pragma unroll
        for (int k = 0; k < KC / 16; ++k)
          asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n"
                       "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem + BN * acc),
                       "l"(umma_desc(a + k * 32, 16, 1024)), "l"(umma_desc(a + A_BYTES + k * 2048, WBOX, 1024)),
                       "r"(idesc), "r"(kc | k));
        commit(empty + 8 * s);
        if (++s == STAGES) s = 0, ph ^= 1;
      }
      commit(tfull + 8 * acc);
    }
  } else if (warp == 2 || warp == 3) {
    // the rope of each first-half item, off the MMA's path: rows 2 tr, 2 tr + 1 a thread
    const int tr = t - 64;
    uint32_t rk = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
      if (item % 2) continue;
      const int h = item / 2 % HEADS, tile = item / 2 / HEADS;
      if (tr == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(rfull), "r"(RIN));
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(rin),
            "l"(&q_map), "r"(NOPE), "r"(h), "r"(tile * BM), "r"(rfull)
            : "memory");
      }
      wait(rfull, rk++ & 1);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = 2 * tr + i;
        const uint4* in = reinterpret_cast<const uint4*>(gen + (rin - base) + r * 128);
        uint4* out = reinterpret_cast<uint4*>(gen + (rout - base) + r * ROPE);
#pragma unroll
        for (int u = 0; u < ROPE / 16; ++u) {
          const uint4 lo = in[(2 * u) ^ (r % 8)], hi = in[(2 * u + 1) ^ (r % 8)];
          out[u] = make_uint4(e4m3_bf16x4(make_uint2(lo.x, lo.y)), e4m3_bf16x4(make_uint2(lo.z, lo.w)),
                              e4m3_bf16x4(make_uint2(hi.x, hi.y)), e4m3_bf16x4(make_uint2(hi.z, hi.w)));
        }
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 2, 64;" ::: "memory");
      if (tr == 0) {
        asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(&r_map),
                     "r"(LAT), "r"(h), "r"(tile * BM), "r"(rout)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
    }
    if (tr == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  } else if (warp >= 4) {
    // warp 4 + 4 x + q: the tile's rows 32 q .. +32 (TMEM lanes), latent columns 128 x .. +128 of the half
    const int q4 = warp % 4, x = (warp - 4) / 4, r = 32 * q4 + lane;
    uint32_t it = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, ++it) {
      const int half = item % 2, h = item / 2 % HEADS, tile = item / 2 / HEADS;
      const uint32_t acc = it & 1, tph = (it >> 1) & 1;
      wait(tfull + 8 * acc, tph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      uint4* const rowp = reinterpret_cast<uint4*>(gen + (ybuf - base) + x * YBOX + r * 128);
#pragma unroll 1
      for (int c = 0; c < 2; ++c) {
        uint32_t v[64];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc)
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
              "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(v[32 * cc + 0]), "=r"(v[32 * cc + 1]), "=r"(v[32 * cc + 2]), "=r"(v[32 * cc + 3]),
                "=r"(v[32 * cc + 4]), "=r"(v[32 * cc + 5]), "=r"(v[32 * cc + 6]), "=r"(v[32 * cc + 7]),
                "=r"(v[32 * cc + 8]), "=r"(v[32 * cc + 9]), "=r"(v[32 * cc + 10]), "=r"(v[32 * cc + 11]),
                "=r"(v[32 * cc + 12]), "=r"(v[32 * cc + 13]), "=r"(v[32 * cc + 14]), "=r"(v[32 * cc + 15]),
                "=r"(v[32 * cc + 16]), "=r"(v[32 * cc + 17]), "=r"(v[32 * cc + 18]), "=r"(v[32 * cc + 19]),
                "=r"(v[32 * cc + 20]), "=r"(v[32 * cc + 21]), "=r"(v[32 * cc + 22]), "=r"(v[32 * cc + 23]),
                "=r"(v[32 * cc + 24]), "=r"(v[32 * cc + 25]), "=r"(v[32 * cc + 26]), "=r"(v[32 * cc + 27]),
                "=r"(v[32 * cc + 28]), "=r"(v[32 * cc + 29]), "=r"(v[32 * cc + 30]), "=r"(v[32 * cc + 31])
              : "r"(tmem + ((32 * q4) << 16) + BN * acc + 128 * x + 64 * c + 32 * cc));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (c == 0) wait(yempty, (it & 1) ^ 1);  // the previous item's store has read the staging tile
        // 64 columns: 16-byte chunks 4 c .. +4 of the row, each at chunk ^ (row % 8)
#pragma unroll
        for (int u = 0; u < 4; ++u)
          rowp[(4 * c + u) ^ (r % 8)] =
              make_uint4(e4m3x4(v[16 * u + 0], v[16 * u + 1], v[16 * u + 2], v[16 * u + 3]),
                         e4m3x4(v[16 * u + 4], v[16 * u + 5], v[16 * u + 6], v[16 * u + 7]),
                         e4m3x4(v[16 * u + 8], v[16 * u + 9], v[16 * u + 10], v[16 * u + 11]),
                         e4m3x4(v[16 * u + 12], v[16 * u + 13], v[16 * u + 14], v[16 * u + 15]));
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tempty + 8 * acc) : "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (warp == 4 && lane == 0) {
        for (int y = 0; y < 2; ++y)
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(&y_map),
                       "r"(half * BN + 128 * y), "r"(h), "r"(tile * BM), "r"(ybuf + y * YBOX)
                       : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(yempty) : "memory");
      }
    }
    if (warp == 4 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}
