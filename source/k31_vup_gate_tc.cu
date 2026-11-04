// K3.1 MLA epilogue for a prefill call on tcgen05: k31_vup_gate's math, a GEMM a head (rows x 512 by 512 x
// 128) on the 5th-generation tensor cores, for the thousands of rows of a chunk. D's batch keeps
// k31_vup_gate.
//
//   extern "C" __global__ void kern_k31_vup_gate_tc(
//       CUtensorMap o_map,     // o_lat [rows, IN_HEADS, 512] bf16: dims (512, IN_HEADS, rows), box (64, 1, 128)
//       CUtensorMap w_map,     // w_kv_b [HEADS*256, 512] bf16: box (64, 128); W_UV = rows h*256+128..256
//       CUtensorMap g_map,     // the gate [rows, HEADS*128] bf16 at its row stride: box (64, 128)
//       CUtensorMap y_map,     // gated [rows, HEADS*128] bf16: box (64, 128)
//       int B);                // every map 128 B swizzle
//
//   grid (SMs, 1, 1)   block (384, 1, 1)   smem SMEM_BYTES dynamic   one block an SM (TMEM)
//
// A block walks the items (128-row tile, head) from blockIdx.x by gridDim.x, head fastest. Warp 0 streams
// each item's eight 64-column chunks of the tile's latent rows and of W_UV through STAGES shared-memory
// stages, and the item's gate tile into one of two gate buffers (TMA, rows past the buffer zero). Warp 1
// issues M 128 x N 128 x K 16 MMAs (bf16, f32 accumulate, both operands K-major) into one of two TMEM
// accumulators. Warps 4..11 each take 32 rows x 64 columns of the other accumulator, one row a thread:
// the gated row overwrites its gate in the buffer, which one thread then stores by TMA (rows past B clipped) before
// handing the buffer back. The accumulation order is fixed per item.
//
//   o[dv]  = bf16(sum_j lat[j] * W_UV_h[dv, j])                      f32 acc
//   gated  = bf16(o[dv] * bf16(sigmoid(f32(gate[h*128 + dv]))))      sigmoid_mul's arithmetic
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#define HEADS 96
#define NOPE 128
#define LAT 512
#define BM 128
#define KC 64
#define CHUNKS (LAT / KC)
#define STAGES 4
#define TILE_BYTES (BM * KC * 2)  // 16 KB: 128 rows x 128 B
#define STAGE_BYTES (2 * TILE_BYTES)
#define GATE_BYTES (2 * TILE_BYTES)  // a gate tile: 128 rows x 128 columns, two 64-column boxes
#define SMEM_BYTES (1024 + STAGES * STAGE_BYTES + 2 * GATE_BYTES + 256)

__device__ __forceinline__ uint64_t umma_desc(uint32_t addr) {
  // K-major, 128 B swizzle: start >> 4 | LBO 16 >> 4 << 16 | SBO 1024 >> 4 << 32 | version 1 << 46 | 2 << 61
  return (uint64_t)((addr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
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

__device__ __forceinline__ void load2d(uint32_t dst, const CUtensorMap* m, int x, int y, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(m), "r"(x), "r"(y), "r"(bar)
               : "memory");
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

extern "C" __global__ void __launch_bounds__(384, 1) kern_k31_vup_gate_tc(const __grid_constant__ CUtensorMap o_map,
                                                                         const __grid_constant__ CUtensorMap w_map,
                                                                         const __grid_constant__ CUtensorMap g_map,
                                                                         const __grid_constant__ CUtensorMap y_map,
                                                                         int B) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  const uint32_t gbuf = base + STAGES * STAGE_BYTES, bars = gbuf + 2 * GATE_BYTES;
  const uint32_t full = bars, empty = full + 8 * STAGES, tfull = empty + 8 * STAGES, tempty = tfull + 16,
                 gfull = tempty + 16, gempty = gfull + 16, hold = gempty + 16;
  uint8_t* const gen = smem_raw + (base - raw);
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int items = (B + BM - 1) / BM * HEADS;
  if (t == 0) {
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(full + 8 * s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(empty + 8 * s));
    }
    for (int a = 0; a < 2; ++a) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tfull + 8 * a));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(tempty + 8 * a));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gfull + 8 * a));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gempty + 8 * a));
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(hold));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(smem_raw + (hold - raw));

  if (warp == 0) {
    if (lane == 0) {
      uint32_t s = 0, ph = 0, it = 0;
      for (int item = blockIdx.x; item < items; item += gridDim.x, ++it) {
        const int tile = item / HEADS, h = item % HEADS;
        const uint32_t g = it & 1, gph = (it >> 1) & 1;
        wait(gempty + 8 * g, gph ^ 1);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(gfull + 8 * g), "r"(GATE_BYTES));
        for (int x = 0; x < 2; ++x) load2d(gbuf + g * GATE_BYTES + x * TILE_BYTES, &g_map, h * NOPE + x * KC, tile * BM, gfull + 8 * g);
        for (int kc = 0; kc < CHUNKS; ++kc) {
          wait(empty + 8 * s, ph ^ 1);
          const uint32_t a = base + s * STAGE_BYTES, f = full + 8 * s;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(f), "r"(STAGE_BYTES));
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(a),
              "l"(&o_map), "r"(kc * KC), "r"(h), "r"(tile * BM), "r"(f)
              : "memory");
          load2d(a + TILE_BYTES, &w_map, kc * KC, h * 256 + NOPE, f);
          if (++s == STAGES) s = 0, ph ^= 1;
        }
      }
    }
  } else if (warp == 1) {
    // bf16 A and B, both K-major, f32 D, M 128, N 128
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(NOPE >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);
    uint32_t s = 0, ph = 0, it = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, ++it) {
      const uint32_t acc = it & 1, tph = (it >> 1) & 1;
      wait(tempty + 8 * acc, tph ^ 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for (int kc = 0; kc < CHUNKS; ++kc) {
        wait(full + 8 * s, ph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t a = base + s * STAGE_BYTES;
#pragma unroll
        for (int k = 0; k < KC / 16; ++k)
          asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n"
                       "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem + NOPE * acc),
                       "l"(umma_desc(a + k * 32)), "l"(umma_desc(a + TILE_BYTES + k * 32)), "r"(idesc), "r"(kc | k));
        commit(empty + 8 * s);
        if (++s == STAGES) s = 0, ph ^= 1;
      }
      commit(tfull + 8 * acc);
    }
  } else if (warp >= 4) {
    // warp 4 + 4 x + q: the tile's rows 32 q .. +32 (TMEM lanes), columns 64 x .. +64 (gate box x)
    const int q = warp % 4, x = (warp - 4) / 4, r = 32 * q + lane;
    uint32_t it = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, ++it) {
      const int tile = item / HEADS, h = item % HEADS;
      const uint32_t acc = it & 1, tph = (it >> 1) & 1;
      wait(gfull + 8 * acc, tph);
      wait(tfull + 8 * acc, tph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      uint32_t v[64];
#pragma unroll
      for (int c = 0; c < 2; ++c)
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=r"(v[32 * c + 0]), "=r"(v[32 * c + 1]), "=r"(v[32 * c + 2]), "=r"(v[32 * c + 3]), "=r"(v[32 * c + 4]),
              "=r"(v[32 * c + 5]), "=r"(v[32 * c + 6]), "=r"(v[32 * c + 7]), "=r"(v[32 * c + 8]), "=r"(v[32 * c + 9]),
              "=r"(v[32 * c + 10]), "=r"(v[32 * c + 11]), "=r"(v[32 * c + 12]), "=r"(v[32 * c + 13]),
              "=r"(v[32 * c + 14]), "=r"(v[32 * c + 15]), "=r"(v[32 * c + 16]), "=r"(v[32 * c + 17]),
              "=r"(v[32 * c + 18]), "=r"(v[32 * c + 19]), "=r"(v[32 * c + 20]), "=r"(v[32 * c + 21]),
              "=r"(v[32 * c + 22]), "=r"(v[32 * c + 23]), "=r"(v[32 * c + 24]), "=r"(v[32 * c + 25]),
              "=r"(v[32 * c + 26]), "=r"(v[32 * c + 27]), "=r"(v[32 * c + 28]), "=r"(v[32 * c + 29]),
              "=r"(v[32 * c + 30]), "=r"(v[32 * c + 31])
            : "r"(tmem + ((32 * q) << 16) + NOPE * acc + 64 * x + 32 * c));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tempty + 8 * acc) : "memory");
      // the row's 128 bytes of box x: 16-byte chunk u at u ^ (row % 8)
      uint4* const rowp = reinterpret_cast<uint4*>(gen + (gbuf - base) + acc * GATE_BYTES + x * TILE_BYTES + r * 128);
      uint4 gv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) gv[u] = rowp[u ^ (r % 8)];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv[u]);
        uint4 ov;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 g = __bfloat1622float2(g2[e]);
          const float s0 = bf(1.0f / (1.0f + expf(-g.x))), s1 = bf(1.0f / (1.0f + expf(-g.y)));
          o2[e] = __floats2bfloat162_rn(bf(__uint_as_float(v[8 * u + 2 * e])) * s0,
                                        bf(__uint_as_float(v[8 * u + 2 * e + 1])) * s1);
        }
        rowp[u ^ (r % 8)] = ov;
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (warp == 4 && lane == 0) {
        for (int b = 0; b < 2; ++b)
          asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(&y_map),
                       "r"(h * NOPE + b * KC), "r"(tile * BM), "r"(gbuf + acc * GATE_BYTES + b * TILE_BYTES)
                       : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(gempty + 8 * acc) : "memory");
      }
    }
    if (warp == 4 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem));
}
