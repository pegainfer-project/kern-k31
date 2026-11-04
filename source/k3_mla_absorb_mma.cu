// K3 MLA decode, absorb on tensor cores: the same contract as k3_mla_absorb.cu
// (docs/k3-kernel-abi.md K5a), for a decode batch of at most 64 rows.
//
//   extern "C" __global__ void kern_k3_mla_absorb_mma(
//       const f32*  q_partial,   // [B, HEADS*192]  f32 partial, nope 128 | rope 64
//       const bf16* w_kv_b,      // [HEADS*256, 512]  W_UK = rows h*256+0..128
//       bf16*       q_abs,       // [B, HEADS, 576]  latent 512 | rope 64 (e4m3 bytes under Q_FP8)
//       int B);                  // B <= 64
//
//   grid  (1, HEADS, LAT / CS = 4)   block (2 * CS = 256, 1, 1)   smem 0 dynamic (52 KB static)
//   3 blocks an SM (80 registers): the 384 blocks are one wave
//
// Block (h, cs) owns columns cs*CS .. +CS of head h for every row: W_UK's
// 128 x CS slice is read once (the scalar kernel re-read it per 8-row group,
// and its 16-way smem reduction made a 16-row step ~13 us for 12 MB),
// staged in shared memory with the batch's bf16 query, and each of the CS / 16
// warps multiplies its 16 columns with mma.sync m16n8k16 (bf16 in, f32
// accumulate; ldmatrix fragments, W's transposed) over ceil(B/16) row tiles,
// its accumulators stored straight to q_abs. Rows past B are zero in shared
// memory and never stored. Each block also copies its eighth of the rope
// columns (2 threads x 4 columns per row, loaded with everything else).
//
//   q_h     = bf16(q_partial[b, h*192 .. +192])
//   q_abs   = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) for j<512 | q_h[128..192] ]   f32 acc
//
// The f32 sum runs in the tensor core's order, not the scalar kernel's: the
// same products, rounded once to bf16 at the end. Every block's order is
// fixed, so a result never depends on the schedule.
//
// -DQ_FP8: q_abs is the fp8 query of SGLang's fp8 cache, one e4m3 byte an
// element: each bf16 above goes on through f32 to e4m3 by cvt.rn without
// saturation, the aten cast SGLang applies to its bf16 query (k3_e4m3.cuh).
//
//   nvcc -cubin -arch=sm_103a -O3 source/k3_mla_absorb_mma.cu
#include <cuda_bf16.h>

#ifdef Q_FP8
#include "k3_e4m3.cuh"
typedef unsigned char q_t;
__device__ __forceinline__ void put2(q_t* p, __nv_bfloat162 v) {
  const float2 f = __bfloat1622float2(v);
  *reinterpret_cast<unsigned short*>(p) = e4m3x2_nosat(f.x, f.y);
}
#define PUT2(p, i, v) put2((p) + 2 * (i), v)
#else
typedef __nv_bfloat16 q_t;
#define PUT2(p, i, v) (reinterpret_cast<__nv_bfloat162*>(p)[i] = (v))
#endif

#define HEADS 96
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define BMAX 64
#ifndef CS
#define CS 128     // columns per block
#endif
#define THREADS (2 * CS)  // a warp per 16 columns
#define QP (NOPE + 8)  // padded smem rows keep the ldmatrix loads off one bank
#define WP (CS + 8)

extern "C" __global__ void __launch_bounds__(THREADS, 3) kern_k3_mla_absorb_mma(const float* __restrict__ q_partial,
                                                                        const __nv_bfloat16* __restrict__ w_kv_b,
                                                                        q_t* __restrict__ q_abs, int B) {
  __shared__ __align__(32) __nv_bfloat16 qs[BMAX][QP];
  __shared__ __align__(32) __nv_bfloat16 ws[NOPE][WP];
  const int h = blockIdx.y, c0 = blockIdx.z * CS, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int tiles = (B + 15) / 16;
  // W slice: 128 rows x CS columns = 16 * CS sixteen-byte vectors, 8 per thread, all in flight.
  const __nv_bfloat16* __restrict__ w = w_kv_b + (size_t)h * 256 * LAT + c0;
  uint4 wv[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
    wv[k] = *reinterpret_cast<const uint4*>(w + (size_t)d * LAT + v * 8);
  }
  // This block's CS / 8 rope columns of every row (ROPE / gridDim.z), a float4 per thread, copied at the end.
  const int rr = t / (CS / 32), rc = blockIdx.z * (CS / 8) + t % (CS / 32) * 4;
  const float4 rv = rr < B ? *reinterpret_cast<const float4*>(q_partial + (size_t)rr * (HEADS * QW) + (size_t)h * QW + NOPE + rc)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
  // The query: up to 64 rows x 128 f32 = 16 float4 per thread, every load in
  // flight before the first store (one after another they cost a round trip each).
  float4 qv[BMAX * NOPE / 4 / THREADS];
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d4 = i % (NOPE / 4);
    qv[k] = r < B ? *reinterpret_cast<const float4*>(q_partial + (size_t)r * (HEADS * QW) + (size_t)h * QW + d4 * 4)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
  }
#pragma unroll
  for (int k = 0; k < BMAX * NOPE / 4 / THREADS; ++k) {
    const int i = t + k * THREADS, r = i / (NOPE / 4), d4 = i % (NOPE / 4);
    if (r < tiles * 16) {
      reinterpret_cast<__nv_bfloat162*>(&qs[r][d4 * 4])[0] = __floats2bfloat162_rn(qv[k].x, qv[k].y);
      reinterpret_cast<__nv_bfloat162*>(&qs[r][d4 * 4])[1] = __floats2bfloat162_rn(qv[k].z, qv[k].w);
    }
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
    *reinterpret_cast<uint4*>(&ws[d][v * 8]) = wv[k];
  }
  __syncthreads();
  // The warp's 16 W columns are the same for every row tile: their B fragments (ldmatrix.trans of the
  // row-major [k][n] slice, two n8 tiles a k16 step) stay in registers.
  const unsigned qs0 = (unsigned)__cvta_generic_to_shared(&qs[0][0]), ws0 = (unsigned)__cvta_generic_to_shared(&ws[0][0]);
  unsigned bw[NOPE / 16][4];
#pragma unroll
  for (int k = 0; k < NOPE / 16; ++k)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(bw[k][0]), "=r"(bw[k][1]), "=r"(bw[k][2]), "=r"(bw[k][3])
                 : "r"(ws0 + ((k * 16 + lane % 16) * WP + warp * 16 + 8 * (lane / 16)) * 2));
  const int g = lane / 4, tq = lane % 4;
  for (int m = 0; m < tiles; ++m) {
    float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int k = 0; k < NOPE / 16; ++k) {
      unsigned a[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                   : "r"(qs0 + ((m * 16 + lane % 8 + 8 * (lane / 8 % 2)) * QP + k * 16 + 8 * (lane / 16)) * 2));
#pragma unroll
      for (int n = 0; n < 2; ++n)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bw[k][2 * n]), "r"(bw[k][2 * n + 1]));
    }
    // c[n] holds rows g and g + 8 of the tile, columns n * 8 + 2 tq and + 1
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int b = m * 16 + g + 8 * hr;
      if (b < B) {
        q_t* o = q_abs + ((size_t)b * HEADS + h) * ROW + c0 + warp * 16 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 2; ++n) PUT2(o + n * 8, 0, __floats2bfloat162_rn(c[n][2 * hr], c[n][2 * hr + 1]));
      }
    }
  }
  if (rr < B) {
    q_t* o = q_abs + ((size_t)rr * HEADS + h) * ROW + LAT + rc;
    PUT2(o, 0, __floats2bfloat162_rn(rv.x, rv.y));
    PUT2(o, 1, __floats2bfloat162_rn(rv.z, rv.w));
  }
}
