// K3.1 MLA absorb for a long call: P's bf16 query folded into the latent space for the sparse attention,
// e4m3 rows of HEADS_PAD heads. The same products as k3_mla_absorb_mma (D's, at most 64 rows), arranged
// for thousands of rows.
//
//   extern "C" __global__ void kern_k31_absorb(
//       const bf16* q,         // [B, HEADS*192]  nope 128 | rope 64
//       const bf16* w_kv_b,    // [HEADS*256, 512]  W_UK = rows h*256+0..128
//       uint8_t*    q_abs,     // [B, HEADS_PAD, 576]  e4m3: latent 512 | rope 64; heads HEADS.. untouched
//       int B);
//
//   grid (any, HEADS, LAT / CS = 4)   block (256, 1, 1)   smem 0 dynamic
//
// Block (x, h, cs) owns columns cs*CS .. +CS of head h: W_UK's 128 x CS slice is staged once and each
// warp keeps its 16 columns' B fragments in registers, then the block walks the 64-row tiles x, x +
// gridDim.x, ...: the tile's query lands in shared memory (16-byte loads, the next tile's in flight
// while this one multiplies), each warp runs mma.sync m16n8k16 (bf16 in, f32 accumulate) over its four
// 16-row slices, and the tile leaves through shared memory in 16-byte rows. The block's eighth of the
// rope columns is copied alongside.
//
//   q_abs = e4m3([ bf16(sum_d q_h[d] W_UK_h[d, j]) for j < 512 | q_h[128..192] ])   by cvt.rn, no saturation
//
//   nvcc -cubin -arch=sm_103a -O3 source/k31_absorb.cu
#include <cuda_bf16.h>

#include "k3_e4m3.cuh"

#define HEADS 96
#ifndef HEADS_PAD
#define HEADS_PAD 128
#endif
#define NOPE 128
#define ROPE 64
#define LAT 512
#define ROW 576
#define QW 192
#define BM 64
#define CS 128
#define THREADS 256
#define QP (NOPE + 8)  // padded rows keep ldmatrix off one bank
#define WP (CS + 8)
#define OP (CS + 16)   // e4m3 tile rows, 16 bytes longer: a fragment store's 8 rows in distinct banks
#define QV (BM * NOPE / 8 / THREADS)  // 4 sixteen-byte query vectors a thread a tile
#define RC (ROPE / (LAT / CS))        // 16 rope columns a block

struct Tile {
  uint4 q[QV];
  uint4 rope;
};

__device__ __forceinline__ Tile load_tile(const __nv_bfloat16* __restrict__ q, int h, int cs, int r0, int B, int t) {
  Tile x;
#pragma unroll
  for (int k = 0; k < QV; ++k) {
    const int i = t + k * THREADS, r = r0 + i / (NOPE / 8), v = i % (NOPE / 8);
    x.q[k] = r < B ? *reinterpret_cast<const uint4*>(q + (size_t)r * (HEADS * QW) + h * QW + v * 8)
                   : make_uint4(0, 0, 0, 0);
  }
  const int r = r0 + t / (RC / 8), v = t % (RC / 8);
  x.rope = t < BM * RC / 8 && r < B
               ? *reinterpret_cast<const uint4*>(q + (size_t)r * (HEADS * QW) + h * QW + NOPE + cs * RC + v * 8)
               : make_uint4(0, 0, 0, 0);
  return x;
}

__device__ __forceinline__ unsigned short e4m3x2(unsigned w) {
  const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
  return e4m3x2_nosat(f.x, f.y);
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) kern_k31_absorb(const __nv_bfloat16* __restrict__ q,
                                                                        const __nv_bfloat16* __restrict__ w_kv_b,
                                                                        unsigned char* __restrict__ q_abs, int B) {
  __shared__ __align__(32) __nv_bfloat16 qs[BM][QP];
  __shared__ __align__(32) __nv_bfloat16 ws[NOPE][WP];
  __shared__ __align__(16) unsigned char ost[BM][OP];
  const int h = blockIdx.y, cs = blockIdx.z, c0 = cs * CS, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int tiles = (B + BM - 1) / BM;
  if ((int)blockIdx.x >= tiles) return;
  {
    const __nv_bfloat16* __restrict__ w = w_kv_b + (size_t)h * 256 * LAT + c0;
    uint4 wv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
      wv[k] = *reinterpret_cast<const uint4*>(w + (size_t)d * LAT + v * 8);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = t + k * THREADS, d = i / (CS / 8), v = i % (CS / 8);
      *reinterpret_cast<uint4*>(&ws[d][v * 8]) = wv[k];
    }
  }
  __syncthreads();
  const unsigned qs0 = (unsigned)__cvta_generic_to_shared(&qs[0][0]), ws0 = (unsigned)__cvta_generic_to_shared(&ws[0][0]);
  unsigned bw[NOPE / 16][4];
#pragma unroll
  for (int k = 0; k < NOPE / 16; ++k)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(bw[k][0]), "=r"(bw[k][1]), "=r"(bw[k][2]), "=r"(bw[k][3])
                 : "r"(ws0 + ((k * 16 + lane % 16) * WP + warp * 16 + 8 * (lane / 16)) * 2));
  const int g = lane / 4, tq = lane % 4;
  Tile next = load_tile(q, h, cs, blockIdx.x * BM, B, t);
#pragma unroll 1
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int r0 = tile * BM, rows = min(BM, B - r0);
    const Tile cur = next;
    __syncthreads();  // the previous tile's qs and ost are read
#pragma unroll
    for (int k = 0; k < QV; ++k) {
      const int i = t + k * THREADS;
      *reinterpret_cast<uint4*>(&qs[i / (NOPE / 8)][i % (NOPE / 8) * 8]) = cur.q[k];
    }
    __syncthreads();
    if (tile + (int)gridDim.x < tiles) next = load_tile(q, h, cs, (tile + gridDim.x) * BM, B, t);
#pragma unroll
    for (int m = 0; m < BM / 16; ++m) {
      float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int k = 0; k < NOPE / 16; ++k) {
        unsigned a[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"(qs0 + ((m * 16 + lane % 8 + 8 * (lane / 8 % 2)) * QP + k * 16 + 8 * (lane / 16)) * 2));
#pragma unroll
        for (int n = 0; n < 2; ++n)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                       "{%8, %9}, {%0, %1, %2, %3};"
                       : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bw[k][2 * n]), "r"(bw[k][2 * n + 1]));
      }
      // c[n] holds rows g and g + 8 of the slice, columns n * 8 + 2 tq and + 1, landed bf16 then e4m3
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const float2 f = __bfloat1622float2(__floats2bfloat162_rn(c[n][2 * hr], c[n][2 * hr + 1]));
          *reinterpret_cast<unsigned short*>(&ost[m * 16 + g + 8 * hr][warp * 16 + n * 8 + 2 * tq]) =
              e4m3x2_nosat(f.x, f.y);
        }
    }
    __syncthreads();
    for (int i = t; i < rows * (CS / 16); i += THREADS) {
      const int b = i / (CS / 16), v = i % (CS / 16);
      *reinterpret_cast<uint4*>(q_abs + ((size_t)(r0 + b) * HEADS_PAD + h) * ROW + c0 + v * 16) =
          *reinterpret_cast<const uint4*>(&ost[b][v * 16]);
    }
    const int rr = t / (RC / 8), rv = t % (RC / 8);
    if (t < BM * RC / 8 && rr < rows)
      *reinterpret_cast<uint2*>(q_abs + ((size_t)(r0 + rr) * HEADS_PAD + h) * ROW + LAT + cs * RC + rv * 8) =
          make_uint2(e4m3x2(cur.rope.x) | (unsigned)e4m3x2(cur.rope.y) << 16,
                     e4m3x2(cur.rope.z) | (unsigned)e4m3x2(cur.rope.w) << 16);
  }
}
