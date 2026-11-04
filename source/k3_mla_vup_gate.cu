// K3 MLA decode, epilogue: the attention's latent output expands through
// W_UV to the head dim and meets the sigmoid gate (docs/k3-kernel-abi.md K5c).
//
//   extern "C" __global__ void kern_k3_mla_vup_gate(
//       const bf16* o_lat,     // [B, HEADS, 512]  attention output in latent space
//       const bf16* w_kv_b,    // [HEADS*256, 512]  W_UV = rows h*256+128..256
//       const bf16* mla_gate,  // [B, HEADS*128]
//       bf16*       gated,     // [B, HEADS*128]
//       int B);
//
//   grid  (ceil(B/32), HEADS, 4)   block (256, 1, 1)   smem 0 dynamic
//
// HEADS is the heads this rank expands: 96 whole, or its tp share of a
// head-sharded o side (`-DHEADS=12` at TP8, where o_lat is the DCP combine's
// [B, 12, 512] and w_kv_b / mla_gate / gated are this rank's heads only).
//
// Block (bt, h, ds) produces dv = ds*32 .. +32 for rows 32*bt .. +32 of head
// h, eight rows at a time. Like the absorb, a bandwidth problem on the 12 MB
// of W_UV: a thread owns one dv and an eighth of its 512-long row (eight
// 16-byte loads in flight, kept in registers across the row groups), dots it
// against the group's 8 staged latent rows, and the eight slices meet in
// shared memory. Staging loads are unconditional (rows past B re-read the
// last row and are zeroed) so they are all in flight at once.
//
//   o[dv]   = bf16(sum_j W_UV_h[dv,j]*lat[j])          f32 acc
//   gated   = o[dv] * bf16(sigmoid(f32(mla_gate[...])))
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_mla_vup_gate.cu
#include <cuda_bf16.h>
#ifndef HEADS
#define HEADS 96
#endif
#define NOPE 128
#define LAT 512
#define RB 8      // rows per group
#define GROUPS 4  // row groups per block
#define DS 32     // dv per block
#define JS 8      // j-slices per block
#define JW (LAT / JS)

extern "C" __global__ void __launch_bounds__(256) kern_k3_mla_vup_gate(const __nv_bfloat16* __restrict__ o_lat,
                                                                      const __nv_bfloat16* __restrict__ w_kv_b,
                                                                      const __nv_bfloat16* __restrict__ mla_gate,
                                                                      __nv_bfloat16* __restrict__ gated, int B) {
  __shared__ __align__(16) __nv_bfloat16 lat[RB][LAT];
  __shared__ float red[JS][RB][DS];
  const int h = blockIdx.y, dv0 = blockIdx.z * DS, t = threadIdx.x;
  const int js = t / DS, dl = t - js * DS, dv = dv0 + dl, j0 = js * JW;
  const uint4* __restrict__ w = reinterpret_cast<const uint4*>(w_kv_b + ((size_t)h * 256 + NOPE + dv) * LAT + j0);
  uint4 wv[JW / 8];
#pragma unroll
  for (int k = 0; k < JW / 8; ++k) wv[k] = w[k];
  for (int g = 0; g < GROUPS; ++g) {
    const int b0 = (blockIdx.x * GROUPS + g) * RB;
    const int nr = min(RB, B - b0);
    if (nr <= 0) break;
    // RB*LAT/8 / 256 = 2 vectors per thread, all loads in flight before any store
    uint4 lv[RB * (LAT / 8) / 256];
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8);
      lv[k] = reinterpret_cast<const uint4*>(o_lat + ((size_t)min(b0 + r, B - 1) * HEADS + h) * LAT)[c];
    }
    if (g) __syncthreads();  // the previous group is done reading lat / red
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8);
      reinterpret_cast<uint4*>(&lat[r][0])[c] = r < nr ? lv[k] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    float acc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r] = 0.f;
#pragma unroll
    for (int k = 0; k < JW / 8; ++k) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[k]);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(w2[j]);
        wf[2 * j] = f.x;
        wf[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint4 l = *reinterpret_cast<const uint4*>(&lat[r][j0 + k * 8]);
        const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&l);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(l2[j]);
          acc[r] += wf[2 * j] * f.x + wf[2 * j + 1] * f.y;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) red[js][r][dl] = acc[r];
    __syncthreads();
    for (int i = t; i < nr * DS; i += 256) {
      const int r = i / DS, d = i - r * DS;
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < JS; ++k) a += red[k][r][d];
      const size_t oi = (size_t)(b0 + r) * (HEADS * NOPE) + (size_t)h * NOPE + dv0 + d;
      const float gf = __bfloat162float(mla_gate[oi]);
      gated[oi] = __hmul(__float2bfloat16_rn(a), __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
    }
  }
}
