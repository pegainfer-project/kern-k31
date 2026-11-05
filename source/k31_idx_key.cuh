// K3.1's indexer key, a warp a row, as vLLM's indexer_k_norm_quant_insert makes it: LayerNorm in f32 (eps
// 1e-5) over the row's 128 bf16 values, rounded to bf16, then e4m3 with one UE8M0 scale
// 2^ceil(log2(max(amax, 1e-4) / 448)) into its 144-byte cache row (128 e4m3, the scale's packed word).
// k31_dsa.cu's k31_idx_key and k3_mla_prep.cu's IDX_KEY build share it.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// 2^ceil(log2(max(amax, eps) / 448)), exact on the bits.
__device__ __forceinline__ float ue8m0(float amax, float eps) {
  const unsigned b = __float_as_uint(fmaxf(amax, eps) / 448.f);
  const unsigned x = ((b >> 23) & 0xff) + ((b & 0x7fffff) != 0);
  return __uint_as_float(x << 23);
}

__device__ __forceinline__ unsigned ue8m0x4(float sc) { return (__float_as_uint(sc) >> 23) * 0x01010101u; }

__device__ __forceinline__ unsigned fp8x4(float a, float b, float c, float d) {
  const __nv_fp8x4_e4m3 q(make_float4(a, b, c, d));
  return *reinterpret_cast<const unsigned*>(&q);
}

__device__ __forceinline__ void bf16x4(const __nv_bfloat16* p, float v[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
}

// Lane l's four values x (columns 4l .. 4l + 3, bf16 values) of a row's key, with the norm's weight w and
// bias b, into `row`. Every lane of the warp calls it.
__device__ __forceinline__ void k31_key_row(const float x[4], const __nv_bfloat16* __restrict__ w,
                                            const __nv_bfloat16* __restrict__ b, unsigned char* row, int l) {
  float g[4], c[4];
  bf16x4(w + l * 4, g);
  bf16x4(b + l * 4, c);
  const float mean = warp_sum(x[0] + x[1] + x[2] + x[3]) / 128;
  float d2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) d2 += (x[j] - mean) * (x[j] - mean);
  const float rstd = rsqrtf(warp_sum(d2) / 128 + 1e-5f);
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) y[j] = rnd((x[j] - mean) * rstd * g[j] + c[j]);
  const float sc = ue8m0(warp_max(fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])))), 1e-4f);
  *reinterpret_cast<unsigned*>(row + l * 4) = fp8x4(y[0] / sc, y[1] / sc, y[2] / sc, y[3] / sc);
  if (l == 0) *reinterpret_cast<unsigned*>(row + 128) = ue8m0x4(sc);
}
