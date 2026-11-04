// K11 `k3_kda_out_gate` — the K3 epilogue on its own, for span rows: the
// FlashKDA span kernel (K8) writes the raw attention of the span's rows
// 0..span into `attn`, and this finishes them into the batch rows
// [*span_at, +span) of `gated` with the output rms, gamma_o and the sigmoid
// gate, with K3's landing points. Contract: docs/k3-kernel-abi.md section K11.
//
//   extern "C" __global__ void kern_k3_kda_out_gate(
//       const bf16* attn,          // [span, INNER]
//       const part* gate_partial,  // [rows, KDA_FUSED]  band 3 only, rows at..at+span; f32, or bf16
//                                  // with -DPARTIAL_BF16 (then the landing below is exact)
//       const float* gamma_o,      // [128]
//       bf16* gated,               // [rows, INNER]  rows at..at+span written
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid (span, HEADS / 4, 1)   block 128   (i = blockIdx.x, b = at + i; warp w is head
//   h = 4 * blockIdx.y + w, lane l its dv 4l .. 4l + 4)
//
//   a       = f32(attn[i, h*128 + dv])
//   r       = rsqrt(mean_dv(a^2) + 1e-5)
//   o       = bf16(a * r * gamma_o[dv])
//   gt      = bf16(sigmoid(f32(bf16(gate_partial[b, 3*INNER + h*128 + dv]))))
//   gated[b, h*128 + dv] = bf16(f32(o) * f32(gt))
//
// The sum of squares is the tree of one thread per dv: four 32-lane xor
// butterflies, (((w0 + w1) + w2) + w3). Lane l's dv 4l + j is that tree's lane
// 4(l & 7) + j of warp l >> 3, so three butterfly steps over l & 7 per j,
// then j's two bits in the same order, give each warp's sum bit for bit.
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 96
#endif
#define K11_INNER (HEADS * 128)
#define K11_KDA_FUSED (4 * K11_INNER)
#define K11_RMS_EPS 1e-5f

typedef __nv_bfloat16 bf16;

// -DPARTIAL_BF16: the gate arrives landed (a bf16 GEMM output).
#ifdef PARTIAL_BF16
typedef bf16 part_t;
#else
typedef float part_t;
#endif

extern "C" __global__ __launch_bounds__(128) void kern_k3_kda_out_gate(
    const bf16* __restrict__ attn, const part_t* __restrict__ gate_partial, const float* __restrict__ gamma_o,
    bf16* __restrict__ gated, const int* __restrict__ span_at, int span) {
  const int i = blockIdx.x, h = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, d = lane * 4;
  const int b = span_at[0] + i;
  const uint2 raw = *(const uint2*)(attn + (size_t)i * K11_INNER + (size_t)h * 128 + d);
  const __nv_bfloat162* a2 = (const __nv_bfloat162*)&raw;
  float a[4], v[4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const float2 f = __bfloat1622float2(a2[j]);
    a[2 * j] = f.x;
    a[2 * j + 1] = f.y;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = a[j] * a[j];
#pragma unroll
  for (int o = 4; o > 0; o >>= 1)
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] += __shfl_xor_sync(0xffffffffu, v[j], o);
  const float w = (v[0] + v[2]) + (v[1] + v[3]);
  const float tot = __shfl_sync(0xffffffffu, w, 0) + __shfl_sync(0xffffffffu, w, 8) +
                    __shfl_sync(0xffffffffu, w, 16) + __shfl_sync(0xffffffffu, w, 24);
  const float r = rsqrtf(tot * (1.0f / 128.0f) + K11_RMS_EPS);
  const float4 go = *(const float4*)(gamma_o + d);
  const part_t* gp = gate_partial + (size_t)b * K11_KDA_FUSED + 3 * K11_INNER + (size_t)h * 128 + d;
  const float gam[4] = {go.x, go.y, go.z, go.w};
  float gpart[4];
#ifdef PARTIAL_BF16
  const uint2 graw = *(const uint2*)gp;
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&graw;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const float2 f = __bfloat1622float2(g2[j]);
    gpart[2 * j] = f.x;
    gpart[2 * j + 1] = f.y;
  }
#else
  const float4 g4 = *(const float4*)gp;
  gpart[0] = g4.x; gpart[1] = g4.y; gpart[2] = g4.z; gpart[3] = g4.w;
#endif
  bf16 out[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const bf16 o = __float2bfloat16(a[j] * r * gam[j]);
    const float g = __bfloat162float(__float2bfloat16(gpart[j]));
    const bf16 gt = __float2bfloat16(1.0f / (1.0f + __expf(-g)));
    out[j] = __float2bfloat16(__bfloat162float(o) * __bfloat162float(gt));
  }
  *(uint2*)(gated + (size_t)b * K11_INNER + (size_t)h * 128 + d) = *(const uint2*)out;
}
