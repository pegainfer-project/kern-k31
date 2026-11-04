// Rows of bf16 to MXFP8 for cuBLASLt's block-scaled GEMM (extern:cublaslt_mxfp8_tn), the way vLLM's online
// mxfp8 quantizes a weight at load and an activation per call (FlashInfer's mxfp8_quantize, the same per
// block as k3_moe_prefill.cu's kern_k3_moe_quant):
//
//   kern_k3_mxfp8_quant(const bf16* x, int ld, u8* q, u8* sf, int rows, int cols)
//     per 32 elements of a row: e = ceil(log2(amax / 448)) (0 when amax is 0), sf = e + 127 (UE8M0),
//     q = e4m3(x / 2^e) round-to-nearest; x rows `ld` apart, q dense [rows, cols], sf in cuBLAS's 128x4
//     tiled layout: block (r, b) at ((r / 128) * ceil(cols / 128) + b / 4) * 512 + (r % 32) * 16 +
//     (r % 128 / 32) * 4 + b % 4. A thread a block, grid-stride.
//   grid (any, 1, 1)   block (256, 1, 1)
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ unsigned short e4m3x2(float lo, float hi) {
  unsigned short d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_mxfp8_quant(const __nv_bfloat16* __restrict__ x, int ld,
                                                                      uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                                                      int rows, int cols) {
  const int kb = cols / 32, kb_tiles = (kb + 3) / 4;
  const long long n = (long long)rows * kb;
  for (long long i = blockIdx.x * 256ll + threadIdx.x; i < n; i += gridDim.x * 256ll) {
    const int r = (int)(i / kb), b = (int)(i % kb);
    const uint4* src = reinterpret_cast<const uint4*>(x + (long long)r * ld + b * 32);
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 u = src[j];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const float2 f = __bfloat1622float2(p[l]);
        v[j * 8 + l * 2] = f.x;
        v[j * 8 + l * 2 + 1] = f.y;
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : 0;
    e = min(max(e, -127), 127);
    const float s = exp2f((float)-e);
    sf[(long long)(r / 128 * kb_tiles + b / 4) * 512 + r % 32 * 16 + r % 128 / 32 * 4 + b % 4] = (uint8_t)(e + 127);
    unsigned w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      w[j] = e4m3x2(v[4 * j] * s, v[4 * j + 1] * s) | (unsigned)e4m3x2(v[4 * j + 2] * s, v[4 * j + 3] * s) << 16;
    uint4* dst = reinterpret_cast<uint4*>(q + i * 32);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  }
}
