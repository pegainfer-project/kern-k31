// K6/K7 -- generic rms + the two f32-partial landing kernels.
//
//   nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
//        -o target/cubins/k3_land.cubin tools/kernels-src/k3_land.cu
//
// Entries (docs/k3-kernel-abi.md sections K6 / K7):
//
//   extern "C" __global__ void kern_k3_rms(
//       const bf16* x, const bf16* gamma, bf16* o, int h, int B);
//   grid (B, 1, 1)   block 1024   smem 132 B static, 0 dynamic
//
//   extern "C" __global__ void kern_k3_land(
//       const f32* p, bf16* o, int n, int off, int ldc, int B);
//   grid (B, ceil(n/1024))   block 1024   smem 0
//
//   extern "C" __global__ void kern_k3_land_situ(
//       const f32* p, bf16* act, int n, int B);
//   grid (B, ceil(n/1024))   block 1024   smem 0
//
// Math / landing points
// ---------------------
// rms  (round-before-scale, EPS = 1e-5, generic row width h, used with h=3584):
//        y[i] = bf16( f32(x[i]) * rsqrt( mean_i(f32(x[i])^2) + 1e-5 ) )
//        o[i] = y[i] * gamma[i]                     <- bf16 * bf16 -> bf16
//      Two landings, exactly as pegainfer: one on x*rsqrt, one on the product.
//      Sum of squares is f32 (per-thread serial, then shuffle + smem tree).
//
// land: o[b,i] = bf16( p[b*ldc + off + i] ),  i < n         one landing.
//
// land_situ (gate in the first n columns, up in the next n; ldc == 2n, or
// SITU_LDS when the pair sits inside a fused GEMM's wider row):
//        g = f32(bf16( p[b*2n + i] ));  u = f32(bf16( p[b*2n + n + i] ))
//        act[b,i] = bf16( 4*tanh(g/4) * sigmoid(g) * 25*tanh(u/25) )
//      Three landings: both operands land to bf16 *before* the activation (the
//      pegainfer chain lands the f32 partial and only then applies situ), and
//      the result lands.
//
// Access pattern / performance notes
// ----------------------------------
// * All three are pure streams. The documented grid gives exactly one element
//   per thread for land/land_situ; we hand each thread one float4 instead and
//   keep a grid-stride loop, so the same block count covers the row with 4x
//   fewer, 4x wider memory instructions (16 B loads, 8 B stores). Bytes in
//   flight per block are unchanged. Any n/off/ldc that is not a multiple of 4
//   falls back to a scalar grid-stride loop, so the kernels stay correct for
//   shapes outside the K3 set.
// * land_situ is transcendental-bound, not bandwidth-bound: three special
//   functions per element. `tanh.approx.f32` is one SFU instruction and is
//   accurate to 7.8e-6 absolute / 1.05e-5 relative over |x| <= 8 (measured on
//   sm_103a), i.e. ~11 bits below one bf16 ULP even after the x25 scale, and
//   __expf likewise. That took n=33792 B=64 from 11.2 us to 6.8 us.
// * rms prefetches gamma alongside x in pass 1, so the gamma round trip hides
//   behind the sum-of-squares reduction instead of serialising after it
//   (1.93 -> 1.53 us at B=1).
#include <cuda_bf16.h>

typedef unsigned int u32;

// ------------------------------------------------------------------------ rms
//
// Pass 1 keeps up to RMS_REGS * blockDim.x uint4s of x AND gamma (8 bf16 each)
// live in registers, so each is read exactly once; rows longer than that
// (h > 32768 at block 1024) stream the overflow twice, out of L1. No dynamic
// indexing into the register arrays -- the slot loop is unrolled.
#define RMS_REGS 4

__device__ __forceinline__ void bf16x8_to_f32(const uint4& w, float* f) {
  const __nv_bfloat162* h2 = (const __nv_bfloat162*)&w;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 t = __bfloat1622float2(h2[k]);
    f[2 * k] = t.x;
    f[2 * k + 1] = t.y;
  }
}
__device__ __forceinline__ uint4 rms_scale8(const uint4& xw, const uint4& gw,
                                            float rs) {
  const __nv_bfloat162* x2 = (const __nv_bfloat162*)&xw;
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&gw;
  uint4 ow;
  __nv_bfloat162* o2 = (__nv_bfloat162*)&ow;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(x2[k]);
    o2[k] = __hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), g2[k]);
  }
  return ow;
}

__device__ __forceinline__ float block_sum(float v, float* sm) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < nwarps) ? sm[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) sm[32] = v;
  }
  __syncthreads();
  return sm[32];
}

extern "C" __global__ void __launch_bounds__(1024, 1) kern_k3_rms(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ o, int h, int B) {
  __shared__ float sm[33];

  const int b = blockIdx.x;
  if (b >= B) return;
  const __nv_bfloat16* __restrict__ xr = x + (long long)b * h;
  __nv_bfloat16* __restrict__ orow = o + (long long)b * h;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  float sum = 0.0f;

  if ((h & 7) == 0) {  // 16 B aligned rows: 8 bf16 per vector
    const int hv = h >> 3;
    const uint4* xv = (const uint4*)xr;
    const uint4* gv = (const uint4*)gamma;
    uint4* ov = (uint4*)orow;
    uint4 rx[RMS_REGS], rg[RMS_REGS];
    float f[8];

#pragma unroll
    for (int r = 0; r < RMS_REGS; ++r) {
      int u = tid + r * nt;
      if (u < hv) {
        rx[r] = xv[u];
        rg[r] = gv[u];  // prefetched: hides behind the reduction below
        bf16x8_to_f32(rx[r], f);
#pragma unroll
        for (int k = 0; k < 8; ++k) sum += f[k] * f[k];
      }
    }
    for (int u = tid + RMS_REGS * nt; u < hv; u += nt) {  // overflow: re-read
      bf16x8_to_f32(xv[u], f);
#pragma unroll
      for (int k = 0; k < 8; ++k) sum += f[k] * f[k];
    }

    const float rs = rsqrtf(block_sum(sum, sm) / (float)h + 1e-5f);

#pragma unroll
    for (int r = 0; r < RMS_REGS; ++r) {
      int u = tid + r * nt;
      if (u < hv) ov[u] = rms_scale8(rx[r], rg[r], rs);
    }
    for (int u = tid + RMS_REGS * nt; u < hv; u += nt)
      ov[u] = rms_scale8(xv[u], gv[u], rs);
  } else {  // scalar fallback for row widths that are not a multiple of 8
    for (int i = tid; i < h; i += nt) {
      float v = __bfloat162float(xr[i]);
      sum += v * v;
    }
    const float rs = rsqrtf(block_sum(sum, sm) / (float)h + 1e-5f);
    for (int i = tid; i < h; i += nt) {
      __nv_bfloat16 y = __float2bfloat16(__bfloat162float(xr[i]) * rs);
      orow[i] = __hmul(y, gamma[i]);
    }
  }
}

// ----------------------------------------------------------------------- land
extern "C" __global__ void __launch_bounds__(1024, 1) kern_k3_land(
    const float* __restrict__ p, __nv_bfloat16* __restrict__ o, int n, int off,
    int ldc, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const float* __restrict__ pr = p + (long long)b * ldc + off;
  __nv_bfloat16* __restrict__ orow = o + (long long)b * n;
  const int stride = gridDim.y * blockDim.x;
  const int i0 = blockIdx.y * blockDim.x + threadIdx.x;

  if (((n | off | ldc) & 3) == 0) {  // 16 B loads / 8 B stores
    const int nv = n >> 2;
    const float4* pv = (const float4*)pr;
    for (int u = i0; u < nv; u += stride) {
      float4 v = pv[u];
      __nv_bfloat162 out2[2];
      out2[0] = __floats2bfloat162_rn(v.x, v.y);
      out2[1] = __floats2bfloat162_rn(v.z, v.w);
      *(uint2*)(orow + (u << 2)) = *(const uint2*)out2;
    }
  } else {
    for (int i = i0; i < n; i += stride) orow[i] = __float2bfloat16(pr[i]);
  }
}

// ------------------------------------------------------------------ land_situ
__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float situ_f(float pg, float pu) {
  // land both operands to bf16 first, then the activation in f32
  float g = __bfloat162float(__float2bfloat16(pg));
  float u = __bfloat162float(__float2bfloat16(pu));
  float a = 4.0f * tanh_approx(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

extern "C" __global__ void __launch_bounds__(1024, 2) kern_k3_land_situ(
    const float* __restrict__ p, __nv_bfloat16* __restrict__ act, int n, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
#ifdef SITU_LDS
  const float* __restrict__ pg = p + (long long)b * SITU_LDS;
#else
  const float* __restrict__ pg = p + (long long)b * 2 * n;
#endif
  const float* __restrict__ pu = pg + n;
  __nv_bfloat16* __restrict__ orow = act + (long long)b * n;
  const int stride = gridDim.y * blockDim.x;
  const int i0 = blockIdx.y * blockDim.x + threadIdx.x;

  if ((n & 3) == 0) {
    const int nv = n >> 2;
    const float4* gv = (const float4*)pg;
    const float4* uv = (const float4*)pu;
    for (int k = i0; k < nv; k += stride) {
      float4 g = gv[k];
      float4 u = uv[k];
      __nv_bfloat162 out2[2];
      out2[0] = __floats2bfloat162_rn(situ_f(g.x, u.x), situ_f(g.y, u.y));
      out2[1] = __floats2bfloat162_rn(situ_f(g.z, u.z), situ_f(g.w, u.w));
      *(uint2*)(orow + (k << 2)) = *(const uint2*)out2;
    }
  } else {
    for (int i = i0; i < n; i += stride)
      orow[i] = __float2bfloat16(situ_f(pg[i], pu[i]));
  }
}
