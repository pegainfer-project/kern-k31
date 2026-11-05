// `k3_kda_epilogue` — what a packed P's KDA layer does after its core: K11's
// output gate on every row, in place, and the conv windows advanced (K9's last
// step). The core (cuDNN's FROST KDA, scripts/cudnn_kda_abi.py) reads q / k / v
// to the end of its persistent run, so its output lands in `o` and is gated
// there; the windows wait for it too, since the gather before the core read them.
//
//   extern "C" __global__ void kern_k3_kda_epilogue(
//       bf16* o,                  // [rows, INNER]  the core's output, gated in place
//       const bf16* partial,      // [rows, LD]  q | k | v | gate projections (then wsm's, if LD > 4 * INNER)
//       const float* gamma_o,     // [128]
//       void* kda_base, const int* line_index, long long line_bytes,  // sequence j's line line_index[j]
//       const long long* cu_seqlens,  // sequence j: rows cu[j]..cu[j+1]
//       int nseq, int rows);
//
//   grid (rows + nseq, 3)   block 512   programmatic dependent of the core
//
// Block (i < rows, y) gates row i's heads 32y..32y+32, 16 lanes a head: lane k
// sums the squares of K11's lane group (columns 32(k/4) + 4l + k%4, l < 8) in
// K11's butterfly order, then finishes columns 8k..8k+8 with K11's operations,
// so the result is K11's bit for bit. Block
// (rows + j, s) moves sequence j's last three rows of stream s into its window,
// each thread reading a column's three rows before writing them, so a
// sequence shorter than three rows shifts its old window correctly.
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef HEADS
#define HEADS 96
#endif
#define KE_INNER (HEADS * 128)
#ifndef KE_LD
#define KE_LD (4 * KE_INNER)
#endif
#define KE_REC_BYTES ((long long)HEADS * 128 * 128 * 2)  // the line's bf16 recurrent state ahead of the windows
#define KE_WIN_BYTES ((long long)3 * KE_INNER * 2)
#define KE_THREADS 512
#define KE_HEADS_A_BLOCK (KE_THREADS / 16)
#define KE_RMS_EPS 1e-5f

static_assert(HEADS % KE_HEADS_A_BLOCK == 0, "a block gates 32 heads of a row");
static_assert(HEADS / KE_HEADS_A_BLOCK == 3, "grid.y covers the head groups and the three streams alike");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float ke_lo(unsigned x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float ke_hi(unsigned x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ unsigned ke_pack(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *(const unsigned*)&p;
}

// 1 / x as IEEE division computes it (reciprocal, two Newton steps) for x in [1, 2^126)
__device__ __forceinline__ float ke_recip(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  r = fmaf(r, fmaf(-x, r, 1.0f), r);
  return fmaf(r, fmaf(-x, r, 1.0f), r);
}

__device__ __forceinline__ void ke_gate(bf16* __restrict__ o, const bf16* __restrict__ partial,
                                        const float* __restrict__ gamma_o, int i, int y) {
  const int h = KE_HEADS_A_BLOCK * y + (threadIdx.x >> 4), k = threadIdx.x & 15, grp = 32 * (k >> 2);
  bf16* row = o + (size_t)i * KE_INNER + h * 128;
  uint4 rq[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) rq[q] = *(const uint4*)(row + grp + 8 * q);
  const uint4 graw = *(const uint4*)(partial + (size_t)i * KE_LD + 3 * KE_INNER + h * 128 + 8 * k);
  float gam[8];
  {
    const float4 g0 = *(const float4*)(gamma_o + 8 * k), g1 = *(const float4*)(gamma_o + 8 * k + 4);
    gam[0] = g0.x, gam[1] = g0.y, gam[2] = g0.z, gam[3] = g0.w, gam[4] = g1.x, gam[5] = g1.y, gam[6] = g1.z,
    gam[7] = g1.w;
  }
  float y2[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    // columns 32G + 8q .. +8: l = 2q at +0..3, l = 2q + 1 at +4..7
    const unsigned w0 = (k & 2) ? rq[q].y : rq[q].x, w1 = (k & 2) ? rq[q].w : rq[q].z;
    const float x0 = (k & 1) ? ke_hi(w0) : ke_lo(w0), x1 = (k & 1) ? ke_hi(w1) : ke_lo(w1);
    y2[2 * q] = x0 * x0;
    y2[2 * q + 1] = x1 * x1;
  }
  const float sj = ((y2[0] + y2[4]) + (y2[2] + y2[6])) + ((y2[1] + y2[5]) + (y2[3] + y2[7]));
  const float t1 = sj + __shfl_xor_sync(0xffffffffu, sj, 2);
  const float wsum = t1 + __shfl_xor_sync(0xffffffffu, t1, 1);
  const int base = threadIdx.x & 16;
  const float tot = __shfl_sync(0xffffffffu, wsum, base) + __shfl_sync(0xffffffffu, wsum, base + 4) +
                    __shfl_sync(0xffffffffu, wsum, base + 8) + __shfl_sync(0xffffffffu, wsum, base + 12);
  const float rr = rsqrtf(tot * (1.0f / 128.0f) + KE_RMS_EPS);
  const uint4 araw = (k & 2) ? ((k & 1) ? rq[3] : rq[2]) : ((k & 1) ? rq[1] : rq[0]);  // rq[k % 4], in registers
  const unsigned aw[4] = {araw.x, araw.y, araw.z, araw.w}, gwd[4] = {graw.x, graw.y, graw.z, graw.w};
  float ov[8], den[8], sig[8];
  bool far = false;
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const float a = (x & 1) ? ke_hi(aw[x / 2]) : ke_lo(aw[x / 2]);
    const float g = (x & 1) ? ke_hi(gwd[x / 2]) : ke_lo(gwd[x / 2]);
    ov[x] = __bfloat162float(__float2bfloat16(a * rr * gam[x]));
    // K11's sigmoid: __expf (its ex2 wherever the result is not subnormal, and then 1 + it is 1 either way)
    float ex;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(-g * 1.4426950408889634f));
    den[x] = 1.0f + ex;
    sig[x] = ke_recip(den[x]);
    far |= den[x] >= 0x1p126f;
  }
  if (__any_sync(0xffffffffu, far)) {
#pragma unroll
    for (int x = 0; x < 8; ++x)
      if (den[x] >= 0x1p126f) sig[x] = 1.0f / den[x];
  }
  uint4 res;
  unsigned* rw = (unsigned*)&res;
#pragma unroll
  for (int x = 0; x < 8; x += 2) {
    const unsigned gt = ke_pack(sig[x], sig[x + 1]);
    rw[x / 2] = ke_pack(ov[x] * ke_lo(gt), ov[x + 1] * ke_hi(gt));
  }
  // every lane of the half-warp has read the head's row before any writes it
  __syncwarp();
  *(uint4*)(row + 8 * k) = res;
}

__device__ __forceinline__ void ke_window(const bf16* __restrict__ partial, void* __restrict__ kda_base,
                                          const int* __restrict__ line_index, long long line_bytes,
                                          const long long* __restrict__ cu_seqlens, int j, int s) {
  const long long bos = cu_seqlens[j];
  const int len = (int)(cu_seqlens[j + 1] - bos);
  if (len <= 0) return;
  bf16* win = (bf16*)((char*)kda_base + (long long)line_index[j] * line_bytes + KE_REC_BYTES + s * KE_WIN_BYTES);
  for (int c = 8 * threadIdx.x; c < KE_INNER; c += 8 * KE_THREADS) {
    uint4 nt[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int i = len - 3 + k;
      nt[k] = i >= 0 ? *(const uint4*)(partial + (bos + i) * KE_LD + s * KE_INNER + c)
                     : *(const uint4*)(win + (i + 3) * KE_INNER + c);
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) *(uint4*)(win + k * KE_INNER + c) = nt[k];
  }
}

extern "C" __global__ void __launch_bounds__(KE_THREADS) kern_k3_kda_epilogue(
    bf16* __restrict__ o, const bf16* __restrict__ partial, const float* __restrict__ gamma_o,
    void* __restrict__ kda_base, const int* __restrict__ line_index, long long line_bytes,
    const long long* __restrict__ cu_seqlens, int nseq, int rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if ((int)blockIdx.x < rows)
    ke_gate(o, partial, gamma_o, blockIdx.x, blockIdx.y);
  else
    ke_window(partial, kda_base, line_index, line_bytes, cu_seqlens, blockIdx.x - rows, blockIdx.y);
}
