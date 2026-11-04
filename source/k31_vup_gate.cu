// K3.1 MLA epilogue for a long call: the sparse attention's latent output through W_UV to the head dim, met
// by the sigmoid gate, on tensor cores. D's batch takes k3_mla_vup_gate's bandwidth form (or k3_dcp's
// exchange); a prefill chunk's thousands of rows make it a GEMM a head (rows x 512 by 512 x 128).
//
//   extern "C" __global__ void kern_k31_vup_gate(
//       const bf16* o_lat,     // [B, IN_HEADS, 512]  the sparse kernel's output, heads padded to IN_HEADS
//       const bf16* w_kv_b,    // [HEADS*256, 512]    W_UV = rows h*256+128..256
//       const bf16* gate,      // row b's gate columns at gate + b * gate_stride, [HEADS*128]
//       int gate_stride,
//       bf16*       gated,     // [B, HEADS*128]
//       int B);
//
//   grid (ceil(B / BM), HEADS, 1)   block (256, 1, 1)   smem SMEM_BYTES dynamic
//
// Block (bt, h) owns rows BM bt .. +BM of head h, all 128 dv: the 512-long reduction streams through
// shared memory 64 columns at a time (cp.async, STAGES deep; rows past B zero-filled), and each of the
// 8 warps multiplies a BM/2-row x 32-dv tile with mma.sync m16n8k16 (bf16 in, f32 accumulate), A and W
// both K-major, so neither ldmatrix transposes. Every block's order is fixed: a result never depends on
// the schedule.
//
//   o[dv]  = bf16(sum_j lat[j] * W_UV_h[dv, j])                      f32 acc, the tensor core's order
//   gated  = bf16(o[dv] * bf16(sigmoid(f32(gate[h*128 + dv]))))      sigmoid_mul's arithmetic
//
//   nvcc -cubin -arch=sm_103a -O3 source/k31_vup_gate.cu
#include <cuda_bf16.h>

#define HEADS 96
#ifndef IN_HEADS
#define IN_HEADS 128
#endif
#define NOPE 128
#define LAT 512
#define BM 64
#define MT (BM / 32)  // a warp's 16-row slices: 2 warps of rows x 4 of dv
#define KC 64
#define PITCH (KC + 8)  // 144-byte rows: the eight rows of an ldmatrix land in distinct banks
#define STAGES 3
#define A_ELEMS (BM * PITCH)
#define W_ELEMS (NOPE * PITCH)
#define SMEM_BYTES (STAGES * (A_ELEMS + W_ELEMS) * 2)

__device__ __forceinline__ void cp16(unsigned dst, const void* src, bool full) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(full ? 16 : 0));
}

extern "C" __global__ void __launch_bounds__(256) kern_k31_vup_gate(const __nv_bfloat16* __restrict__ o_lat,
                                                                   const __nv_bfloat16* __restrict__ w_kv_b,
                                                                   const __nv_bfloat16* __restrict__ gate,
                                                                   int gate_stride, __nv_bfloat16* __restrict__ gated,
                                                                   int B) {
  extern __shared__ __align__(128) __nv_bfloat16 sm[];
  const int h = blockIdx.y, r0 = blockIdx.x * BM, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int wm = warp / 4 * (BM / 2), wn = warp % 4 * 32;
  const unsigned s0 = (unsigned)__cvta_generic_to_shared(sm);
  const __nv_bfloat16* __restrict__ w = w_kv_b + ((size_t)h * 256 + NOPE) * LAT;

  // stage s, chunk kc: A BM rows x 8 vectors (BM / 32 a thread), W 128 rows x 8 vectors (4 a thread)
  auto load = [&](int s, int kc) {
    const unsigned a = s0 + s * (A_ELEMS + W_ELEMS) * 2, wb = a + A_ELEMS * 2;
#pragma unroll
    for (int k = 0; k < BM / 32; ++k) {
      const int i = t + k * 256, r = i / 8, v = i % 8, b = r0 + r;
      cp16(a + (r * PITCH + v * 8) * 2, o_lat + ((size_t)min(b, B - 1) * IN_HEADS + h) * LAT + kc * KC + v * 8,
           b < B);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = t + k * 256, r = i / 8, v = i % 8;
      cp16(wb + (r * PITCH + v * 8) * 2, w + (size_t)r * LAT + kc * KC + v * 8, true);
    }
    asm volatile("cp.async.commit_group;");
  };

  float c[MT][4][4] = {};
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) load(s, s);
#pragma unroll 1
  for (int kc = 0; kc < LAT / KC; ++kc) {
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
    __syncthreads();
    if (kc + STAGES - 1 < LAT / KC) load((kc + STAGES - 1) % STAGES, kc + STAGES - 1);
    else asm volatile("cp.async.commit_group;");
    const unsigned a = s0 + kc % STAGES * (A_ELEMS + W_ELEMS) * 2, wb = a + A_ELEMS * 2;
#pragma unroll
    for (int k = 0; k < KC / 16; ++k) {
      unsigned af[MT][4], bf[2][4];
#pragma unroll
      for (int m = 0; m < MT; ++m)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3])
                     : "r"(a + ((wm + m * 16 + lane % 8 + 8 * (lane / 8 % 2)) * PITCH + k * 16 + 8 * (lane / 16)) * 2));
      // W rows are dv, columns j: matrices (n 0-7, k 0-7), (n 0-7, k 8-15), (n 8-15, k 0-7), (n 8-15, k 8-15)
#pragma unroll
      for (int n = 0; n < 2; ++n)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3])
                     : "r"(wb + ((wn + n * 16 + lane % 8 + 8 * (lane / 16)) * PITCH + k * 16 + 8 * (lane / 8 % 2)) * 2));
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                       "{%8, %9}, {%0, %1, %2, %3};"
                       : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]), "+f"(c[m][n][3])
                       : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]), "r"(bf[n / 2][n % 2 * 2]),
                         "r"(bf[n / 2][n % 2 * 2 + 1]));
    }
  }

  // c[m][n] holds rows g and g + 8 of tile m, columns n * 8 + 2 tq and + 1
  const int g = lane / 4, tq = lane % 4;
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int b = r0 + wm + m * 16 + g + 8 * hr;
      if (b >= B) continue;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int col = h * NOPE + wn + n * 8 + 2 * tq;
        const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + (size_t)b * gate_stride + col));
        const float2 o = __bfloat1622float2(__floats2bfloat162_rn(c[m][n][2 * hr], c[m][n][2 * hr + 1]));
        const float2 s = __bfloat1622float2(__floats2bfloat162_rn(1.0f / (1.0f + expf(-x.x)), 1.0f / (1.0f + expf(-x.y))));
        *reinterpret_cast<__nv_bfloat162*>(gated + (size_t)b * (HEADS * NOPE) + col) =
            __floats2bfloat162_rn(o.x * s.x, o.y * s.y);
      }
    }
}
