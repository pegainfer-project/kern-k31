// Where a CTA pair's tcgen05 MMA (cta_group::2, kind::f16, M = 128) reads A from TMEM: each CTA writes A at TMEM
// lane L, column CA + j as bf16x2 (lo, hi) = probe values, B is the identity (N = 32, 16 a CTA), so D[r][n] =
// A[r][n % 16]; two probes give the lane (value L + 1) and the column half (value 2j + h + 1) each A[r][k] came from.
//   $K3_NVCC -O2 -gencode arch=compute_103a,code=sm_103a harness/attn/tmem_a_layout.cu -o OUT && loop/withgpu -n 1 OUT
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>

__device__ __forceinline__ uint64_t desc(uint32_t addr) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
}

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) probe(float* out, int which) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t base = ((uint32_t)__cvta_generic_to_shared(sm) + 1023) & ~1023u;
  uint8_t* gen = sm + (base - (uint32_t)__cvta_generic_to_shared(sm));
  const uint32_t b = base, bar = base + 16384, hold = bar + 8;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int t = threadIdx.x, warp = t / 32;
  // B: this CTA's 16 N rows, K-major 128 B swizzle, the identity over K 0..15
  for (int n = t; n < 16; n += 128) {
    __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(gen + (n / 8) * 1024 + (n % 8) * 128);
    for (int k = 0; k < 64; ++k) row[((k / 8) ^ (n % 8)) * 8 + k % 8] = __float2bfloat16(k == n ? 1.f : 0.f);
  }
  if (t == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(hold));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(gen + 16384 + 8);
  {
    uint32_t v[8];
    for (int j = 0; j < 8; ++j) {
      const float lo = which ? 2 * j + 1 : t + 1, hi = which ? 2 * j + 2 : t + 1;
      __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
      v[j] = *reinterpret_cast<uint32_t*>(&p);
    }
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                 ::"r"(tmem + ((32 * warp) << 16) + 256), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]),
                   "r"(v[6]), "r"(v[7]));
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (rank == 0 && warp == 0) {
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((32u >> 3) << 17) | ((128u >> 4) << 24);
    asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, 0;\n"
                 "@e tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%4], %5;\n}"
                 ::"r"(tmem), "r"(tmem + 256), "l"(desc(b)), "r"(idesc), "r"(bar), "h"((uint16_t)3));
  }
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}" ::"r"(bar) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t v[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]),
                 "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
               : "r"(tmem + ((32 * warp) << 16)));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int i = 0; i < 16; ++i) out[(rank * 128 + t) * 16 + i] = __uint_as_float(v[i]);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}

int main() {
  float* d; cudaMalloc(&d, 2 * 128 * 16 * 4);
  static float h[2][2 * 128 * 16];
  cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 20480);
  for (int w = 0; w < 2; ++w) {
    cudaMemset(d, 0, sizeof h[w]);
    probe<<<2, 128, 20480>>>(d, w);
    printf("probe %d: %s\n", w, cudaGetErrorString(cudaDeviceSynchronize()));
    cudaMemcpy(h[w], d, sizeof h[w], cudaMemcpyDeviceToHost);
  }
  // D lane L (CTA c), column n: A[row][k] with the D layout's row = L % 64, k = n (+ 16 for lanes 64..127 being n 16..31)
  for (int c = 0; c < 2; ++c)
    for (int L : {0, 1, 31, 32, 63, 64, 65, 127}) {
      printf("cta %d D lane %3d:", c, L);
      for (int n = 0; n < 16; ++n) printf(" %g/%g", h[0][(c * 128 + L) * 16 + n], h[1][(c * 128 + L) * 16 + n]);
      printf("\n");
    }
}
