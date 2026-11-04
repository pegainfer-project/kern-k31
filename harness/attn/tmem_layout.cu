// Where a CTA pair's tcgen05 MMA (cta_group::2, kind::f16) puts D in each CTA's TMEM, for M = 128 and 256.
// A row r (of M, split by CTA) is [r + 1, 256, 0 ...], B column n (of N, split by CTA) is [1, n + 1, 0 ...], so
// D[r][n] = r + 1 + 256 (n + 1); every CTA dumps its 128 lanes x N columns and the host decodes (r, n) per lane.
//   $K3_NVCC -O2 -gencode arch=compute_103a,code=sm_103a harness/attn/tmem_layout.cu -o OUT && loop/withgpu -n 1 OUT
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>

constexpr int N = 64;

__device__ __forceinline__ uint64_t desc(uint32_t addr) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
}

template <int M>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) probe(float* out) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t base = ((uint32_t)__cvta_generic_to_shared(sm) + 1023) & ~1023u;
  uint8_t* gen = sm + (base - (uint32_t)__cvta_generic_to_shared(sm));
  const uint32_t a = base, b = base + 16384, bar = base + 32768, hold = bar + 8;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int t = threadIdx.x, warp = t / 32, rows = M / 2, cols = N / 2;
  // A: this CTA's rows, K-major 128 B swizzle (8-row atoms of 128 B, chunk c of row r at c ^ (r % 8))
  for (int r = t; r < rows; r += 128) {
    __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(gen + (r / 8) * 1024 + (r % 8) * 128);
    for (int k = 0; k < 64; ++k) row[((k / 8) ^ (r % 8)) * 8 + k % 8] = __float2bfloat16(k == 0 ? rank * rows + r + 1 : k == 1 ? 256 : 0);
  }
  for (int n = t; n < cols; n += 128) {
    __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(gen + 16384 + (n / 8) * 1024 + (n % 8) * 128);
    for (int k = 0; k < 64; ++k) row[((k / 8) ^ (n % 8)) * 8 + k % 8] = __float2bfloat16(k == 0 ? 1 : k == 1 ? rank * cols + n + 1 : 0);
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
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(gen + 32768 + 8);
  if (rank == 0 && warp == 0) {
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
    asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, 0;\n"
                 "@e tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%4], %5;\n}"
                 ::"r"(tmem), "l"(desc(a)), "l"(desc(b)), "r"(idesc), "r"(bar), "h"((uint16_t)3));
  }
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}" ::"r"(bar) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (int c = 0; c < N; c += 8) {
    uint32_t v[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "r"(tmem + ((32 * warp) << 16) + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    for (int i = 0; i < 8; ++i) out[(rank * 128 + t) * N + c + i] = __uint_as_float(v[i]);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}

template <int M>
void run(float* d) {
  cudaFuncSetAttribute(probe<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960);
  cudaMemset(d, 0, 2 * 128 * N * 4);
  probe<M><<<2, 128, 40960>>>(d);
  printf("M=%d: %s\n", M, cudaGetErrorString(cudaDeviceSynchronize()));
  static float h[2 * 128 * N];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  for (int cta = 0; cta < 2; ++cta) {
    printf(" CTA %d lane: row, column range (lanes with zeros skipped)\n", cta);
    for (int lane = 0; lane < 128; lane += 1) {
      const float* row = h + (cta * 128 + lane) * N;
      int lo = 1 << 30, hi = -1, r = -1, ok = 1;
      for (int c = 0; c < N; ++c) {
        const int v = (int)row[c];
        if (!v) { ok = 0; continue; }
        const int rr = v % 256 - 1, nn = v / 256 - 1;
        if (r < 0) r = rr; else if (r != rr) ok = 0;
        lo = nn < lo ? nn : lo, hi = nn > hi ? nn : hi;
      }
      if (r >= 0 && (lane % 16 == 0 || lane % 32 == 31)) printf("  lane %3d: row %3d, columns %d..%d%s\n", lane, r, lo, hi, ok ? "" : " (gaps)");
      if (r < 0 && lane % 32 == 0) printf("  lane %3d: empty\n", lane);
    }
  }
}

int main() {
  float* d; cudaMalloc(&d, 2 * 128 * N * 4);
  run<128>(d); run<256>(d);
}
