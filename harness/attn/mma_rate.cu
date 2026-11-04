// tcgen05.mma kind::f16 (bf16 in, f32 acc) issue throughput per SM by shape.
// One CTA (or a CTA pair) an SM, operands in smem (SW128 K-major, contents irrelevant),
// ITERS x 4 k16 MMAs into one accumulator, then commit + wait.
//   $K3_NVCC -O2 -gencode arch=compute_103a,code=sm_103a harness/attn/mma_rate.cu -o OUT && loop/withgpu -n 1 OUT
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITERS 4096

__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(a), "r"(ph) : "memory");
}

template <int M, int N, int CG>
__global__ void __launch_bounds__(128, 1) rate(long long* cyc) {
  extern __shared__ __align__(1024) char sm[];
  uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  base = (base + 1023) & ~1023u;
  uint32_t bar = base, hold = base + 16, a = base + 1024, b = a + 32768;
  int warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (CG == 2) {
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (CG == 1) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(hold));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    } else {
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(hold));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
  }
  __syncthreads();
  if (CG == 2) asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;");
  uint32_t tmem = *(volatile uint32_t*)(sm + (hold - (uint32_t)__cvta_generic_to_shared(sm)));
  uint32_t rank = 0;
  if (CG == 2) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  // bf16 A, bf16 B, f32 D, K-major both
  const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
  const uint32_t hi = 0x40004040u;
  long long t0 = clock64();
  if (warp == 0 && rank == 0) {
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        uint64_t da = ((uint64_t)hi << 32) | (((a >> 4) & 0x3FFF) + 2 * k);
        uint64_t db = ((uint64_t)hi << 32) | (((b >> 4) & 0x3FFF) + 2 * k);
        uint32_t acc = (i | k) ? 1 : 0;
        if (CG == 1)
          asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                       ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(acc));
        else
          asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
                       ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(acc));
      }
    }
    if (CG == 1)
      asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(bar));
    else
      asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n@e tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n}" ::"r"(bar), "h"((uint16_t)3));
  }
  if (warp == 0 && (CG == 2 || rank == 0)) mbar_wait(bar, 0);
  long long t1 = clock64();
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (CG == 2) asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 0) {
    if (CG == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
    else asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
  }
}

template <int M, int N, int CG>
void run(long long* d, int nsm) {
  auto k = rate<M, N, CG>;
  int smem = 1024 + 1024 + 32768 + 65536;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(nsm); c.blockDim = dim3(128); c.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CG; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  c.attrs = at; c.numAttrs = 1;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int w = 0; w < 2; w++) cudaLaunchKernelEx(&c, k, d);
  cudaEventRecord(e0);
  cudaLaunchKernelEx(&c, k, d);
  cudaEventRecord(e1);
  cudaError_t err = cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  long long h[256]; cudaMemcpy(h, d, nsm * 8, cudaMemcpyDeviceToHost);
  double cyc = h[0];
  // FLOPs one SM performs: M rows of the instruction are spread over CG SMs
  double flop_sm = 2.0 * M * N * 16 * 4.0 * ITERS / CG;
  printf("cg%d M=%3d N=%3d  %s  %.0f cycles  %.0f flop/clk/SM  grid %.1f TF\n", CG, M, N, cudaGetErrorString(err), cyc,
         flop_sm / cyc, flop_sm * nsm / (ms * 1e-3) / 1e12);
}

int main() {
  long long* d; cudaMalloc(&d, 256 * 8);
  int nsm = 152;
  run<64, 96, 1>(d, nsm); run<64, 128, 1>(d, nsm); run<64, 256, 1>(d, nsm);
  run<128, 96, 1>(d, nsm); run<128, 128, 1>(d, nsm); run<128, 256, 1>(d, nsm);
  run<128, 96, 2>(d, nsm); run<128, 128, 2>(d, nsm); run<128, 256, 2>(d, nsm);
  run<256, 96, 2>(d, nsm); run<256, 128, 2>(d, nsm); run<256, 256, 2>(d, nsm);
  return 0;
}
