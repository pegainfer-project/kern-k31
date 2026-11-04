// The dependent chain one split-KV MLA decode item pays with no math: a 2-CTA cluster, the DSL's 232 KB of
// shared memory (one CTA an SM), launch -> page-table entry -> the row's q half (L2) and TILES 64-position KV
// pages a CTA (HBM, 73.7 KB each, two in flight) by bulk copy -> a 96 x 512
// f32 partial a pair stored (98 KB a CTA) and its lse. Items = rows x splits; the graph times LAUNCHES calls.
//   $K3_NVCC -O2 -gencode arch=compute_103a,code=sm_103a harness/attn/mla_chain.cu -o OUT && loop/withgpu -n 1 OUT
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

constexpr int ROW = 1152, PAGE_POS = 64, PAGE_BYTES = PAGE_POS * ROW, HEADS = 96, LORA = 512;
constexpr int Q_BYTES = HEADS * ROW, SMEM = 232448, POOL_PAGES = 32768, LAUNCHES = 20;

__device__ __forceinline__ void wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__global__ void __launch_bounds__(384, 1) __cluster_dims__(2, 1, 1)
chain(const uint8_t* q, const uint8_t* kv, const int* table, int max_pages, int tiles, float* acc_o, float* acc_lse) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t bar = s, qbuf = s + 1024, kbuf = qbuf + Q_BYTES / 2;  // two page slots after q
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int row = blockIdx.y, split = blockIdx.z, item = row * gridDim.z + split;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar + 8));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar + 16));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  float acc = 0.f;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(Q_BYTES / 2));
    bulk(qbuf, q + (size_t)row * Q_BYTES + rank * (Q_BYTES / 2), Q_BYTES / 2, bar);
    for (int t = 0; t < min(tiles, 2); t++) {
      const int page = table[row * max_pages + (split * tiles + t) * 2 + rank];
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar + 8 + 8 * t), "r"(PAGE_BYTES));
      bulk(kbuf + t * PAGE_BYTES, kv + (size_t)page * PAGE_BYTES, PAGE_BYTES, bar + 8 + 8 * t);
    }
  }
  wait(bar, 0);
  for (int t = 0; t < tiles; t++) {
    const int slot = t & 1;
    wait(bar + 8 + 8 * slot, (t >> 1) & 1);
    acc += *reinterpret_cast<const float*>(sm + 1024 + Q_BYTES / 2 + slot * PAGE_BYTES + threadIdx.x * 4);
    __syncthreads();
    if (threadIdx.x == 0 && t + 2 < tiles) {
      const int page = table[row * max_pages + (split * tiles + t + 2) * 2 + rank];
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar + 8 + 8 * slot), "r"(PAGE_BYTES));
      bulk(kbuf + slot * PAGE_BYTES, kv + (size_t)page * PAGE_BYTES, PAGE_BYTES, bar + 8 + 8 * slot);
    }
  }
  // the pair's 96 x 512 f32 partial, 48 heads a CTA
  float4* o = reinterpret_cast<float4*>(acc_o + ((size_t)item * HEADS + rank * HEADS / 2) * LORA);
  for (int i = threadIdx.x; i < HEADS / 2 * LORA / 4; i += blockDim.x) o[i] = make_float4(acc, acc, acc, acc);
  if (threadIdx.x < HEADS / 2) acc_lse[item * HEADS + rank * HEADS / 2 + threadIdx.x] = acc;
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

int main() {
  uint8_t *q, *kv; int* table; float *o, *lse;
  const int rows_max = 64, splits_max = 16, max_pages = 1024;
  cudaMalloc(&q, (size_t)rows_max * Q_BYTES);
  cudaMalloc(&kv, (size_t)POOL_PAGES * PAGE_BYTES);
  cudaMalloc(&table, (size_t)LAUNCHES * rows_max * max_pages * 4);
  cudaMalloc(&o, (size_t)rows_max * splits_max * HEADS * LORA * 4);
  cudaMalloc(&lse, (size_t)rows_max * splits_max * HEADS * 4);
  std::vector<int> h((size_t)LAUNCHES * rows_max * max_pages);
  uint32_t x = 12345;
  for (auto& p : h) x = x * 1664525u + 1013904223u, p = (x >> 8) % POOL_PAGES;
  cudaMemcpy(table, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(chain, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaStream_t st; cudaStreamCreate(&st);
  struct Shape { int rows, splits, tiles; };
  for (Shape sh : {Shape{1, 8, 1}, Shape{8, 8, 1}, Shape{32, 2, 4}, Shape{64, 1, 8}, Shape{32, 2, 16},
                   Shape{8, 8, 8}, Shape{24, 3, 22}, Shape{36, 2, 32}, Shape{48, 3, 43}}) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < LAUNCHES; l++)
      chain<<<dim3(2, sh.rows, sh.splits), 384, SMEM, st>>>(q, kv, table + (size_t)l * rows_max * max_pages, max_pages,
                                                           sh.tiles, o, lse);
    cudaStreamEndCapture(st, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e9;
    for (int r = 0; r < 5; r++) {
      cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
    }
    printf("rows %2d splits %2d tiles/split %d (pages a CTA): %6.2f us a call  %s\n", sh.rows, sh.splits, sh.tiles,
           best * 1e3 / LAUNCHES, cudaGetErrorString(cudaGetLastError()));
  }
  return 0;
}
