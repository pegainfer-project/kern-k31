// vLLM's prefill top-k over the indexer's logits (csrc/sampler.cu, `top_k_per_row_prefill`), its
// device code unmodified in sampler_topk.cuh. This unit only gives the two launches the launcher
// makes stable entry names: a block of 512 threads a row, topK * 4 bytes of dynamic shared memory,
// the histogram path for the first 12288 rows of a call and the radix sort past them.
//   glm_topk_prefill{,_radix}(logits, row_starts, row_ends, out, stride0, stride1, topK, offset)
#include <cstdint>
#include <cfloat>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32  // vLLM core/cuda_compat.h on CUDA

namespace vllm {
#include "sampler_topk.cuh"
}

template <bool kRadix>
__device__ __forceinline__ void row(const float* logits, const int* starts, const int* ends, int* out, int stride0,
                                    int stride1, int topK, int offset) {
  const int r = blockIdx.x + offset;
  vllm::topKPerRowJob<512, 2048, kRadix>(nullptr, logits + static_cast<int64_t>(r) * stride0, starts[r], ends[r],
                                         out + static_cast<int64_t>(r) * topK, nullptr, stride1, topK);
}

extern "C" __global__ void __launch_bounds__(512) glm_topk_prefill(const float* logits, const int* starts,
                                                                   const int* ends, int* out, int stride0,
                                                                   int stride1, int topK, int offset) {
  row<false>(logits, starts, ends, out, stride0, stride1, topK, offset);
}

extern "C" __global__ void __launch_bounds__(512) glm_topk_prefill_radix(const float* logits, const int* starts,
                                                                         const int* ends, int* out, int stride0,
                                                                         int stride1, int topK, int offset) {
  row<true>(logits, starts, ends, out, stride0, stride1, topK, offset);
}
