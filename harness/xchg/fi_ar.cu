// FlashInfer's TRT-LLM allreduce fusion (Lamport one-shot, two-shot), as written, plain all-reduce pattern.
// logging.h needs spdlog; nothing here logs
#include <array>
#define FLASHINFER_LOGGING_H_
#define FLASHINFER_LOG_WARN(...)
#define FLASHINFER_LOG_INFO(...)
#define FLASHINFER_LOG_DEBUG(...)
#define FLASHINFER_LOG_ERROR(...)
#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"
using namespace flashinfer::trtllm_allreduce_fusion;
extern "C" int fi_allreduce_bf16(void** workspace, int rank, int tokens, int hidden, void* in, void* out, bool oneshot,
                                 bool pdl, cudaStream_t s) {
  AllReduceFusionParams<__nv_bfloat16> p{};
  p.nranks = 8;
  p.rank = rank;
  p.size = tokens * hidden;
  p.hidden_dim = hidden;
  p.workspace = workspace;
  p.allreduce_in = in;
  p.allreduce_out = out;
  p.rms_eps = 1e-5f;
  p.use_oneshot = oneshot;
  p.stream = s;
  p.pattern = AllReduceFusionPattern::kAllReduce;
  p.trigger_completion_at_end = false;
  return (int)allreduce_fusion_op(p, pdl, false);
}
