// DeepGEMM's sm100 paged fp8 MQA logits at K3.1's indexer shape, over the 144-byte token-major key
// cache D keeps (128 e4m3, the f32 scale, 12 pad bytes a token, 64-token pages): one query token a
// request, 32 heads of 128, f32 head weights and logits, a row's logits at [r][0, context_len).
// source/deepgemm-k31/ patches upstream's kernel only in how a page's scales land in shared memory:
// TMA cannot gather 4 bytes every 144, so a token's 16 scale bytes land whole (DG_SF_KV_STRIDE 4
// floats apart), which takes one KV stage less (4) to fit. Its metadata kernel is upstream's.
#define DG_SF_KV_STRIDE 4
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;

static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, false, 4, 256, 3, 4, 3, e4m3, float>) == 198656);

void k31_instantiate() {
  auto p = reinterpret_cast<void*>(&sm100_paged_mqa_logits<1, 32, 128, 64, false, true, false, 3, 4, 256, 16, 128,
                                                           256, e4m3, float, float>);
  auto m = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<1, true, false, 256, 152>);
  asm volatile("" ::"l"(p), "l"(m));
}
