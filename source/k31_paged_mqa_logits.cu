// DeepGEMM's sm100 paged MQA logits at K3.1's indexer shape over the 144-byte token-major key cache D
// keeps (128 e4m3, the key's UE8M0 scale as four equal bytes, 12 pad bytes a token, 64-token pages):
// one query token a request, 32 heads of 128, bf16 head weights and logits, a row's logits at
// [r][0, context_len). source/deepgemm-k31/ patches upstream's kernel only in how a page's scales land
// in shared memory: TMA cannot gather 4 bytes every 144, so a token's 16 bytes land whole
// (DG_SF_KV_STRIDE 4 words apart) and the warp that feeds UTCCP compacts them. Its metadata kernel is
// upstream's.
#define DG_SF_KV_STRIDE 4
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;

static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 3, 3, 3, e4m3>) == 218112);

void k31_instantiate() {
  auto p = reinterpret_cast<void*>(&sm100_paged_mqa_logits<32, 128, 64, 4, 128, 3, 3, 3, 384, 8, 128, 384, e4m3>);
  auto m = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<384, 152, 4, 256>);
  asm volatile("" ::"l"(p), "l"(m));
}
