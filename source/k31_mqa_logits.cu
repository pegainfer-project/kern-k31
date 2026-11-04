// DeepGEMM's sm100 MQA logits at K3.1's indexer shape over P's gathered key stream, as vLLM runs it on
// SM100 (upstream's kernel: source/deepgemm-k31 changes only the paged path): 32 heads of 128, e4m3 q
// and keys with UE8M0 scales (four equal bytes, one a 32-element block), bf16 head weights, the heads'
// relu-weighted sum in bf16 into compressed bf16 logits (row r's at [r][0, ke - ks)), no schedule. The
// generator names the instance by its mangled template arguments and passes exactly the shared memory
// asserted here.
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;

static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 3, 3, 3, e4m3>) == 203776);

void k31_instantiate() {
  auto p = reinterpret_cast<void*>(&sm100_mqa_logits<32, 128, false, 4, 384, 128, 3, 3, 3, 152, 128, 384, e4m3>);
  asm volatile("" ::"l"(p));
}
