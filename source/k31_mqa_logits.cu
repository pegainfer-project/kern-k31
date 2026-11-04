// DeepGEMM's sm100 fp8 MQA logits at K3.1's indexer shape, unmodified: 32 heads of 128, e4m3 q and
// keys with one f32 scale a key, f32 head weights, compressed f32 logits (row r's at [r][0, ke - ks)).
// The generator names the instance by its mangled template arguments and passes exactly the shared
// memory asserted here. The same instance kern-glm53-pp-prefill runs for GLM-5.3's indexer.
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;

static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, false, 4, 256, 3, 5, 3, e4m3, float>) == 220160);

void k31_instantiate() {
  auto p = reinterpret_cast<void*>(&sm100_mqa_logits<32, 128, false, true, false, false, 4, 256, 3, 5, 152,
                                                     128, 256, e4m3, float, float>);
  asm volatile("" ::"l"(p));
}
