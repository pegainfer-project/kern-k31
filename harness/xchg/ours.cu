// main's kernels verbatim, each source in a translation unit of its own (they share helper names).
#if defined(OURS_AR)
#include "../../source/peer_allreduce_bf16.cu"
#elif defined(OURS_FUSED)
#include "../../source/k3_ar_fused.cu"
#elif defined(OURS_DCP)
#include "../../source/k3_dcp.cu"
#elif defined(OURS_HEAD)
#include "../../source/k3_head.cu"
#endif
