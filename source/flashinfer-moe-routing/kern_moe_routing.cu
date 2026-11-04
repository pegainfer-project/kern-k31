// FlashInfer's TRT-LLM gen MoE routing kernels over precomputed top-k, the
// multi-kernel path (init counts, histogram, offsets), instantiated for kern.
// The kernels are upstream's, unmodified; this unit only picks the parameter
// struct (routingPrecomputed::KernelParams<float, 1024, 16>: f32 top-k
// weights, up to 1024 experts, top-k up to 16) and instantiates them, so the
// cubin's entries are the mangled template names kern launches with the
// struct packed as one by-value parameter (abi.cu prints its layout).
#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"

namespace routing = moe::dev::routing;
using Params = routing::routingPrecomputed::KernelParams<float, 1024, 16>;

template __global__ void routing::routingInitExpertCounts<Params>(Params);
template __global__ void routing::routingIndicesHistogramKernel<Params>(Params);
template __global__ void routing::routingIndicesOffsetsKernel<Params>(Params);
