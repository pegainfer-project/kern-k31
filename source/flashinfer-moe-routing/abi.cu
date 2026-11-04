// Prints, as JSON, the layout of the parameter struct kern_moe_routing.cu's
// kernels take: every field's byte offset, the struct's size, and the bytes
// of IntFastDiv(top_k) for the top-k values given on the command line. The
// manifest generator packs the struct from this file (abi.json).
//
//   nvcc -std=c++17 <the build's includes and defines> -o abi abi.cu && ./abi 16 > abi.json
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"

using Params = moe::dev::routing::routingPrecomputed::KernelParams<float, 1024, 16>;

#define FIELD(f) std::printf("%s    \"" #f "\": %zu", first ? "" : ",\n", offsetof(Params, f)), first = false

int main(int argc, char** argv) {
  bool first = true;
  std::printf("{\n  \"size\": %zu,\n  \"max_experts\": %d,\n  \"max_top_k\": %d,\n  \"fields\": {\n", sizeof(Params),
              Params::MaxNumExperts, Params::MaxNumTopExperts);
  FIELD(mUsePdl);
  FIELD(mIsPow2);
  FIELD(mPtrExpertCounts);
  FIELD(mPtrPermutedIdxSize);
  FIELD(mPtrExpandedIdxToPermutedIdx);
  FIELD(mPtrPermutedIdxToExpandedIdx);
  FIELD(mPtrPermutedIdxToTokenIdx);
  FIELD(mPtrCtaIdxXyToBatchIdx);
  FIELD(mPtrCtaIdxXyToMnLimit);
  FIELD(mPtrNumNonExitingCtas);
  FIELD(mPtrTopKWeights);
  FIELD(mPtrTopKIds);
  FIELD(mPtrScores);
  FIELD(mNumTokens);
  FIELD(mNumExperts);
  FIELD(mPaddingLog2);
  FIELD(mTileTokensDim);
  FIELD(mLocalExpertsStartIdx);
  FIELD(mLocalExpertsStrideLog2);
  FIELD(mNumLocalExperts);
  FIELD(mNumFusedSharedExperts);
  FIELD(mSharedExpertTokenOffset);
  FIELD(mSharedExpertNumTokens);
  FIELD(mTotalExpertsPerToken);
  FIELD(mPtrRoutingReplayOut);
  FIELD(mUseContiguousRouteWindows);
  FIELD(mPtrNumTokensPerExpert);
  FIELD(mPtrTopKPacked);
  FIELD(mTopK);
  std::printf("\n  },\n  \"int_fast_div\": {");
  for (int i = 1; i < argc; ++i) {
    trtllm::dev::IntFastDiv d(std::atoi(argv[i]));
    int words[sizeof d / 4];
    std::memcpy(words, &d, sizeof d);
    std::printf("%s\n    \"%s\": [", i > 1 ? "," : "", argv[i]);
    for (size_t w = 0; w < sizeof d / 4; ++w) std::printf("%s%d", w ? ", " : "", words[w]);
    std::printf("]");
  }
  std::printf("\n  }\n}\n");
}
