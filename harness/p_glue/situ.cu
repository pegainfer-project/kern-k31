// situ harness: the committed kern_k3g_situ against a candidate source's, timed, bit-compared
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include "k3_fmha_plan.cuh"
typedef __nv_bfloat16 bf16_t;
namespace base {
#include "base_g.cu"
}
namespace cand {
#include "cand_g.cu"
}
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%d %s\n", __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
int main() {
  const int B = 8192, n = 33792;
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16_t> h((size_t)B * 2 * n); for (auto& x : h) x = __float2bfloat16(nd(rng) * 3.f);
  bf16_t *p, *o0, *o1;
  CK(cudaMalloc(&p, h.size() * 2)); CK(cudaMalloc(&o0, (size_t)B * n * 2)); CK(cudaMalloc(&o1, (size_t)B * n * 2));
  CK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto timeit = [&](auto f) { for (int i = 0; i < 3; ++i) f(); cudaEventRecord(a); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / 20; };
  for (int rows : {10, 354, 8192}) {
    dim3 g(rows, (n + 2047) / 2048);
    auto k0 = [&] { base::base_k3g_situ<<<g, 256>>>(p, o0, n, rows); };
    auto k1 = [&] { cand::cand_k3g_situ<<<g, 256>>>(p, o1, n, rows); };
    float t0 = timeit(k0), t1 = timeit(k1);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> x((size_t)rows * n), y((size_t)rows * n);
    cudaMemcpy(x.data(), o0, x.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), o1, y.size() * 2, cudaMemcpyDeviceToHost);
    long bad = 0; for (size_t i = 0; i < x.size(); ++i) bad += x[i] != y[i];
    printf("rows %5d  base %8.2f  cand %8.2f us  mismatch %ld\n", rows, t0, t1, bad);
  }
}
