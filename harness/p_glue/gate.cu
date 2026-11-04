// mla gate harness: the committed kern_k3g_mla_gate (base) against a candidate source's, 8192 / 4096 / 354 rows
// of normal data (gate x ~ N(0, 3)), timed, bit-compared
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdint>
#include <random>
typedef __nv_bfloat16 bf16_t;
namespace base {
#include "base_g.cu"
}
namespace cand {
#include "cand_g.cu"
}
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%d %s\n", __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
int main() {
  const int B = 8192, n = 12288, F = 14400;
  bf16_t *partial, *o, *out0, *out1; float2* stats; int* lens; int ns = 8;
  CK(cudaMalloc(&partial, (size_t)B * F * 2)); CK(cudaMalloc(&o, (size_t)(B + 1024) * n * 2));
  CK(cudaMalloc(&out0, (size_t)B * n * 2)); CK(cudaMalloc(&out1, (size_t)B * n * 2));
  CK(cudaMalloc(&stats, (size_t)(B + 1024) * 96 * 8)); CK(cudaMalloc(&lens, 64 * 4)); CK(cudaMemset(lens, 0, 256));
  std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16_t> h((size_t)B * F); for (auto& x : h) x = __float2bfloat16(nd(rng) * 3.f);
  CK(cudaMemcpy(partial, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  std::vector<bf16_t> ho((size_t)B * n); for (auto& x : ho) x = __float2bfloat16(nd(rng));
  CK(cudaMemcpy(o, ho.data(), ho.size() * 2, cudaMemcpyHostToDevice));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto timeit = [&](auto f) { for (int i = 0; i < 3; ++i) f(); cudaEventRecord(a); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / 20; };
  for (int rows : {354, 4096, 8192}) {
    auto k0 = [&] { base::base_k3g_mla_gate<<<(rows + 7) / 8 * (n / 2048) + 4 * 96 * 4, 256>>>(out0, o, partial, nullptr, nullptr, nullptr, nullptr, 0, nullptr, stats, lens, ns, n, 128, rows); };
    auto k1 = [&] { cand::cand_k3g_mla_gate<<<(rows + 7) / 8 * (n / 2048) + 4 * 96 * 4, 256>>>(out1, o, partial, nullptr, nullptr, nullptr, nullptr, 0, nullptr, stats, lens, ns, n, 128, rows); };
    float t0 = timeit(k0), t1 = timeit(k1);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> x((size_t)rows * n), y((size_t)rows * n);
    cudaMemcpy(x.data(), out0, x.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), out1, y.size() * 2, cudaMemcpyDeviceToHost);
    long bad = 0; for (size_t i = 0; i < x.size(); ++i) bad += x[i] != y[i];
    printf("rows %5d  base %8.2f  cand %8.2f us  mismatch %ld\n", rows, t0, t1, bad);
  }
}
