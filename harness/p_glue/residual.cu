// residual kernels harness: K1b (land_add_attnres_rms) and K1d (land_add2_attnres_rms) of a candidate
// source (CAND) against the committed one (BASE), bit-compared, timed.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
typedef __nv_bfloat16 bf16_t;
#include "k3_fmha_plan.cuh"
#ifndef TWO
#define TWO 0
#endif
namespace base {
#include "base_g.cu"
}
namespace cand {
#include "cand_g.cu"
}
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
int main() {
  const int B = 8192, H = 7168, NB = 8;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto rnd = [&](size_t n, float s) { std::vector<bf16_t> v(n); for (auto& x : v) x = __float2bfloat16(nd(rng) * s); return v; };
  auto up = [&](const std::vector<bf16_t>& h) { bf16_t* d; CK(cudaMalloc(&d, h.size() * 2)); CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); return d; };
  bf16_t *p1 = up(rnd((size_t)B * H, 0.3f)), *p2 = up(rnd((size_t)B * H, 0.3f)), *pre = up(rnd((size_t)B * H, 1.f));
  bf16_t *blocks = up(rnd((size_t)B * NB * H, 1.f)), *gamma = up(rnd(H, 0.2f));
  std::vector<float> hsw(H); for (auto& x : hsw) x = nd(rng) * 0.05f;
  float* sw; CK(cudaMalloc(&sw, H * 4)); CK(cudaMemcpy(sw, hsw.data(), H * 4, cudaMemcpyHostToDevice));
  bf16_t *o1[2], *o2[2];
  for (int i = 0; i < 2; ++i) { CK(cudaMalloc(&o1[i], (size_t)B * H * 2)); CK(cudaMalloc(&o2[i], (size_t)B * H * 2)); }


  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto cmp = [&](const char* name) {
    std::vector<uint16_t> x((size_t)B * H), y((size_t)B * H); int bad = 0;
    for (int k = 0; k < 2; ++k) {
      bf16_t* p = k ? o2[0] : o1[0]; bf16_t* q = k ? o2[1] : o1[1];
      cudaMemcpy(x.data(), p, x.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), q, y.size() * 2, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < x.size(); ++i) bad += x[i] != y[i];
    }
    printf("%s mismatch %d\n", name, bad);
  };
  auto timeit = [&](auto f) { for (int i = 0; i < 3; ++i) f(); cudaEventRecord(a); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / 20; };
  for (int rows : {10, 354, 8192}) for (int nb : {1, 2, 4}) {
    // K1b: partial = p1, prefix = pre, snapshot 0
    auto kb0 = [&] { base::base_k3g_land_add_attnres_rms<<<rows, 448>>>(p1, pre, blocks, sw, gamma, o1[0], o2[0], nb, 0, rows); };
    auto kb1 = [&] { cand::cand_k3g_land_add_attnres_rms<<<rows, 448>>>(p1, pre, blocks, sw, gamma, o1[1], o2[1], nb, 0, rows); };
    auto kd0 = [&] { base::base_k3g_land_add2_attnres_rms<<<rows, 448>>>(p1, p2, pre, o1[0], TWO, blocks, sw, gamma, o2[0], nb, rows); };
    auto kd1 = [&] { cand::cand_k3g_land_add2_attnres_rms<<<rows, 448>>>(p1, p2, pre, o1[1], TWO, blocks, sw, gamma, o2[1], nb, rows); };
    float t0 = timeit(kb0), t1 = timeit(kb1);
    kb0(); kb1(); CK(cudaDeviceSynchronize());
    printf("rows %5d nb %d  K1b base %8.2f cand %8.2f us  ", rows, nb, t0, t1); cmp("");
    t0 = timeit(kd0); t1 = timeit(kd1);
    kd0(); kd1(); CK(cudaDeviceSynchronize());
    printf("rows %5d nb %d  K1d base %8.2f cand %8.2f us  ", rows, nb, t0, t1); cmp("");
  }
}
