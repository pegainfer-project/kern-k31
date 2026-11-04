// p_gemm_algos.cu: nvcc -O2 -std=c++17 -arch=sm_103a p_gemm_algos.cu -o p_gemm_algos -lcublasLt -lcublas
// stdin lines "bf16|f32 m n k ldc lda" (one per GEMM shape of a manifest); see kern-k3-pp-prefill's notes/d-attn.md for the table.
// cuBLASLt algorithm pinning, measure only: for each GEMM shape on stdin ("kind m n k ldc lda", kind bf16 =
// kern's cublaslt_bf16_tn, f32 = its cublas_bf16_tn_f32), the heuristic's first answer (what kern runs; for
// f32 cublasGemmEx itself) against every candidate of a 32-deep heuristic request (cublasLt, f32 output for f32).
// Row-major C[m, n] = A[m, k] W[n, k]^T as kern maps it; W rotates over copies past L2 so it streams from HBM.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

#define CK(x) do { auto r_ = (x); if ((int)r_ != 0) { printf("%s:%d %s = %d\n", __FILE__, __LINE__, #x, (int)r_); exit(1); } } while (0)

int main() {
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  cublasHandle_t bl; CK(cublasCreate(&bl)); CK(cublasSetMathMode(bl, CUBLAS_TENSOR_OP_MATH));
  const size_t WS = 32 << 20;
  void* ws; CK(cudaMalloc(&ws, WS)); CK(cublasSetWorkspace(bl, ws, WS));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  char kind[8]; long m, n, k, ldc, lda;
  while (scanf("%7s %ld %ld %ld %ld %ld", kind, &m, &n, &k, &ldc, &lda) == 6) {
    const bool f32 = std::string(kind) == "f32";
    const size_t wbytes = (size_t)n * k * 2;
    const int copies = (int)std::max<size_t>(1, std::min<size_t>(8, (512ull << 20) / wbytes + 1));
    void *A, *C; std::vector<void*> W(copies);
    CK(cudaMalloc(&A, (size_t)m * lda * 2)); CK(cudaMemset(A, 0, (size_t)m * lda * 2));
    CK(cudaMalloc(&C, (size_t)m * ldc * (f32 ? 4 : 2)));
    for (auto& w : W) { CK(cudaMalloc(&w, wbytes)); CK(cudaMemset(w, 0, wbytes)); }
    cudaDataType ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
    cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
    cublasLtMatrixLayout_t la, lb, lc;
    CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, lda));
    CK(cublasLtMatrixLayoutCreate(&lc, ct, n, m, ldc));
    cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res[32]; int nres = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 32, res, &nres));
    float alpha = 1, beta = 0;
    const double est = std::max(2.0 * m * n * k / 1.5e15, wbytes / 7e12);
    const int iters = std::max(3, std::min(50, (int)(0.05 / est)));
    auto time = [&](auto fn) {
      fn(0); CK(cudaDeviceSynchronize());
      std::vector<float> v;
      for (int r = 0; r < 3; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) fn(i % copies);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms * 1000 / iters);
      }
      std::sort(v.begin(), v.end()); return v[1];
    };
    float base;
    if (f32)
      base = time([&](int c) {
        cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, W[c], CUDA_R_16BF, k, A, CUDA_R_16BF, lda, &beta, C,
                     CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      });
    else
      base = time([&](int c) { cublasLtMatmul(lt, op, &alpha, W[c], la, A, lb, &beta, C, lc, C, lc, &res[0].algo, ws, WS, 0); });
    float best = base; int bi = -1;
    for (int a = 0; a < nres; ++a) {
      if (res[a].state != CUBLAS_STATUS_SUCCESS || res[a].workspaceSize > WS) continue;
      float t = time([&](int c) { cublasLtMatmul(lt, op, &alpha, W[c], la, A, lb, &beta, C, lc, C, lc, &res[a].algo, ws, WS, 0); });
      if (t < best) best = t, bi = a;
    }
    if (bi >= 0) {  // the winner of many noisy timings is biased low: re-time both, interleaved
      float b2 = 0, w2 = 0;
      for (int r = 0; r < 3; ++r) {
        b2 += f32 ? time([&](int c) {
          cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, W[c], CUDA_R_16BF, k, A, CUDA_R_16BF, lda, &beta, C,
                       CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }) : time([&](int c) { cublasLtMatmul(lt, op, &alpha, W[c], la, A, lb, &beta, C, lc, C, lc, &res[0].algo, ws, WS, 0); });
        w2 += time([&](int c) { cublasLtMatmul(lt, op, &alpha, W[c], la, A, lb, &beta, C, lc, C, lc, &res[bi].algo, ws, WS, 0); });
      }
      base = b2 / 3, best = std::min(base, w2 / 3);
    }
    printf("%s %ld %ld %ld %ld %ld  heuristic %.1f us  best %.1f us (#%d of %d)  saving %.1f us (%.1f%%)\n", kind, m, n, k,
           ldc, lda, base, best, bi, nres, base - best, 100 * (base - best) / base);
    fflush(stdout);
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    cudaFree(A); cudaFree(C); for (auto w : W) cudaFree(w);
  }
}
