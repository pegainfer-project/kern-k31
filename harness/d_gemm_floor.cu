// One D GEMM shape alone: row-major C[m,n] = A[m,k] @ W[n,k]^T, bf16 in. d_gemm_floor N K [m...]
// 1. kern's call: cublasGemmEx, f32 out, CUBLAS_COMPUTE_32F, DEFAULT_TENSOR_OP.
// 2. cublasLt f32 out and bf16 out, every heuristic candidate.
// 3. a plain read of the same weight bytes (the HBM ceiling for this size).
// W rotates over COPIES copies (> L2), A fixed; each timing is a graph of CALLS calls, median of REPS.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e = (x); if (e != 0) { printf("%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)e); exit(1); } } while (0)

static int N, K; constexpr int COPIES = 8, CALLS = 32, REPS = 30;
static size_t WB;

__global__ void fill(__nv_bfloat16* p, size_t n, unsigned s) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16(((unsigned)(i * 2654435761u + s) >> 8) * (1.0f / 16777216.0f) - 0.5f);
}

__global__ void readall(const int4* p, size_t n, int* sink) {
  int4 acc = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int4 v = __ldcs(p + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *sink = 1;
}

template <int U>
__global__ void readu(const int4* p, size_t n, int* sink) {
  int4 acc = {0, 0, 0, 0};
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += stride * U) {
    int4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = i + u * stride < n ? __ldcs(p + i + u * stride) : int4{0, 0, 0, 0};
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *sink = 1;
}

__global__ void empty() {}

// Each CTA streams a contiguous slice with cp.async.bulk into a ring of ST stages of BOX bytes.
template <int BOX, int ST>
__global__ void __launch_bounds__(32) readbulk(const char* p, size_t bytes, int* sink) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) unsigned long long bar[ST];
  size_t per = (bytes / gridDim.x + BOX - 1) / BOX * BOX, beg = blockIdx.x * per, end = beg + per < bytes ? beg + per : bytes;
  int nb = beg < end ? int((end - beg + BOX - 1) / BOX) : 0;
  if (threadIdx.x == 0)
    for (int i = 0; i < ST; ++i) asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(&bar[i])));
  asm volatile("fence.proxy.async.shared::cta;");
  __syncwarp();
  auto issue = [&](int b) {
    int st = b % ST; size_t left = end - beg - size_t(b) * BOX; unsigned len = (unsigned)(left < BOX ? left : BOX);
    unsigned br = (unsigned)__cvta_generic_to_shared(&bar[st]), dst = (unsigned)__cvta_generic_to_shared(sm + st * BOX);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(br), "r"(len));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(p + beg + size_t(b) * BOX), "r"(len), "r"(br) : "memory");
  };
  if (threadIdx.x == 0) for (int b = 0; b < min(ST, nb); ++b) issue(b);
  unsigned acc = 0;
  for (int b = 0; b < nb; ++b) {
    int st = b % ST; unsigned br = (unsigned)__cvta_generic_to_shared(&bar[st]), ph = (b / ST) & 1, ok = 0;
    while (!ok) asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }" : "=r"(ok) : "r"(br), "r"(ph));
    acc ^= ((unsigned*)(sm + st * BOX))[threadIdx.x];
    __syncwarp();
    if (threadIdx.x == 0 && b + ST < nb) issue(b + ST);
  }
  if (acc == 0x12345678) *sink = 1;
}

template <class F>
float time_us(cudaStream_t s, F body) {
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int c = 0; c < CALLS; ++c) body(c);
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> t;
  for (int r = 0; r < REPS + 3; ++r) {
    cudaEventRecord(a, s); CK(cudaGraphLaunch(ge, s)); cudaEventRecord(b, s); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (r >= 3) t.push_back(ms * 1000 / CALLS);
  }
  CK(cudaGetLastError());
  std::sort(t.begin(), t.end());
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s N K [m...]\n", argv[0]); return 2; }
  N = atoi(argv[1]); K = atoi(argv[2]); WB = size_t(N) * K * 2;
  std::vector<int> ms;
  for (int i = 3; i < argc; ++i) ms.push_back(atoi(argv[i]));
  if (ms.empty()) ms = {1, 8, 16, 24, 32, 48, 64};
  cudaStream_t s; cudaStreamCreate(&s);
  __nv_bfloat16 *w, *a; float* cf; __nv_bfloat16* cb; int* sink;
  CK(cudaMalloc(&w, WB * COPIES)); CK(cudaMalloc(&a, size_t(64) * K * 2));
  CK(cudaMalloc(&cf, size_t(64) * N * 4)); CK(cudaMalloc(&cb, size_t(64) * N * 2)); CK(cudaMalloc(&sink, 4));
  fill<<<1024, 256>>>(w, WB / 2 * COPIES, 1); fill<<<64, 256>>>(a, size_t(64) * K, 2);
  CK(cudaDeviceSynchronize());
  auto W = [&](int c) { return w + size_t(c % COPIES) * N * K; };
  auto tbs = [](float us) { return WB / (us * 1e-6) / 1e12; };

  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int bl : {1, 2, 4}) {
    float us = time_us(s, [&](int c) { readall<<<sms * bl, 1024, 0, s>>>((const int4*)W(c), WB / 16, sink); });
    printf("read      %d blk/SM          %7.2f us  %5.2f TB/s\n", bl, us, tbs(us));
  }

  {
    float us = time_us(s, [&](int) { empty<<<sms, 128, 0, s>>>(); });
    printf("empty launch                %7.2f us\n", us);
  }
  for (int bl : {1, 2}) for (int th : {512, 1024}) {
    float u4 = time_us(s, [&](int c) { readu<4><<<sms * bl, th, 0, s>>>((const int4*)W(c), WB / 16, sink); });
    float u8 = time_us(s, [&](int c) { readu<8><<<sms * bl, th, 0, s>>>((const int4*)W(c), WB / 16, sink); });
    printf("read x4/x8 %d blk/SM %4d thr  %7.2f / %7.2f us  %5.2f / %5.2f TB/s\n", bl, th, u4, u8, tbs(u4), tbs(u8));
  }
  {
    auto run = [&](auto kern, int box, int st, int bl) {
      int smem = box * st;
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      float us = time_us(s, [&](int c) { kern<<<sms * bl, 32, smem, s>>>((const char*)W(c), WB, sink); });
      printf("bulk box %5d x%d st, %d blk/SM  %7.2f us  %5.2f TB/s\n", box, st, bl, us, tbs(us));
    };
    run(readbulk<16384, 4>, 16384, 4, 1); run(readbulk<16384, 4>, 16384, 4, 2); run(readbulk<16384, 4>, 16384, 4, 3);
    run(readbulk<32768, 4>, 32768, 4, 1); run(readbulk<32768, 6>, 32768, 6, 1);
    run(readbulk<8192, 8>, 8192, 8, 2); run(readbulk<8192, 8>, 8192, 8, 3);
  }
  if (getenv("READ_ONLY")) return 0;

  cublasHandle_t h; CK(cublasCreate(&h)); CK(cublasSetStream(h, s));
  void* ws; size_t wsb = 32 << 20; CK(cudaMalloc(&ws, wsb)); CK(cublasSetWorkspace(h, ws, wsb));
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  const float one = 1, zero = 0;

  for (int m : ms) {
    float us = time_us(s, [&](int c) {
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, m, K, &one, W(c), CUDA_R_16BF, K, a, CUDA_R_16BF, K, &zero,
                      cf, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });
    printf("m=%2d gemmex f32 (kern)       %7.2f us  %5.2f TB/s\n", m, us, tbs(us));

    for (auto [ct, name] : {std::pair{CUDA_R_32F, "f32 "}, std::pair{CUDA_R_16BF, "bf16"}}) {
      cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t la, lb, lc;
      CK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t T = CUBLAS_OP_T;
      CK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
      CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
      CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, m, K));
      CK(cublasLtMatrixLayoutCreate(&lc, ct, N, m, N));
      cublasLtMatmulPreference_t p; CK(cublasLtMatmulPreferenceCreate(&p));
      CK(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
      cublasLtMatmulHeuristicResult_t r[32]; int got = 0;
      CK(cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, p, 32, r, &got));
      void* c0 = ct == CUDA_R_32F ? (void*)cf : (void*)cb;
      float best = 1e9; int bi = -1, first = 0;
      for (int i = 0; i < got; ++i) {
        float us = time_us(s, [&](int c) {
          CK(cublasLtMatmul(lt, d, &one, W(c), la, a, lb, &zero, c0, lc, c0, lc, &r[i].algo, ws, wsb, s));
        });
        if (i == 0) first = 0, printf("m=%2d lt %s heuristic #0     %7.2f us  %5.2f TB/s\n", m, name, us, tbs(us));
        if (us < best) best = us, bi = i;
      }
      printf("m=%2d lt %s best of %2d (#%d)  %7.2f us  %5.2f TB/s\n", m, name, got, bi, best, tbs(best));
      cublasLtMatmulPreferenceDestroy(p); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
      cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(d);
    }
  }
  int v; cublasGetVersion(h, &v); printf("cublas %d, %d SMs, weight %.1f MB x %d copies\n", v, sms, WB / 1e6, COPIES);
}
