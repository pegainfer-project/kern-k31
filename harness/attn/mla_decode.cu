// harness/attn/mla_decode.cu: k3_mla_decode against a plain f32 reference (every split's normalized o and lse) and
// timed in a CUDA graph over a 2.4 GB page pool (KV cold), at rows x positions a member with the DSL plan's splits.
//   $K3_NVCC -O2 -std=c++17 -arch=sm_103a -Isource harness/attn/mla_decode.cu -o OUT -lcuda && loop/withgpu -n 1 OUT [rows@pos@splits ...]
#include <cstdio>
#include "k3_mla_decode.cu"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto r_ = (x); if (r_ != 0) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
constexpr int POOL = 32768, BMAX = 64, MAXP = 4096, SPLIT_MAX = 16, LAUNCHES = 20;

__global__ void reference(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int* table, const int* seq_lens,
                          const int* bsk, float* o, float* lse, float scale_log2) {
  const int b = blockIdx.z, h = blockIdx.x, split = blockIdx.y, n = seq_lens[b];
  if (n == 0) return;
  const int tiles = (n + 127) / 128, per = (tiles + bsk[b] - 1) / bsk[b], lo = split * per * 128, hi = min(n, lo + per * 128);
  if (lo >= hi) return;
  extern __shared__ float sc[];
  __shared__ float red[2];
  const __nv_bfloat16* qh = q + ((size_t)b * HEADS + h) * ROW;
  for (int p = lo + threadIdx.x; p < hi; p += blockDim.x) {
    const __nv_bfloat16* k = kv + ((size_t)table[b * MAXP + p / 64] * 64 + p % 64) * ROW;
    float s = 0.f;
    for (int d = 0; d < ROW; ++d) s += __bfloat162float(qh[d]) * __bfloat162float(k[d]);
    sc[p - lo] = s * scale_log2;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY, sum = 0.f;
    for (int p = 0; p < hi - lo; ++p) m = fmaxf(m, sc[p]);
    for (int p = 0; p < hi - lo; ++p) sum += exp2f(sc[p] - m);
    red[0] = m, red[1] = sum;
  }
  __syncthreads();
  const size_t rec = ((size_t)b * M_TILE + h) * SPLIT_MAX + split;
  for (int d = threadIdx.x; d < LAT; d += blockDim.x) {
    float acc = 0.f;
    for (int p = lo; p < hi; ++p)
      acc += exp2f(sc[p - lo] - red[0]) * __bfloat162float(kv[((size_t)table[b * MAXP + p / 64] * 64 + p % 64) * ROW + d]);
    o[rec * LAT + d] = acc / red[1];
  }
  if (threadIdx.x == 0) lse[rec] = red[0] + log2f(red[1]);
}

__global__ void fill(__nv_bfloat16* x, size_t n, uint32_t seed, float amp) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    x[i] = __float2bfloat16(((h >> 8) / 16777216.f * 2 - 1) * amp);
  }
}

int main(int argc, char** argv) {
  struct Shape { int rows, pos, splits; };
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) { Shape s; sscanf(argv[i], "%d@%d@%d", &s.rows, &s.pos, &s.splits); shapes.push_back(s); }
  if (shapes.empty())
    shapes = {{1, 1024, 8}, {8, 1024, 8}, {32, 1024, 2}, {64, 1024, 1}, {32, 4096, 2}, {8, 8192, 8}, {24, 8192, 3},
              {36, 8192, 2}, {48, 16384, 3}, {5, 1000, 3}};
  setvbuf(stdout, 0, _IONBF, 0);
  CK(cuInit(0));
  __nv_bfloat16 *q, *kv; int *table, *lens, *bsk; float *o, *lse, *ro, *rlse;
  CK(cudaMalloc(&q, (size_t)BMAX * HEADS * ROW * 2));
  CK(cudaMalloc(&kv, (size_t)POOL * 64 * ROW * 2));
  CK(cudaMalloc(&table, (size_t)LAUNCHES * BMAX * MAXP * 4));
  CK(cudaMalloc(&lens, BMAX * 4)); CK(cudaMalloc(&bsk, BMAX * 4));
  const size_t on = (size_t)BMAX * M_TILE * SPLIT_MAX * LAT, ln = (size_t)BMAX * M_TILE * SPLIT_MAX;
  CK(cudaMalloc(&o, on * 4)); CK(cudaMalloc(&lse, ln * 4)); CK(cudaMalloc(&ro, on * 4)); CK(cudaMalloc(&rlse, ln * 4));
  fill<<<1024, 256>>>(q, (size_t)BMAX * HEADS * ROW, 1, 2.f);
  fill<<<4096, 256>>>(kv, (size_t)POOL * 64 * ROW, 2, 1.f);
  std::vector<int> ht((size_t)LAUNCHES * BMAX * MAXP);
  uint32_t x = 12345;
  for (auto& p : ht) x = x * 1664525u + 1013904223u, p = (x >> 8) % POOL;
  CK(cudaMemcpy(table, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice));
  CUtensorMap qm, km;
  {
    cuuint64_t dims[3] = {ROW, HEADS, BMAX}, strides[2] = {ROW * 2, HEADS * ROW * 2};
    cuuint32_t box[3] = {64, 64, 1}, es[3] = {1, 1, 1};
    CK(cuTensorMapEncodeTiled(&qm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, q, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    cuuint64_t kd[3] = {ROW, 64, POOL}, ks[2] = {ROW * 2, 64 * ROW * 2};
    CK(cuTensorMapEncodeTiled(&km, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, kv, kd, ks, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  CK(cudaFuncSetAttribute(kern_k3_mla_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  CK(cudaFuncSetAttribute(reference, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
  const float scale_log2 = 0.0601f;
  cudaStream_t st; CK(cudaStreamCreate(&st));
  printf("rows pos splits | max|do| relRMS(o) max|dlse| | us/call\n");
  for (auto sh : shapes) {
    std::vector<int> hl(BMAX, 0), hb(BMAX, 1);
    for (int r = 0; r < sh.rows; ++r) hl[r] = sh.pos - (r % 3) * 37, hb[r] = sh.splits;  // ragged last pages
    if (sh.rows > 2) hl[1] = 0;
    CK(cudaMemcpy(lens, hl.data(), BMAX * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bsk, hb.data(), BMAX * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(o, 0, on * 4)); CK(cudaMemset(ro, 0, on * 4)); CK(cudaMemset(lse, 0, ln * 4)); CK(cudaMemset(rlse, 0, ln * 4));
    kern_k3_mla_decode<<<dim3(2, sh.rows, SPLIT_MAX), THREADS, SMEM_BYTES, st>>>(qm, km, table, lens, bsk, o, lse, SPLIT_MAX, MAXP, scale_log2);
    reference<<<dim3(HEADS, SPLIT_MAX, sh.rows), 256, 128 * 1024, st>>>(q, kv, table, lens, bsk, ro, rlse, scale_log2);
    CK(cudaStreamSynchronize(st));
    std::vector<float> a(on), r(on), al(ln), rl(ln);
    CK(cudaMemcpy(a.data(), o, on * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), ro, on * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(al.data(), lse, ln * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rl.data(), rlse, ln * 4, cudaMemcpyDeviceToHost));
    double md = 0, se = 0, sr = 0, ml = 0;
    for (size_t i = 0; i < on; ++i) {
      const double d = std::fabs((double)a[i] - r[i]);
      md = std::max(md, std::isnan(d) ? 1e30 : d), se += d * d, sr += (double)r[i] * r[i];
    }
    for (size_t i = 0; i < ln; ++i) { const double d = std::fabs((double)al[i] - rl[i]); ml = std::max(ml, std::isnan(d) ? 1e30 : d); }
#ifdef MLA_PROF
    {
      unsigned long long z[32] = {}, h[32];
      cudaMemcpyToSymbol(mla_prof, z, sizeof z);
      kern_k3_mla_decode<<<dim3(2, sh.rows, SPLIT_MAX), THREADS, SMEM_BYTES, st>>>(qm, km, table + BMAX * MAXP, lens, bsk, o, lse, SPLIT_MAX, MAXP, scale_log2);
      cudaStreamSynchronize(st);
      cudaMemcpyFromSymbol(h, mla_prof, sizeof h);
      const char* nm[16] = {"mma:s_empty", "mma:kfull", "mma:pfull", "mma:vfull", "smx:sfull", "smx:odone", "smx:ldS", "mma:total", "smx:compute", "smx:sincepre", "rescales", "@tmem_ready", "@q_full", "@staged", "@last_O", "@epilogue_done"};
      for (int c = 0; c < 2; ++c) { printf("  cta %d:", c); for (int i = 0; i < 16; ++i) if (h[16 * c + i]) printf(" %s %.2f", nm[i], h[16 * c + i] / 1940.0); printf(" (us)\n"); }
    }
#endif
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < LAUNCHES; ++l)
      kern_k3_mla_decode<<<dim3(2, sh.rows, SPLIT_MAX), THREADS, SMEM_BYTES, st>>>(qm, km, table + (size_t)l * BMAX * MAXP, lens, bsk, o,
                                                                                    lse, SPLIT_MAX, MAXP, scale_log2);
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e9;
    for (int k = 0; k < 5; ++k) {
      cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
    }
    printf("%3d %6d %2d | %.2e %.2e %.2e | %7.2f  %s\n", sh.rows, sh.pos, sh.splits, md, std::sqrt(se / std::max(sr, 1e-30)), ml,
           best * 1e3 / LAUNCHES, cudaGetErrorString(cudaGetLastError()));
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  }
}
