// p_fmha_window.cu: one-GPU harness for the packed prefill's windowed context attention: a packed call's
// rows attend their own rows through the causal TRT-LLM gen context FMHA (h192 / v128, Q256 x Kv128) and
// their sequences' cached context in rounds of at most S keys a sequence through its Dense sibling, the
// partials merged by the kernels' softmax stats into an f32 accumulator (k3_mla_glue.cu's merge), against
// the plain causal call over the whole context and an f32 reference. A round in which a sequence has no
// keys left is the case the Dense kernel must survive: that sequence's rows have a q span and an empty kv span.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a p_fmha_window.cu -o p_fmha_window -lcuda
//   ./p_fmha_window <causal.cubin> <dense.cubin> check [S]        the packed example vs the reference
//   ./p_fmha_window <causal.cubin> <dense.cubin> time ROWS PREFIX S   plain vs own + rounds + merges
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)
#define RT(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
  cudaGetErrorString(e_)); exit(1); } } while (0)
constexpr int H = 96, HQK = 192, HV = 128, KVR = 320, TILE = 256, SMEM = 199296, PARAMS = 1344;
constexpr float SCALE_LOG2 = 0.10411754995584488f;
const char* CAUSAL = "fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ256Kv128PersistentContext";
const char* DENSE = "fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvDenseVarSeqQ256Kv128PersistentContext";
static void tmap(uint8_t* at, void* ptr, std::vector<uint64_t> dims, std::vector<uint64_t> strides, std::vector<uint32_t> box) {
  CUtensorMap m;
  std::vector<uint32_t> es(dims.size(), 1);
  CK(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dims.size(), ptr, dims.data(), strides.data(), box.data(),
                            es.data(), CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  memcpy(at, &m, 128);
}
struct Call {
  void *q, *kv, *o, *seqlens, *cumq, *cumkv, *stats, *scratch;
  int qcap, kvcap, batch, maxq, sumq;
};
static void params(uint8_t* p, const Call& c) {
  memset(p, 0, PARAMS);
  auto i32 = [&](int at, int v) { memcpy(p + at, &v, 4); };
  auto f32 = [&](int at, float v) { memcpy(p + at, &v, 4); };
  auto ptr = [&](int at, void* v) { memcpy(p + at, &v, 8); };
  tmap(p + 0, c.q, {HQK, 1, H, (uint64_t)c.qcap}, {HQK * 2, HQK * 2, H * HQK * 2}, {64, 1, 1, 128});
  tmap(p + 128, c.kv, {HQK, (uint64_t)c.kvcap, H, 1}, {H * KVR * 2, KVR * 2, 16}, {64, 128, 1, 1});
  tmap(p + 384, (char*)c.kv + HQK * 2, {HV, (uint64_t)c.kvcap, H, 1}, {H * KVR * 2, KVR * 2, 16}, {64, 128, 1, 1});
  tmap(p + 512, c.o, {HV, (uint64_t)c.qcap, H, 1, 1}, {H * HV * 2, HV * 2, H * HV * 2, 16}, {64, 128, 1, 1, 1});
  ptr(912, c.o); ptr(936, c.cumq); ptr(944, c.cumkv); ptr(1024, (char*)c.scratch + 311296); ptr(1032, c.scratch);
  ptr(1096, c.seqlens); ptr(1112, c.stats);
  i32(1128, 0x7FFFFFFF); i32(1132, c.batch); i32(1172, c.maxq); i32(1176, c.kvcap); i32(1180, (c.maxq + TILE - 1) / TILE);
  i32(1184, 1); i32(1192, H); i32(1196, H); i32(1200, 1); i32(1204, 1); i32(1208, INT32_MIN); i32(1216, -1);
  long long hid = (long long)H * HQK; memcpy(p + 1224, &hid, 8);
  i32(1236, TILE); i32(1240, -1); i32(1244, 1); f32(1248, 1.f); f32(1252, SCALE_LOG2); f32(1260, 1.f);
  i32(1276, c.sumq); i32(1280, c.kvcap);
}
static CUfunction fn_causal, fn_dense;
static void launch(CUfunction fn, const Call& c) {
  alignas(128) static uint8_t p[PARAMS];
  params(p, c);
  void* args[] = {p};
  CK(cuLaunchKernel(fn, (c.maxq + TILE - 1) / TILE, H, c.batch, 512, 1, 1, SMEM, 0, args, nullptr));
}

__global__ void fill(__nv_bfloat16* x, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    float u1 = ((h & 0xffff) + 0.5f) / 65536.f, u2 = ((h >> 16) + 0.5f) / 65536.f;
    x[i] = __float2bfloat16(scale * sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2));
  }
}

// f32 reference: row r of sequence `seq` (rows cq[seq].., keys ck[seq].. of length kl[seq]) sees keys [0, kl - ql + i]
__global__ void reference(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int* cq, const int* ck, const int* kl, int seq,
                          float* o) {
  int r = blockIdx.x, h = blockIdx.y, i = r - cq[seq], ql = cq[seq + 1] - cq[seq];
  if (i < 0 || i >= ql) return;
  int n = kl[seq] - ql + i + 1;
  extern __shared__ float sm[];
  float* qs = sm; float* red = sm + HQK;
  for (int d = threadIdx.x; d < HQK; d += blockDim.x) qs[d] = __bfloat162float(q[((size_t)r * H + h) * HQK + d]);
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const __nv_bfloat16* k = kv + ((size_t)(ck[seq] + j) * H + h) * KVR;
    float acc = 0; for (int d = 0; d < HQK; ++d) acc += qs[d] * __bfloat162float(k[d]);
    m = fmaxf(m, acc * SCALE_LOG2);
  }
  red[threadIdx.x] = m; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + w]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float acc_o[HV] = {}; float sum = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const __nv_bfloat16* k = kv + ((size_t)(ck[seq] + j) * H + h) * KVR;
    float a = 0; for (int d = 0; d < HQK; ++d) a += qs[d] * __bfloat162float(k[d]);
    float p = exp2f(a * SCALE_LOG2 - m); sum += p;
    for (int d = 0; d < HV; ++d) acc_o[d] += p * __bfloat162float(k[HQK + d]);
  }
  red[threadIdx.x] = sum; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w]; __syncthreads(); }
  sum = red[0]; __syncthreads();
  float* out = o + ((size_t)r * H + h) * HV;
  if (threadIdx.x == 0) for (int d = 0; d < HV; ++d) out[d] = 0;
  __syncthreads();
  for (int d = 0; d < HV; ++d) atomicAdd(out + d, acc_o[d] / sum);
}

// The accumulator's merge of one partial (o bf16, stats (max, sum) natural) for the rows whose sequence has
// keys in it (kl[seq(row)] > 0): acc = acc e^(m - m') + s_p e^(m_p - m') o_p, s likewise, m' the new max.
__global__ void merge(const __nv_bfloat16* o, const float2* st, const int* cq, const int* kl, int nseq, float* acc, float* am,
                      float* as, int T) {
  int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  int j = 0;
  while (j + 1 < nseq && cq[j + 1] <= r) ++j;
  if (kl[j] == 0) return;
  float2 s = st[(size_t)r * H + h];
  float m0 = am[(size_t)r * H + h], s0 = as[(size_t)r * H + h], m1 = fmaxf(m0, s.x);
  float wa = expf(m0 - m1), wp = s.y * expf(s.x - m1);
  float* a = acc + ((size_t)r * H + h) * HV;
  a[d] = a[d] * wa + wp * __bfloat162float(o[((size_t)r * H + h) * HV + d]);
  if (d == 0) am[(size_t)r * H + h] = m1, as[(size_t)r * H + h] = s0 * wa + wp;
}
__global__ void finish(const float* acc, const float* as, __nv_bfloat16* o, int T) {
  int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  o[((size_t)r * H + h) * HV + d] = __float2bfloat16(acc[((size_t)r * H + h) * HV + d] / as[(size_t)r * H + h]);
}
__global__ void reset(float* am, float* as, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) am[i] = -INFINITY, as[i] = 0.f;
}

template <class T> static T* dev(size_t n) { T* p; RT(cudaMalloc(&p, n * sizeof(T))); RT(cudaMemset(p, 0, n * sizeof(T))); return p; }
static int* devi(std::vector<int> v) { int* p = dev<int>(v.size()); RT(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p; }

static double compare(const char* what, const __nv_bfloat16* o, const float* ref, size_t n) {
  std::vector<__nv_bfloat16> a(n); std::vector<float> r(n);
  RT(cudaMemcpy(a.data(), o, n * 2, cudaMemcpyDeviceToHost)); RT(cudaMemcpy(r.data(), ref, n * 4, cudaMemcpyDeviceToHost));
  double e2 = 0, r2 = 0, mx = 0; size_t nans = 0;
  for (size_t k = 0; k < n; ++k) {
    float x = __bfloat162float(a[k]);
    if (!std::isfinite(x)) { ++nans; continue; }
    double e = x - r[k]; e2 += e * e; r2 += (double)r[k] * r[k]; mx = fmax(mx, fabs(e));
  }
  printf("  %-36s relRMS %.3e  max abs %.3e  non-finite %zu\n", what, sqrt(e2 / r2), mx, nans);
  return sqrt(e2 / r2);
}

// A packed call: sequences (ql_j own rows over p_j cached keys), kv_full holds every sequence's keys
// (cached then own), kv_own its own keys only, round w of sequence j its keys [w S, min((w + 1) S, p_j)).
struct Packed {
  std::vector<int> ql, p, cq, ck_full, kl_full;
  int T = 0, K = 0, rounds = 0;
};
static Packed packed(std::vector<std::pair<int, int>> seqs, int S) {
  Packed x;
  x.cq.push_back(0); x.ck_full.push_back(0);
  for (auto [ql, p] : seqs) {
    x.ql.push_back(ql); x.p.push_back(p); x.kl_full.push_back(p + ql);
    x.T += ql; x.K += p + ql;
    x.cq.push_back(x.T); x.ck_full.push_back(x.K);
    x.rounds = std::max(x.rounds, (p + S - 1) / S);
  }
  return x;
}

int main(int argc, char** argv) {
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));
  CUmodule mc, md;
  CK(cuModuleLoad(&mc, argv[1])); CK(cuModuleGetFunction(&fn_causal, mc, CAUSAL));
  CK(cuModuleLoad(&md, argv[2])); CK(cuModuleGetFunction(&fn_dense, md, DENSE));
  CK(cuFuncSetAttribute(fn_causal, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM));
  CK(cuFuncSetAttribute(fn_dense, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM));
  bool check = !strcmp(argv[3], "check");
  int S = check ? atoi(argc > 4 ? argv[4] : "4096") : atoi(argv[6]);
  // the check's call: a long context, a fresh sequence (no keys in any round) and one whose last round is 4 keys
  Packed x = check ? packed({{2000, 9000}, {300, 0}, {1200, 4100}}, S) : packed({{atoi(argv[4]), atoi(argv[5])}}, S);
  int nseq = (int)x.ql.size();
  size_t qn = (size_t)x.T * H * HQK, kvn = (size_t)x.K * H * KVR;
  auto q = dev<__nv_bfloat16>(qn), kv_full = dev<__nv_bfloat16>(kvn);
  fill<<<1024, 256>>>(q, qn, 1, 1.f); fill<<<1024, 256>>>(kv_full, kvn, 2, 1.f);
  // own keys: sequence j's rows [p_j, p_j + ql_j) of kv_full, packed in q order (row b's key is row b)
  auto kv_own = dev<__nv_bfloat16>((size_t)std::max(x.T, 256) * H * KVR);
  for (int j = 0; j < nseq; ++j)
    RT(cudaMemcpy(kv_own + (size_t)x.cq[j] * H * KVR, kv_full + (size_t)(x.ck_full[j] + x.p[j]) * H * KVR,
                  (size_t)x.ql[j] * H * KVR * 2, cudaMemcpyDeviceToDevice));
  int rcap = std::max(nseq * S, 256);
  auto kv_round = dev<__nv_bfloat16>((size_t)rcap * H * KVR);
  auto o = dev<__nv_bfloat16>((size_t)x.T * H * HV), op = dev<__nv_bfloat16>((size_t)x.T * H * HV), om = dev<__nv_bfloat16>((size_t)x.T * H * HV);
  auto st = dev<float2>((size_t)x.T * H);
  auto acc = dev<float>((size_t)x.T * H * HV), am = dev<float>((size_t)x.T * H), as = dev<float>((size_t)x.T * H);
  void* scratch = dev<uint8_t>(8 << 20);
  auto cq = devi(x.cq);
  // the plain causal call over the whole context
  auto plain = Call{q, kv_full, o, devi(x.kl_full), cq, devi(x.ck_full), st, scratch, x.T, std::max(x.K, 256), nseq, x.T, x.T};
  // the own rows' causal call
  auto own = Call{q, kv_own, op, devi(x.ql), cq, cq, st, scratch, x.T, std::max(x.T, 256), nseq, x.T, x.T};
  auto own_kl = devi(x.ql);
  // round w: its keys gathered, its tables
  struct Round { Call call; int* kl; int total; };
  std::vector<Round> rounds;
  for (int w = 0; w < x.rounds; ++w) {
    std::vector<int> kl, ck = {0};
    int total = 0;
    for (int j = 0; j < nseq; ++j) {
      int n = std::min(std::max(x.p[j] - w * S, 0), S);
      kl.push_back(n); total += n; ck.push_back(total);
    }
    auto klp = devi(kl);
    rounds.push_back({Call{q, kv_round, op, klp, cq, devi(ck), st, scratch, x.T, std::max(rcap, 256), nseq, x.T, x.T}, klp, total});
  }
  auto gather = [&](int w) {
    int at = 0;
    for (int j = 0; j < nseq; ++j) {
      int n = std::min(std::max(x.p[j] - w * S, 0), S);
      if (n > 0) RT(cudaMemcpy(kv_round + (size_t)at * H * KVR, kv_full + (size_t)(x.ck_full[j] + w * S) * H * KVR,
                               (size_t)n * H * KVR * 2, cudaMemcpyDeviceToDevice));
      at += n;
    }
  };
  auto windowed = [&] {
    reset<<<256, 256>>>(am, as, (size_t)x.T * H);
    launch(fn_causal, own);
    merge<<<dim3(x.T, H), HV>>>(op, st, cq, own_kl, nseq, acc, am, as, x.T);
    for (int w = 0; w < x.rounds; ++w) {
      gather(w);
      launch(fn_dense, rounds[w].call);
      merge<<<dim3(x.T, H), HV>>>(op, st, cq, rounds[w].kl, nseq, acc, am, as, x.T);
    }
    finish<<<dim3(x.T, H), HV>>>(acc, as, om, x.T);
  };
  if (check) {
    auto ref = dev<float>((size_t)x.T * H * HV);
    auto ck_full = devi(x.ck_full), kl_full = devi(x.kl_full);
    for (int j = 0; j < nseq; ++j) reference<<<dim3(x.T, H), 128, (HQK + 128) * 4>>>(q, kv_full, cq, ck_full, kl_full, j, ref);
    RT(cudaDeviceSynchronize());
    printf("packed call: %d sequences, %d rows, %d keys, S %d, %d rounds\n", nseq, x.T, x.K, S, x.rounds);
    launch(fn_causal, plain); RT(cudaDeviceSynchronize());
    compare("plain causal", o, ref, (size_t)x.T * H * HV);
    windowed(); RT(cudaDeviceSynchronize());
    double e = compare("own causal + dense rounds, merged", om, ref, (size_t)x.T * H * HV);
    // the fresh sequence's rows alone (no keys in any round): their o must be the own call's
    size_t from = (size_t)x.cq[1] * H * HV, n1 = (size_t)x.ql[1] * H * HV;
    compare("  the fresh sequence's rows", om + from, ref + from, n1);
    // the dense call of the last round: sequences 1 and 2 have no keys there
    int wl = x.rounds - 1;
    std::vector<float2> s((size_t)x.T * H);
    RT(cudaMemcpy(s.data(), st, s.size() * 8, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (int b = x.cq[1]; b < x.cq[2]; ++b) for (int h = 0; h < H; ++h) if (std::isfinite(s[(size_t)b * H + h].x) && s[(size_t)b * H + h].y != 0) ++bad;
    printf("  last round (%d): an empty sequence's stats are (-inf, 0) on %zu of %d rows x heads otherwise\n", wl,
           (size_t)(x.cq[2] - x.cq[1]) * H - bad, (x.cq[2] - x.cq[1]) * H);
    printf("RESULT\t%s\n", e < 2e-2 ? "PASS" : "FAIL");
    return e < 2e-2 ? 0 : 1;
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time = [&](auto f) {
    f(); RT(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < 5; ++r) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms); }
    std::sort(v.begin(), v.end()); return v[2];
  };
  int ql = x.ql[0], p = x.p[0];
  {  // an empty round: every sequence without keys
    std::vector<int> z(nseq, 0), zc(nseq + 1, 0);
    auto empty = Call{q, kv_round, op, devi(z), cq, devi(zc), st, scratch, x.T, std::max(rcap, 256), nseq, x.T, x.T};
    printf("empty dense round over %d rows: %.3f ms\n", x.T, time([&] { launch(fn_dense, empty); }));
  }
  double flop = 2.0 * ql * ((double)p + (ql + 1) / 2.0) * H * (HQK + HV);
  float t0 = time([&] { launch(fn_causal, plain); });
  for (int w = 0; w < x.rounds; ++w) gather(w);
  float t_own = time([&] { launch(fn_causal, own); });
  float t_dense = time([&] { for (int w = 0; w < x.rounds; ++w) launch(fn_dense, rounds[w].call); });
  float t_merge = time([&] { for (int w = 0; w <= x.rounds; ++w) merge<<<dim3(x.T, H), HV>>>(op, st, cq, own_kl, nseq, acc, am, as, x.T); });
  printf("rows %d prefix %d S %d (%d rounds): plain %.3f ms (%.0f TF/s); own %.3f + dense %.3f + merges %.3f = %.3f ms (%+.1f%%)\n",
         ql, p, S, x.rounds, t0, flop / t0 / 1e9, t_own, t_dense, t_merge, t_own + t_dense + t_merge,
         100 * (t_own + t_dense + t_merge - t0) / t0);
  return 0;
}
