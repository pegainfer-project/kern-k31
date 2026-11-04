// harness/attn/absorb.cu: k3_mla_absorb_mma timed cold on one GPU (W_UK cold, q_partial warm, as in the step) and
// cubins compared bit for bit with the first. Run under nsys (harness/attn/run.sh): each absorb launch's own time.
//   absorb [--rows 8,16] CUBIN[@threads[@gridz]][#dynamic smem bytes][!tc]...
//   !tc: k3_mla_absorb_tc's entry, a block a head, W_UK by a tensor map (box 64 x 128, 128 B swizzle)
#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(2); } } while (0)
static uint64_t rng = 88172645463325252ull;
static float frand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f * 2 - 1; }
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (u + 0x7fff + ((u >> 16) & 1)) >> 16; }
struct K { CUmodule m; CUfunction f; int threads, gz, smem; bool tc; };
int main(int argc, char** argv) {
  std::vector<std::string> cubins; std::vector<int> rows = {8, 16, 24, 36, 48}; int reps = 30;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rows")) { rows.clear(); for (char* s = strtok(argv[++i], ","); s; s = strtok(0, ",")) rows.push_back(atoi(s)); }
    else cubins.push_back(argv[i]);
  }
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  const int H = 96, QW = 192, BM = 64; const size_t WN = (size_t)H * 256 * 512, QN = (size_t)BM * H * QW, ON = (size_t)BM * H * 576;
  std::vector<float> q(QN); for (auto& x : q) x = frand();
  std::vector<uint16_t> w(WN); for (auto& x : w) x = f2b(frand() * 0.05f);
  CUdeviceptr dq, dw, dout, dfl, dsink; CK(cuMemAlloc(&dq, QN * 4)); CK(cuMemcpyHtoD(dq, q.data(), QN * 4));
  CK(cuMemAlloc(&dw, WN * 2)); CK(cuMemcpyHtoD(dw, w.data(), WN * 2)); CK(cuMemAlloc(&dout, ON * 2));
  const size_t FL = 256 << 20; CK(cuMemAlloc(&dfl, FL)); dsink = dout;
  CUmodule fm; CK(cuModuleLoad(&fm, getenv("FLUSH_CUBIN") ? getenv("FLUSH_CUBIN") : "flush.cubin")); CUfunction ff; CK(cuModuleGetFunction(&ff, fm, "flush_read"));
  std::vector<K> ks;
  for (auto& c : cubins) {
    std::string p = c; K k{0, 0, 256, 4, 0, false};
    if (auto a = p.find('!'); a != std::string::npos) { k.tc = true; p.resize(a); }
    if (auto a = p.find('#'); a != std::string::npos) { k.smem = atoi(p.c_str() + a + 1); p.resize(a); }
    if (auto a = p.find('@'); a != std::string::npos) { sscanf(p.c_str() + a + 1, "%d@%d", &k.threads, &k.gz); p.resize(a); }
    CK(cuModuleLoad(&k.m, p.c_str())); CK(cuModuleGetFunction(&k.f, k.m, k.tc ? "kern_k3_mla_absorb_tc" : "kern_k3_mla_absorb_mma"));
    if (k.smem) CK(cuFuncSetAttribute(k.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem));
    if (getenv("CARVEOUT")) CK(cuFuncSetAttribute(k.f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, atoi(getenv("CARVEOUT"))));
    int occ; CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k.f, k.threads, k.smem));
    fprintf(stderr, "%s: %d blocks an SM\n", c.c_str(), occ);
    ks.push_back(k);
  }
  CUstream st; CK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));
  CUtensorMap wmap;
  {
    cuuint64_t dims[2] = {512, (cuuint64_t)H * 256}, strides[1] = {1024};
    cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
    CK(cuTensorMapEncodeTiled(&wmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dw, dims, strides, box, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  auto launch = [&](K& k, int B) {
    if (k.tc) { void* a[] = {&wmap, &dq, &dout, &B}; CK(cuLaunchKernel(k.f, H, 1, 1, k.threads, 1, 1, k.smem, st, a, 0)); }
    else { void* a[] = {&dq, &dw, &dout, &B}; CK(cuLaunchKernel(k.f, 1, H, k.gz, k.threads, 1, 1, k.smem, st, a, 0)); }
  };
  auto timed = [&](K& k, int B) {
    CK(cuStreamBeginCapture(st, CU_STREAM_CAPTURE_MODE_GLOBAL));
    for (int i = 0; i < reps; ++i) {
      long long fn = FL / 16, qn = (long long)B * H * QW / 4;
      void* fa[] = {&dfl, &fn, &dsink}; CK(cuLaunchKernel(ff, 152 * 4, 1, 1, 512, 1, 1, 0, st, fa, 0));
      void* qa[] = {&dq, &qn, &dsink}; CK(cuLaunchKernel(ff, 152, 1, 1, 512, 1, 1, 0, st, qa, 0));
      launch(k, B);
    }
    CUgraph g; CK(cuStreamEndCapture(st, &g)); CUgraphExec ge; CK(cuGraphInstantiate(&ge, g, 0));
    CK(cuGraphLaunch(ge, st)); CK(cuStreamSynchronize(st)); CK(cuGraphExecDestroy(ge)); CK(cuGraphDestroy(g));
  };
  printf("reps %d rounds 3\nrows  floor_us", reps); for (auto& c : cubins) printf("  %s", c.c_str()); printf("\n");
  for (int B : rows) {
    for (int r = 0; r < 3; ++r) for (auto& k : ks) timed(k, B);
    printf("%4d  %8.2f", B, (H * 128.0 * 512 * 2) / 7.33e6);
    std::vector<uint16_t> r0(ON), r1(ON);
    for (size_t i = 0; i < ks.size(); ++i) {
      CK(cuMemsetD8(dout, 0xff, ON * 2)); launch(ks[i], B); CK(cuStreamSynchronize(st));
      CK(cuMemcpyDtoH(i ? r1.data() : r0.data(), dout, ON * 2));
      if (i == 0 || r1 == r0) { printf("  %s", i == 0 ? "base" : "="); continue; }
      // the written elements that differ, and the largest difference in bf16 ulps
      size_t nd = 0; int ulp = 0;
      for (int b = 0; b < B; ++b) for (int e = 0; e < H * 576; ++e) {
        const size_t x = (size_t)b * H * 576 + e; if (r1[x] == r0[x]) continue;
        ++nd; ulp = std::max(ulp, std::abs((int)(int16_t)r1[x] - (int)(int16_t)r0[x]));
      }
      printf("  DIFF %zu/%d max %d ulp", nd, B * H * 576, ulp);
    }
    printf("\n");
  }
}
