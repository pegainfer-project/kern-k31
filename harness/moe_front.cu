// harness/moe_front.cu: kern_k3_moe_front cubins timed in a CUDA graph (S warm in L2 as the front GEMM leaves it), outputs
// compared with the first cubin: bit for bit, the routing tables up to the order of an expert's rows.
//   $K3_NVCC -O2 -std=c++17 harness/moe_front.cu -o OUT -lcuda && loop/withgpu -n 1 OUT [--rows 1,8] [--ties] CUBIN...
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(2); } } while (0)
static uint64_t rng = 88172645463325252ull;
static float frand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f * 2 - 1; }
int main(int argc, char** argv) {
  std::vector<std::string> cubins; std::vector<int> rows = {1, 8, 24, 48, 64}; int ties = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rows")) { rows.clear(); for (char* s = strtok(argv[++i], ","); s; s = strtok(0, ",")) rows.push_back(atoi(s)); }
    else if (!strcmp(argv[i], "--ties")) ties = 1;
    else cubins.push_back(argv[i]);
  }
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  const int E = 896, LDS = 6016, LAT = 3584, SH = 768, TOPK = 16, TM = 64;
  std::vector<float> S((size_t)TM * LDS), bias(E);
  for (auto& x : S) x = ties ? (float)(int)(frand() * 6) * 0.5f : frand() * 4; for (auto& x : bias) x = ties ? 0.f : frand() * 0.1f;
  CUdeviceptr dS, db, drs, didx, dw, dq, dsf, dact, ddone, dcb, dcl, dnn, dtp, drm, de2p;
  CK(cuMemAlloc(&dS, S.size() * 4)); CK(cuMemcpyHtoD(dS, S.data(), S.size() * 4));
  CK(cuMemAlloc(&db, E * 4)); CK(cuMemcpyHtoD(db, bias.data(), E * 4));
  uint16_t rs = 0x3f80; CK(cuMemAlloc(&drs, 2)); CK(cuMemcpyHtoD(drs, &rs, 2));
  CK(cuMemAlloc(&didx, TM * TOPK * 4)); CK(cuMemAlloc(&dw, TM * TOPK * 4)); CK(cuMemAlloc(&dq, TM * LAT));
  CK(cuMemAlloc(&dsf, TM * LAT / 32)); CK(cuMemAlloc(&dact, TM * SH * 2)); CK(cuMemAlloc(&ddone, 4)); CK(cuMemsetD8(ddone, 0, 4));
  const int CT = 2048, RM = CT * 16;
  CK(cuMemAlloc(&dcb, CT * 4)); CK(cuMemAlloc(&dcl, CT * 4)); CK(cuMemAlloc(&dnn, 4)); CK(cuMemAlloc(&dtp, 4));
  CK(cuMemAlloc(&drm, RM * 4)); CK(cuMemAlloc(&de2p, TM * TOPK * 4));
  struct Out { CUdeviceptr p; size_t n; };
  std::vector<Out> outs = {{didx, TM * TOPK * 4u}, {dw, TM * TOPK * 4u}, {dq, (size_t)TM * LAT}, {dsf, (size_t)TM * LAT / 32},
                           {dact, (size_t)TM * SH * 2}, {dcb, CT * 4u}, {dcl, CT * 4u}, {dnn, 4}, {dtp, 4}, {drm, RM * 4u}, {de2p, TM * TOPK * 4u}};
  std::vector<CUfunction> fs;
  for (auto& c : cubins) { CUmodule m; CK(cuModuleLoad(&m, c.c_str())); CUfunction f; CK(cuModuleGetFunction(&f, m, "kern_k3_moe_front")); fs.push_back(f); }
  CUstream st; CK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));
  int first = 0;
  auto launch = [&](CUfunction f, int T) {
    void* a[] = {&dS, &db, &drs, &didx, &dw, &dq, &dsf, &dact, &ddone, &dcb, &dcl, &dnn, &dtp, &drm, &de2p, &first, &T};
    CK(cuLaunchKernel(f, T, 1, 1, 1024, 1, 1, 0, st, a, 0));
  };
  const int reps = 40;
  printf("rows"); for (auto& c : cubins) printf("  %s", c.substr(c.rfind('/') + 1).c_str()); printf("\n");
  for (int T : rows) {
    printf("%4d", T);
    std::vector<std::vector<uint8_t>> ref;
    for (size_t k = 0; k < fs.size(); ++k) {
      CUgraph g; CUgraphExec ge;
      CK(cuStreamBeginCapture(st, CU_STREAM_CAPTURE_MODE_GLOBAL));
      for (int i = 0; i < reps; ++i) launch(fs[k], T);
      CK(cuStreamEndCapture(st, &g)); CK(cuGraphInstantiate(&ge, g, 0));
      float best = 1e9;
      for (int r = 0; r < 7; ++r) {
        CUevent e0, e1; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
        cuEventRecord(e0, st); CK(cuGraphLaunch(ge, st)); cuEventRecord(e1, st); CK(cuEventSynchronize(e1));
        float ms; cuEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
      }
      for (auto& o : outs) CK(cuMemsetD8(o.p, 0xab, o.n));
      launch(fs[k], T); CK(cuStreamSynchronize(st));
      std::vector<std::vector<uint8_t>> got;
      for (auto& o : outs) { got.emplace_back(o.n); CK(cuMemcpyDtoH(got.back().data(), o.p, o.n)); }
      bool same = true;
      if (k == 0) ref = got;
      else {
        for (size_t i = 0; i < 9; ++i) if (got[i] != ref[i]) { same = false; printf(" [out %zu differs]", i); }
        // exp2perm: a row's slot may move within its expert's tiles; route_map must name the row there
        const int* e2 = (const int*)got[10].data(); const int* e2r = (const int*)ref[10].data();
        const int* rm = (const int*)got[9].data(); const int* cb = (const int*)got[5].data();
        for (int i = 0; i < T * TOPK; ++i) {
          const bool ok = (e2[i] < 0) == (e2r[i] < 0) && (e2[i] < 0 || (rm[e2[i]] == i / TOPK && cb[e2[i] / 16] == cb[e2r[i] / 16]));
          if (!ok) { same = false; printf(" [route %d]", i); break; }
        }
      }
      printf("  %7.2f%s", best * 1e3 / reps, k == 0 ? "" : same ? " =" : " DIFF");
      cuGraphExecDestroy(ge); cuGraphDestroy(g);
    }
    printf("\n");
  }
}
