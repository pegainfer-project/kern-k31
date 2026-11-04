// harness/attn/kda.cu: k3_kda_core's D build (HEADS 12, KDA_CONV, LDS 6400) timed cold, and two cubins compared bit for bit.
//   kda [--rows 8,16,...] [--reps N] [--warm] BASE.cubin VARIANT.cubin...
//   a cubin as PATH[#dynamic smem bytes][%] (% = a 2-CTA cluster an item, grid x = 2 B)
// Each timed launch follows a 256 MB read and a read of the partial (state and weights cold, the GEMM's output in L2, as in the step);
// the read alone is timed the same way and subtracted. Prints us per call per cubin, and
// whether out and the state after one call match the base.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(2); } } while (0)
static const int H = 12, KD = 128, INNER = H * KD, LDS = 6400;
static const long long REC = (long long)H * KD * KD * 4, LINE = REC + 3LL * 3 * INNER * 2;
static uint64_t rng = 88172645463325252ull;
static float frand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f * 2 - 1; }
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (u + 0x7fff + ((u >> 16) & 1)) >> 16; }
static CUdeviceptr up(const void* h, size_t n) { CUdeviceptr d; CK(cuMemAlloc(&d, n)); CK(cuMemcpyHtoD(d, h, n)); return d; }
struct K { CUmodule m; CUfunction f; int threads, dsmem, split; };
static int g_dsmem = 0;
static K load(const char* arg) {
  std::string path = arg; K k; k.threads = 128; k.dsmem = 0; k.split = 0;
  if (auto at = path.find('%'); at != std::string::npos) k.split = 1, path.resize(at);
  if (auto at = path.find('#'); at != std::string::npos) k.dsmem = atoi(path.c_str() + at + 1), path.resize(at);
  if (auto at = path.find('@'); at != std::string::npos) k.threads = atoi(path.c_str() + at + 1), path.resize(at);
  const char* p = path.c_str(); CK(cuModuleLoad(&k.m, p)); CK(cuModuleGetFunction(&k.f, k.m, "kern_k3_kda_core"));
  int st; CK(cuFuncGetAttribute(&st, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k.f));
  if (k.dsmem) CK(cuFuncSetAttribute(k.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.dsmem));
  return k; }
int main(int argc, char** argv) {
  std::vector<const char*> cubins; std::vector<int> rows = {8, 16, 24, 36, 48}; int reps = 30, gh = H, threads = 128, dsmem = 0; bool warm = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rows")) { rows.clear(); for (char* s = strtok(argv[++i], ","); s; s = strtok(0, ",")) rows.push_back(atoi(s)); }
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--grid-heads")) gh = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warm")) warm = true;
    else if (!strcmp(argv[i], "--dsmem")) dsmem = atoi(argv[++i]);
    else cubins.push_back(argv[i]);
  }
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  const int BM = 64;
  std::vector<float> partial((size_t)BM * LDS), cw(3 * 4 * INNER), dtb(INNER), alog(H), gam(KD);
  for (auto& x : partial) x = frand() * 2;
  for (auto& x : cw) x = frand() * 0.5f;
  for (auto& x : dtb) x = frand();
  for (auto& x : alog) x = frand() * 0.5f;
  for (auto& x : gam) x = 1 + frand() * 0.2f;
  std::vector<uint16_t> wfb((size_t)INNER * KD); for (auto& x : wfb) x = f2b(frand() * 0.1f);
  CUdeviceptr d_part = up(partial.data(), partial.size() * 4), d_cw = up(cw.data(), cw.size() * 4),
              d_wfb = up(wfb.data(), wfb.size() * 2), d_dtb = up(dtb.data(), INNER * 4), d_alog = up(alog.data(), H * 4),
              d_gam = up(gam.data(), KD * 4);
  int zero[2] = {0, 0}; CUdeviceptr d_span = up(zero, 8);
  const int SETS = 6;  // rotating line sets: 6 x 64 lines x 814 KB = 312 MB
  std::vector<uint8_t> lines((size_t)SETS * BM * LINE);
  { float* f = (float*)lines.data(); for (size_t i = 0; i < lines.size() / 4; ++i) f[i] = frand() * 0.05f; }
  for (int l = 0; l < SETS * BM; ++l) { uint16_t* w = (uint16_t*)(lines.data() + (size_t)l * LINE + REC); for (int i = 0; i < 9 * INNER; ++i) w[i] = f2b(frand()); }
  CUdeviceptr d_lines = up(lines.data(), lines.size());
  std::vector<int> li(SETS * BM); for (int i = 0; i < SETS * BM; ++i) li[i] = i;
  CUdeviceptr d_li = up(li.data(), li.size() * 4);
  CUdeviceptr d_out; CK(cuMemAlloc(&d_out, (size_t)BM * INNER * 2));
  CUdeviceptr d_flush; const size_t FL = 256 << 20; CK(cuMemAlloc(&d_flush, FL));
  g_dsmem = dsmem; std::vector<K> ks; for (auto c : cubins) ks.push_back(load(c));
  CUmodule fm; CK(cuModuleLoad(&fm, getenv("FLUSH_CUBIN") ? getenv("FLUSH_CUBIN") : "flush.cubin")); CUfunction ff; CK(cuModuleGetFunction(&ff, fm, "flush_read"));
  long long fn = FL / 16; CUdeviceptr d_sink = d_out;
  CUstream st; CK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));
  CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
  long long lb = LINE;
  auto launch = [&](K& k, int B, int set) {
    CUdeviceptr li_p = d_li + (size_t)set * BM * 4, wsm = d_part + 6144 * 4;
    void* a[] = {&d_part, &d_cw, &wsm, &d_wfb, &d_dtb, &d_alog, &d_gam, &d_lines, &li_p, &lb, &d_out, &B, &d_span, &zero[0]};
    if (k.split) {
      CUlaunchConfig cfg = {}; CUlaunchAttribute at[1];
      at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION; at[0].value.clusterDim.x = 2; at[0].value.clusterDim.y = 1; at[0].value.clusterDim.z = 1;
      cfg.gridDimX = 2 * B; cfg.gridDimY = gh; cfg.gridDimZ = 1; cfg.blockDimX = k.threads; cfg.blockDimY = 1; cfg.blockDimZ = 1;
      cfg.sharedMemBytes = k.dsmem; cfg.hStream = st; cfg.attrs = at; cfg.numAttrs = 1;
      CK(cuLaunchKernelEx(&cfg, k.f, a, 0));
    } else {
      CK(cuLaunchKernel(k.f, B, gh, 1, k.threads, 1, 1, k.dsmem, st, a, 0));
    }
  };
  // a graph of reps x (flush [, kda]) per (cubin or none)
  // one graph launch of reps x (flush, partial read, kda): under nsys each kda launch's duration is its own
  auto timed = [&](K* k, int B) {
    CK(cuStreamBeginCapture(st, CU_STREAM_CAPTURE_MODE_GLOBAL));
    for (int i = 0; i < reps; ++i) {
      if (!warm) { void* fa[] = {&d_flush, &fn, &d_sink}; CK(cuLaunchKernel(ff, 152 * 4, 1, 1, 512, 1, 1, 0, st, fa, 0));
        long long pn = (long long)B * LDS / 4; void* pa[] = {&d_part, &pn, &d_sink}; CK(cuLaunchKernel(ff, 152, 1, 1, 512, 1, 1, 0, st, pa, 0));
        if (getenv("STATE_WARM")) {  // the rows' lines (state and windows) read into L2 after the flush
          CUdeviceptr lp = d_lines + (size_t)(i % SETS) * BM * LINE; long long ln = (long long)B * LINE / 16;
          void* la[] = {&lp, &ln, &d_sink}; CK(cuLaunchKernel(ff, 152, 1, 1, 512, 1, 1, 0, st, la, 0)); } }
      launch(*k, B, warm ? 0 : i % SETS);
    }
    CUgraph g; CK(cuStreamEndCapture(st, &g)); CUgraphExec ge; CK(cuGraphInstantiate(&ge, g, 0));
    CK(cuGraphLaunch(ge, st)); CK(cuStreamSynchronize(st));
    CK(cuGraphExecDestroy(ge)); CK(cuGraphDestroy(g));
  };
  // the outputs of one call from the same initial lines
  auto result = [&](K& k, int B, std::vector<uint8_t>& o) {
    CK(cuMemcpyHtoD(d_lines, lines.data(), (size_t)B * LINE)); CK(cuMemsetD8(d_out, 0xff, (size_t)BM * INNER * 2));
    launch(k, B, 0); CK(cuStreamSynchronize(st));
    o.resize((size_t)B * INNER * 2 + (size_t)B * LINE);
    CK(cuMemcpyDtoH(o.data(), d_out, (size_t)B * INNER * 2)); CK(cuMemcpyDtoH(o.data() + (size_t)B * INNER * 2, d_lines, (size_t)B * LINE));
  };
  printf("reps %d rounds 3\nrows  floor_us", reps); for (auto c : cubins) printf("  %s", strrchr(c, '/') ? strrchr(c, '/') + 1 : c); printf("\n");
  for (int B : rows) {
    for (int round = 0; round < 3; ++round)
      for (size_t i = 0; i < ks.size(); ++i) timed(&ks[i], B);
    printf("%4d  %8.2f", B, 2.0 * B * (REC + 3.0 * INNER * 2) / 7.33e6);
    std::vector<uint8_t> r0, r;
    for (size_t i = 0; i < ks.size(); ++i) {
      result(ks[i], B, i ? r : r0);
      printf("  %s", i == 0 ? "base" : (r == r0 ? "=" : "DIFF"));
    }
    printf("\n");
  }
  return 0;
}
