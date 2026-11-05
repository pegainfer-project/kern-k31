// kda_core_ab.cu: two `k3_kda_core` builds of the TP8 rank (12 heads, the fused 6400-column
// partial) on the same inputs, bit for bit, and their cold-L2 time.
//
// Equality: both run `--steps` consecutive decode steps from the same states (each step's partial fresh,
// the states carried), and every step's output, every line's rec and conv window must be byte-identical.
// Time: `--pools` copies of the B rows' lines (default: enough for 1 GB, past L2), one launch on each in
// turn, back to back on one stream, the mean per launch over `--reps` rounds; a step's KDA core reads
// state no earlier kernel left in L2, and so does this. A step launches it as a programmatic dependent of
// the projection GEMM, so what precedes its griddepcontrol.wait can hide under the GEMM's tail: `pdl`
// times also put a 5 us kernel (one 128-thread block an SM, no shared memory, launch_dependents at entry)
// before each launch and report what the pair takes beyond that kernel alone.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a kda_core_ab.cu -o kda_core_ab -lcuda
//   ./kda_core_ab --base <cubin> [--smem 65536] [--slots G] --variant <cubin> [--vsmem N] [--vslots G]
//                 --B 64 [--steps 3] [--seed 1234] [--reps 20] [--pools N] [--pdl 0|1]
//
// A build launches (min(B, slots), 12) blocks of 128 threads (B blocks wide with no `--slots`) on `--smem` bytes.
// `--variant` may be omitted to time the base alone. Exit 0 PASS, 1 FAIL, 2 harness or driver error.
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

typedef uint16_t bf16;

static const int HEADS = 12, KD = 128, INNER = HEADS * KD, LDS = 6400;
static const long long REC = (long long)HEADS * KD * KD * 2, WIN = 3LL * 3 * INNER * 2, LINE = REC + WIN;

static void cu(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  std::fprintf(stderr, "%s: %s\n", what, s ? s : "?");
  std::exit(2);
}
#define CU(x) cu((x), #x)

static CUdeviceptr put(const void* h, size_t n) {
  CUdeviceptr d;
  CU(cuMemAlloc(&d, n));
  CU(cuMemcpyHtoD(d, h, n));
  return d;
}
static std::vector<uint8_t> get(CUdeviceptr d, size_t n) {
  std::vector<uint8_t> h(n);
  CU(cuMemcpyDtoH(h.data(), d, n));
  return h;
}
static bf16 f2b(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// The GEMM ahead of the KDA core in a step, as far as overlap goes: it lets its dependents on at entry and
// holds every SM for `ns`.
__global__ void pre(unsigned long long ns) {
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
}

static CUstream g_stream;  // a stream of its own: programmatic serialization does not apply on the legacy one

struct Build {
  CUfunction f = nullptr;
  int smem = 0, slots = 0;
  void launch(int B, void** args, bool pdl = false) const {
    CUlaunchConfig c = {};
    c.gridDimX = slots ? std::min(B, slots) : B, c.gridDimY = HEADS, c.gridDimZ = 1, c.blockDimX = 128, c.blockDimY = 1, c.blockDimZ = 1;
    c.sharedMemBytes = smem, c.hStream = g_stream;
    CUlaunchAttribute a = {};
    a.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    a.value.programmaticStreamSerializationAllowed = 1;
    c.attrs = &a, c.numAttrs = pdl ? 1 : 0;
    CU(cuLaunchKernelEx(&c, f, args, nullptr));
  }
};
static Build load(const std::string& path, int smem, int slots) {
  CUmodule m;
  CU(cuModuleLoad(&m, path.c_str()));
  Build b;
  CU(cuModuleGetFunction(&b.f, m, "kern_k3_kda_core"));
  b.smem = smem, b.slots = slots;
  if (smem) CU(cuFuncSetAttribute(b.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  return b;
}

int main(int argc, char** argv) {
  std::string base, variant;
  int B = 64, steps = 3, reps = 20, pools = 0, smem = 0, vsmem = -1, pdl = 1, slots = 0, vslots = 0;
  unsigned seed = 1234;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const char* v = argv[i + 1];
    if (a == "--base") base = v;
    else if (a == "--variant") variant = v;
    else if (a == "--B") B = std::atoi(v);
    else if (a == "--steps") steps = std::atoi(v);
    else if (a == "--reps") reps = std::atoi(v);
    else if (a == "--pools") pools = std::atoi(v);
    else if (a == "--seed") seed = (unsigned)std::atoi(v);
    else if (a == "--smem") smem = std::atoi(v);
    else if (a == "--vsmem") vsmem = std::atoi(v);
    else if (a == "--pdl") pdl = std::atoi(v);
    else if (a == "--slots") slots = std::atoi(v);
    else if (a == "--vslots") vslots = std::atoi(v);
    else { std::fprintf(stderr, "unknown option `%s`\n", argv[i]); return 2; }
  }
  if (base.empty() || B < 1) {
    std::fprintf(stderr, "usage: kda_core_ab --base <cubin> [--variant <cubin>] --B N\n");
    return 2;
  }
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CU(cuStreamCreate(&g_stream, CU_STREAM_NON_BLOCKING));
  const Build fb = load(base, smem, slots);
  const bool ab = !variant.empty();
  const Build fv = ab ? load(variant, vsmem < 0 ? smem : vsmem, vslots) : fb;

  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto f32 = [&](size_t n, float mean, float sd) {
    std::vector<float> v(n);
    for (auto& x : v) x = mean + sd * n01(rng);
    return v;
  };
  auto b16 = [&](size_t n, float sd) {
    std::vector<bf16> v(n);
    for (auto& x : v) x = f2b(sd * n01(rng));
    return v;
  };
  // Lines: B rows over a random permutation of B slots; rec ~ N(0, 0.1), the window ~ N(0, 1).
  std::vector<int> index(B);
  std::iota(index.begin(), index.end(), 0);
  std::shuffle(index.begin(), index.end(), rng);
  std::vector<uint8_t> lines((size_t)B * LINE);
  for (int s = 0; s < B; ++s) {
    auto r = b16(REC / 2, 0.1f), w = b16(WIN / 2, 1.f);
    std::memcpy(&lines[(size_t)s * LINE], r.data(), REC);
    std::memcpy(&lines[(size_t)s * LINE + REC], w.data(), WIN);
  }
  auto cw = f32((size_t)3 * 4 * INNER, 0.f, 0.5f);
  auto w_f_b = b16((size_t)INNER * KD, 0.044f);
  auto dt_bias = f32(INNER, 0.f, 1.f), a_log = f32(HEADS, 0.f, 0.5f), gamma_o = f32(KD, 1.f, 0.1f);

  CUdeviceptr dcw = put(cw.data(), cw.size() * 4), dwfb = put(w_f_b.data(), w_f_b.size() * 2);
  CUdeviceptr ddt = put(dt_bias.data(), dt_bias.size() * 4), dal = put(a_log.data(), a_log.size() * 4);
  CUdeviceptr dgo = put(gamma_o.data(), gamma_o.size() * 4), dix = put(index.data(), B * 4);
  CUdeviceptr dp;
  CU(cuMemAlloc(&dp, (size_t)B * LDS * 4));
  CUdeviceptr dwsm = dp + (CUdeviceptr)4 * INNER * 4;
  int zero = 0, span = 0, b_ = B;
  CUdeviceptr dat = put(&zero, 4);
  long long lb = LINE;
  const size_t nout = (size_t)B * INNER * 2;
  std::vector<bf16> poison((size_t)B * INNER, 0x7fc0);
  auto args = [&](CUdeviceptr& dl, CUdeviceptr& dout) {
    return std::vector<void*>{&dp, &dcw, &dwsm, &dwfb, &ddt, &dal, &dgo, &dl, &dix, &lb, &dout, &b_, &dat, &span};
  };

  bool ok = true;
  if (ab) {
    CUdeviceptr lbase = put(lines.data(), lines.size()), lvar = put(lines.data(), lines.size());
    CUdeviceptr obase = put(poison.data(), nout), ovar = put(poison.data(), nout);
    auto abase = args(lbase, obase), avar = args(lvar, ovar);
    long bad_out = 0, bad_line = 0, nonzero_rec = 0;
    for (int st = 0; st < steps; ++st) {
      auto partial = f32((size_t)B * LDS, 0.f, 1.f);
      CU(cuMemcpyHtoD(dp, partial.data(), partial.size() * 4));
      fb.launch(B, abase.data());
      fv.launch(B, avar.data());
      CU(cuCtxSynchronize());
      auto x = get(obase, nout), y = get(ovar, nout);
      for (size_t i = 0; i < nout; i += 2) bad_out += std::memcmp(&x[i], &y[i], 2) != 0;
      auto p = get(lbase, lines.size()), q = get(lvar, lines.size());
      for (size_t i = 0; i < p.size(); i += 2) bad_line += std::memcmp(&p[i], &q[i], 2) != 0;
      for (int s = 0; s < B; ++s)
        nonzero_rec += std::memcmp(&p[(size_t)s * LINE], &lines[(size_t)s * LINE], REC) != 0;
    }
    ok = !bad_out && !bad_line && nonzero_rec == (long)B * steps;
    std::printf("RESULT\tkda_core_ab\tB=%d\tsteps=%d\tout_diff=%ld\tline_diff=%ld\tslots_moved=%ld/%ld\t%s\n", B, steps,
                bad_out, bad_line, nonzero_rec, (long)B * steps, ok ? "PASS" : "FAIL");
  }

  if (!pools) pools = (int)std::max<long long>(2, (1LL << 30) / ((long long)B * LINE));
  {
    auto partial = f32((size_t)B * LDS, 0.f, 1.f);
    CU(cuMemcpyHtoD(dp, partial.data(), partial.size() * 4));
  }
  std::vector<CUdeviceptr> pl(pools), po(pools);
  std::vector<std::vector<void*>> pa;
  for (int i = 0; i < pools; ++i) {
    pl[i] = put(lines.data(), lines.size());
    CU(cuMemAlloc(&po[i], nout));
  }
  for (int i = 0; i < pools; ++i) pa.push_back(args(pl[i], po[i]));
  CUevent e0, e1;
  CU(cuEventCreate(&e0, CU_EVENT_DEFAULT));
  CU(cuEventCreate(&e1, CU_EVENT_DEFAULT));
  const double state = (double)B * REC * 2;
  CUdevice dv;
  CU(cuCtxGetDevice(&dv));
  int sms;
  CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dv));
  const unsigned long long pre_ns = 5000;
  // mode 0: back to back; 1: each after `pre`; 2: `pre` alone
  auto round = [&](const Build& f, int mode) {
    for (int i = 0; i < pools; ++i) {
      if (mode) pre<<<sms, 128, 0, g_stream>>>(pre_ns);
      if (mode < 2) f.launch(B, pa[i].data(), mode == 1);
    }
  };
  auto time = [&](const Build& f, int mode) {
    round(f, mode);
    CU(cuCtxSynchronize());
    std::vector<double> us;
    for (int r = 0; r < reps; ++r) {
      CU(cuEventRecord(e0, g_stream));
      round(f, mode);
      CU(cuEventRecord(e1, g_stream));
      CU(cuEventSynchronize(e1));
      float ms;
      CU(cuEventElapsedTime(&ms, e0, e1));
      us.push_back(ms * 1e3 / pools);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
  };
  const double alone = pdl ? time(fb, 2) : 0.0;
  auto report = [&](const char* name, const Build& f, double ref) {
    const double t = time(f, 0);
    std::printf("TIME\t%s\tB=%d\t%.2f us\tstate %.2f TB/s", name, B, t, state / (t * 1e-6) / 1e12);
    if (ref > 0) std::printf("\t%.3fx", ref / t);
    double e = 0;
    if (pdl) {
      e = time(f, 1) - alone;
      std::printf("\tpdl %.2f us\tstate %.2f TB/s", e, state / (e * 1e-6) / 1e12);
    }
    std::printf("\n");
    return std::make_pair(t, e);
  };
  const auto tb = report("base", fb, 0);
  if (ab) report("variant", fv, tb.first);
  return ok ? 0 : 1;
}
