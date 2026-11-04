// kda_conv_layout.cu: `k3_kda_core`'s CONV_TAP_MAJOR + LINE_GUARD build
// against the build it varies (the KDA_CONV line layout), on one GPU, bit for
// bit. Both run the same random rows over the same states, the base with rec
// and the window [stream][tap][c] in one line, the variant with rec in its own
// state and the window [tap][stream][c] in another, each indexed by its own
// table; lines are a random permutation of the slots. The variant also runs a
// padded row (slot -1), whose output must be 0 and which must leave every
// slot as it was: every other row's output, rec and window equal the base's
// exactly, and the one slot nobody names is untouched.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a kda_conv_layout.cu -o kda_conv_layout -lcuda
//   ./kda_conv_layout --base <k3_kda_core+HEADS=12+KDA_CONV=1+...cubin> \
//                     --variant <k3_kda_core+CONV_TAP_MAJOR=1+...+LINE_GUARD=1+...cubin> --B 64 [--smem 65536]
//
// The model constants are the TP8 rank's (12 heads, the fused 6400-column
// partial). Exit 0 PASS, 1 FAIL, 2 harness or driver error.
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

static const int HEADS = 12, KD = 128, INNER = HEADS * KD, LDS = 6400, TAPS = 3, STREAMS = 3;
static const long long REC = (long long)HEADS * KD * KD * 4, WIN = (long long)TAPS * STREAMS * INNER * 2;
static const long long LINE = REC + WIN;

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

static CUfunction entry(const char* cubin) {
  CUmodule m;
  CU(cuModuleLoad(&m, cubin));
  CUfunction f;
  CU(cuModuleGetFunction(&f, m, "kern_k3_kda_core"));
  return f;
}

// Element (stream s, tap t, column c) of a window, in the base's and the variant's layout.
static size_t at_line(int s, int t, int c) { return (size_t)s * TAPS * INNER + (size_t)t * INNER + c; }
static size_t at_taps(int s, int t, int c) { return (size_t)t * STREAMS * INNER + (size_t)s * INNER + c; }

int main(int argc, char** argv) {
  std::string base, variant;
  int B = 64;
  unsigned seed = 1234;
  int smem = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--base") base = argv[i + 1];
    else if (a == "--variant") variant = argv[i + 1];
    else if (a == "--B") B = std::atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)std::atoi(argv[i + 1]);
    else if (a == "--smem") smem = std::atoi(argv[i + 1]);  // dynamic smem of both launches (KDA_SMEM_STATE builds)
    else { std::fprintf(stderr, "unknown option `%s`\n", argv[i]); return 2; }
  }
  if (base.empty() || variant.empty() || B < 2) {
    std::fprintf(stderr, "usage: kda_conv_layout --base <cubin> --variant <cubin> [--B N >= 2] [--seed N]\n");
    return 2;
  }
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CUfunction fb = entry(base.c_str()), fv = entry(variant.c_str());

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
  // Row 1 is the padded row; slots: B rows' worth plus one nobody names. The base runs row 1 on that
  // spare slot, so its own states are only read where the variant's must stay put.
  const int slots = B + 1, pad = 1;
  std::vector<int> perm(slots);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int> base_index(perm.begin(), perm.begin() + B), var_index = base_index;
  var_index[pad] = -1;
  const int spare = perm[pad];

  auto partial = f32((size_t)B * LDS, 0.f, 1.f), cw = f32((size_t)STREAMS * 4 * INNER, 0.f, 0.5f);
  auto w_f_b = b16((size_t)INNER * KD, 0.044f);
  auto dt_bias = f32(INNER, 0.f, 1.f), a_log = f32(HEADS, 0.f, 0.5f), gamma_o = f32(KD, 1.f, 0.1f);
  auto rec = f32((size_t)slots * REC / 4, 0.f, 0.1f);
  auto win = b16((size_t)slots * WIN / 2, 1.f);  // in the variant's layout

  std::vector<uint8_t> lines((size_t)slots * LINE);
  for (int s = 0; s < slots; ++s) {
    std::memcpy(&lines[(size_t)s * LINE], (uint8_t*)rec.data() + (size_t)s * REC, REC);
    bf16* w = (bf16*)&lines[(size_t)s * LINE + REC];
    const bf16* t = win.data() + (size_t)s * WIN / 2;
    for (int st = 0; st < STREAMS; ++st)
      for (int tp = 0; tp < TAPS; ++tp)
        for (int c = 0; c < INNER; ++c) w[at_line(st, tp, c)] = t[at_taps(st, tp, c)];
  }

  CUdeviceptr dp = put(partial.data(), partial.size() * 4), dcw = put(cw.data(), cw.size() * 4);
  CUdeviceptr dwsm = dp + (CUdeviceptr)4 * INNER * 4;
  CUdeviceptr dwfb = put(w_f_b.data(), w_f_b.size() * 2), ddt = put(dt_bias.data(), dt_bias.size() * 4);
  CUdeviceptr dal = put(a_log.data(), a_log.size() * 4), dgo = put(gamma_o.data(), gamma_o.size() * 4);
  CUdeviceptr dlines = put(lines.data(), lines.size());
  CUdeviceptr drec = put(rec.data(), rec.size() * 4), dwin = put(win.data(), win.size() * 2);
  CUdeviceptr dbi = put(base_index.data(), B * 4), dvi = put(var_index.data(), B * 4);
  std::vector<bf16> poison((size_t)B * INNER, 0x7fc0);
  CUdeviceptr dob = put(poison.data(), poison.size() * 2), dov = put(poison.data(), poison.size() * 2);
  int zero = 0, span = 0, b_ = B;
  CUdeviceptr dat = put(&zero, 4);
  long long lb = LINE, rb = REC, wb = WIN;

  void* ab[] = {&dp, &dcw, &dwsm, &dwfb, &ddt, &dal, &dgo, &dlines, &dbi, &lb, &dob, &b_, &dat, &span};
  void* av[] = {&dp, &dcw, &dwsm, &dwfb, &ddt, &dal, &dgo, &drec, &dvi, &rb, &dwin, &dvi, &wb, &dov, &b_, &dat, &span};
  if (smem) {
    CU(cuFuncSetAttribute(fb, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    CU(cuFuncSetAttribute(fv, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  }
  CU(cuLaunchKernel(fb, B, HEADS, 1, 128, 1, 1, smem, 0, ab, nullptr));
  CU(cuLaunchKernel(fv, B, HEADS, 1, 128, 1, 1, smem, 0, av, nullptr));
  CU(cuCtxSynchronize());

  auto ob = get(dob, poison.size() * 2), ov = get(dov, poison.size() * 2);
  auto lb_ = get(dlines, lines.size()), rv = get(drec, rec.size() * 4), wv = get(dwin, win.size() * 2);
  long bad_out = 0, bad_pad = 0, bad_rec = 0, bad_win = 0;
  for (int b = 0; b < B; ++b) {
    const bf16* x = (const bf16*)ob.data() + (size_t)b * INNER;
    const bf16* y = (const bf16*)ov.data() + (size_t)b * INNER;
    for (int c = 0; c < INNER; ++c) {
      if (b == pad) bad_pad += y[c] != 0;
      else bad_out += x[c] != y[c];
    }
  }
  for (int s = 0; s < slots; ++s) {
    // The spare slot: the base ran the padded row on it, the variant must not have touched it.
    const uint8_t* want_rec = s == spare ? (uint8_t*)rec.data() + (size_t)s * REC : &lb_[(size_t)s * LINE];
    bad_rec += std::memcmp(want_rec, &rv[(size_t)s * REC], REC) != 0;
    const bf16* got = (const bf16*)wv.data() + (size_t)s * WIN / 2;
    const bf16* line = (const bf16*)&lb_[(size_t)s * LINE + REC];
    const bf16* init = win.data() + (size_t)s * WIN / 2;
    for (int st = 0; st < STREAMS; ++st)
      for (int tp = 0; tp < TAPS; ++tp)
        for (int c = 0; c < INNER; ++c)
          bad_win += got[at_taps(st, tp, c)] != (s == spare ? init[at_taps(st, tp, c)] : line[at_line(st, tp, c)]);
  }
  const bool ok = !bad_out && !bad_pad && !bad_rec && !bad_win;
  std::printf("RESULT\tkda_conv_layout\tB=%d\tout_diff=%ld\tpad_nonzero=%ld\trec_slots_diff=%ld\twin_diff=%ld\t%s\n",
              B, bad_out, bad_pad, bad_rec, bad_win, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
