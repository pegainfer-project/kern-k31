// d1.cu: acceptance harness for the kernels a full-K3 decode at TP8 / EP8 /
// DCP8 adds beside kern's K3 set (harness/k3 covers the
// HEADS / EXPERTS variants of the existing families, see run.sh). Same
// contract as that harness: random inputs from --seed, a CPU reference in
// the kernel's own buffer layout, every output poisoned before the launch,
// tolerance |err| <= 3 bf16 ULP(|ref|) + 1e-3 and relative RMS <= 2e-3,
// integer tables exact; then a median time.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a d1.cu -o d1 -lcuda
//
//   ./d1 --kernel vup_gate --cubin k3_mla_vup_gate+HEADS=12.cubin --heads 12 --B 64
//   ./d1 --kernel dcp      --cubin k3_dcp.cubin --B 48 --ctx 515
//   ./d1 --kernel routing  --cubin flashinfer_moe_routing.cubin --cubin2 k3_moe_prefill.cubin
//                          --B 48 --rank 3 --tile 16
//   ./d1 --kernel dcp --selftest      (CPU only: the DCP merge reference against unsharded attention)
//
// dcp runs all NRANKS = 8 members on one GPU, one after another: each member's
// partials (the attention over its positions p % 8 == member, bf16 o and f32
// natural-log lse, an empty member's o NaN and lse +inf as the DSL decode
// leaves them) go through fixup and pack, the host plays the all-to-all, and
// every member's combine is checked against the LSE merge of the same bf16
// partials and against the unsharded attention. A second pass skips the fixup
// to show the combine alone keeps empty partials out.
//
// routing runs FlashInfer's three precomputed-top-k routing kernels for the
// rank's 112 of 896 experts — a contiguous slice (mLocalExpertsStartIdx =
// 112 * rank) or, with --stride 8, every 8th from the rank's index (start =
// rank, mLocalExpertsStrideLog2 = 3, the DCP decode's deal) — and kern's
// finalize over the tables: counts, CTA tables and totals exact against the
// host model, every expanded id of a foreign expert -1, every local one a
// distinct row inside its expert's tile-padded segment that routes back to its
// token; the finalize against the weighted sum of the rows it names.
//
// Exit 0 PASS, 1 FAIL, 2 harness or driver error.
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <vector>

typedef uint16_t bf16;

static inline float b2f(bf16 h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static inline bf16 f2b(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline double bf16_ulp(double x) {
  double a = std::fabs(x);
  if (!(a > 1.17549435e-38)) return 9.183549616e-41;
  int e;
  std::frexp(a, &e);
  return std::ldexp(1.0, (e - 1) - 7);
}
static inline double sigmoidd(double x) { return 1.0 / (1.0 + std::exp(-x)); }

template <class F>
static void parallel_for(int n, F f) {
  int nt = std::max(1, std::min(n, (int)std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (int i = t; i < n; i += nt) f(i);
    });
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ driver
static void cu_check(CUresult r, const char* what, int line) {
  if (r == CUDA_SUCCESS) return;
  const char *n = nullptr, *s = nullptr;
  cuGetErrorName(r, &n);
  cuGetErrorString(r, &s);
  std::fprintf(stderr, "CUDA error at line %d: %s -> %s (%s)\n", line, what, n ? n : "?", s ? s : "?");
  std::exit(2);
}
#define CU(x) cu_check((x), #x, __LINE__)

static CUdeviceptr dmalloc(size_t bytes) {
  CUdeviceptr p = 0;
  CU(cuMemAlloc(&p, bytes ? bytes : 1));
  return p;
}
template <class T>
static CUdeviceptr dput(const std::vector<T>& h) {
  CUdeviceptr p = dmalloc(h.size() * sizeof(T));
  if (!h.empty()) CU(cuMemcpyHtoD(p, h.data(), h.size() * sizeof(T)));
  return p;
}
static CUdeviceptr dpoison(size_t bytes) {
  CUdeviceptr p = dmalloc(bytes);
  CU(cuMemsetD8(p, 0x5A, bytes ? bytes : 1));
  return p;
}
template <class T>
static std::vector<T> dget(CUdeviceptr p, size_t n) {
  std::vector<T> h(n);
  if (n) CU(cuMemcpyDtoH(h.data(), p, n * sizeof(T)));
  return h;
}
static CUfunction getfn(CUmodule m, const char* name) {
  CUfunction f = nullptr;
  if (cuModuleGetFunction(&f, m, name) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cubin has no entry `%s`\n", name);
    std::exit(2);
  }
  return f;
}
static void launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, void** args) {
  CU(cuLaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, 0, args, nullptr));
  CU(cuCtxSynchronize());
}

// --------------------------------------------------------------------- rng
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double u01() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    double u1 = std::max(u01(), 1e-12), u2 = u01();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};
static std::vector<bf16> rand_bf16(size_t n, Rng& r, double sd) {
  std::vector<bf16> v(n);
  for (auto& x : v) x = f2b((float)(sd * r.normal()));
  return v;
}
static std::vector<float> rand_f32(size_t n, Rng& r, double mu, double sd) {
  std::vector<float> v(n);
  for (auto& x : v) x = (float)(mu + sd * r.normal());
  return v;
}

// ---------------------------------------------------------------- compare
static bool g_fail = false;

// Against a reference the kernel's inputs already differ from by their own
// rounding, only a relative RMS bound applies: the bf16 partials and the bf16
// output are two roundings of at most 2^-9 each, hence 4e-3 against unsharded
// attention, 2e-3 for the CPU merge alone.
template <class G, class R>
static void cmp_rms(const char* name, size_t n, G got, R ref, double bound) {
  double max_abs = 0, se = 0, sr = 0;
  for (size_t i = 0; i < n; ++i) {
    double e = std::fabs(got(i) - ref(i));
    max_abs = std::max(max_abs, e);
    se += e * e;
    sr += ref(i) * ref(i);
  }
  double rms = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se / (double)(n ? n : 1));
  bool ok = rms <= bound;
  g_fail |= !ok;
  std::printf("  %-18s n=%-10zu max|err|=%.3e  (relRMS only) relRMS=%.3e  %s\n", name, n, max_abs, rms,
              ok ? "PASS" : "FAIL");
}

template <class G, class R>
static void cmp_gen(const char* name, size_t n, G got, R ref) {
  double max_abs = 0, max_ulp = 0, se = 0, sr = 0, bg = 0, br = 0;
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < n; ++i) {
    double g = got(i), rv = ref(i), e = std::fabs(g - rv);
    bool same_inf = std::isinf(g) && std::isinf(rv) && (g > 0) == (rv > 0);
    if (same_inf) continue;
    if (!(e <= 3.0 * bf16_ulp(rv) + 1e-3)) {
      if (!bad++) first = i, bg = g, br = rv;
      continue;
    }
    max_abs = std::max(max_abs, e);
    max_ulp = std::max(max_ulp, e / bf16_ulp(rv));
    se += e * e;
    sr += rv * rv;
  }
  double rms = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se / (double)(n ? n : 1));
  bool ok = bad == 0 && rms <= 2e-3;
  g_fail |= !ok;
  std::printf("  %-18s n=%-10zu max|err|=%.3e  maxULP=%6.2f  relRMS=%.3e  %s\n", name, n, max_abs, max_ulp, rms,
              ok ? "PASS" : "FAIL");
  if (bad) std::printf("      %zu out of tolerance; first at %zu: got %.9g ref %.9g\n", bad, first, bg, br);
}
static void cmp_bf16(const char* name, const std::vector<bf16>& got, const std::vector<double>& ref) {
  cmp_gen(name, ref.size(), [&](size_t i) { return (double)b2f(got[i]); }, [&](size_t i) { return ref[i]; });
}
template <class T>
static void cmp_exact(const char* name, const std::vector<T>& got, const std::vector<T>& ref) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::memcmp(&got[i], &ref[i], sizeof(T))) {
      if (!bad++) first = i;
    }
  g_fail |= bad != 0;
  std::printf("  %-18s n=%-10zu exact                                          %s\n", name, ref.size(),
              bad ? "FAIL" : "PASS");
  if (bad) std::printf("      %zu mismatches, first at %zu\n", bad, first);
}
static void verdict(const char* name, bool ok, const std::string& why = "") {
  g_fail |= !ok;
  std::printf("  %-18s %s%s%s\n", name, ok ? "PASS" : "FAIL", why.empty() ? "" : "  ", why.c_str());
}

static double time_us(int reps, const std::function<void()>& body) {
  for (int i = 0; i < 5; ++i) body();
  CU(cuCtxSynchronize());
  CUevent a, b;
  CU(cuEventCreate(&a, CU_EVENT_DEFAULT));
  CU(cuEventCreate(&b, CU_EVENT_DEFAULT));
  std::vector<double> us;
  for (int i = 0; i < reps; ++i) {
    CU(cuEventRecord(a, 0));
    body();
    CU(cuEventRecord(b, 0));
    CU(cuEventSynchronize(b));
    float ms = 0;
    CU(cuEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1000.0);
  }
  std::sort(us.begin(), us.end());
  CU(cuEventDestroy(a));
  CU(cuEventDestroy(b));
  return us[us.size() / 2];
}

struct Opt {
  std::string kernel, cubin, cubin2;
  int B = 8, heads = 12, ctx = 515, rank = 3, tile = 16, stride = 1, reps = 50;
  uint64_t seed = 1234;
  bool selftest = false;
};

// ======================================================================
// vup_gate: o_lat [B, HEADS, 512] through W_UV (rows h*256+128..256 of
// w_kv_b [HEADS*256, 512]) meets bf16(sigmoid(gate)).
// ======================================================================
static int run_vup_gate(const Opt& o, CUmodule m) {
  const int B = o.B, Hh = o.heads, LAT = 512, NOPE = 128;
  Rng r(o.seed);
  auto lat = rand_bf16((size_t)B * Hh * LAT, r, 1.0);
  auto w = rand_bf16((size_t)Hh * 256 * LAT, r, 0.044);
  auto gate = rand_bf16((size_t)B * Hh * NOPE, r, 1.0);
  std::vector<double> ref((size_t)B * Hh * NOPE);
  parallel_for(B * Hh, [&](int bh) {
    int b = bh / Hh, h = bh % Hh;
    for (int dv = 0; dv < NOPE; ++dv) {
      double acc = 0;
      const bf16* wr = &w[((size_t)h * 256 + NOPE + dv) * LAT];
      const bf16* l = &lat[((size_t)b * Hh + h) * LAT];
      for (int j = 0; j < LAT; ++j) acc += (double)b2f(wr[j]) * b2f(l[j]);
      size_t i = (size_t)b * Hh * NOPE + (size_t)h * NOPE + dv;
      bf16 g = f2b((float)sigmoidd(b2f(gate[i])));
      ref[i] = b2f(f2b(b2f(f2b((float)acc)) * b2f(g)));
    }
  });
  CUdeviceptr dl = dput(lat), dw = dput(w), dg = dput(gate), dout = dpoison(ref.size() * 2);
  CUfunction f = getfn(m, "kern_k3_mla_vup_gate");
  int Bv = B;
  void* args[] = {&dl, &dw, &dg, &dout, &Bv};
  unsigned gx = (B + 31) / 32;
  std::printf("  launch             grid(%u,%d,4) block 256\n", gx, Hh);
  launch(f, gx, Hh, 4, 256, args);
  cmp_bf16("gated", dget<bf16>(dout, ref.size()), ref);
  double t = time_us(o.reps, [&] { CU(cuLaunchKernel(f, gx, Hh, 4, 256, 1, 1, 0, 0, args, nullptr)); });
  double bytes = (double)Hh * 128 * LAT * 2 + (double)B * Hh * (LAT + 2 * NOPE) * 2;
  std::printf("  TIME               median %8.2f us   %.2f MB   %.1f GB/s\n", t, bytes / 1e6, bytes / t / 1e3);
  return g_fail;
}

// ======================================================================
// absorb: q_abs [B, 96, 576] = bf16(q_h) . W_UK_h (rows h*256+0..128 of
// w_kv_b) | bf16(q_h rope), q_h from the f32 partial. --cubin2 times the
// scalar kernel (GROUPS=1, its decode geometry) beside it.
// ======================================================================
static int run_absorb(const Opt& o, CUmodule m, CUmodule old) {
  const int B = o.B, Hh = 96, LAT = 512, NOPE = 128, QW = 192, ROW = 576;
  Rng r(o.seed);
  auto q = rand_f32((size_t)B * Hh * QW, r, 0.0, 1.0);
  auto w = rand_bf16((size_t)Hh * 256 * LAT, r, 0.044);
  std::vector<double> ref((size_t)B * Hh * ROW);
  parallel_for(B * Hh, [&](int bh) {
    int b = bh / Hh, h = bh % Hh;
    const float* qh = &q[((size_t)b * Hh + h) * QW];
    for (int j = 0; j < LAT; ++j) {
      double acc = 0;
      for (int d = 0; d < NOPE; ++d) acc += (double)b2f(f2b(qh[d])) * b2f(w[((size_t)h * 256 + d) * LAT + j]);
      ref[(size_t)bh * ROW + j] = b2f(f2b((float)acc));
    }
    for (int k = 0; k < QW - NOPE; ++k) ref[(size_t)bh * ROW + LAT + k] = b2f(f2b(qh[NOPE + k]));
  });
  CUdeviceptr dq = dput(q), dw = dput(w), dout = dpoison(ref.size() * 2);
  CUfunction f = getfn(m, "kern_k3_mla_absorb_mma");
  int Bv = B;
  void* args[] = {&dq, &dw, &dout, &Bv};
  std::printf("  launch             grid(1,96,8) block 128\n");
  launch(f, 1, Hh, 8, 128, args);
  cmp_bf16("q_abs", dget<bf16>(dout, ref.size()), ref);
  double t = time_us(o.reps, [&] { CU(cuLaunchKernel(f, 1, Hh, 8, 128, 1, 1, 0, 0, args, nullptr)); });
  double bytes = (double)Hh * NOPE * LAT * 2 + (double)B * Hh * (QW * 4 + ROW * 2);
  std::printf("  TIME               median %8.2f us   %.2f MB   %.1f GB/s\n", t, bytes / 1e6, bytes / t / 1e3);
  if (old) {
    CUfunction g = getfn(old, "kern_k3_mla_absorb");
    unsigned gx = (B + 7) / 8;
    double t0 = time_us(o.reps, [&] { CU(cuLaunchKernel(g, gx, Hh, 8, 128, 1, 1, 0, 0, args, nullptr)); });
    std::printf("  TIME scalar        median %8.2f us   grid(%u,96,8)\n", t0, gx);
  }
  return g_fail;
}

// ======================================================================
// dcp: fixup / pack / combine over NRANKS members.
// ======================================================================
namespace dcp {
const int HEADS = 96, NRANKS = 8, HL = HEADS / NRANKS, LAT = 512;
inline size_t chunk(int R) { return (size_t)R * HL * (LAT + 2); }

struct Case {
  int B;
  std::vector<int> len;            // [B] the row's positions
  std::vector<bf16> q;             // [B, HEADS, LAT]
  std::vector<bf16> kv;            // [B, ctx_max, LAT]; the latent is K and V
  int ctx_max;
};

// The row lengths cover every member state: empty rows, rows shorter than
// NRANKS (some members empty), lengths around multiples of NRANKS, and long.
Case make(int B, int ctx, uint64_t seed) {
  Rng r(seed);
  Case c;
  c.B = B;
  const int pick[] = {0, 1, 3, 7, 8, 9, 16, 17};
  for (int b = 0; b < B; ++b)
    c.len.push_back(b == 0 ? ctx : (b % 3 == 1 ? pick[r.below(8)] : 1 + (int)r.below((uint32_t)ctx)));
  c.ctx_max = std::max(1, *std::max_element(c.len.begin(), c.len.end()));
  c.q = rand_bf16((size_t)B * HEADS * LAT, r, 1.0);
  c.kv = rand_bf16((size_t)B * c.ctx_max * LAT, r, 1.0);
  return c;
}

const double SCALE = 0.07216878364870322;  // 192^-0.5

// Attention of (b, h) over the positions p with p % stride == member
// (stride 1: all of them): o in double and the natural-log lse; count == 0
// leaves o = 0, lse = -inf.
int attend(const Case& c, int b, int h, int member, int stride, double* o, double* lse) {
  const bf16* q = &c.q[((size_t)b * HEADS + h) * LAT];
  std::vector<double> s;
  std::vector<int> pos;
  double mx = -INFINITY;
  for (int p = member; p < c.len[b]; p += stride) {
    const bf16* k = &c.kv[((size_t)b * c.ctx_max + p) * LAT];
    double d = 0;
    for (int j = 0; j < LAT; ++j) d += (double)b2f(q[j]) * b2f(k[j]);
    s.push_back(d * SCALE);
    pos.push_back(p);
    mx = std::max(mx, s.back());
  }
  std::fill(o, o + LAT, 0.0);
  if (pos.empty()) {
    *lse = -INFINITY;
    return 0;
  }
  double sum = 0;
  for (size_t i = 0; i < pos.size(); ++i) {
    double w = std::exp(s[i] - mx);
    sum += w;
    const bf16* v = &c.kv[((size_t)b * c.ctx_max + pos[i]) * LAT];
    for (int j = 0; j < LAT; ++j) o[j] += w * b2f(v[j]);
  }
  for (int j = 0; j < LAT; ++j) o[j] /= sum;
  *lse = mx + std::log(sum);
  return (int)pos.size();
}

struct Partials {
  std::vector<bf16> o;     // [B, HEADS, LAT]  bf16 as the decode writes it
  std::vector<float> lse;  // [B, HEADS]
  std::vector<int> len;    // [B]  this member's positions
};

// Member q's decode output; an empty member's rows are what the DSL decode
// leaves (o NaN, lse +inf), for the fixup and the combine to deal with.
Partials member(const Case& c, int q) {
  Partials m;
  m.o.resize((size_t)c.B * HEADS * LAT);
  m.lse.resize((size_t)c.B * HEADS);
  m.len.resize(c.B);
  parallel_for(c.B * HEADS, [&](int bh) {
    int b = bh / HEADS, h = bh % HEADS;
    std::vector<double> o(LAT);
    double lse;
    int n = attend(c, b, h, q, NRANKS, o.data(), &lse);
    if (h == 0) m.len[b] = n;
    for (int j = 0; j < LAT; ++j) m.o[(size_t)bh * LAT + j] = n ? f2b((float)o[j]) : (bf16)0x7fc0;
    m.lse[bh] = n ? (float)lse : INFINITY;
  });
  return m;
}

// The LSE merge of the members' (bf16, f32) partials for member `me`'s heads.
std::vector<double> merge(const std::vector<Partials>& ps, int me, int B) {
  std::vector<double> out((size_t)B * HL * LAT, 0.0);
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < HL; ++j) {
      int h = me * HL + j;
      double mx = -INFINITY;
      for (auto& p : ps)
        if (p.len[b]) mx = std::max(mx, (double)p.lse[(size_t)b * HEADS + h]);
      if (mx == -INFINITY) continue;
      double den = 0;
      double* o = &out[((size_t)b * HL + j) * LAT];
      for (auto& p : ps) {
        if (!p.len[b]) continue;
        double w = std::exp((double)p.lse[(size_t)b * HEADS + h] - mx);
        den += w;
        for (int k = 0; k < LAT; ++k) o[k] += w * b2f(p.o[((size_t)b * HEADS + h) * LAT + k]);
      }
      for (int k = 0; k < LAT; ++k) o[k] /= den;
    }
  return out;
}

std::vector<double> whole(const Case& c, int me) {
  std::vector<double> out((size_t)c.B * HL * LAT);
  parallel_for(c.B * HL, [&](int bj) {
    int b = bj / HL, j = bj % HL;
    double lse;
    attend(c, b, me * HL + j, 0, 1, &out[(size_t)bj * LAT], &lse);
  });
  return out;
}

// The pack a member's (fixed-up) partials make, built on the host.
std::vector<bf16> pack(const Partials& p, int B) {
  std::vector<bf16> s(NRANKS * chunk(B));
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < HEADS; ++h) {
      int q = h / HL, j = h % HL;
      std::memcpy(&s[q * chunk(B) + ((size_t)b * HL + j) * LAT], &p.o[((size_t)b * HEADS + h) * LAT], LAT * 2);
      std::memcpy(&s[q * chunk(B) + (size_t)B * HL * LAT + 2 * ((size_t)b * HL + j)], &p.lse[(size_t)b * HEADS + h],
                  4);
    }
  return s;
}

Partials fixed(Partials p) {
  for (size_t b = 0; b < p.len.size(); ++b)
    if (!p.len[b]) {
      std::fill(p.o.begin() + b * HEADS * LAT, p.o.begin() + (b + 1) * HEADS * LAT, (bf16)0);
      std::fill(p.lse.begin() + b * HEADS, p.lse.begin() + (b + 1) * HEADS, -INFINITY);
    }
  return p;
}

int selftest(const Opt& o) {
  Case c = make(o.B, o.ctx, o.seed);
  std::vector<Partials> ps;
  for (int q = 0; q < NRANKS; ++q) ps.push_back(member(c, q));
  for (int me = 0; me < NRANKS; me += NRANKS - 1)
    cmp_rms(me ? "merge vs whole m7" : "merge vs whole m0", (size_t)c.B * HL * LAT,
            [&, m = merge(ps, me, c.B)](size_t i) { return m[i]; },
            [&, w = whole(c, me)](size_t i) { return w[i]; }, 2e-3);
  return g_fail;
}

int run(const Opt& o, CUmodule m) {
  const int B = o.B;
  Case c = make(B, o.ctx, o.seed);
  std::printf("  rows               B=%d ctx=%d, lengths", B, o.ctx);
  for (int b = 0; b < std::min(B, 12); ++b) std::printf(" %d", c.len[b]);
  std::printf("%s\n", B > 12 ? " ..." : "");
  std::vector<Partials> ps;
  for (int q = 0; q < NRANKS; ++q) ps.push_back(member(c, q));
  CUfunction ffix = getfn(m, "kern_k3_dcp_fixup"), fpack = getfn(m, "kern_k3_dcp_pack"),
             fcomb = getfn(m, "kern_k3_dcp_combine");
  int R = B;
  for (int fix = 1; fix >= 0; --fix) {
    std::printf(" %s\n", fix ? "fixup -> pack -> all-to-all -> combine" : "pack -> all-to-all -> combine (no fixup)");
    std::vector<std::vector<bf16>> sends;
    for (int q = 0; q < NRANKS; ++q) {
      CUdeviceptr dlo = dput(ps[q].o), dlse = dput(ps[q].lse), dlen = dput(ps[q].len);
      if (fix) {
        void* a[] = {&dlo, &dlse, &dlen, &R};
        launch(ffix, B, 1, 1, 256, a);
        Partials want = fixed(ps[q]);
        if (q == 0 || q == NRANKS - 1) {
          cmp_exact(q ? "fixup o m7" : "fixup o m0", dget<bf16>(dlo, want.o.size()), want.o);
          cmp_exact(q ? "fixup lse m7" : "fixup lse m0", dget<float>(dlse, want.lse.size()), want.lse);
        }
      }
      CUdeviceptr dsend = dpoison(NRANKS * chunk(B) * 2);
      void* a[] = {&dlo, &dlse, &dsend, &R};
      launch(fpack, B, HEADS / 4, 1, 256, a);
      sends.push_back(dget<bf16>(dsend, NRANKS * chunk(B)));
      if (fix && (q == 0 || q == NRANKS - 1))
        cmp_exact(q ? "pack m7" : "pack m0", sends.back(), pack(fixed(ps[q]), B));
      CU(cuMemFree(dlo));
      CU(cuMemFree(dlse));
      CU(cuMemFree(dlen));
      CU(cuMemFree(dsend));
    }
    for (int me = 0; me < NRANKS; ++me) {
      std::vector<bf16> recv(NRANKS * chunk(B));
      for (int q = 0; q < NRANKS; ++q)
        std::memcpy(&recv[q * chunk(B)], &sends[q][me * chunk(B)], chunk(B) * 2);
      CUdeviceptr drecv = dput(recv), dout = dpoison((size_t)B * HL * LAT * 2);
      void* a[] = {&drecv, &dout, &R};
      launch(fcomb, B, HL / 4, 1, 256, a);
      auto got = dget<bf16>(dout, (size_t)B * HL * LAT);
      char name[32];
      std::snprintf(name, sizeof name, "combine m%d", me);
      cmp_bf16(name, got, merge(ps, me, B));
      if (me == 0 || me == NRANKS - 1) {
        std::snprintf(name, sizeof name, "vs whole m%d", me);
        auto w = whole(c, me);
        cmp_rms(name, got.size(), [&](size_t i) { return (double)b2f(got[i]); }, [&](size_t i) { return w[i]; }, 4e-3);
      }
      CU(cuMemFree(drecv));
      CU(cuMemFree(dout));
    }
  }
  CUdeviceptr dlo = dput(ps[0].o), dlse = dput(ps[0].lse), dsend = dmalloc(NRANKS * chunk(B) * 2),
              dout = dmalloc((size_t)B * HL * LAT * 2);
  void* ap[] = {&dlo, &dlse, &dsend, &R};
  void* ac[] = {&dsend, &dout, &R};
  double tp = time_us(o.reps, [&] { CU(cuLaunchKernel(fpack, B, HEADS / 4, 1, 256, 1, 1, 0, 0, ap, nullptr)); });
  double tc = time_us(o.reps, [&] { CU(cuLaunchKernel(fcomb, B, HL / 4, 1, 256, 1, 1, 0, 0, ac, nullptr)); });
  double bp = 2.0 * B * HEADS * (LAT * 2 + 4), bc = (double)NRANKS * chunk(B) * 2 + (double)B * HL * LAT * 2;
  std::printf("  TIME pack          median %8.2f us   %.2f MB   %.1f GB/s\n", tp, bp / 1e6, bp / tp / 1e3);
  std::printf("  TIME combine       median %8.2f us   %.2f MB   %.1f GB/s\n", tc, bc / 1e6, bc / tc / 1e3);
  return g_fail;
}
}  // namespace dcp

// ======================================================================
// routing: FlashInfer precomputed-top-k routing for a rank's slice of the
// experts, then kern_k3_moe_finalize over its tables.
// ======================================================================
namespace routing {
const int EXPERTS = 896, LOCAL = 112, TOPK = 16, H = 3584, THREADS = 1024, PER_CTA = 8 * THREADS;
// routingPrecomputed::KernelParams<float, 1024, 16> (source/flashinfer-moe-routing/abi.json)
const int SIZE = 192;
enum {
  mIsPow2 = 1, mPtrExpertCounts = 8, mPtrPermutedIdxSize = 16, mPtrExpandedIdxToPermutedIdx = 24,
  mPtrPermutedIdxToTokenIdx = 40, mPtrCtaIdxXyToBatchIdx = 48, mPtrCtaIdxXyToMnLimit = 56,
  mPtrNumNonExitingCtas = 64, mPtrTopKIds = 80, mNumTokens = 96, mNumExperts = 100, mPaddingLog2 = 104,
  mTileTokensDim = 108, mLocalExpertsStartIdx = 112, mLocalExpertsStrideLog2 = 116, mNumLocalExperts = 120,
  mTopK = 176,
};
const int32_t TOPK_DIV[4] = {16, -2147483647, 3, 1};
const char* INIT = "_ZN3moe3dev7routing23routingInitExpertCountsINS1_18routingPrecomputed12KernelParamsIfLi1024ELi16EEEEEvT_";
const char* HIST =
    "_ZN3moe3dev7routing29routingIndicesHistogramKernelINS1_18routingPrecomputed12KernelParamsIfLi1024ELi16EEEEEvT_";
const char* OFFS =
    "_ZN3moe3dev7routing27routingIndicesOffsetsKernelINS1_18routingPrecomputed12KernelParamsIfLi1024ELi16EEEEEvT_";

// Distinct top-16 ids per token; every fourth token takes as many of the
// rank's own experts as it can (16), the rest draw uniformly over all 896.
std::vector<int> make_ids(int T, int first, int stride, Rng& r) {
  std::vector<int> ids((size_t)T * TOPK);
  for (int t = 0; t < T; ++t) {
    std::vector<char> taken(EXPERTS, 0);
    for (int k = 0; k < TOPK; ++k) {
      int e;
      do e = (t % 4 == 0) ? first + stride * (int)r.below(LOCAL) : (int)r.below(EXPERTS);
      while (taken[e]);
      taken[e] = 1;
      ids[(size_t)t * TOPK + k] = e;
    }
  }
  return ids;
}

int run(const Opt& o, CUmodule m, CUmodule glue) {
  const int T = o.B, tile = o.tile, stride = o.stride, first = stride == 1 ? LOCAL * o.rank : o.rank;
  int log2 = 0, slog2 = 0;
  while ((1 << log2) < tile) ++log2;
  while ((1 << slog2) < stride) ++slog2;
  if ((1 << log2) != tile || (1 << slog2) != stride || LOCAL * stride > EXPERTS) {
    std::fprintf(stderr, "--tile and --stride must be powers of two, --stride at most %d\n", EXPERTS / LOCAL);
    return 2;
  }
  // the rank's local index of global expert e, or -1
  auto local = [&](int e) { return e >= first && (e - first) % stride == 0 && (e - first) / stride < LOCAL ? (e - first) / stride : -1; };
  Rng r(o.seed);
  auto ids = make_ids(T, first, stride, r);
  const int expanded = T * TOPK, filled = std::min(LOCAL, expanded);
  const int ctas_max = filled + (expanded - filled + tile - 1) / tile, padded = ctas_max * tile;
  const int counts_len = std::max(512, 2 * EXPERTS);
  std::printf("  routing            T=%d rank=%d (experts %d..%d step %d) tile=%d ctas_max=%d\n", T, o.rank, first,
              first + stride * (LOCAL - 1), stride, tile, ctas_max);

  // host model: counts, tile-padded segments, CTA tables
  std::vector<int> count(LOCAL, 0), off(LOCAL + 1, 0);
  for (int e : ids)
    if (local(e) >= 0) ++count[local(e)];
  for (int e = 0; e < LOCAL; ++e) off[e + 1] = off[e] + (count[e] + tile - 1) / tile;
  const int nne = off[LOCAL];
  std::vector<int> cta_batch(nne), cta_limit(nne);
  for (int i = 0; i < nne; ++i) {
    int e = int(std::upper_bound(off.begin(), off.end(), i) - off.begin()) - 1;
    cta_batch[i] = e;
    cta_limit[i] = std::min((i + 1) * tile, off[e] * tile + count[e]);
  }

  CUdeviceptr d_ids = dput(ids), d_counts = dpoison(counts_len * 4), d_batch = dpoison(ctas_max * 4),
              d_limit = dpoison(ctas_max * 4), d_nne = dpoison(4), d_total = dpoison(4),
              d_map = dpoison((size_t)padded * 4), d_e2p = dpoison((size_t)expanded * 4);
  std::vector<uint8_t> p(SIZE, 0);
  auto put64 = [&](int at, CUdeviceptr v) { std::memcpy(&p[at], &v, 8); };
  auto put32 = [&](int at, int32_t v) { std::memcpy(&p[at], &v, 4); };
  p[mIsPow2] = 1;
  put64(mPtrExpertCounts, d_counts);
  put64(mPtrPermutedIdxSize, d_total);
  put64(mPtrExpandedIdxToPermutedIdx, d_e2p);
  put64(mPtrPermutedIdxToTokenIdx, d_map);
  put64(mPtrCtaIdxXyToBatchIdx, d_batch);
  put64(mPtrCtaIdxXyToMnLimit, d_limit);
  put64(mPtrNumNonExitingCtas, d_nne);
  put64(mPtrTopKIds, d_ids);
  put32(mNumTokens, T);
  put32(mNumExperts, EXPERTS);
  put32(mPaddingLog2, log2);
  put32(mTileTokensDim, tile);
  put32(mLocalExpertsStartIdx, first);
  put32(mLocalExpertsStrideLog2, slog2);
  put32(mNumLocalExperts, LOCAL);
  for (int j = 0; j < 4; ++j) put32(mTopK + 4 * j, TOPK_DIV[j]);
  void* args[] = {p.data()};
  const unsigned grid_x = (unsigned)((expanded + PER_CTA - 1) / PER_CTA);
  auto route = [&](bool sync) {
    CUfunction fs[] = {getfn(m, INIT), getfn(m, HIST), getfn(m, OFFS)};
    unsigned g[] = {(unsigned)((counts_len + THREADS - 1) / THREADS), grid_x, grid_x};
    for (int i = 0; i < 3; ++i) CU(cuLaunchKernel(fs[i], g[i], 1, 1, THREADS, 1, 1, 0, 0, args, nullptr));
    if (sync) CU(cuCtxSynchronize());
  };
  route(true);

  cmp_exact("num_non_exiting", dget<int>(d_nne, 1), std::vector<int>{nne});
  cmp_exact("total_padded", dget<int>(d_total, 1), std::vector<int>{nne * tile});
  cmp_exact("cta_batch", dget<int>(d_batch, nne), cta_batch);
  cmp_exact("cta_limit", dget<int>(d_limit, nne), cta_limit);
  auto e2p = dget<int>(d_e2p, expanded);
  auto map = dget<int>(d_map, padded);
  std::vector<char> used(padded, 0);
  std::string why;
  for (int t = 0; t < T && why.empty(); ++t)
    for (int k = 0; k < TOPK && why.empty(); ++k) {
      int e = local(ids[(size_t)t * TOPK + k]), row = e2p[(size_t)t * TOPK + k];
      char buf[160];
      if (e < 0 || e >= LOCAL) {
        if (row != -1) std::snprintf(buf, sizeof buf, "token %d k %d: foreign expert got row %d", t, k, row), why = buf;
      } else if (row < off[e] * tile || row >= off[e] * tile + count[e]) {
        std::snprintf(buf, sizeof buf, "token %d k %d: row %d outside expert %d's [%d, %d)", t, k, row, e,
                      off[e] * tile, off[e] * tile + count[e]);
        why = buf;
      } else if (used[row]++) {
        std::snprintf(buf, sizeof buf, "row %d given twice", row), why = buf;
      } else if (map[row] != t) {
        std::snprintf(buf, sizeof buf, "row %d routes token %d, not %d", row, map[row], t), why = buf;
      }
    }
  verdict("exp2perm/route_map", why.empty(), why);

  // finalize over the tables: out[t] = sum over local k of w * fc2[row]
  auto fc2 = rand_bf16((size_t)padded * H, r, 1.0);
  auto wts = rand_f32(expanded, r, 0.0, 0.3);
  std::vector<double> ref((size_t)T * H, 0.0);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < TOPK; ++k) {
      int row = e2p[(size_t)t * TOPK + k];
      if (row < 0) continue;
      for (int c = 0; c < H; ++c) ref[(size_t)t * H + c] += (double)wts[(size_t)t * TOPK + k] * b2f(fc2[(size_t)row * H + c]);
    }
  for (auto& x : ref) x = b2f(f2b((float)x));
  CUdeviceptr d_fc2 = dput(fc2), d_w = dput(wts), d_out = dpoison((size_t)T * H * 2);
  int Tv = T, cols = H;
  void* fa[] = {&d_fc2, &d_e2p, &d_w, &d_out, &Tv, &cols};
  launch(getfn(glue, "kern_k3_moe_finalize"), T, 1, 1, 256, fa);
  cmp_bf16("finalize", dget<bf16>(d_out, (size_t)T * H), ref);

  double t = time_us(o.reps, [&] { route(false); });
  std::printf("  TIME routing       median %8.2f us (3 launches)\n", t);
  return g_fail;
}
}  // namespace routing

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--kernel") o.kernel = next();
    else if (a == "--cubin") o.cubin = next();
    else if (a == "--cubin2") o.cubin2 = next();
    else if (a == "--B") o.B = std::stoi(next());
    else if (a == "--heads") o.heads = std::stoi(next());
    else if (a == "--ctx") o.ctx = std::stoi(next());
    else if (a == "--rank") o.rank = std::stoi(next());
    else if (a == "--tile") o.tile = std::stoi(next());
    else if (a == "--stride") o.stride = std::stoi(next());
    else if (a == "--reps") o.reps = std::stoi(next());
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--selftest") o.selftest = true;
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  std::printf("== %s B=%d\n", o.kernel.c_str(), o.B);
  if (o.selftest) {
    if (o.kernel != "dcp") return 2;
    int rc = dcp::selftest(o);
    std::printf("RESULT\t%s-selftest\tB=%d\tctx=%d\t%s\n", o.kernel.c_str(), o.B, o.ctx, rc ? "FAIL" : "PASS");
    return rc;
  }
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CUmodule m, glue = nullptr;
  CU(cuModuleLoad(&m, o.cubin.c_str()));
  if (!o.cubin2.empty()) CU(cuModuleLoad(&glue, o.cubin2.c_str()));
  int rc;
  if (o.kernel == "vup_gate") rc = run_vup_gate(o, m);
  else if (o.kernel == "absorb") rc = run_absorb(o, m, glue);
  else if (o.kernel == "dcp") rc = dcp::run(o, m);
  else if (o.kernel == "routing" && glue) rc = routing::run(o, m, glue);
  else {
    std::fprintf(stderr, "--kernel vup_gate | dcp | routing (routing also needs --cubin2 k3_moe_prefill.cubin)\n");
    return 2;
  }
  std::printf("RESULT\t%s\tB=%d\theads=%d\tctx=%d\trank=%d\ttile=%d\tstride=%d\t%s\n", o.kernel.c_str(), o.B, o.heads,
              o.ctx, o.rank, o.tile, o.stride, rc ? "FAIL" : "PASS");
  return rc ? 1 : 0;
}
