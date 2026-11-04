// p_split_layout.cu: the packed prefill's split layout on the real cubins. kern_k3_fmha_lens_varlen's
// tables, kern_k3g_mla_prep_gather's latent_g and k3_mla_absorb's (-DQBF16) q copies for single- and
// multi-sequence calls (the split sequence first, in the middle, last; unsplit calls), every latent_g
// row checked against the slab row of the (sequence, key) its FMHA piece should hold, the q copies
// against their rows.
//
//   nvcc -O2 -std=c++17 -arch=sm_103a p_split_layout.cu -o p_split_layout -lcuda
//   ./p_split_layout build/k3_prefill.cubin build/k3_mla_glue.cubin 'build/k3_mla_absorb+QBF16=1.cubin'
//   prints PASS / FAIL
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); \
  printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)

constexpr int KV_A = 576, PAGE = 64, MLA_FUSED = 14400, Q_LORA = 1536, PACK = 16, NS = PACK + 3, SHORT = 128, EXTRA = 512;
typedef unsigned short u16;

static CUfunction lens_fn, prep_fn, absorb_fn;
constexpr int QW = 96 * 192;
static int failures = 0;

static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }
static void up(CUdeviceptr d, const void* h, size_t n) { CK(cuMemcpyHtoD(d, h, n)); }
static void down(void* h, CUdeviceptr d, size_t n) { CK(cuMemcpyDtoH(h, d, n)); }

static void run(const char* name, std::vector<int> lq, std::vector<int> ctx) {
  const int n = lq.size();
  std::vector<long long> cu(n + 1, 0);
  for (int j = 0; j < n; ++j) cu[j + 1] = cu[j] + lq[j];
  const int T = cu[n];
  int total = 0, max_pages = 0;
  for (int j = 0; j < n; ++j) total += ctx[j], max_pages = std::max(max_pages, (ctx[j] + PAGE - 1) / PAGE);
  const int ctx_rows = (total + 127) / 128 * 128, rows = ctx_rows + EXTRA;
  // pages: sequence j's page p is slab page j * max_pages + p, filled with noise unique to (j, key)
  const long long page_stride = (long long)PAGE * KV_A;
  const size_t slab_elems = (size_t)n * max_pages * page_stride;
  std::vector<u16> slab(slab_elems);
  for (size_t e = 0; e < slab_elems; ++e) slab[e] = (u16)(0x3c00 + ((e * 2654435761u) >> 20) % 0x300);
  std::vector<int> table(n * max_pages);
  for (int j = 0; j < n; ++j) for (int p = 0; p < max_pages; ++p) table[j * max_pages + p] = j * max_pages + p;
  std::vector<long long> slot(T);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lq[j]; ++i) {
      const int key = ctx[j] - lq[j] + i;
      slot[cu[j] + i] = (long long)table[j * max_pages + key / PAGE] * PAGE + key % PAGE;
    }
  std::vector<u16> partial((size_t)T * MLA_FUSED);
  for (size_t e = 0; e < partial.size(); ++e) partial[e] = (u16)(0x3c00 + ((e * 40503u) >> 7) % 0x200);
  std::vector<u16> gamma(Q_LORA, 0x3f80);

  CUdeviceptr d_seq = dalloc(4 * n), d_cu = dalloc(8 * (n + 1)), d_lens = dalloc(4 * (3 * NS + 8));
  CUdeviceptr d_slab = dalloc(slab_elems * 2), d_table = dalloc(4 * table.size()), d_slot = dalloc(8 * T);
  CUdeviceptr d_part = dalloc(partial.size() * 2), d_gq = dalloc(Q_LORA * 2), d_gkv = dalloc(Q_LORA * 2);
  CUdeviceptr d_qn = dalloc((size_t)(T + 1024) * Q_LORA * 2), d_lat = dalloc((size_t)rows * KV_A * 2);
  CUdeviceptr d_rt = dalloc(4 * SHORT * max_pages), d_rl = dalloc(4 * SHORT), d_bsk = dalloc(4 * SHORT);
  std::vector<u16> q((size_t)(T + 1024) * QW);
  for (size_t e = 0; e < (size_t)T * QW; ++e) q[e] = (u16)(0x3c00 + ((e * 2246822519u) >> 19) % 0x300);
  CUdeviceptr d_q = dalloc(q.size() * 2), d_w = dalloc((size_t)96 * 256 * 512 * 2), d_qabs = dalloc((size_t)T * 96 * 576 * 2);
  up(d_q, q.data(), q.size() * 2);
  up(d_seq, ctx.data(), 4 * n); up(d_cu, cu.data(), 8 * (n + 1)); up(d_slab, slab.data(), slab_elems * 2);
  up(d_table, table.data(), 4 * table.size()); up(d_slot, slot.data(), 8 * T); up(d_part, partial.data(), partial.size() * 2);
  up(d_gq, gamma.data(), Q_LORA * 2); up(d_gkv, gamma.data(), Q_LORA * 2);
  CK(cuMemsetD8(d_lat, 0xff, (size_t)rows * KV_A * 2));  // stale rows must be overwritten or zeroed

  int ns = NS, sm = SHORT, nn = n;
  void* la[] = {&d_seq, &d_cu, &d_lens, &nn, &ns, &sm};
  CK(cuLaunchKernel(lens_fn, 1, 1, 1, 32, 1, 1, 0, 0, la, nullptr));
  int split_max = 16, B = T, nrows = rows;
  void* pa[] = {&d_part, &d_gq, &d_gkv, &d_slot, &d_slab, (void*)&page_stride, &d_qn, &d_table, &max_pages, &d_lens, &ns,
                &nn, &d_lat, &nrows, &d_rt, &d_rl, &d_bsk, &split_max, &sm, &B};
  CK(cuLaunchKernel(prep_fn, 2 * T + 1 + (rows + 27) / 28, 1, 1, 512, 1, 1, 0, 0, pa, nullptr));
  if (T <= 512) {
    void* aa[] = {&d_q, &d_w, &d_qabs, &B, &d_lens, &ns, &sm};
    CK(cuLaunchKernel(absorb_fn, (T + 31) / 32, 96, 8, 128, 1, 1, 0, 0, aa, nullptr));
  }
  CK(cuCtxSynchronize());

  std::vector<int> L(3 * NS + 8);
  down(L.data(), d_lens, 4 * L.size());
  down(slab.data(), d_slab, slab_elems * 2);  // now holds the chunk rows the heads appended
  std::vector<u16> lat((size_t)rows * KV_A), qn((size_t)(T + 1024) * Q_LORA);
  down(lat.data(), d_lat, lat.size() * 2); down(qn.data(), d_qn, qn.size() * 2); down(q.data(), d_q, q.size() * 2);
  const int* kl = L.data(); const int* cq = kl + NS; const int* ck = kl + 2 * NS; const int* rec = kl + 3 * NS;
  const int s = rec[3] > 0 ? rec[0] : -1, c = rec[1], lqs = rec[3];
  int bad = 0;
  auto fail = [&](const std::string& m) { if (bad++ < 5) printf("  FAIL %s: %s\n", name, m.c_str()); };
  // expected (sequence, key) of every FMHA sequence's rows (a short chunk gathers nothing)
  for (int z = 0; z < n + 2 && T > SHORT; ++z) {
    if (ck[z + 1] - ck[z] != kl[z]) fail("span != seq_lens_kv at z " + std::to_string(z));
    for (int local = 0; local < kl[z]; ++local) {
      int j, key;
      if (z < n) j = z, key = z == s ? local + c + 1 : local;
      else j = s, key = z == n ? c - local : local;
      const u16* want = slab.data() + (size_t)table[j * max_pages + key / PAGE] * page_stride + (size_t)(key % PAGE) * KV_A;
      if (memcmp(lat.data() + (size_t)(ck[z] + local) * KV_A, want, KV_A * 2))
        fail("latent_g row " + std::to_string(ck[z] + local) + " (z " + std::to_string(z) + " local " + std::to_string(local) + ")");
    }
  }
  for (size_t r = ck[n + 2]; r < (size_t)rows && T > SHORT; ++r)
    for (int e = 0; e < KV_A; ++e)
      if (lat[r * KV_A + e]) { fail("row past the last span not zero: " + std::to_string(r)); break; }
  // the pieces' q rows and the sequences' spans of q
  for (int z = 0; z < n; ++z) if (cq[z] != cu[z]) fail("cum_q");
  if (s >= 0) {
    if (cq[n] != T || cq[n + 1] != T + lqs || cq[n + 2] != T + 2 * lqs || rec[2] != cu[s] || lqs != lq[s]) fail("split q spans");
    for (int i = 0; i < lqs; ++i) {
      const u16* row = q.data() + (size_t)(cu[s] + i) * QW;
      if (memcmp(q.data() + (size_t)(T + lqs - 1 - i) * QW, row, QW * 2)) fail("reversed q copy " + std::to_string(i));
      if (memcmp(q.data() + (size_t)(T + lqs + i) * QW, row, QW * 2)) fail("q copy " + std::to_string(i));
    }
    // every key of the split sequence exactly once over its pieces for each row: N + R + A = [0, P + i]
    const int P = ctx[s] - lq[s];
    if (kl[s] != ctx[s] - c - 1 || kl[n] != c || kl[n + 1] != lqs || c < lqs || c > P - 1) fail("split spans");
  } else if (cq[n] != T || cq[n + 2] != T || kl[n] || kl[n + 1]) fail("unsplit extra sequences not empty");
  printf("%-34s T %4d seqs %d split %2d c %6d: %s\n", name, T, n, s, c, bad ? "FAIL" : "ok");
  failures += bad > 0;
  for (CUdeviceptr p : {d_q, d_w, d_qabs, d_seq, d_cu, d_lens, d_slab, d_table, d_slot, d_part, d_gq, d_gkv, d_qn, d_lat, d_rt, d_rl, d_bsk}) cuMemFree(p);
}

int main(int argc, char** argv) {
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));
  CUmodule m1, m2, m3; CK(cuModuleLoad(&m1, argv[1])); CK(cuModuleLoad(&m2, argv[2])); CK(cuModuleLoad(&m3, argv[3]));
  CK(cuModuleGetFunction(&absorb_fn, m3, "kern_k3_mla_absorb"));
  CK(cuModuleGetFunction(&lens_fn, m1, "kern_k3_fmha_lens_varlen")); CK(cuModuleGetFunction(&prep_fn, m2, "kern_k3g_mla_prep_gather"));
  run("one sequence", {354}, {131072 + 354});
  run("split in the middle", {100, 150, 104}, {5000, 20000, 3104});
  run("split first", {300, 50}, {9000, 2050});
  run("split last", {60, 70, 200}, {700, 1070, 30200});
  run("one tile, short prefix", {200}, {200 + 1000});
  run("prefix too short: unsplit", {300}, {300 + 400});
  run("over 512 rows: unsplit", {400, 300}, {9400, 5300});
  run("short chunk (absorbed): unsplit", {100}, {50100});
  printf("%s\n", failures ? "FAIL" : "PASS");
  return failures != 0;
}
