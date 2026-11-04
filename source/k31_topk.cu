// K3.1's indexer top-k: each row's 2048 largest bf16 logits among its first `lens[r]`, as positions in
// ascending order, -1 past min(lens, 2048). The set is the exact top 2048 that vLLM's
// top_k_per_row_prefill selects; where equal logits straddle the cut, the smaller positions are taken.
// Unlike vLLM's kernel, whose output order follows its shared-memory atomics, the result is the same
// bytes on every run.
//
//   extern "C" __global__ void k31_topk(const bf16* logits, int stride, const int* lens, int* out);
//   extern "C" __global__ void k31_topk_held(...the same...);
//   grid (rows), block NT, dynamic smem k31_topk_smem (a __device__ int of the module)
//
// k31_topk runs two blocks an SM, for P's thousands of rows. k31_topk_held runs one, for D's few
// rows, each latency-bound: a row of at most HELD_MAX is read once into registers and every digit
// below is counted and collected from there.
//
// A logit is ranked by its key, the bits of the float it widens to ordered as the float is, read in
// digits: bits 31..20, 19..8, 7..0. A digit's 4096-bin histogram over the keys that share the digits
// above it finds the bin holding the cut. Once that bin holds at most CAP keys, one read takes every
// position above it and keeps the bin's (key, position) pairs in shared memory, where a radix select
// by key and then by position settles the cut.
//
// A row of at least SAMPLE_MIN is first tried on a sample: its ends whole and an eighth of the rest in
// chunks spread evenly, weighted to the row, two digits deep, puts a threshold where 1.5x 2048 keys
// are expected above it, and one read keeps every key at or above it. When the kept count lands in
// [2048, CAP], that is the row: one read and an eighth. Otherwise (a sample that misleads) the row
// takes the exact route above, a read a digit plus the one that collects; a cut inside one exact key
// held by more than CAP positions takes the ties in position order on one more read.
//
// The 2048 positions leave in ascending order: through a bitmap of the row for a row of at most
// BITMAP_MAX, a radix sort past it.
#include <cub/cub.cuh>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float4 widen(uint2 v) {
  return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u), __uint_as_float(v.y << 16),
                     __uint_as_float(v.y & 0xffff0000u));
}
// four logits at p (8-byte aligned) widened, through the read-only path or plainly
__device__ __forceinline__ float4 ld4v(const bf16* p) { return widen(__ldg(reinterpret_cast<const uint2*>(p))); }
__device__ __forceinline__ float4 ld4(const bf16* p) { return widen(*reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ float lv(const bf16* row, unsigned i) { return __bfloat162float(row[i]); }

#define TK 2048
#define NT 512
#define BINS 4096
#define CAP 6144
#define BPT (BINS / NT)
#define UNROLL 4
#define SAMPLE_CHUNK 1024  // logits a sample chunk
#define SAMPLE_FRAC 8      // the sample is about 1 / SAMPLE_FRAC of the row
#define HEAD 1024
#define TAIL 8192
#define SAMPLE_MIN 32768

__device__ __forceinline__ unsigned okey(float x) {
  const unsigned b = __float_as_uint(x + 0.0f);  // -0 taken as +0
  return b ^ ((b & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

using Scan = cub::BlockScan<unsigned, NT>;
using Sort = cub::BlockRadixSort<int, NT, TK / NT>;

#define BITMAP_MAX 262144  // rows up to this long order their 2048 by a bitmap, longer ones by a sort

struct Smem {
  union {
    unsigned hist[BINS];
    unsigned tie[CAP];
    unsigned bits[BITMAP_MAX / 32];
    typename Sort::TempStorage sort;
  } u;
  typename Scan::TempStorage scan;
  unsigned h256[256];
  unsigned wsum[8];
  int sel[TK];
  unsigned ckey[CAP];
  unsigned cpos[CAP];
  unsigned nsel, ncand, ntie, cut, above, count, digit, gt;
};
extern "C" __device__ const int k31_topk_smem = sizeof(Smem);

// The digits a level reads: key >> SHIFT & 0xfff among the keys whose bits above FSHIFT match.
__constant__ const int SHIFT[3] = {20, 8, 0};
__constant__ const int FSHIFT[3] = {32, 20, 8};

__device__ __forceinline__ bool in_play(unsigned k, int level, unsigned fpre) {
  return level == 0 || k >> FSHIFT[level] == fpre;
}

// The key prefix at SHIFT[level] of a bin.
__device__ __forceinline__ unsigned prefix_of(int level, unsigned fpre, unsigned bin) {
  const int w = FSHIFT[level] - SHIFT[level];
  return level == 0 ? bin : fpre << w | (bin & ((1u << w) - 1));
}

// Thread t < 256 owns bin 255 - t of h256: the digit holding the rem-th largest (rem >= 1) and the
// count strictly above it, into s.digit / s.gt. Every thread calls it.
__device__ __forceinline__ void cut256(Smem& s, unsigned rem) {
  const int t = threadIdx.x;
  if (t < 256) {
    const unsigned h = s.h256[255 - t];
    unsigned c = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
      if ((t & 31) >= o) c += y;
    }
    if ((t & 31) == 31) s.wsum[t >> 5] = c;
    asm volatile("bar.sync 1, 256;");
    for (int w = 0; w < (t >> 5); ++w) c += s.wsum[w];
    if (c >= rem && c - h < rem) s.digit = 255 - t, s.gt = c - h;
  }
  __syncthreads();
}

// The rem-th largest (rem in [1, m]) of v[0, m) and how many lie above it, by 8-bit passes over the
// low `low` bits; every v shares the bits of `base` above them.
__device__ __forceinline__ void select_smem(Smem& s, const unsigned* v, unsigned m, unsigned rem, unsigned& thr,
                                            unsigned& gt, int low, unsigned base) {
  const int top = (low + 7) / 8 * 8 - 8;
  unsigned mask = top + 8 >= 32 ? 0u : ~0u << (top + 8), prefix = base & mask;
  gt = 0;
  for (int shift = top; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += NT) s.h256[i] = 0;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < m; i += NT)
      if ((v[i] & mask) == prefix) atomicAdd(&s.h256[(v[i] >> shift) & 255], 1u);
    __syncthreads();
    cut256(s, rem);
    prefix |= s.digit << shift, mask |= 255u << shift;
    gt += s.gt, rem -= s.gt;
    __syncthreads();
  }
  thr = prefix;
}

// The four logits at i (fewer at the row's end, the rest -inf).
__device__ __forceinline__ void load4(const bf16* row, unsigned i, unsigned n, float x[4]) {
  if (i + 3 < n) {
    const float4 v = ld4(row + i);
    x[0] = v.x, x[1] = v.y, x[2] = v.z, x[3] = v.w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = i + j < n ? lv(row, i + j) : -INFINITY;
  }
}

// f(x[4], i) for the four logits at i. Whole tiles issue UNROLL unguarded loads of each thread before
// any is used (a guard between them would serialize them); the tail goes four logits a thread.
template <class F>
__device__ __forceinline__ void each4(const bf16* row, unsigned n, F f) {
  constexpr unsigned TILE = NT * 4 * UNROLL;
  const unsigned full = n / TILE * TILE;
  for (unsigned base = 0; base < full; base += TILE) {
    float4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      v[u] = ld4v(row + base + (u * NT + threadIdx.x) * 4);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const float x[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
      f(x, base + (u * NT + threadIdx.x) * 4);
    }
  }
  for (unsigned i = full + threadIdx.x * 4; i < n; i += NT * 4) {
    float x[4];
    load4(row, i, n, x);
    f(x, i);
  }
}

// The sample: the head (HEAD logits) and the tail (TAIL) whole, weight 1, and the interior's chunks
// spread evenly at 1 / SAMPLE_FRAC, weighted up to the interior's length; weights in 1/16ths. Indexer
// logits crowd the ends (the first tokens, the recent window), so an unweighted sample overcounts.
struct Sample {
  unsigned chunks, interior, w16;
};

__device__ __forceinline__ Sample sample_of(unsigned n) {
  const unsigned len = n - HEAD - TAIL, interior = max(2u, len / (SAMPLE_CHUNK * SAMPLE_FRAC));
  return {(HEAD + TAIL) / SAMPLE_CHUNK + interior, interior,
          (unsigned)((16ull * len + interior * SAMPLE_CHUNK / 2) / (interior * SAMPLE_CHUNK))};
}

// f(x[4], i, weight) over the sample, a thread's loads issued four at a time.
template <class F>
__device__ __forceinline__ void each_sample(const bf16* row, unsigned n, Sample m, F f) {
  constexpr unsigned PER = SAMPLE_CHUNK / 4, ENDS = (HEAD + TAIL) / SAMPLE_CHUNK, STEP = NT / PER;
  const auto at = [&](unsigned c) {
    return c == 0     ? 0
           : c < ENDS ? ((n - TAIL) & ~3u) + (c - 1) * SAMPLE_CHUNK
                      : HEAD + (unsigned)((unsigned long long)(c - ENDS) * (n - HEAD - TAIL - SAMPLE_CHUNK) /
                                          (m.interior - 1)) & ~3u;
  };
  for (unsigned c0 = threadIdx.x / PER; c0 < m.chunks; c0 += 4 * STEP) {
    float x[4][4];
    unsigned i[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const unsigned c = c0 + u * STEP;
      i[u] = c < m.chunks ? at(c) + threadIdx.x % PER * 4 : n;
      load4(row, i[u], n, x[u]);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (i[u] < n) f(x[u], i[u], c0 + u * STEP < ENDS ? 16u : m.w16);
  }
}

// One level's histogram over `scan` (the row, each key weight 1, or the sample).
template <class Over>
__device__ __forceinline__ void histogram(Smem& s, Over scan, unsigned n, int level, unsigned fpre) {
  for (int i = threadIdx.x; i < BINS; i += NT) s.u.hist[i] = 0;
  __syncthreads();
  const int sh = SHIFT[level];
  scan([&](const float* x, unsigned i, unsigned w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned k = okey(x[j]);
      if (i + j < n && in_play(k, level, fpre)) atomicAdd(&s.u.hist[k >> sh & 0xfff], w);
    }
  });
  __syncthreads();
}

// The bin where the count from the top first reaches `want` (at most the histogram's total), into
// s.cut, with the count above it and its own into s.above / s.count.
__device__ __forceinline__ void cut_bin(Smem& s, unsigned want) {
  const int t = threadIdx.x;
  unsigned c[BPT], sum = 0;
#pragma unroll
  for (int k = 0; k < BPT; ++k) c[k] = s.u.hist[BINS - 1 - BPT * t - k], sum += c[k];
  __syncthreads();
  unsigned ab;
  Scan(s.scan).ExclusiveSum(sum, ab);
#pragma unroll
  for (int k = 0; k < BPT; ++k) {
    if (ab < want && ab + c[k] >= want) s.cut = BINS - 1 - BPT * t - k, s.above = ab, s.count = c[k];
    ab += c[k];
  }
  __syncthreads();
}

// Positions whose key >> sh is above `hi` into s.sel, those in [lo, hi] into the candidates; s.ncand
// counts every candidate, CAP or not. Hits are a few percent of a row, so each takes its own atomic.
__device__ __forceinline__ void collect(Smem& s, const bf16* row, unsigned n, int sh, unsigned lo, unsigned hi) {
  if (threadIdx.x == 0) s.nsel = 0, s.ncand = 0;
  __syncthreads();
  each4(row, n, [&](const float* x, unsigned i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned k = okey(x[j]), d = k >> sh;
      if (i + j >= n || d < lo) continue;
      if (d > hi) {
        s.sel[atomicAdd(&s.nsel, 1u)] = i + j;
      } else {
        const unsigned c = atomicAdd(&s.ncand, 1u);
        if (c < CAP) s.ckey[c] = k, s.cpos[c] = i + j;
      }
    }
  });
  __syncthreads();
}

// The candidates' rem largest by key, ties by smallest position, appended to s.sel. Every candidate
// shares the key bits of `base` above the low `low`, and every position lies below n.
__device__ __forceinline__ void finish(Smem& s, unsigned rem, unsigned n, int low = 32, unsigned base = 0) {
  const int t = threadIdx.x;
  const unsigned m = s.ncand;
  unsigned kthr, kgt;
  select_smem(s, s.ckey, m, rem, kthr, kgt, low, base);
  if (t == 0) s.ntie = 0;
  __syncthreads();
  for (unsigned i = t; i < m; i += NT) {
    if (s.ckey[i] > kthr) s.sel[atomicAdd(&s.nsel, 1u)] = s.cpos[i];
    else if (s.ckey[i] == kthr) s.u.tie[atomicAdd(&s.ntie, 1u)] = ~s.cpos[i];
  }
  __syncthreads();
  const unsigned e = s.ntie, need = rem - kgt;
  unsigned pthr = 0, pgt;
  if (need < e) select_smem(s, s.u.tie, e, need, pthr, pgt, 32 - __clz(n - 1), ~0u);
  for (unsigned i = t; i < e; i += NT)
    if (need == e || s.u.tie[i] >= pthr) s.sel[atomicAdd(&s.nsel, 1u)] = ~s.u.tie[i];
  __syncthreads();
}

// Every key above `key` and the first `rem` positions holding it, in one in-order read.
__device__ void ties_in_order(Smem& s, const bf16* row, unsigned n, unsigned key, unsigned rem) {
  const int t = threadIdx.x;
  if (t == 0) s.nsel = 0;
  __syncthreads();
  unsigned taken = 0;  // ties taken by earlier tiles
  for (unsigned base = 0; base < n; base += NT * 4) {
    const unsigned i = base + t * 4;
    float x[4];
    load4(row, i, n, x);
    unsigned e = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) e += i + j < n && okey(x[j]) == key;
    unsigned off, tot;
    Scan(s.scan).ExclusiveSum(e, off, tot);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned k = okey(x[j]);
      const bool tie = i + j < n && k == key;
      if ((i + j < n && k > key) || (tie && taken + off < rem)) s.sel[atomicAdd(&s.nsel, 1u)] = i + j;
      off += tie;
    }
    taken += tot;
    __syncthreads();
  }
}

// The sampled route: true when s.sel holds the row's 2048.
__device__ __forceinline__ bool sampled(Smem& s, const bf16* row, unsigned n) {
  const Sample m = sample_of(n);
  const auto over = [&](auto f) { each_sample(row, n, m, f); };
  const unsigned want = 16 * TK * 3 / 2;
  histogram(s, over, n, 0, 0);
  cut_bin(s, want);
  const unsigned b1 = s.cut, left = want - s.above;
  histogram(s, over, n, 1, b1);
  cut_bin(s, left);
  collect(s, row, n, 8, prefix_of(1, b1, s.cut), 0xffffffu);
  const unsigned kept = s.ncand;
  if (kept < TK || kept > CAP) return false;
  finish(s, TK, n);
  return true;
}

#define HELD_U 16  // float4s a thread holds: rows of at most NT * 4 * HELD_U stay in registers
#define HELD_MAX (NT * 4 * HELD_U)

// The exact route over a row held in registers, one read for every level; a cut inside one key held
// by more than CAP positions takes the ties in position order on one more read. A key of 0 (a negative NaN's) marks a slot past n: it
// sits in the lowest bin, under the 2049 real keys, so no cut reaches it.
__device__ __forceinline__ void held(Smem& s, const bf16* row, unsigned n) {
  constexpr int Q = 4 * HELD_U;
  const unsigned t = threadIdx.x, lane = t & 31, last = n >= 4 ? (n - 4) & ~3u : 0;
  const auto at = [&](int u) { return (u * NT + t) * 4; };
  float4 v[HELD_U];
#pragma unroll
  for (int u = 0; u < HELD_U; ++u) v[u] = ld4v(row + min((unsigned)at(u), last));
  unsigned k[Q];
#pragma unroll
  for (int u = 0; u < HELD_U; ++u) {
    float x[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
    if (at(u) < n && at(u) + 3 >= n)
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] = at(u) + j < n ? lv(row, at(u) + j) : 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) k[4 * u + j] = at(u) + j < n ? okey(x[j]) : 0u;
  }
  unsigned rem = TK, fpre = 0;
  for (int level = 0; level < 3; ++level) {
    const int sh = SHIFT[level];
    for (int i = t; i < BINS; i += NT) s.u.hist[i] = 0;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < Q; ++q)
      if (in_play(k[q], level, fpre)) atomicAdd(&s.u.hist[k[q] >> sh & 0xfff], 1u);
    __syncthreads();
    cut_bin(s, rem);
    const unsigned p = prefix_of(level, fpre, s.cut);
    if (s.count <= CAP) {
      // every hit of the row claims its slot at once: one warp scan, one atomic a kind a warp; a
      // candidate's key is read again rather than indexed out of the registers
      unsigned long long ma = 0, mc = 0;
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        const unsigned d = k[q] >> sh;
        ma |= (unsigned long long)(d > p) << q, mc |= (unsigned long long)(d == p) << q;
      }
      const unsigned na = __popcll(ma), nc = __popcll(mc);
      unsigned c = na | nc << 16;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
        if (lane >= o) c += y;
      }
      const unsigned tot = __shfl_sync(0xffffffffu, c, 31);
      if (t == 0) s.nsel = 0, s.ncand = 0;
      __syncthreads();
      unsigned a0 = 0, c0 = 0;
      if (lane == 31) a0 = atomicAdd(&s.nsel, tot & 0xffff), c0 = atomicAdd(&s.ncand, tot >> 16);
      unsigned a = __shfl_sync(0xffffffffu, a0, 31) + (c & 0xffff) - na,
               b = __shfl_sync(0xffffffffu, c0, 31) + (c >> 16) - nc;
      const auto pos = [&](int q) { return (unsigned)at(q >> 2) + (q & 3); };
      for (; ma; ma &= ma - 1) s.sel[a++] = pos(__ffsll(ma) - 1);
      for (; mc; mc &= mc - 1, ++b) {
        const unsigned i = pos(__ffsll(mc) - 1);
        s.ckey[b] = okey(lv(row, i)), s.cpos[b] = i;
      }
      __syncthreads();
      finish(s, TK - s.nsel, n, sh, p << sh);
      return;
    }
    rem -= s.above, fpre = p;
    __syncthreads();
  }
  ties_in_order(s, row, n, fpre, rem);
}

// The exact route: a level a read until the cut's bin fits CAP.
__device__ __forceinline__ void exact(Smem& s, const bf16* row, unsigned n) {
  const auto over = [&](auto f) { each4(row, n, [&](const float* x, unsigned i) { f(x, i, 1u); }); };
  unsigned rem = TK, fpre = 0;
  for (int level = 0; level < 3; ++level) {
    histogram(s, over, n, level, fpre);
    cut_bin(s, rem);
    const unsigned p = prefix_of(level, fpre, s.cut), above = s.above;
    if (s.count <= CAP) {
      collect(s, row, n, SHIFT[level], p, p);
      finish(s, TK - s.nsel, n, SHIFT[level], p << SHIFT[level]);
      return;
    }
    rem -= above, fpre = p;
  }
  ties_in_order(s, row, n, fpre, rem);
}

template <bool HELD>
__device__ __forceinline__ void topk(const bf16* __restrict__ logits, int stride, const int* __restrict__ lens,
                                     int* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char raw[];
  Smem& s = *reinterpret_cast<Smem*>(raw);
  const int t = threadIdx.x;
  const unsigned n = (unsigned)max(lens[blockIdx.x], 0);
  const bf16* row = logits + (size_t)blockIdx.x * stride;
  int* o = out + (size_t)blockIdx.x * TK;
  if (n <= TK) {
    for (int i = t; i < TK; i += NT) o[i] = i < (int)n ? i : -1;
    return;
  }
  if (HELD && n <= HELD_MAX) held(s, row, n);
  else if (n < SAMPLE_MIN || !sampled(s, row, n)) exact(s, row, n);
  if (n <= BITMAP_MAX) {
    // ascending: each thread a run of words, its bits' rank from a scan of the runs' counts
    const unsigned words = (n + 31) / 32, per = (words + NT - 1) / NT, w0 = min(t * per, words), w1 = min(w0 + per, words);
    for (unsigned w = t; w < words; w += NT) s.u.bits[w] = 0;
    __syncthreads();
    for (int i = t; i < TK; i += NT) atomicOr(&s.u.bits[s.sel[i] >> 5], 1u << (s.sel[i] & 31));
    __syncthreads();
    unsigned c = 0, at;
    for (unsigned w = w0; w < w1; ++w) c += __popc(s.u.bits[w]);
    Scan(s.scan).ExclusiveSum(c, at);
    for (unsigned w = w0; w < w1; ++w)
      for (unsigned b = s.u.bits[w]; b; b &= b - 1) o[at++] = (int)(w * 32 + __ffs(b) - 1);
    return;
  }
  int k[TK / NT];
#pragma unroll
  for (int j = 0; j < TK / NT; ++j) k[j] = s.sel[t * (TK / NT) + j];
  __syncthreads();
  Sort(s.u.sort).Sort(k, 0, 32 - __clz(n - 1));
#pragma unroll
  for (int j = 0; j < TK / NT; ++j) o[t * (TK / NT) + j] = k[j];
}

extern "C" __global__ void __launch_bounds__(NT, 2) k31_topk(const bf16* __restrict__ logits, int stride,
                                                            const int* __restrict__ lens, int* __restrict__ out) {
  topk<false>(logits, stride, lens, out);
}

// One block an SM, a row of at most HELD_MAX read once into registers: few rows, each as soon as can be.
extern "C" __global__ void __launch_bounds__(NT, 1) k31_topk_held(const bf16* __restrict__ logits, int stride,
                                                                 const int* __restrict__ lens, int* __restrict__ out) {
  topk<true>(logits, stride, lens, out);
}
