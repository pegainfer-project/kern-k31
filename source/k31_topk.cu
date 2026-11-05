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
// k31_topk is for P's thousands of rows. k31_topk_held is for D's few rows, each latency-bound: a
// row of at most HELD_MAX is read once into registers, and from there a histogram finds the cut
// value and bitmaps of the row place the 2048 (held, below); a longer row takes P's route.
//
// A logit is ranked by its key, the bits of the float it widens to ordered as the float is, read in
// digits: bits 31..20, 19..8, 7..0. A digit's 4096-bin histogram over the keys that share the digits
// above it finds the bin holding the cut. Once that bin holds at most CAP keys, one read takes every
// position above it and keeps the bin's (key, position) pairs in shared memory, where a radix select
// by key and then by position settles the cut.
//
// A row of at least SAMPLE_MIN is first tried on a sample: its ends whole and an eighth of the rest in
// chunks spread evenly, weighted to the row, puts a threshold where 1.5x 2048 keys are expected above
// it, and one read keeps every key at or above it. The threshold is the bottom of the first digit's
// bin when the bin and all above it are expected to fit 3/4 of CAP (a bf16 key has 16 bits, so that
// bin is 16 values wide), the second digit's cut otherwise. When the kept count lands in
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

// The float whose key k is.
__device__ __forceinline__ float unkey(unsigned k) {
  return __uint_as_float(k ^ ((k & 0x80000000u) ? 0x80000000u : 0xffffffffu));
}

using Scan = cub::BlockScan<unsigned, NT>;
using Sort = cub::BlockRadixSort<int, NT, TK / NT>;

#define BITMAP_MAX 262144  // rows up to this long order their 2048 by a bitmap, longer ones by a sort

#define HELD_MAX 32768

struct Smem {
  union {
    unsigned hist[BINS];
    unsigned tie[CAP];
    // held's: the histogram, then each warp's 32 words of digits
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

#define CU 2  // 16-byte loads a thread a collect tile

// Positions whose key >> sh is above `hi` into s.sel, those in [lo, hi] into the candidates; s.ncand
// counts every candidate, CAP or not. Hits are a few percent of a row, so each takes its own atomic.
// Keying every logit made the read bound by issue: eight logits at a time are ruled out by four pair
// compares against `lo` as a bf16 rounded down, and only a group with a logit at or above it (or a
// NaN) is keyed. The next tile's loads leave before this tile is looked at. A row starts 16-byte
// aligned (`stride` a multiple of 8).
__device__ __forceinline__ void collect(Smem& s, const bf16* row, unsigned n, int sh, unsigned lo, unsigned hi) {
  if (threadIdx.x == 0) s.nsel = 0, s.ncand = 0;
  __syncthreads();
  const __nv_bfloat162 thr = __bfloat162bfloat162(__float2bfloat16_rd(unkey(lo << sh)));
  constexpr unsigned TILE = NT * 8 * CU;
  const auto load = [&](unsigned base, uint4* v) {
#pragma unroll
    for (int u = 0; u < CU; ++u) {
      const unsigned i = base + (u * NT + threadIdx.x) * 8;
      v[u] = i < n ? __ldg(reinterpret_cast<const uint4*>(row + i)) : make_uint4(0, 0, 0, 0);
    }
  };
  uint4 cur[CU], nxt[CU];
  load(0, cur);
  for (unsigned base = 0; base < n; base += TILE) {
    if (base + TILE < n) load(base + TILE, nxt);
#pragma unroll
    for (int u = 0; u < CU; ++u) {
      const unsigned i = base + (u * NT + threadIdx.x) * 8, w[4] = {cur[u].x, cur[u].y, cur[u].z, cur[u].w};
      unsigned below = ~0u;
#pragma unroll
      for (int q = 0; q < 4; ++q) below &= __hlt2_mask(*reinterpret_cast<const __nv_bfloat162*>(&w[q]), thr);
      if (below == ~0u) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned k = okey(__uint_as_float(j & 1 ? w[j / 2] & 0xffff0000u : w[j / 2] << 16)), d = k >> sh;
        if (i + j >= n || d < lo) continue;
        if (d > hi) {
          s.sel[atomicAdd(&s.nsel, 1u)] = i + j;
        } else {
          const unsigned c = atomicAdd(&s.ncand, 1u);
          if (c < CAP) s.ckey[c] = k, s.cpos[c] = i + j;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < CU; ++u) cur[u] = nxt[u];
  }
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
  if (s.above + s.count <= 16 * (CAP * 3 / 4)) {
    collect(s, row, n, 20, b1, 0xfffu);
  } else {
    histogram(s, over, n, 1, b1);
    cut_bin(s, left);
    collect(s, row, n, 8, prefix_of(1, b1, s.cut), 0xffffffu);
  }
  const unsigned kept = s.ncand;
  if (kept < TK || kept > CAP) return false;
  finish(s, TK, n);
  return true;
}


// D's route's shared memory, over Smem's: the histogram and later the bitmaps, the output slots' marks.
struct Held {
  unsigned bits[BINS + 32];
  unsigned mark[TK];
  unsigned wsum[NT / 32];
  unsigned cut, above, digit, gt;
};
static_assert(sizeof(Held) <= sizeof(Smem), "held's layout fits the module's shared memory");

// The exclusive sum of x over the block in thread order; every thread calls it. A warp's total is in
// h.wsum, so it is free until the next barrier.
__device__ __forceinline__ unsigned held_scan(Held& h, unsigned x) {
  const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned c = x;
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    const unsigned y = __shfl_up_sync(0xffffffffu, c, i);
    if (lane >= i) c += y;
  }
  if (lane == 31) h.wsum[w] = c;
  __syncthreads();
  unsigned before = 0;
#pragma unroll
  for (int i = 0; i < NT / 32; i += 4) {
    const uint4 q = *reinterpret_cast<const uint4*>(h.wsum + i);
    before += (i < w ? q.x : 0) + (i + 1 < w ? q.y : 0) + (i + 2 < w ? q.z : 0) + (i + 3 < w ? q.w : 0);
  }
  return before + c - x;
}

// The position of x's k-th set bit, k below its count.
__device__ __forceinline__ unsigned nth_bit(unsigned x, unsigned k) {
  unsigned p = 0;
#pragma unroll
  for (int w = 16; w; w >>= 1) {
    const unsigned c = __popc(x >> p & (1u << w) - 1);
    if (k >= c) k -= c, p += w;
  }
  return p;
}

// D's route for a row of at most HELD_MAX, read once. Warp w holds positions [32 C w, 32 C (w + 1)),
// lane l the eight from 32 C w + 256 j + 8 l for each j < C / 8, so every read is coalesced. Adding
// +0 makes -0 +0 and every NaN 0x7fff, as widening and adding +0 does for the key (every NaN keys as
// 0xffffffff, above +inf), and leaves the rest alone, subnormals too. Each value counts into the
// 4096-bin histogram of its top 12 bits, the bins read in key order (a negative half's reversed),
// whose cut bin a 17-slot histogram of the low 4 bits splits to the cut value K. One HSET2 compares
// two values with K: the row's positions above K and at K become two bitmaps, a byte a group. A scan
// of their words' counts places each word's taken positions (every one above K, the first `need` at
// K), and each thread fills four output slots from the words holding them. The 2048 come out
// ascending with no sort and in balanced work, the same for any ties.
//
// A slot past n is read as a NaN and taken back out: from the NaN bin and digit by count, from the
// bitmaps by position. A row with 2048 NaNs or more cuts at NaN, which no ordered compare finds.
template <int C>
__device__ __forceinline__ void held(Held& h, const bf16* row, unsigned n, int* o) {
  constexpr int V = C / 8, W = C / 2;
  const unsigned t = threadIdx.x, pads = NT * C - n;
  const auto at = [&](int j) { return 32 * C * (t >> 5) + 256 * j + 8 * (t & 31); };  // group j's first position
  unsigned a[W];
#pragma unroll
  for (int j = 0; j < V; ++j) {
    const uint4 v = at(j) < n ? __ldg(reinterpret_cast<const uint4*>(row + at(j))) : make_uint4(~0u, ~0u, ~0u, ~0u);
    a[4 * j] = v.x, a[4 * j + 1] = v.y, a[4 * j + 2] = v.z, a[4 * j + 3] = v.w;
  }
  for (int i = t; i < (BINS + 32) / 4; i += NT) reinterpret_cast<uint4*>(h.bits)[i] = make_uint4(0, 0, 0, 0);
  unsigned* dig = h.bits + BINS;
  // the group straddling n holds real values past it: NaNs from here on
#pragma unroll
  for (int j = 0; j < V; ++j)
    if (at(j) == (n & ~7u))
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const unsigned i = at(j) + 2 * q;
        a[4 * j + q] |= i < n ? (i + 1 < n ? 0u : 0xffff0000u) : ~0u;
      }
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const __nv_bfloat162 x = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&a[i]), __bfloat162bfloat162(0.0f));
    a[i] = *reinterpret_cast<const unsigned*>(&x);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < W; ++i) atomicAdd(&h.bits[a[i] >> 4 & 0xfff], 1u), atomicAdd(&h.bits[a[i] >> 20], 1u);
  __syncthreads();
  {
    // thread t reads key bins 4095 - 8t down to 4088 - 8t: raw bins 2047 - 8t down for t < 256 (the
    // NaN bin first), 8t up for the negative half
    const uint4* h4 = reinterpret_cast<const uint4*>(h.bits);
    const bool pos = t < 256;
    const uint4 x = h4[pos ? (0x7f8 - 8 * t) / 4 + 1 : 2 * t], y = h4[pos ? (0x7f8 - 8 * t) / 4 : 2 * t + 1];
    const unsigned c[8] = {(pos ? x.w : x.x) - (t == 0 ? pads : 0), pos ? x.z : x.y, pos ? x.y : x.z, pos ? x.x : x.w,
                           pos ? y.w : y.x, pos ? y.z : y.y, pos ? y.y : y.z, pos ? y.x : y.w};
    unsigned sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) sum += c[k];
    unsigned ab = held_scan(h, sum);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (ab < TK && ab + c[k] >= TK) h.cut = BINS - 1 - 8 * t - k, h.above = ab;
      ab += c[k];
    }
  }
  __syncthreads();
  const unsigned rb = h.cut ^ (h.cut & 0x800 ? 0x800 : 0xfff), rem = TK - h.above, neg = rb >> 11;
  // a value's top 12 bits are the cut bin's when (value ^ rb << 4) is below 16: its digit
#pragma unroll
  for (int i = 0; i < W; ++i) {
    atomicAdd(dig + min((a[i] ^ rb << 4) & 0xffffu, 16u), 1u);
    atomicAdd(dig + min(a[i] >> 16 ^ rb << 4, 16u), 1u);
  }
  __syncthreads();
  if (t < 32) {
    // lane l < 16 holds key digit 15 - l; the count from the top first reaching rem marks the cut
    const unsigned d = 15 - t, r = neg ? 15 - d : d, c0 = t < 16 ? dig[r] - (rb == 0x7ff && r == 15 ? pads : 0) : 0;
    unsigned c = c0;
#pragma unroll
    for (int i = 1; i < 16; i <<= 1) {
      const unsigned y = __shfl_up_sync(0xffffffffu, c, i);
      if (t >= i) c += y;
    }
    if (t < 16 && c >= rem && c - c0 < rem) h.digit = r, h.gt = c - c0;
  }
  __syncthreads();
  const unsigned K = rb << 4 | h.digit, need = rem - h.gt;
  // above K by an unordered compare (a NaN is above every value), at K by an ordered one; for a NaN K,
  // the unordered compare with +inf finds the NaNs, and nothing is above
  const bool nan = K == 0x7fff;
  const __nv_bfloat162 ka = __bfloat162bfloat162(__ushort_as_bfloat16((unsigned short)(nan ? 0x7f80 : K))),
                       kb = __bfloat162bfloat162(__ushort_as_bfloat16((unsigned short)K));
  // the bitmaps over the histogram's words, word q's low value at bit 2q of its group's byte and its
  // high one at 2q + 1
  constexpr unsigned NW = NT * C / 32;
  unsigned *gbits = h.bits, *ebits = gbits + NW, *take = ebits + NW;
#pragma unroll
  for (int j = 0; j < V; ++j) {
    unsigned g = 0, e = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&a[4 * j + q]);
      const unsigned bit = 1u << 2 * q | 1u << 17 + 2 * q;
      g |= __hgtu2_mask(x, ka) & bit, e |= __heq2_mask(x, kb) & bit;
    }
    const unsigned keep = 0xffu >> 8 - min(max((int)(n - at(j)), 0), 8);
    g = (g | g >> 16) & keep, e = (e | e >> 16) & keep;
    reinterpret_cast<unsigned char*>(gbits)[at(j) / 8] = nan ? 0 : g;
    reinterpret_cast<unsigned char*>(ebits)[at(j) / 8] = nan ? g : e;
  }
  __syncthreads();
  // thread t owns PW words; one scan of their counts places them, the first `need` ties kept
  constexpr int PW = C >= 32 ? C / 32 : 1;
  const bool owns = t * PW < NW;
  unsigned gw[PW], ew[PW], cg = 0, ce = 0;
#pragma unroll
  for (int i = 0; i < PW; ++i) {
    gw[i] = owns ? gbits[t * PW + i] : 0, ew[i] = owns ? ebits[t * PW + i] : 0;
    cg += __popc(gw[i]), ce += __popc(ew[i]);
  }
  const unsigned pre = held_scan(h, cg | ce << 16), ties = pre >> 16;
  unsigned r = need > ties ? need - ties : 0, q = (pre & 0xffffu) + min(ties, need);
  if (ce > r)
#pragma unroll
    for (int i = 0; i < PW; ++i) {
      const unsigned c = __popc(ew[i]);
      ew[i] = r >= c ? ew[i] : r ? ew[i] & (1u << nth_bit(ew[i], r)) - 1 : 0;
      r -= min(r, c);
    }
  // a word's owner marks the slot it starts at and each of its slots a multiple of 4: the word and
  // how many of its positions come before the slot
  if (owns)
#pragma unroll
    for (int i = 0; i < PW; ++i) {
      const unsigned x = gw[i] | ew[i], c = __popc(x), wd = t * PW + i;
      take[wd] = x;
      if (c) h.mark[q] = wd << 5;
      for (unsigned k = (q + 3) & ~3u; k < q + c; k += 4) h.mark[k] = wd << 5 | (k - q);
      q += c;
    }
  __syncthreads();
  // output slots 4t to 4t + 3, a word read again where a slot starts one
  unsigned k = 4 * t, m = h.mark[k], wd = m >> 5, x = take[wd];
  x &= ~0u << nth_bit(x, m & 31);
  int v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i, ++k) {
    if (!x) m = h.mark[k], wd = m >> 5, x = take[wd];
    v[i] = wd * 32 + __ffs(x) - 1, x &= x - 1;
  }
  reinterpret_cast<int4*>(o)[t] = make_int4(v[0], v[1], v[2], v[3]);
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
  // D launches the held build as a programmatic dependent: the next kernel onto the SMs at once, the
  // logits read once the kernel before is done
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  extern __shared__ __align__(16) unsigned char raw[];
  Smem& s = *reinterpret_cast<Smem*>(raw);
  const int t = threadIdx.x;
  const bf16* row = logits + (size_t)blockIdx.x * stride;
  const unsigned n = (unsigned)max(lens[blockIdx.x], 0);
  int* o = out + (size_t)blockIdx.x * TK;
  if (n <= TK) {
    for (int i = t; i < TK; i += NT) o[i] = i < (int)n ? i : -1;
    return;
  }
  if (HELD && n <= HELD_MAX) {
    Held& h = *reinterpret_cast<Held*>(raw);
    if (n <= NT * 8) held<8>(h, row, n, o);
    else if (n <= NT * 16) held<16>(h, row, n, o);
    else if (n <= NT * 32) held<32>(h, row, n, o);
    else held<64>(h, row, n, o);
    return;
  }
  if (n < SAMPLE_MIN || !sampled(s, row, n)) exact(s, row, n);
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
