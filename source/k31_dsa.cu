// K3.1's DSA glue: the indexer between its projections, DeepGEMM's MQA logits, vLLM's top-k and the
// sparse attention. Each entry is a plain extern "C" ABI with its grid noted; T is the call's rows.
//
// Numerics follow vLLM's Kimi K3.1 path (vllm/models/kimi_k3/nvidia/ops/dsa_indexer.py):
//   - the key: LayerNorm in f32 (eps 1e-5, the model's rms_norm_eps), rounded to bf16, then e4m3
//     with one UE8M0 scale 2^ceil(log2(max(amax, 1e-4) / 448)) (indexer_k_norm_quant_insert);
//   - the query: e4m3 per (row, head) with a UE8M0 scale over eps 1e-10, clamped to +-448, and the
//     head weight bf16(weights_proj * 128^-0.5 * 32^-0.5). vLLM folds the scale into the weight and
//     hands DeepGEMM unit scales; a power of two commutes with every rounding after it, so passing
//     it as the query's UE8M0 scale gives the same logits.
// Neither has a position encoding (dsa_config.index_rope_dim 0).
//
// D launches the decode entries as programmatic dependents (pdl_entry): each lets the next kernel
// onto the SMs at once and reads nothing before the kernel before it is done, its own state included.
//
// A UE8M0 scale travels as DeepGEMM's packed word: four equal exponent bytes, one a 32-element block.
// The key cache is token-major, 144 bytes a token: 128 e4m3, the scale's word, 12 bytes of padding,
// so a row is a multiple of 16 bytes as kern's P -> D push and TMA require. A page is 64 tokens.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef unsigned char u8;

#define IH 32
#define ID 128
#define IDX_ROW 144
#define TOPK 2048
#define PAGE 64
#define K_EPS 1e-5f

__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void pdl_entry() {
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// 2^ceil(log2(max(amax, eps) / 448)), exact on the bits.
__device__ __forceinline__ float ue8m0(float amax, float eps) {
  const unsigned b = __float_as_uint(fmaxf(amax, eps) / 448.f);
  const unsigned x = ((b >> 23) & 0xff) + ((b & 0x7fffff) != 0);
  return __uint_as_float(x << 23);
}

__device__ __forceinline__ unsigned ue8m0x4(float sc) { return (__float_as_uint(sc) >> 23) * 0x01010101u; }

__device__ __forceinline__ unsigned fp8x4(float a, float b, float c, float d) {
  const __nv_fp8x4_e4m3 q(make_float4(a, b, c, d));
  return *reinterpret_cast<const unsigned*>(&q);
}

__device__ __forceinline__ void ld4(const bf16* p, float v[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
}

// The key of row r (k [T, 128] bf16, lda apart) into its slot's cache row; a negative slot (a
// padding row, or a position another member holds) writes nothing.
//   grid (ceil(T / 4)), block 128: a warp a row
extern "C" __global__ void __launch_bounds__(128) k31_idx_key(
    const bf16* __restrict__ k, int lda, const bf16* __restrict__ w, const bf16* __restrict__ b,
    const long long* __restrict__ slot, u8* __restrict__ cache, int T) {
  pdl_entry();
  const int r = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (r >= T) return;
  const long long s = slot[r];
  if (s < 0) return;
  float x[4], g[4], c[4];
  ld4(k + (size_t)r * lda + l * 4, x);
  ld4(w + l * 4, g);
  ld4(b + l * 4, c);
  const float mean = warp_sum(x[0] + x[1] + x[2] + x[3]) / ID;
  float d2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) d2 += (x[j] - mean) * (x[j] - mean);
  const float rstd = rsqrtf(warp_sum(d2) / ID + K_EPS);
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) y[j] = rnd((x[j] - mean) * rstd * g[j] + c[j]);
  const float sc = ue8m0(warp_max(fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])))), 1e-4f);
  u8* row = cache + s * IDX_ROW;
  *reinterpret_cast<unsigned*>(row + l * 4) = fp8x4(y[0] / sc, y[1] / sc, y[2] / sc, y[3] / sc);
  if (l == 0) *reinterpret_cast<unsigned*>(row + ID) = ue8m0x4(sc);
}

// The indexer query of row r (q [T, 32 x 128] bf16, ldq apart) as e4m3 per head into iq [T, 32, 128]
// with its scales into sfq [T, 32], each head's weight (iw [T, 32] bf16, ldw apart) into wout [T, 32],
// and, when rid is given, the row's request id r (DeepGEMM's paged scheduler: one token a request).
//   grid (T), block 256: warp w owns heads 4w .. 4w + 3
extern "C" __global__ void __launch_bounds__(256) k31_idx_q(
    const bf16* __restrict__ q, int ldq, const bf16* __restrict__ iw, int ldw, u8* __restrict__ iq,
    unsigned* __restrict__ sfq, bf16* __restrict__ wout, unsigned* __restrict__ rid, int T) {
  pdl_entry();
  const int r = blockIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int h = w * 4 + i;
    float v[4];
    ld4(q + (size_t)r * ldq + h * ID + l * 4, v);
    const float sc = ue8m0(warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])))), 1e-10f);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = fminf(fmaxf(v[j] / sc, -448.f), 448.f);
    *reinterpret_cast<unsigned*>(iq + ((size_t)r * IH + h) * ID + l * 4) = fp8x4(v[0], v[1], v[2], v[3]);
    if (l == 0)
      sfq[(size_t)r * IH + h] = ue8m0x4(sc), wout[(size_t)r * IH + h] = __float2bfloat16(bf(iw[(size_t)r * ldw + h]) * 0.015625f);
  }
  if (rid && threadIdx.x == 0) rid[r] = r;
}

// A row's position and its keys in the gathered stream: seq_lens are the sequences' lengths with
// this call's rows, the keys are gathered sequence after sequence (k31_idx_gather), so row r scores
// stream keys [ks, ke), klen = ke - ks of them, and row_seq is its sequence.
//   grid (ceil(T / 256)), block 256
extern "C" __global__ void k31_idx_plan(const long long* __restrict__ cu, const int* __restrict__ seq_lens, int nseq,
                                        unsigned* __restrict__ ks, unsigned* __restrict__ ke,
                                        int* __restrict__ klen, int* __restrict__ row_seq, int T) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= T) return;
  int s = 0;
  unsigned goff = 0;
  while (s + 1 < nseq && cu[s + 1] <= r) goff += seq_lens[s++];
  const int p = seq_lens[s] - (int)(cu[s + 1] - r);
  ks[r] = goff;
  ke[r] = goff + p + 1;
  klen[r] = p + 1;
  row_seq[r] = s;
}

// The keys of every sequence, back to back: page j of sequence s (its tokens
// [64 j, min(64 j + 64, len))) lands at stream rows goff(s) + 64 j, values in kg [., 128] and
// scale words in ksf, the contiguous layout the MQA logits read. A block a page; blocks past the last
// page return.
//   grid (ceil(ctx / 64) + seqs), block 256
extern "C" __global__ void k31_idx_gather(const u8* __restrict__ cache, const int* __restrict__ block_table,
                                          int max_pages, const int* __restrict__ seq_lens, int nseq,
                                          u8* __restrict__ kg, unsigned* __restrict__ ksf) {
  int b = blockIdx.x, s = 0;
  unsigned goff = 0;
  for (; s < nseq; ++s) {
    const int pages = (seq_lens[s] + PAGE - 1) / PAGE;
    if (b < pages) break;
    b -= pages;
    goff += seq_lens[s];
  }
  if (s == nseq) return;
  const int n = min(PAGE, seq_lens[s] - b * PAGE);
  const u8* src = cache + (size_t)block_table[(size_t)s * max_pages + b] * PAGE * IDX_ROW;
  const size_t d = goff + (size_t)b * PAGE;
  uint4* dv = reinterpret_cast<uint4*>(kg + d * ID);
  for (int i = threadIdx.x; i < n * (ID / 16); i += blockDim.x)
    dv[i] = reinterpret_cast<const uint4*>(src + (i / (ID / 16)) * IDX_ROW)[i % (ID / 16)];
  if (threadIdx.x < n) ksf[d + threadIdx.x] = *reinterpret_cast<const unsigned*>(src + threadIdx.x * IDX_ROW + ID);
}

// The top-k's sequence-relative key indices (-1 past a short row's keys) as cache slots through the
// row's page table, in place, and the row's live count min(klen, 2048) for the attention. A null
// row_seq is D's batch, row r sequence r.
//   grid (T), block 256
extern "C" __global__ void k31_topk_slots(int* __restrict__ ind, const int* __restrict__ block_table, int max_pages,
                                          const int* __restrict__ row_seq, const int* __restrict__ klen,
                                          int* __restrict__ tlen, int T) {
  const int r = blockIdx.x;
  const int* bt = block_table + (size_t)(row_seq ? row_seq[r] : r) * max_pages;
  int* row = ind + (size_t)r * TOPK;
  for (int i = threadIdx.x; i < TOPK; i += blockDim.x) {
    const int k = row[i];
    row[i] = k < 0 ? -1 : bt[k / PAGE] * PAGE + k % PAGE;
  }
  if (threadIdx.x == 0) tlen[r] = min(klen[r], TOPK);
}

// D, under DCP: the group's top 2048 of a row from its members' local top-ks (local positions, -1 past a
// member's count), exact, while only each member's best HOT candidates cross the wire (SGLang #48588's
// merge, over unicast pushes). Local position l of member `rank` is global position l * W + rank (dealt one
// position a member). The order is vLLM's (dcp_indexer_cutedsl.py stable_key): a higher score first, a
// smaller global position among equal scores; an empty candidate is below everything.
//
// k31_dcp_pack pushes the best HOT of a member's local list into slot `rank` of every member's stage
// ([W][R][HOT], member-major; a TensorRT-LLM Lamport one-shot on kern's peer ABI, as peer_allreduce_bf16.cu
// runs it). k31_dcp_select takes the 2048-th key t over the W x HOT. A member's unsent candidates lie below
// its HOT-th, so when no member that sent HOT has all of them at or above t, the stage holds the whole top
// 2048. Otherwise those members' lists are pulled from their ind and logits and t taken again over them; t
// only rises, so every other member's certificate still holds. Every member reads the same stage and so
// pulls the same lists. A pulled list is still in place: its member writes the next one after its
// exchange, which waits on every member's attention, so on every member's select; the select's slots go to
// their own buffer, not over ind.
//
// A candidate is two words, the score's ordered bf16 bits (16 wide) and its global position + 1 (0: an
// empty slot); the CAND32 build, for a member of fewer than 65534 positions, packs it into one, the ordered
// bits over local position + 1 (the member is its slot in the stage). Either way no word is ever
// 0xffffffff, the poison that says "not arrived": a position + 1 is below 2^24, a local one + 1 below
// 0xffff. Stages rotate through state[2], which k31_dcp_select advances; every stage is poisoned once by
// k31_dcp_poison.
#define DCP_POISON 0xffffffffu
#ifndef CAND32
#define CAND32 0
#endif
#define CAND_BYTES (CAND32 ? 4 : 8)
#ifndef HOT
#define HOT 384  // the candidates a member sends a row: its share of a top 2048 is ~256, 339 at most on real rows
#endif
#ifndef DCP_W
#define DCP_W 8  // the DCP group k31_dcp_select merges
#endif
#define SEL_THREADS 1024
#define SEL_BINS 4096  // a 12-bit digit, four bins a thread
#define SEL_HOT (DCP_W * HOT / SEL_THREADS)  // the stage's candidates a thread: 3 at W = 8
// A global position is below 2^24, so the merge's order is that of a 40-bit key, the score's 16 ordered
// bits over p = 2^24 - (position + 1).
#define KEY_POS 24

__device__ __forceinline__ unsigned dcp_ordered(bf16 x) {
  const unsigned u = __bfloat16_as_ushort(x);
  return u & 0x8000u ? ~u & 0xffffu : u | 0x8000u;
}

// A candidate as the stage holds it: whether it is there, its score bits, its global position + 1 given
// its member m of W.
#if CAND32
typedef unsigned cand;
__device__ __forceinline__ cand ld_cand(const uint8_t* p) {
  cand w;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(w) : "l"(p));
  return w;
}
__device__ __forceinline__ cand make_cand(unsigned score, int l, int, int) { return score << 16 | (unsigned)(l + 1); }
__device__ __forceinline__ bool arrived(cand w) { return w != DCP_POISON; }
__device__ __forceinline__ bool cand_live(cand w) { return w & 0xffffu; }
__device__ __forceinline__ unsigned cand_score(cand w) { return w >> 16; }
__device__ __forceinline__ unsigned cand_g1(cand w, int W, int m) { return ((w & 0xffffu) - 1) * W + m + 1; }
#else
typedef uint2 cand;
__device__ __forceinline__ cand ld_cand(const uint8_t* p) {
  cand w;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(w.x), "=r"(w.y) : "l"(p));
  return w;
}
__device__ __forceinline__ cand make_cand(unsigned score, int l, int W, int m) {
  return make_uint2(score, (unsigned)(l * W + m) + 1u);
}
__device__ __forceinline__ bool arrived(cand w) { return w.x != DCP_POISON && w.y != DCP_POISON; }
__device__ __forceinline__ bool cand_live(cand w) { return w.y; }
__device__ __forceinline__ unsigned cand_score(cand w) { return w.x; }
__device__ __forceinline__ unsigned cand_g1(cand w, int, int) { return w.y; }
#endif

struct Key {
  bool live;
  unsigned score, p;
};
__device__ __forceinline__ Key key_of(cand w, int W, int m) {
  return {cand_live(w), cand_score(w), (1u << KEY_POS) - cand_g1(w, W, m)};
}
// at or above the threshold (score, p); (0, 0) takes every live key
__device__ __forceinline__ bool taken(Key k, uint2 thr) {
  return k.live && (k.score > thr.x || (k.score == thr.x && k.p >= thr.y));
}

struct Sel {
  unsigned hist[2][SEL_BINS];
  unsigned low[16];
  unsigned totals[SEL_THREADS / 32];
  unsigned res[4][2];
};

// The threshold search's shared state zeroed; the caller syncs before sel_threshold.
__device__ __forceinline__ void sel_clear(Sel& s) {
  for (int i = threadIdx.x; i < 2 * SEL_BINS; i += SEL_THREADS) (&s.hist[0][0])[i] = 0;
  if (threadIdx.x < 16) s.low[threadIdx.x] = 0;
  if (threadIdx.x == 0) s.res[0][0] = ~0u;
}

// Of a 4096-bin histogram h (a larger bin holds larger keys), the bin of the need-th largest key into
// res[0] and how many of that bin to take into res[1], 0 meaning all of it; nothing written when the
// histogram holds fewer than `need` keys. Thread t sums bins 4092 - 4t .. 4095 - 4t, a block scan from
// the top finds the thread whose bins the need-th key falls in. Every thread of the block calls it; the
// result is there after the caller's next __syncthreads.
__device__ __forceinline__ void sel_find(const unsigned* h, unsigned need, unsigned* totals, unsigned* res) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint4 v = reinterpret_cast<const uint4*>(h)[SEL_BINS / 4 - 1 - t];
  const unsigned sum = v.x + v.y + v.z + v.w;
  unsigned c = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
    if (lane >= o) c += y;
  }
  if (lane == 31) totals[warp] = c;
  __syncthreads();
  unsigned off = lane < warp ? totals[lane] : 0u;
#pragma unroll
  for (int o = 16; o; o >>= 1) off += __shfl_xor_sync(0xffffffffu, off, o);
  c += off;
  if (c >= need && c - sum < need) {
    unsigned acc = c - sum, h = v.w;
    int j = 0;
    if (acc + h < need) acc += h, h = v.z, j = 1;
    if (j == 1 && acc + h < need) acc += h, h = v.y, j = 2;
    if (j == 2 && acc + h < need) acc += h, h = v.x, j = 3;
    res[0] = SEL_BINS - 1 - 4 * t - j;
    res[1] = need - acc == h ? 0u : need - acc;
  }
}

// The need-th largest key over the block's keys, get(0 .. N - 1) of every thread, as (score, p); (0, 0)
// when there are fewer. Found digit by digit, each digit a histogram over the keys still tied with the
// threshold, stopping once the threshold's bin is taken whole: the score's top 12 bits, its low 4, then
// p's top 12 and low 12 among the keys of the threshold's score. s cleared (sel_clear) and synced on
// entry, dirty on return; every thread returns after the block's last read of s.
template <int N, class Get>
__device__ __forceinline__ uint2 sel_threshold(Sel& s, unsigned need, Get get) {
  const int t = threadIdx.x;
  unsigned thr_score = 0, thr_p = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const Key k = get(i);
    if (k.live) atomicAdd(&s.hist[0][k.score >> 4], 1u);
  }
  __syncthreads();
  sel_find(s.hist[0], need, s.totals, s.res[0]);
  __syncthreads();
  if (s.res[0][0] != ~0u) {
    const unsigned hi = s.res[0][0];
    need = s.res[0][1];
    thr_score = hi << 4;
    if (need) {
      for (int i = t; i < SEL_BINS; i += SEL_THREADS) s.hist[0][i] = 0;  // for p's low digit
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const Key k = get(i);
        if (k.live && k.score >> 4 == hi) atomicAdd(&s.low[k.score & 15], 1u);
      }
      __syncthreads();
      if (t < 32) {
        const unsigned h = t < 16 ? s.low[15 - t] : 0u;
        unsigned c = h;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) {
          const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
          if (t >= o) c += y;
        }
        if (t < 16 && c >= need && c - h < need)
          s.res[1][0] = 15 - t, s.res[1][1] = need - (c - h) == h ? 0u : need - (c - h);
      }
      __syncthreads();
      thr_score |= s.res[1][0];
      need = s.res[1][1];
    }
    if (need) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const Key k = get(i);
        if (k.live && k.score == thr_score) atomicAdd(&s.hist[1][k.p >> 12], 1u);
      }
      __syncthreads();
      sel_find(s.hist[1], need, s.totals, s.res[2]);
      __syncthreads();
      thr_p = s.res[2][0] << 12;
      need = s.res[2][1];
    }
    if (need) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const Key k = get(i);
        if (k.live && k.score == thr_score && k.p >> 12 == thr_p >> 12) atomicAdd(&s.hist[0][k.p & 0xfff], 1u);
      }
      __syncthreads();
      sel_find(s.hist[0], need, s.totals, s.res[3]);
      __syncthreads();
      thr_p |= s.res[3][0];  // a position is one key: its bin is taken whole
    }
  }
  return make_uint2(thr_score, thr_p);
}

// The exclusive prefix of x over the block in thread order, and the block's total; totals is free again
// after the caller's next __syncthreads.
__device__ __forceinline__ unsigned block_scan(unsigned x, unsigned* totals, unsigned& total) {
  const int t = threadIdx.x;
  const unsigned mine = x;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
    if ((t & 31) >= o) x += y;
  }
  if ((t & 31) == 31) totals[t >> 5] = x;
  __syncthreads();
  if (t < 32) {
    unsigned v = totals[t];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned y = __shfl_up_sync(0xffffffffu, v, o);
      if (t >= o) v += y;
    }
    totals[t] = v;
  }
  __syncthreads();
  total = totals[SEL_THREADS / 32 - 1];
  return x - mine + ((t >> 5) ? totals[(t >> 5) - 1] : 0u);
}

// Row r's local list (ind, logits) as candidates, thread t's slots t and t + 1024; bf16 / int loads through
// `ld` (plain for this member's own list, volatile for a peer's).
template <class Ld>
__device__ __forceinline__ void load_list(const bf16* logits, int stride, const int* ind, int W, int m, int r, cand w[2],
                                          Ld ld) {
  const bf16* row = logits + (size_t)r * stride;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int l = ld(ind + (size_t)r * TOPK + threadIdx.x + j * SEL_THREADS);
    w[j] = l < 0 ? cand{} : make_cand(dcp_ordered(ld(row + l)), l, W, m);
  }
}
__device__ __forceinline__ int ld_own(const int* p) { return *p; }
__device__ __forceinline__ bf16 ld_own(const bf16* p) { return *p; }
__device__ __forceinline__ int ld_peer(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ bf16 ld_peer(const bf16* p) {
  return __ushort_as_bfloat16(*reinterpret_cast<const volatile unsigned short*>(p));
}

// The best HOT of this member's list w (its slots t and t + 1024 a thread) pushed to slot `at` of every
// member's stage, empty slots after them.
__device__ __forceinline__ void push_hot(Sel& s, cand* hot, const cand w[2], int rank, int W,
                                         const unsigned long long* __restrict__ peers, long long at) {
  const int t = threadIdx.x;
  const uint2 thr = sel_threshold<2>(s, HOT, [&](int j) { return key_of(w[j], W, rank); });
  const bool k0 = taken(key_of(w[0], W, rank), thr), k1 = taken(key_of(w[1], W, rank), thr);
  unsigned total;
  unsigned at_hot = block_scan(k0 + k1, s.totals, total);
  if (k0) hot[at_hot++] = w[0];
  if (k1) hot[at_hot] = w[1];
  for (int i = total + t; i < HOT; i += SEL_THREADS) hot[i] = cand{};
  __syncthreads();
  constexpr int VEC = HOT * CAND_BYTES / 16;
  for (int v = t; v < W * VEC; v += SEL_THREADS)
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[v / VEC]) + at + (long long)(v % VEC) * 16) =
        reinterpret_cast<const uint4*>(hot)[v % VEC];
}

// Pushes row r's best HOT of its local list (k31_dcp_pack above), a block a row.
//   grid (R), block 1024
extern "C" __global__ void __launch_bounds__(SEL_THREADS) k31_dcp_pack(
    const bf16* __restrict__ logits, int stride, const int* __restrict__ ind, int rank, int W, int R,
    const unsigned long long* __restrict__ peers, const int* __restrict__ state, long long stage_bytes) {
  pdl_entry();
  __shared__ __align__(16) Sel s;
  __shared__ __align__(16) cand hot[HOT];
  sel_clear(s);
  const int r = blockIdx.x;
  cand w[2];
  load_list(logits, stride, ind, W, rank, r, w, [](auto p) { return ld_own(p); });
  __syncthreads();
  push_hot(s, hot, w, rank, W, peers,
           (long long)(state[2] % 3) * stage_bytes + ((long long)rank * R + r) * HOT * CAND_BYTES);
}

// Every stage of a member's candidate buffer, poisoned once before its first step.
extern "C" __global__ void k31_dcp_poison(uint4* __restrict__ buf, long long bytes) {
  const uint4 p = make_uint4(DCP_POISON, DCP_POISON, DCP_POISON, DCP_POISON);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16; i += (long long)gridDim.x * blockDim.x)
    buf[i] = p;
}

// The group's top 2048 of row r, waited for in this member's stage, and of those the positions this member
// holds (own: its list, thread t's slots t and t + 1024) as cache slots through its page table into sel,
// -1 after them, in (t, slot) order, a fixed order of the candidates that never depends on the schedule.
// lens is their count, at least 1 for the attention (a row with none attends slot 0, live 0 says to
// discard it); live the true count. A pulled list is read from its member's ind and logits (peers):
// the kernel boundary on that member orders them before its push. cnt is 2 W shared counters, zeroed and
// synced on entry. A member whose candidates do not show within `timeout_ns` is recorded in `err` as
// 1 + its rank, then the kernel traps.
__device__ __forceinline__ void select_row(Sel& s, unsigned* cnt, const uint8_t* stage, int* err, long long timeout_ns,
                                           int R, int rank, int stride,
                                           const unsigned long long* __restrict__ logits_peers,
                                           const unsigned long long* __restrict__ ind_peers,
                                           const int* __restrict__ block_table, int max_pages, int* __restrict__ sel,
                                           int* __restrict__ lens, int* __restrict__ live, int r, const cand own[2]) {
  constexpr int W = DCP_W;
  static_assert(SEL_HOT * SEL_THREADS == W * HOT && TOPK == 2 * SEL_THREADS, "the stage splits evenly over the block");
  const int t = threadIdx.x;
  // candidate c = t + 1024 i of the stage: member c / HOT, slot c % HOT
  const auto member = [&](int i) { return (t + i * SEL_THREADS) / HOT; };
  const auto slot = [&](int i) {
    const int c = t + i * SEL_THREADS;
    return stage + (((size_t)(c / HOT) * R + r) * HOT + c % HOT) * CAND_BYTES;
  };
  // every candidate's load in flight at once, then a spin on those not yet arrived
  cand w[SEL_HOT];
#pragma unroll
  for (int i = 0; i < SEL_HOT; ++i) w[i] = ld_cand(slot(i));
  int fail = 0;
#pragma unroll
  for (int i = 0; i < SEL_HOT; ++i) {
    for (unsigned long long t0 = 0; !arrived(w[i]); w[i] = ld_cand(slot(i))) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (t0 == 0) {
        t0 = now;
      } else if ((long long)(now - t0) > timeout_ns) {
        fail = 1 + member(i), w[i] = cand{};
        break;
      }
    }
  }
  if (fail) {
    atomicMax(err, fail);
    __threadfence_system();
    __trap();
  }
  uint2 thr = sel_threshold<SEL_HOT>(s, TOPK, [&](int i) { return key_of(w[i], W, member(i)); });
  // the certificate: a member that sent HOT, every one of them taken, may hold more of the top 2048
#pragma unroll
  for (int i = 0; i < SEL_HOT; ++i) {
    const Key k = key_of(w[i], W, member(i));
    if (k.live) atomicAdd(&cnt[member(i)], 1u);
    if (taken(k, thr)) atomicAdd(&cnt[W + member(i)], 1u);
  }
  __syncthreads();
  unsigned pull = 0;
#pragma unroll
  for (int m = 0; m < W; ++m) pull |= (cnt[m] == HOT && cnt[W + m] == HOT) << m;
  if (pull) {
    // candidates 2048 m + t + 1024 j: member m's list if pulled, else what it sent
    sel_clear(s);
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    cand x[2 * W];
#pragma unroll
    for (int m = 0; m < W; ++m) {
      if (pull >> m & 1) {
        load_list(reinterpret_cast<const bf16*>(logits_peers[m]), stride, reinterpret_cast<const int*>(ind_peers[m]), W,
                  m, r, x + 2 * m, [](auto p) { return ld_peer(p); });
      } else {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int k = t + j * SEL_THREADS;
          x[2 * m + j] = k < HOT ? ld_cand(stage + (((size_t)m * R + r) * HOT + k) * CAND_BYTES) : cand{};
        }
      }
    }
    __syncthreads();
    thr = sel_threshold<2 * W>(s, TOPK, [&](int i) { return key_of(x[i], W, i / 2); });
  }
  // this member's candidates at or above the threshold, as local positions
  int pos[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) pos[j] = taken(key_of(own[j], W, rank), thr) ? (int)(cand_g1(own[j], W, rank) - 1) / W : -1;
  unsigned total;
  unsigned at = block_scan((pos[0] >= 0) + (pos[1] >= 0), s.totals, total);
  const int* bt = block_table + (size_t)r * max_pages;
  int* row = sel + (size_t)r * TOPK;
#pragma unroll
  for (int j = 0; j < 2; ++j)
    if (pos[j] >= 0) row[at++] = bt[pos[j] / PAGE] * PAGE + pos[j] % PAGE;
  for (int i = total + t; i < TOPK; i += SEL_THREADS) row[i] = i == 0 ? 0 : -1;
  if (t == 0) lens[r] = total ? (int)total : 1, live[r] = (int)total;
}

// The blocks re-poison the stage the previous call read, and the last to arrive advances the stage: state
// [8] starts zeroed, [0] block counter, [2] stage, [4..5] i64 bytes to re-poison next call.
__device__ __forceinline__ void repoison(const uint8_t* lamport, int flag, long long stage_bytes, long long clear_count,
                                         int r) {
  uint4* clear = reinterpret_cast<uint4*>(const_cast<uint8_t*>(lamport) + (long long)((flag + 2) % 3) * stage_bytes);
  const uint4 poison = make_uint4(DCP_POISON, DCP_POISON, DCP_POISON, DCP_POISON);
  for (long long i = (long long)r * SEL_THREADS + threadIdx.x; i < clear_count / 16; i += (long long)gridDim.x * SEL_THREADS)
    clear[i] = poison;
}
__device__ __forceinline__ void advance(int* state, int flag, int R) {
  while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
  }
  state[2] = (flag + 1) % 3;
  *reinterpret_cast<long long*>(state + 4) = (long long)DCP_W * R * HOT * CAND_BYTES;
  state[0] = 0;
}

// k31_dcp_select above, a block a row, this member's list read again.
//   grid (R), block 1024, W = DCP_W
extern "C" __global__ void __launch_bounds__(SEL_THREADS) k31_dcp_select(
    const uint8_t* __restrict__ lamport, int* state, int* err, long long stage_bytes, long long timeout_ns,
    int R, int rank, const int* __restrict__ block_table, int max_pages, int* __restrict__ sel,
    int* __restrict__ lens, int* __restrict__ live, const bf16* __restrict__ logits, int stride,
    const int* __restrict__ ind, const unsigned long long* __restrict__ logits_peers,
    const unsigned long long* __restrict__ ind_peers) {
  pdl_entry();
  __shared__ __align__(16) Sel s;
  __shared__ unsigned cnt[2 * DCP_W];
  const int r = blockIdx.x, t = threadIdx.x;
  const int flag = state[2];
  const long long clear_count = *reinterpret_cast<const long long*>(state + 4);
  sel_clear(s);
  if (t < 2 * DCP_W) cnt[t] = 0;
  cand own[2];
  load_list(logits, stride, ind, DCP_W, rank, r, own, [](auto p) { return ld_own(p); });
  __syncthreads();
  if (t == 0) atomicAdd(state, 1);
  repoison(lamport, flag, stage_bytes, clear_count, r);
  select_row(s, cnt, lamport + (long long)(flag % 3) * stage_bytes, err, timeout_ns, R, rank, stride, logits_peers,
             ind_peers, block_table, max_pages, sel, lens, live, r, own);
  if (r == 0 && t == 0) advance(state, flag, R);
}
