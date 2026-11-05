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

// D, under DCP: a member's local top-k (local positions, -1 past its count) as candidates for the
// group's merge, pushed into slot `rank` of every member's stage (a TensorRT-LLM Lamport one-shot on
// kern's peer ABI, as peer_allreduce_bf16.cu runs it): a stage is [W][R][2048] candidates, member-major.
// Local position l of member `rank` is global position l * W + rank (dealt one position a member). A
// candidate is two words, the score's ordered bf16 bits (16 wide) and its global position + 1 (0: an
// empty slot); the CAND32 build, for a member of fewer than 65534 positions, packs it into one, the
// ordered bits over local position + 1 (the member is its slot in the stage). Either way no word is ever
// 0xffffffff, the poison that says "not arrived": a position + 1 is below 2^24, a local one + 1 below
// 0xffff. Stages rotate through state[2], which k31_dcp_select advances; every stage is poisoned once by
// k31_dcp_poison.
//   grid (R), block 256: a thread pushes two (CAND32: four) adjacent candidates as one 16-byte word
#define DCP_POISON 0xffffffffu
#ifndef CAND32
#define CAND32 0
#endif
#define CAND_BYTES (CAND32 ? 4 : 8)
#define CAND_VEC (16 / CAND_BYTES)
__device__ __forceinline__ unsigned dcp_ordered(const bf16* __restrict__ row, int l) {
  const unsigned u = __bfloat16_as_ushort(row[l]);
  return u & 0x8000u ? ~u & 0xffffu : u | 0x8000u;
}

extern "C" __global__ void k31_dcp_pack(const bf16* __restrict__ logits, int stride, const int* __restrict__ ind,
                                        int rank, int W, int R, const unsigned long long* __restrict__ peers,
                                        const int* __restrict__ state, long long stage_bytes) {
  pdl_entry();
  const int r = blockIdx.x;
  const long long at = (long long)(state[2] % 3) * stage_bytes + (((long long)rank * R + r) * TOPK) * CAND_BYTES;
  const bf16* row = logits + (size_t)r * stride;
  for (int i = CAND_VEC * threadIdx.x; i < TOPK; i += CAND_VEC * blockDim.x) {
#if CAND32
    const int4 l = *reinterpret_cast<const int4*>(ind + (size_t)r * TOPK + i);
    const auto word = [&](int l) { return l < 0 ? 0u : dcp_ordered(row, l) << 16 | (unsigned)(l + 1); };
    const uint4 v = make_uint4(word(l.x), word(l.y), word(l.z), word(l.w));
#else
    const auto word = [&](int l) {
      return l < 0 ? make_uint2(0u, 0u) : make_uint2(dcp_ordered(row, l), (unsigned)(l * W + rank) + 1u);
    };
    const uint2 a = word(ind[(size_t)r * TOPK + i]), b = word(ind[(size_t)r * TOPK + i + 1]);
    const uint4 v = make_uint4(a.x, a.y, b.x, b.y);
#endif
    for (int q = 0; q < W; ++q)
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[q]) + at + (long long)i * CAND_BYTES) = v;
  }
}

// Every stage of a member's candidate buffer, poisoned once before its first step.
extern "C" __global__ void k31_dcp_poison(uint4* __restrict__ buf, long long bytes) {
  const uint4 p = make_uint4(DCP_POISON, DCP_POISON, DCP_POISON, DCP_POISON);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16; i += (long long)gridDim.x * blockDim.x)
    buf[i] = p;
}

// vLLM's order of the merge (dcp_indexer_cutedsl.py stable_key): a higher score first, a smaller
// global position among equal scores; an empty candidate is below everything. A global position is below
// 2^24, so the order is that of a 40-bit key, the score's 16 ordered bits over p = 2^24 - (position + 1).
#define KEY_POS 24

// A candidate as the stage holds it: whether it is there, its score bits, its global position + 1 given
// its member m of W.
#if CAND32
typedef unsigned cand;
__device__ __forceinline__ cand ld_cand(const uint8_t* p) {
  cand w;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(w) : "l"(p));
  return w;
}
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
__device__ __forceinline__ bool arrived(cand w) { return w.x != DCP_POISON && w.y != DCP_POISON; }
__device__ __forceinline__ bool cand_live(cand w) { return w.y; }
__device__ __forceinline__ unsigned cand_score(cand w) { return w.x; }
__device__ __forceinline__ unsigned cand_g1(cand w, int, int) { return w.y; }
#endif

#ifndef DCP_W
#define DCP_W 8  // the DCP group k31_dcp_select merges
#endif
#define SEL_THREADS 1024
#define SEL_PER (DCP_W * TOPK / SEL_THREADS)  // 16 candidates a thread at W = 8
#define SEL_BINS 4096                        // a 12-bit digit, four bins a thread

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

// The group's top 2048 of a row from every member's candidates, waited for in this member's stage
// (k31_dcp_pack), and of those the positions this member holds, as cache slots through its page
// table, -1 after them: thread t owns candidates t + 1024 i, and the slots go out in (t, i) order, a
// fixed order of the candidates, so the loads coalesce and the result never depends on the schedule.
// lens is their count, at least 1 for the attention (a row with none attends slot 0, live 0 says to
// discard it); live the true count. The blocks re-poison the stage the previous call read, and the
// last to arrive advances the stage: state [8] starts zeroed, [0] block counter, [2] stage, [4..5]
// i64 bytes to re-poison next call. A member whose candidates do not show within `timeout_ns` is
// recorded in `err` as 1 + its rank, then the kernel traps.
//
// The 2048-th largest key is found digit by digit, each digit a histogram over the candidates still
// tied with the threshold, stopping once the threshold's bin is taken whole: the score's top 12 bits,
// its low 4, then p's top 12 and low 12 among the candidates of the threshold's score. Member m's
// candidates are thread t's i = 2m and 2m + 1, so this member's share is two candidates a thread.
//   grid (R), every block resident at once; block 1024, W = DCP_W
extern "C" __global__ void __launch_bounds__(SEL_THREADS) k31_dcp_select(
    const uint8_t* __restrict__ lamport, int* state, int* err, long long stage_bytes, long long timeout_ns,
    int R, int rank, const int* __restrict__ block_table, int max_pages,
    int* __restrict__ ind, int* __restrict__ lens, int* __restrict__ live) {
  pdl_entry();
  constexpr int W = DCP_W;
  static_assert(SEL_PER == 2 * W && TOPK == 2 * SEL_THREADS, "member m is candidates 2m and 2m + 1 of a thread");
  __shared__ __align__(16) unsigned hist[2][SEL_BINS];
  __shared__ unsigned low[16];
  __shared__ unsigned totals[SEL_THREADS / 32];
  __shared__ unsigned res[4][2];
  const int r = blockIdx.x, t = threadIdx.x;
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
  for (int i = t; i < 2 * SEL_BINS; i += SEL_THREADS) (&hist[0][0])[i] = 0;
  if (t < 16) low[t] = 0;
  if (t == 0) res[0][0] = ~0u;
  __syncthreads();
  if (t == 0) atomicAdd(state, 1);
  const uint8_t* stage = lamport + (long long)(flag % 3) * stage_bytes;
  // every candidate's load in flight at once, then a spin on those not yet arrived
  const auto slot = [&](int i) {  // candidate c = t + 1024 i of the row: member c / 2048, slot c % 2048
    const int c = t + i * SEL_THREADS;
    return stage + (((size_t)(c / TOPK) * R + r) * TOPK + c % TOPK) * CAND_BYTES;
  };
  cand w[SEL_PER];
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i) w[i] = ld_cand(slot(i));
  int fail = 0;
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i) {
    for (unsigned long long t0 = 0; !arrived(w[i]); w[i] = ld_cand(slot(i))) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (t0 == 0) {
        t0 = now;
      } else if ((long long)(now - t0) > timeout_ns) {
        fail = 1 + i / 2, w[i] = cand{};
        break;
      }
    }
  }
  if (fail) {
    atomicMax(err, fail);
    __threadfence_system();
    __trap();
  }
  {
    uint4* clear = reinterpret_cast<uint4*>(const_cast<uint8_t*>(lamport) + (long long)((flag + 2) % 3) * stage_bytes);
    const uint4 poison = make_uint4(DCP_POISON, DCP_POISON, DCP_POISON, DCP_POISON);
    for (long long i = (long long)r * SEL_THREADS + t; i < clear_count / 16; i += (long long)gridDim.x * SEL_THREADS)
      clear[i] = poison;
  }
  const auto p_of = [&](int i) { return (1u << KEY_POS) - cand_g1(w[i], W, i / 2); };
  // the threshold key: (score, p) at or above it are the top 2048; (0, 0) takes every candidate there is
  unsigned thr_score = 0, thr_p = 0;
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i)
    if (cand_live(w[i])) atomicAdd(&hist[0][cand_score(w[i]) >> 4], 1u);
  __syncthreads();
  sel_find(hist[0], TOPK, totals, res[0]);
  __syncthreads();
  if (res[0][0] != ~0u) {
    const unsigned hi = res[0][0];
    unsigned need = res[0][1];
    thr_score = hi << 4;
    if (need) {
      for (int i = t; i < SEL_BINS; i += SEL_THREADS) hist[0][i] = 0;  // for p's low digit
#pragma unroll
      for (int i = 0; i < SEL_PER; ++i)
        if (cand_live(w[i]) && cand_score(w[i]) >> 4 == hi) atomicAdd(&low[cand_score(w[i]) & 15], 1u);
      __syncthreads();
      if (t < 32) {
        const unsigned h = t < 16 ? low[15 - t] : 0u;
        unsigned c = h;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) {
          const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
          if (t >= o) c += y;
        }
        if (t < 16 && c >= need && c - h < need) res[1][0] = 15 - t, res[1][1] = need - (c - h) == h ? 0u : need - (c - h);
      }
      __syncthreads();
      thr_score |= res[1][0];
      need = res[1][1];
    }
    if (need) {
#pragma unroll
      for (int i = 0; i < SEL_PER; ++i)
        if (cand_live(w[i]) && cand_score(w[i]) == thr_score) atomicAdd(&hist[1][p_of(i) >> 12], 1u);
      __syncthreads();
      sel_find(hist[1], need, totals, res[2]);
      __syncthreads();
      thr_p = res[2][0] << 12;
      need = res[2][1];
    }
    if (need) {
#pragma unroll
      for (int i = 0; i < SEL_PER; ++i)
        if (cand_live(w[i]) && cand_score(w[i]) == thr_score && p_of(i) >> 12 == thr_p >> 12)
          atomicAdd(&hist[0][p_of(i) & 0xfff], 1u);
      __syncthreads();
      sel_find(hist[0], need, totals, res[3]);
      __syncthreads();
      thr_p |= res[3][0];  // a position is one candidate: its bin is taken whole
    }
  }
  // this member's two candidates a thread, taken at or above the threshold, as local positions
  int pos[2] = {-1, -1};
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i)
    if (i / 2 == rank && cand_live(w[i])) {
      const unsigned sc = cand_score(w[i]), p = p_of(i);
      if (sc > thr_score || (sc == thr_score && p >= thr_p)) pos[i & 1] = (int)(cand_g1(w[i], W, i / 2) - 1) / W;
    }
  const int mine = (pos[0] >= 0) + (pos[1] >= 0);
  // exclusive scan of `mine` over the block, candidate order
  unsigned x = mine;
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
  unsigned at = x - mine + ((t >> 5) ? totals[(t >> 5) - 1] : 0u);
  const unsigned total = totals[SEL_THREADS / 32 - 1];
  const int* bt = block_table + (size_t)r * max_pages;
  int* row = ind + (size_t)r * TOPK;
#pragma unroll
  for (int j = 0; j < 2; ++j)
    if (pos[j] >= 0) row[at++] = bt[pos[j] / PAGE] * PAGE + pos[j] % PAGE;
  for (int i = total + t; i < TOPK; i += SEL_THREADS) row[i] = i == 0 ? 0 : -1;
  if (t == 0) lens[r] = total ? (int)total : 1, live[r] = (int)total;
  if (r == 0 && t == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
    }
    state[2] = (flag + 1) % 3;
    *clear_ptr = (long long)W * R * TOPK * CAND_BYTES;
    state[0] = 0;
  }
}
