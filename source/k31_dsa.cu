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
// global position among equal scores; an empty candidate is 0, below everything. A global position is
// below 2^24, so the key is the score's 16 ordered bits over 24 of position: 40 bits, five radix digits.
#define KEY_POS 24
__device__ __forceinline__ unsigned long long stable_key(unsigned score, unsigned g1) {
  return g1 ? ((unsigned long long)score << KEY_POS) | ((1u << KEY_POS) - g1) : 0ull;
}

// A candidate as the stage holds it, and its key given its member m of W.
#if CAND32
typedef unsigned cand;
__device__ __forceinline__ cand ld_cand(const uint8_t* p) {
  cand w;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(w) : "l"(p));
  return w;
}
__device__ __forceinline__ bool arrived(cand w) { return w != DCP_POISON; }
__device__ __forceinline__ unsigned long long cand_key(cand w, int W, int m) {
  const unsigned l1 = w & 0xffffu;
  return stable_key(w >> 16, l1 ? (l1 - 1) * W + m + 1 : 0u);
}
#else
typedef uint2 cand;
__device__ __forceinline__ cand ld_cand(const uint8_t* p) {
  cand w;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(w.x), "=r"(w.y) : "l"(p));
  return w;
}
__device__ __forceinline__ bool arrived(cand w) { return w.x != DCP_POISON && w.y != DCP_POISON; }
__device__ __forceinline__ unsigned long long cand_key(cand w, int, int) { return stable_key(w.x, w.y); }
#endif

#ifndef DCP_W
#define DCP_W 8  // the DCP group k31_dcp_select merges
#endif
#define SEL_THREADS 1024
#define SEL_PER (DCP_W * TOPK / SEL_THREADS)  // 16 candidates a thread at W = 8

// The group's top 2048 of a row from every member's candidates, waited for in this member's stage
// (k31_dcp_pack), and of those the positions this member holds, as cache slots through its page
// table, -1 after them: thread t owns candidates t + 1024 i, and the slots go out in (t, i) order, a
// fixed order of the candidates, so the loads coalesce and the result never depends on the schedule.
// lens is their count, at least 1 for the attention (a row with none attends slot 0, live 0 says to
// discard it); live the true count. The blocks re-poison the stage the previous call read, and the
// last to arrive advances the stage: state [8] starts zeroed, [0] block counter, [2] stage, [4..5]
// i64 bytes to re-poison next call. A member whose candidates do not show within `timeout_ns` is
// recorded in `err` as 1 + its rank, then the kernel traps.
//   grid (R), every block resident at once; block 1024, W = DCP_W
extern "C" __global__ void __launch_bounds__(SEL_THREADS) k31_dcp_select(
    const uint8_t* __restrict__ lamport, int* state, int* err, long long stage_bytes, long long timeout_ns,
    int R, int rank, const int* __restrict__ block_table, int max_pages,
    int* __restrict__ ind, int* __restrict__ lens, int* __restrict__ live) {
  pdl_entry();
  constexpr int W = DCP_W;
  __shared__ unsigned whist[SEL_THREADS / 32][256];  // a warp's own: the top digit is mostly one bin
  __shared__ unsigned hist[256];
  __shared__ unsigned scan[SEL_THREADS / 32];
  __shared__ unsigned long long s_prefix;
  __shared__ unsigned s_need, s_found;
  const int r = blockIdx.x, t = threadIdx.x;
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
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
  unsigned long long key[SEL_PER];
  int fail = 0;
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i) {
    const int m = (t + i * SEL_THREADS) / TOPK;
    for (unsigned long long t0 = 0; !arrived(w[i]); w[i] = ld_cand(slot(i))) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (t0 == 0) {
        t0 = now;
      } else if ((long long)(now - t0) > timeout_ns) {
        fail = 1 + m, w[i] = cand{};
        break;
      }
    }
    key[i] = cand_key(w[i], W, m);
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
  // the 2048-th largest key by an 8-bit MSB-first radix select; keys are distinct but 0. A pass ends
  // the search once the threshold's digit bin is taken whole.
  unsigned long long prefix = 0;
  unsigned need = TOPK;
  for (int shift = KEY_POS + 8; shift >= 0; shift -= 8) {
    for (int i = t & 31; i < 256; i += 32) whist[t >> 5][i] = 0;
    __syncwarp();
    const unsigned long long hi = ~0ull << (shift + 8);
#pragma unroll
    for (int i = 0; i < SEL_PER; ++i)
      if (key[i] != 0 && (key[i] & hi) == prefix) atomicAdd(&whist[t >> 5][(key[i] >> shift) & 0xff], 1u);
    __syncthreads();
    if (t < 256) {
      unsigned h = 0;
#pragma unroll 8
      for (int w = 0; w < SEL_THREADS / 32; ++w) h += whist[w][t];
      hist[t] = h;
    }
    __syncthreads();
    if (t == 0) s_found = 0;
    if (t < 256) {  // cum: keys whose digit is 255 - t or above, a scan over the 256 bins
      const unsigned h = hist[255 - t];
      unsigned c = h;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, c, o);
        if ((t & 31) >= o) c += y;
      }
      if ((t & 31) == 31) scan[t >> 5] = c;
      __syncwarp();
      asm volatile("bar.sync 1, 256;");
      for (int w = 0; w < (t >> 5); ++w) c += scan[w];
      if (c >= need && c - h < need) {
        s_prefix = prefix | ((unsigned long long)(255 - t) << shift);
        s_need = need - (c - h) == h ? 0u : need - (c - h);
        s_found = 1;
      }
    }
    __syncthreads();
    if (!s_found) break;  // fewer than 2048 candidates in all (first pass only): take them all
    prefix = s_prefix, need = s_need;
    if (need == 0) break;
  }
  // `prefix` is now the threshold key (or 0: fewer than 2048 candidates, take them all)
  const unsigned long long thr = prefix;
  int mine = 0;
  int pos[SEL_PER];
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i) {
    const bool take = key[i] != 0 && key[i] >= thr;
    const int g = take ? (int)((1u << KEY_POS) - 1 - (unsigned)(key[i] & ((1u << KEY_POS) - 1))) : -1;
    pos[i] = take && g % W == rank ? g / W : -1;
    mine += pos[i] >= 0;
  }
  // exclusive scan of `mine` over the block, candidate order
  unsigned x = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
    if ((t & 31) >= o) x += y;
  }
  if ((t & 31) == 31) scan[t >> 5] = x;
  __syncthreads();
  if (t < 32) {
    unsigned w = scan[t];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned y = __shfl_up_sync(0xffffffffu, w, o);
      if (t >= o) w += y;
    }
    scan[t] = w;
  }
  __syncthreads();
  unsigned at = x - mine + ((t >> 5) ? scan[(t >> 5) - 1] : 0u);
  const unsigned total = scan[SEL_THREADS / 32 - 1];
  const int* bt = block_table + (size_t)r * max_pages;
  int* row = ind + (size_t)r * TOPK;
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i)
    if (pos[i] >= 0) row[at++] = bt[pos[i] / PAGE] * PAGE + pos[i] % PAGE;
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
