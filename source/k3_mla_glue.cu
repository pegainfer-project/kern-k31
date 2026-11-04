// The glue of a packed prefill chunk's MLA layer around TensorRT-LLM's context FMHA, whose rows attend
// their own rows through the causal kernel and their sequences' cached context through its dense
// sibling in rounds of `window` keys (k3_fmha_plan.cuh), the pieces merged by the kernels' softmax stats:
//
//   kern_k3g_mla_prep(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm, block_table,
//                     max_pages, lens, ns, nseq, latent_own, row_table, row_lens, bsk, split_max, short_max, B)
//     blocks [0, B): k3_mla_prep.cu's head for chunk row b (q_norm, the latent row kv_norm | rope appended to
//       the slab), the latent row also written to row b of latent_own, the chunk's own keys in row order;
//       rows [B, B + 128) of latent_own zero, the FMHA's tile past the last sequence's end.
//     blocks [B, 2B + 1): a short chunk's (B <= short_max) rows for the absorbed attention (k3_mla_absorb →
//       FlashInfer's MLA decode → k3_mla_vup_gate) instead of the expansion + FMHA: row b's page table row
//       (its sequence's) and causal length (the cached rows, the chunk's rows up to b), then block 2B
//       k3_mla_split_plan.cu's KV split plan over those lengths. A long chunk's head blocks skip latent_own.
//     partial bf16 [B, MLA_FUSED] the fused projection (q_a | kv_a | rope | gate), landed by its GEMM;
//     slab, page_stride: the layer's latent pages; lens: the FMHA tables (k3_fmha_plan.cuh, ns words a row).
//   grid (2 B + 1, 1, 1)   block (512, 1, 1)
//
//   kern_k3g_mla_gather(slab, page_stride, block_table, max_pages, lens, ns, nseq, latent_g, w, window, n)
//     rows [0, n) of latent_g: round w's keys, the sequences' cached latent rows [w window, w window + n) of
//     the stream (set 0's aux row: sequence j's keys at [C_j, C_j + P_j)), every row past the stream's end
//     zero. GITER rows a thread in flight, 28 a block.
//   grid (ceil(n / 28), 1, 1)   block (512, 1, 1)
//
//   kern_k3g_mla_merge(o, stats, lens, ns, nseq, acc_o, acc_ms, init, n, B)
//     one FMHA pass's partial (o bf16 [B, n], stats float2 (max, sum) per row and head) folded into the
//     f32 accumulator (acc_o [B, n], acc_ms float2 per row and head) for the rows whose sequence had keys
//     in it (lens: the pass's set): acc = acc e^(m - m') + s_p e^(m_p - m') o_p, likewise the sum, m' the
//     new max; `init` (the own pass, every row's first) takes the partial as is. 8 columns a thread.
//   grid (B n / 2048, 1, 1)   block (256, 1, 1)
//
//   kern_k3g_mla_gate(out, acc_o, partial, ..., acc_ms, n, short_max, B)
//     sigmoid_mul.cu's kern_sigmoid_mul_bf16 on the gate columns of the fused projection over the merged
//     attention: out = bf16(bf16(acc_o / s) * bf16(sigmoid(bf16(partial[b, HEADC + j])))), out [B, n].
//   grid (ceil(B / 8) * n / 2048 + the absorbed form's blocks, 1, 1)   block (256, 1, 1)
//
//   kern_k3g_e4m3(in, out, rows, width)   (KV_FP8 build)
//     out = in's first rows x width bf16 as e4m3 bytes by aten's cast (k3_e4m3.cuh): the FMHA's fp8 q and
//     expanded k | v, vLLM's `.to(float8_e4m3fn)` at scale 1. 8 elements a thread a pass, grid-stride.
//   grid (any, 1, 1)   block (256, 1, 1)
//
// Every value is k3_mla_prep's and sigmoid_mul's bit for bit (the same landings, sums and roundings);
// the attention's o is an f32 merge of bf16 pieces, bf16-landed once.
#include <cuda_bf16.h>

#define Q_LORA 1536
#define KV_LORA 512
#define ROPE 64
#define KV_A 576
#ifndef MLA_FUSED
#define MLA_FUSED 14400
#endif
#define HEADC (Q_LORA + KV_LORA + ROPE)  // 2112 head columns
#define EPSV 1e-5f
#define PAGE 64
#define QU (Q_LORA / 4)  // 384 float4 units of q
#define NT 512
#define QW (QU / 32)  // 12 warps of q
#define LANES (KV_A * 2 / 16)  // 72 sixteen-byte pieces a latent row
#define GROWS (NT / LANES)      // 7 latent rows a gather block pass
#ifndef GITER
#define GITER 4                 // passes: 28 rows a gather block
#endif

typedef __nv_bfloat16 bf16_t;
// -DKV_FP8: the latent pages are SGLang's fp8 cache, one e4m3 byte an element (k3_mla_prep.cu's build: each
// landed bf16 to e4m3 by cvt.rn without saturation); the gather lifts a row back to bf16, exact.
#ifdef KV_FP8
#include <cuda_fp8.h>
#include "k3_e4m3.cuh"
typedef unsigned char kv_t;
__device__ __forceinline__ unsigned fp8x2(unsigned w) {
  const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
  return e4m3x2_nosat(f.x, f.y);
}
__device__ __forceinline__ void put4(kv_t* p, uint2 v) { *reinterpret_cast<unsigned*>(p) = fp8x2(v.x) | fp8x2(v.y) << 16; }
__device__ __forceinline__ unsigned bf16x2_of_e4m3x2(unsigned short x) {
  const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(x, __NV_E4M3));
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *reinterpret_cast<const unsigned*>(&b);
}
// the 8 e4m3 elements at p as 8 bf16
__device__ __forceinline__ uint4 get8(const kv_t* p) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  return make_uint4(bf16x2_of_e4m3x2(v.x & 0xffffu), bf16x2_of_e4m3x2(v.x >> 16), bf16x2_of_e4m3x2(v.y & 0xffffu),
                    bf16x2_of_e4m3x2(v.y >> 16));
}
#define PUT4(p, v) put4(p, v)
#define GET8(p) get8(p)
#else
typedef bf16_t kv_t;
#define PUT4(p, v) (*reinterpret_cast<uint2*>(p) = (v))
#define GET8(p) (*reinterpret_cast<const uint4*>(p))
#endif

__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&p);
}
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned g) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&g));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ float landf(float x) { return __bfloat162float(__float2bfloat16(x)); }

// k3_mla_prep.cu's fast head for chunk row b: q_norm, kv_norm | rope appended to the slab, and the
// latent row also written to `grow` unless null. The whole 512-thread block.
__device__ __forceinline__ void prep_head(const bf16_t* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a,
                                          const bf16_t* __restrict__ gamma_kv_a, const long long* __restrict__ slot_mapping,
                                          kv_t* __restrict__ slab, long long page_stride, bf16_t* __restrict__ q_norm,
                                          bf16_t* __restrict__ grow, int b, int t) {
  __shared__ float red[NT / 32];
  const bf16_t* __restrict__ P = partial + (long long)b * MLA_FUSED;
  const bool isq = (t < QU);
  const int col = isq ? (4 * t) : (Q_LORA + 4 * (t - QU));
  const uint2 v = *reinterpret_cast<const uint2*>(P + col);
  const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  const float x0 = v01.x, x1 = v01.y, x2 = v23.x, x3 = v23.y;

  float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
  if ((t & 31) == 0) red[t >> 5] = ss;

  const long long slot = slot_mapping[b];
  const bool append = (slot >= 0);
  kv_t* const row = slab + (slot / PAGE) * page_stride + (slot % PAGE) * KV_A;
  const bf16_t* const gsrc = isq ? (gamma_q_a + col) : (gamma_kv_a + (col - Q_LORA));
  const uint2 g = *reinterpret_cast<const uint2*>(gsrc);
  uint2 rope4 = make_uint2(0u, 0u);
  if (t < ROPE / 4) {
    rope4 = *reinterpret_cast<const uint2*>(P + Q_LORA + KV_LORA + 4 * t);
  }

  __syncthreads();

  float tot = 0.f;
  if (isq) {
#pragma unroll
    for (int w = 0; w < QW; ++w) tot += red[w];
  } else {
#pragma unroll
    for (int w = QW; w < NT / 32; ++w) tot += red[w];
  }
  const float sc = isq ? rsqrtf(tot * (1.f / Q_LORA) + EPSV) : rsqrtf(tot * (1.f / KV_LORA) + EPSV);
  const uint2 o = make_uint2(mul2(pack2(x0 * sc, x1 * sc), g.x), mul2(pack2(x2 * sc, x3 * sc), g.y));
  if (isq) {
    *reinterpret_cast<uint2*>(q_norm + (long long)b * Q_LORA + col) = o;
    if (t < ROPE / 4) {
      if (grow) *reinterpret_cast<uint2*>(grow + KV_LORA + 4 * t) = rope4;
      if (append) PUT4(row + KV_LORA + 4 * t, rope4);
    }
  } else {
    if (grow) *reinterpret_cast<uint2*>(grow + (col - Q_LORA)) = o;
    if (append) PUT4(row + (col - Q_LORA), o);
  }
}

#define TILE_KV 128  // the decode attention's KV tile (k3_mla_split_plan.cu)
#define KV_PAD 128   // the FMHA's KV tile: rows past the last sequence's end it reads, zero

// row b's sequence among nseq (cum_q: the set's cum_seq_lens_q row)
__device__ __forceinline__ int seq_of(const int* __restrict__ cum_q, int nseq, int b) {
  int j = 0;
  while (j + 1 < nseq && cum_q[j + 1] <= b) ++j;
  return j;
}

extern "C" __global__ void __launch_bounds__(NT) kern_k3g_mla_prep(
    const bf16_t* __restrict__ partial, const bf16_t* __restrict__ gamma_q_a, const bf16_t* __restrict__ gamma_kv_a,
    const long long* __restrict__ slot_mapping, kv_t* __restrict__ slab, long long page_stride,
    bf16_t* __restrict__ q_norm, const int* __restrict__ block_table, int max_pages, const int* __restrict__ lens,
    int ns, int nseq, bf16_t* __restrict__ latent_own, int* __restrict__ row_table, int* __restrict__ row_lens,
    int* __restrict__ bsk, int split_max, int short_max, int B) {
  const int t = threadIdx.x;
  const int* __restrict__ cum_q = lens + ns;
  const int* __restrict__ cached = lens + 3 * ns;  // set 0's aux: the cached-key prefix sums
  const bool absorbed = B <= short_max;
  // row b of sequence j: its place among the sequence's rows (the cached ones first), its causal length
  const auto pos_of = [&](int b, int j) { return cached[j + 1] - cached[j] + (b - cum_q[j]); };

  if (blockIdx.x < B) {  // ---- head: q_norm, kv_norm, rope, the latent row (also to latent_own for the expansion)
    const int b = blockIdx.x;
    prep_head(partial, gamma_q_a, gamma_kv_a, slot_mapping, slab, page_stride, q_norm,
              absorbed ? nullptr : latent_own + (long long)b * KV_A, b, t);
    if (!absorbed && b < KV_PAD && t < LANES)
      *reinterpret_cast<uint4*>(latent_own + (long long)(B + b) * KV_A + t * 8) = make_uint4(0, 0, 0, 0);
    return;
  }
  if (!absorbed) return;  // ---- a short chunk's rows for the absorbed attention
  if (blockIdx.x < 2 * B) {  // row b's page table (its sequence's pages) and causal length
    const int b = blockIdx.x - B, j = seq_of(cum_q, nseq, b), len = pos_of(b, j) + 1;
    for (int p = t; p < (len + PAGE - 1) / PAGE; p += NT) row_table[b * max_pages + p] = block_table[j * max_pages + p];
    if (t == 0) row_lens[b] = len;
    return;
  }
  // k3_mla_split_plan.cu's KV split plan over those lengths
  __shared__ long long wsum[NT / 32];
  const int warp = t >> 5, lane = t & 31;
  long long mine = 0;
  for (int b = t; b < B; b += NT) mine += (pos_of(b, seq_of(cum_q, nseq, b)) + TILE_KV) / TILE_KV;
#pragma unroll
  for (int o = 16; o; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
  if (lane == 0) wsum[warp] = mine;
  __syncthreads();
  long long total = 0;
#pragma unroll
  for (int w = 0; w < NT / 32; ++w) total += wsum[w];
  unsigned nsm;
  asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
  long long budget = (long long)(nsm / 2) - B;
  if (budget < 1) budget = 1;
  long long per = (total + budget - 1) / budget;
  if (per < 1) per = 1;
  for (int b = t; b < B; b += NT) {
    const long long tiles = (pos_of(b, seq_of(cum_q, nseq, b)) + TILE_KV) / TILE_KV;
    bsk[b] = (int)min(max((tiles + per - 1) / per, 1ll), (long long)split_max);
  }
}

extern "C" __global__ void __launch_bounds__(NT) kern_k3g_mla_gather(
    const kv_t* __restrict__ slab, long long page_stride, const int* __restrict__ block_table, int max_pages,
    const int* __restrict__ lens, int ns, int nseq, bf16_t* __restrict__ latent_g, int w, int window, int n) {
  const int t = threadIdx.x;
  if (t >= GROWS * LANES) return;
  const int* __restrict__ cached = lens + 3 * ns;
  const long long total = cached[nseq], base = (long long)w * window;
  const int lane = t % LANES, r0 = blockIdx.x * GROWS * GITER + t / LANES;
  uint4 v[GITER];
#pragma unroll
  for (int k = 0; k < GITER; ++k) {
    const long long s = base + r0 + k * GROWS;
    v[k] = make_uint4(0, 0, 0, 0);
    if (r0 + k * GROWS < n && s < total) {
      int j = 0;
      while (j + 1 < nseq && cached[j + 1] <= s) ++j;
      const long long key = s - cached[j];
      v[k] = GET8(slab + block_table[j * max_pages + key / PAGE] * page_stride + (key % PAGE) * KV_A + lane * 8);
    }
  }
#pragma unroll
  for (int k = 0; k < GITER; ++k)
    if (r0 + k * GROWS < n) *reinterpret_cast<uint4*>(latent_g + (long long)(r0 + k * GROWS) * KV_A + lane * 8) = v[k];
}

// k3_mla_vup_gate.cu's kernel as the absorbed form's blocks: the attention's latent output through W_UV,
// then the gate, read from the fused projection and landed as k3_mla_prep would have.
#ifndef HEADS
#define HEADS 96
#endif
#define NOPE 128
#define LAT 512
#define RB 8      // rows per group
#define GROUPS 4  // row groups per block
#define DS 32     // dv per block
#define JS 8      // j-slices per block
#define JW (LAT / JS)

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The decode attention's KV splits of row b, head h merged as the DSL's reduction kernel merges them
// (its SASS: lane s holds split s, the max and the sum of exp2(l_s - m) by xor butterflies 16 .. 1,
// l = m + lg2(sum), split s scaled by exp2(l_s - l)): row b ran S = ceil(t / ceil(t / bsk[b])) splits of
// its t = ceil(len / 128) tiles. One warp a (row, head); scale[s] for s < S.
#define M_TILE 128
__device__ __forceinline__ int splits_of(const int* __restrict__ row_lens, const int* __restrict__ bsk, int b) {
  const int tiles = (row_lens[b] + M_TILE - 1) / M_TILE, per = (tiles + bsk[b] - 1) / bsk[b];
  return (tiles + per - 1) / per;
}
__device__ __forceinline__ void split_scales(const float* __restrict__ acc_lse, int split_max, int splits, int b, int h,
                                             int lane, float* __restrict__ scale) {
  const float l = lane < splits ? acc_lse[((long long)b * M_TILE + h) * split_max + lane] : -__int_as_float(0x7f800000);
  float m = l;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (m == -__int_as_float(0x7f800000)) m = 0.f;
  float sum = 0.f + ex2(l - m);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float g = m + lg2(sum);
  if (lane < splits) scale[lane] = ex2(l - g);
}

__device__ __forceinline__ void vup_gate_block(const float* __restrict__ acc_o, const float* __restrict__ acc_lse,
                                               const int* __restrict__ row_lens, const int* __restrict__ bsk,
                                               int split_max, const __nv_bfloat16* __restrict__ w_kv_b,
                                               const bf16_t* __restrict__ partial, __nv_bfloat16* __restrict__ gated,
                                               int B, int bx, int h, int dz) {
  __shared__ __align__(16) __nv_bfloat16 lat[RB][LAT];
  __shared__ float red[JS][RB][DS];
  __shared__ float scale[RB][32];
  const int dv0 = dz * DS, t = threadIdx.x;
  const int js = t / DS, dl = t - js * DS, dv = dv0 + dl, j0 = js * JW;
  const uint4* __restrict__ w = reinterpret_cast<const uint4*>(w_kv_b + ((size_t)h * 256 + NOPE + dv) * LAT + j0);
  uint4 wv[JW / 8];
#pragma unroll
  for (int k = 0; k < JW / 8; ++k) wv[k] = w[k];
  for (int g = 0; g < GROUPS; ++g) {
    const int b0 = (bx * GROUPS + g) * RB;
    const int nr = min(RB, B - b0);
    if (nr <= 0) break;
    if (g) __syncthreads();  // the previous group is done reading lat / red / scale
    // warp r: row r's split scales
    if ((t >> 5) < RB) {
      const int b = min(b0 + (t >> 5), B - 1);
      split_scales(acc_lse, split_max, splits_of(row_lens, bsk, b), b, h, t & 31, scale[t >> 5]);
    }
    __syncthreads();
    // RB*LAT/8 / 256 = 2 vectors per thread: the splits' sum in split order, as the reduction runs it
    uint4 lv[RB * (LAT / 8) / 256];
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8), b = min(b0 + r, B - 1);
      const int splits = splits_of(row_lens, bsk, b);
      float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float4* src = reinterpret_cast<const float4*>(acc_o + ((long long)b * M_TILE + h) * split_max * LAT + c * 8);
      for (int sp = 0; sp < splits; ++sp, src += LAT / 4) {
        const float w = scale[r][sp];
        const float4 x = src[0], y = src[1];
        a[0] = fmaf(x.x, w, a[0]), a[1] = fmaf(x.y, w, a[1]), a[2] = fmaf(x.z, w, a[2]), a[3] = fmaf(x.w, w, a[3]);
        a[4] = fmaf(y.x, w, a[4]), a[5] = fmaf(y.y, w, a[5]), a[6] = fmaf(y.z, w, a[6]), a[7] = fmaf(y.w, w, a[7]);
      }
      __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&lv[k]);
#pragma unroll
      for (int q = 0; q < 4; ++q) v2[q] = __floats2bfloat162_rn(a[2 * q], a[2 * q + 1]);
    }
#pragma unroll
    for (int k = 0; k < RB * (LAT / 8) / 256; ++k) {
      const int i = t + k * 256, r = i / (LAT / 8), c = i - r * (LAT / 8);
      reinterpret_cast<uint4*>(&lat[r][0])[c] = r < nr ? lv[k] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    float acc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r] = 0.f;
#pragma unroll
    for (int k = 0; k < JW / 8; ++k) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[k]);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(w2[j]);
        wf[2 * j] = f.x;
        wf[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint4 l = *reinterpret_cast<const uint4*>(&lat[r][j0 + k * 8]);
        const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&l);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(l2[j]);
          acc[r] += wf[2 * j] * f.x + wf[2 * j + 1] * f.y;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) red[js][r][dl] = acc[r];
    __syncthreads();
    for (int i = t; i < nr * DS; i += 256) {
      const int r = i / DS, d = i - r * DS;
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < JS; ++k) a += red[k][r][d];
      const size_t oi = (size_t)(b0 + r) * (HEADS * NOPE) + (size_t)h * NOPE + dv0 + d;
      const float gf = __bfloat162float(partial[(size_t)(b0 + r) * MLA_FUSED + HEADC + (size_t)h * NOPE + dv0 + d]);
      gated[oi] = __hmul(__float2bfloat16_rn(a), __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
    }
  }
}


#define NEG_INF (-__int_as_float(0x7f800000))

extern "C" __global__ void __launch_bounds__(256) kern_k3g_mla_merge(
    const bf16_t* __restrict__ o, const float2* __restrict__ stats, const int* __restrict__ lens, int ns, int nseq,
    float* __restrict__ acc_o, float2* __restrict__ acc_ms, int init, int n, int B) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int b = (int)(i / (n / 8)), c = (int)(i % (n / 8)) * 8, h = c / NOPE;
  if (b >= B || lens[seq_of(lens + ns, nseq, b)] == 0) return;
  const float2 st = stats[(long long)b * HEADS + h];
  const float2 prev = init ? make_float2(NEG_INF, 0.f) : acc_ms[(long long)b * HEADS + h];
  const float m = fmaxf(prev.x, st.x), wa = init ? 0.f : expf(prev.x - m), wp = st.y * expf(st.x - m);
  float* __restrict__ a = acc_o + (long long)b * n + c;
  const uint4 pv = *reinterpret_cast<const uint4*>(o + (long long)b * n + c);
  const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
  float4 x = init ? make_float4(0.f, 0.f, 0.f, 0.f) : *reinterpret_cast<const float4*>(a);
  float4 y = init ? make_float4(0.f, 0.f, 0.f, 0.f) : *reinterpret_cast<const float4*>(a + 4);
  const float2 p0 = __bfloat1622float2(pp[0]), p1 = __bfloat1622float2(pp[1]), p2 = __bfloat1622float2(pp[2]),
               p3 = __bfloat1622float2(pp[3]);
  x = make_float4(fmaf(wp, p0.x, x.x * wa), fmaf(wp, p0.y, x.y * wa), fmaf(wp, p1.x, x.z * wa), fmaf(wp, p1.y, x.w * wa));
  y = make_float4(fmaf(wp, p2.x, y.x * wa), fmaf(wp, p2.y, y.y * wa), fmaf(wp, p3.x, y.z * wa), fmaf(wp, p3.y, y.w * wa));
  *reinterpret_cast<float4*>(a) = x;
  *reinterpret_cast<float4*>(a + 4) = y;
  if (c % NOPE == 0) acc_ms[(long long)b * HEADS + h] = make_float2(m, prev.y * wa + wp);
}

// GATE_ROWS rows a thread, every load issued before the first use: the kernel's registers and shared
// memory are the absorbed form's, two blocks an SM, so one row's loads alone leave HBM idle
#define GATE_ROWS 8

// 1 / d without the IEEE division's slow-path branch, equal to it for every d in [2^-126, 2^126)
// (k3_moe_route.cu); outside, the division
__device__ __forceinline__ float rcp_fast(float d) {
  float q;
  asm("{\n .reg .f32 r, e;\n rcp.approx.ftz.f32 r, %1;\n fma.rn.f32 e, %1, r, 0fBF800000;\n neg.f32 e, e;\n"
      " fma.rn.f32 %0, r, e, r;\n}"
      : "=f"(q)
      : "f"(d));
  return q;
}

// bf16(a * bf16(sigmoid(g))) for 8 columns; FAST: every denominator in rcp_fast's range
template <bool FAST>
__device__ __forceinline__ uint4 gate8(const uint4& gv, const uint4& av) {
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&av);
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 x = __bfloat1622float2(gp[i]);
    const float dx = 1.0f + expf(-x.x), dy = 1.0f + expf(-x.y);
    const __nv_bfloat162 g =
        FAST ? __floats2bfloat162_rn(rcp_fast(dx), rcp_fast(dy)) : __floats2bfloat162_rn(1.0f / dx, 1.0f / dy);
    op[i] = __floats2bfloat162_rn(__bfloat162float(ap[i].x) * __bfloat162float(g.x),
                                  __bfloat162float(ap[i].y) * __bfloat162float(g.y));
  }
  return ov;
}

// sigmoid's denominator 1 + e^-x is in rcp_fast's range for every x >= -87 (e^87 < 2^126), NaN not
__device__ __forceinline__ bool gate_ok(const uint4& gv) {
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  bool ok = true;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 x = __bfloat1622float2(gp[i]);
    ok = ok && x.x >= -87.0f && x.y >= -87.0f;
  }
  return ok;
}

// rows b0 .. b0 + GATE_ROWS of columns j .. j + 8, the merged attention landed to bf16. A warp's whole
// tile with every x in rcp_fast's range runs as straight-line code: no division branch to keep its
// rows apart.
__device__ __forceinline__ void gate_rows(bf16_t* __restrict__ out, const float* __restrict__ acc_o,
                                          const float2* __restrict__ acc_ms, const bf16_t* __restrict__ partial,
                                          int n, int B) {
  const long long b0 = (long long)(blockIdx.x / (n / 2048)) * GATE_ROWS;
  const int j = (blockIdx.x % (n / 2048) * blockDim.x + threadIdx.x) * 8, h = j / NOPE;
  uint4 gv[GATE_ROWS], av[GATE_ROWS];
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r)
    if (b0 + r < B) {
      gv[r] = *reinterpret_cast<const uint4*>(partial + (b0 + r) * MLA_FUSED + HEADC + j);
      const float inv = 1.f / acc_ms[(b0 + r) * HEADS + h].y;
      const float4 x = *reinterpret_cast<const float4*>(acc_o + (b0 + r) * n + j);
      const float4 y = *reinterpret_cast<const float4*>(acc_o + (b0 + r) * n + j + 4);
      av[r] = make_uint4(pack2(x.x * inv, x.y * inv), pack2(x.z * inv, x.w * inv), pack2(y.x * inv, y.y * inv),
                         pack2(y.z * inv, y.w * inv));
    }
  bool ok = b0 + GATE_ROWS <= B;
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r) ok = ok && gate_ok(gv[r]);
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int r = 0; r < GATE_ROWS; ++r) *reinterpret_cast<uint4*>(out + (b0 + r) * n + j) = gate8<true>(gv[r], av[r]);
    return;
  }
#pragma unroll
  for (int r = 0; r < GATE_ROWS; ++r) {
    const long long b = b0 + r;
    if (b >= B) break;
    *reinterpret_cast<uint4*>(out + b * n + j) = gate8<false>(gv[r], av[r]);
  }
}

extern "C" __global__ void __launch_bounds__(256) kern_k3g_mla_gate(
    bf16_t* __restrict__ out, const float* __restrict__ acc_o, const bf16_t* __restrict__ partial,
    const float* __restrict__ s_acc_o, const float* __restrict__ s_acc_lse, const int* __restrict__ row_lens,
    const int* __restrict__ bsk, int split_max, const bf16_t* __restrict__ w_kv_b, const float2* __restrict__ acc_ms,
    int n, int short_max, int B) {
  const int gate_blocks = (B + GATE_ROWS - 1) / GATE_ROWS * (n / 2048);
  if (B <= short_max) {  // a short chunk: o_lat through W_UV, then the gate (blocks past the gate's)
    if (blockIdx.x < gate_blocks) return;
    const int i = blockIdx.x - gate_blocks;
    vup_gate_block(s_acc_o, s_acc_lse, row_lens, bsk, split_max, w_kv_b, partial, out, B, i / (HEADS * 4), i / 4 % HEADS,
                   i % 4);
    return;
  }
  if (blockIdx.x >= gate_blocks) return;
  gate_rows(out, acc_o, acc_ms, partial, n, B);
}

#ifdef KV_FP8
extern "C" __global__ void __launch_bounds__(256) kern_k3g_e4m3(const bf16_t* __restrict__ in, kv_t* __restrict__ out,
                                                               int rows, int width) {
  const long long n8 = (long long)rows * width / 8;
  for (long long i = blockIdx.x * 256ll + threadIdx.x; i < n8; i += gridDim.x * 256ll) {
    const uint4 v = reinterpret_cast<const uint4*>(in)[i];
    reinterpret_cast<uint2*>(out)[i] = make_uint2(fp8x2(v.x) | fp8x2(v.y) << 16, fp8x2(v.z) | fp8x2(v.w) << 16);
  }
}
#endif
