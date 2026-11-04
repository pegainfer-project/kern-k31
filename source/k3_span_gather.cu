// K9 `k3_span_gather` — the K2 conv + SiLU for a span: batch rows
// [*span_at, *span_at + span) are consecutive tokens of one sequence, so the
// conv taps of span row i are the sequence's window (the line of the span's
// first row) for i < 3 and the span's own earlier rows after that; the window
// leaves holding the span's last three inputs. The results land in the span's
// own buffers, rows 0..span, where the FlashKDA span kernel's descriptors
// (fixed at load) find them, together with what it wants transposed: beta as
// [HEADS][span] and the f_a flow as bf16 [span][128] (the f_b GEMM's input).
// Contract: docs/k3-kernel-abi.md section K9.
//
//   extern "C" __global__ void kern_k3_span_gather(
//       const part* partial,       // [rows, LDP]  q | k | v | g, rows at..at+span read; f32, or bf16 with
//                                  // -DPARTIAL_BF16 (the GEMM landed it, so bf16() below is exact)
//       const float* cw,           // [3 stream][4 tap][INNER]
//       void* kda_base, const int* line_index, long long line_bytes,  // line_index[at]'s line
//       const part* wsm_partial,   // [rows, LDW]  col h = b_proj, 96.. = f_a
//         (LDP = KDA_FUSED, LDW = WSM; both K9_LD when one projection made q | k | v | g | wsm)
//       bf16* span_q, bf16* span_k, bf16* span_v,   // [span, INNER]
//       bf16* span_beta,           // [HEADS * span]   h*span + i
//       bf16* span_flow,           // [span, 128]
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid  (INNER/512, 4, ceil(span/8))   block 128   smem 0
//   blockIdx.y < 3: stream y, 4 consecutive columns per thread, rows
//   8*blockIdx.z .. +8;  blockIdx.y == 3: beta / flow for the same 8 rows.
//
// Per stream s, column c, with x_{-3..-1} = win_s[0..2][c] and
// x_i = bf16(partial[i, s*INNER + c]) for 0 <= i < span:
//   y_i        = sum_{t<3} f32(x_{i-3+t}) * cw[s][t][c] + f32(x_i) * cw[s][3][c]
//   sb         = bf16(y_i);  out_s[i, c] = bf16(sb * sigmoid(sb))
//   win_s[t][c] = x_{span-3+t}                                   (t < 3)
// which is exactly K2 applied to the rows one after another (same landing
// points), so a span of n tokens leaves the window as n decode steps would.
// Rows are independent given the inputs, so they run in parallel; only the
// z == 0 block reads the old window (rows 0..2 need it) and only it writes
// the new one, after its reads, so no two blocks touch the window.
#include <cuda_bf16.h>
#include <climits>

#ifndef HEADS
#define HEADS 96
#endif
#define K9_INNER (HEADS * 128)
#define K9_KDA_FUSED (4 * K9_INNER)
#ifndef KDA_STATE_BF16  // the line's recurrent state ahead of the windows: f32, or bf16 (k3_kda_rec.cu's)
#define KDA_STATE_BF16 0
#endif
#define K9_REC_BYTES ((long long)HEADS * 128 * 128 * (KDA_STATE_BF16 ? 2 : 4))
#define K9_WIN_BYTES ((long long)3 * K9_INNER * 2)
#define K9_WSM 256
#define K9_WSM_FA 96
#ifdef K9_LD  // q | k | v | g and wsm as one projection: both slices of its K9_LD-wide rows
#define K9_LDP K9_LD
#define K9_LDW K9_LD
#else
#define K9_LDP K9_KDA_FUSED
#define K9_LDW K9_WSM
#endif
#define K9_BLOCK 128
#define K9_VEC 4
#define K9_ROWS 8

typedef __nv_bfloat16 bf16;

// -DPARTIAL_BF16: the projections arrive landed (a bf16 GEMM output), the
// value the f32 form rounds to first.
#ifdef PARTIAL_BF16
typedef bf16 part_t;
__device__ __forceinline__ bf16 k9_land(bf16 x) { return x; }
#else
typedef float part_t;
__device__ __forceinline__ bf16 k9_land(float x) { return __float2bfloat16(x); }
#endif

__device__ __forceinline__ float k9_silu_bf16(float y) {
  float sb = __bfloat162float(__float2bfloat16(y));
  return sb / (1.0f + __expf(-sb));
}

__device__ __forceinline__ void k9_unpack(uint2 raw, float* x) {
  x[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x & 0xffffu)));
  x[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x >> 16)));
  x[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y & 0xffffu)));
  x[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y >> 16)));
}

__device__ __forceinline__ void k9_pack(const float* x, bf16* o) {
#pragma unroll
  for (int k = 0; k < K9_VEC; ++k) o[k] = __float2bfloat16(x[k]);
}

// x_i for i in [-3, span): the window for i < 0 (z == 0 blocks only), the
// bf16 landing of the partial otherwise.
__device__ __forceinline__ void k9_input(const part_t* __restrict__ partial, const bf16* win, int s, int c, int i,
                                         float* x) {
  if (i < 0) {
    k9_unpack(*(const uint2*)(win + (size_t)(i + 3) * K9_INNER + c), x);
  } else {
    const part_t* p = partial + (size_t)i * K9_LDP + (size_t)s * K9_INNER + c;
#ifdef PARTIAL_BF16
    k9_unpack(*(const uint2*)p, x);
#else
    const float4 v = *(const float4*)p;
    x[0] = __bfloat162float(__float2bfloat16(v.x));
    x[1] = __bfloat162float(__float2bfloat16(v.y));
    x[2] = __bfloat162float(__float2bfloat16(v.z));
    x[3] = __bfloat162float(__float2bfloat16(v.w));
#endif
  }
}

extern "C" __global__ __launch_bounds__(K9_BLOCK) void kern_k3_span_gather(
    const part_t* __restrict__ partial,
    const float* __restrict__ cw,
    void* __restrict__ kda_base,
    const int* __restrict__ line_index,
    long long line_bytes,
    const part_t* __restrict__ wsm_partial,
    bf16* __restrict__ span_q, bf16* __restrict__ span_k, bf16* __restrict__ span_v,
    bf16* __restrict__ span_beta,
    bf16* __restrict__ span_flow,
    const int* __restrict__ span_at,
    int span) {
  const int at = span_at[0];
  const int row0 = blockIdx.z * K9_ROWS;
  partial += (size_t)at * K9_LDP;
  wsm_partial += (size_t)at * K9_LDW;
  if (blockIdx.y == 3) {
    // 16 threads per row: thread k writes flow[8k..8k+8) and beta for heads k, k+16, ...
    const int i = row0 + (threadIdx.x >> 4), k = threadIdx.x & 15;
    if (i >= span) return;
    const part_t* row = wsm_partial + (size_t)i * K9_LDW;
    for (int h = k; h < HEADS; h += 16) span_beta[(size_t)h * span + i] = k9_land(row[h]);
    bf16 f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = k9_land(row[K9_WSM_FA + k * 8 + j]);
    *(uint4*)(span_flow + (size_t)i * 128 + k * 8) = *(const uint4*)f;
    return;
  }
  const int s = blockIdx.y;
  const int c = (int)(blockIdx.x * K9_BLOCK + threadIdx.x) * K9_VEC;
  bf16* __restrict__ out = s == 0 ? span_q : s == 1 ? span_k : span_v;
  bf16* win = (bf16*)((char*)kda_base + (long long)line_index[at] * line_bytes + K9_REC_BYTES +
                      (long long)s * K9_WIN_BYTES);
  const float* __restrict__ w = cw + (size_t)s * 4 * K9_INNER + c;
  const float4 w0 = *(const float4*)(w), w1 = *(const float4*)(w + K9_INNER),
               w2 = *(const float4*)(w + 2 * K9_INNER), w3 = *(const float4*)(w + 3 * K9_INNER);
  const float wt[4][K9_VEC] = {{w0.x, w0.y, w0.z, w0.w}, {w1.x, w1.y, w1.z, w1.w},
                               {w2.x, w2.y, w2.z, w2.w}, {w3.x, w3.y, w3.z, w3.w}};

  // The block's rows plus the three before them, in a sliding register set;
  // the rows' loads all issue before the first one is used.
  float t0[K9_VEC], t1[K9_VEC], t2[K9_VEC];
  k9_input(partial, win, s, c, row0 - 3, t0);
  k9_input(partial, win, s, c, row0 - 2, t1);
  k9_input(partial, win, s, c, row0 - 1, t2);
  const int rows = min(K9_ROWS, span - row0);
  float xs[K9_ROWS][K9_VEC];
#pragma unroll
  for (int r = 0; r < K9_ROWS; ++r)
    if (r < rows) k9_input(partial, win, s, c, row0 + r, xs[r]);
#pragma unroll
  for (int r = 0; r < K9_ROWS; ++r) {
    if (r >= rows) break;
    const float* x = xs[r];
    bf16 o[K9_VEC];
#pragma unroll
    for (int k = 0; k < K9_VEC; ++k) {
      const float y = t0[k] * wt[0][k] + t1[k] * wt[1][k] + t2[k] * wt[2][k] + x[k] * wt[3][k];
      o[k] = __float2bfloat16(k9_silu_bf16(y));
      t0[k] = t1[k];
      t1[k] = t2[k];
      t2[k] = x[k];
    }
    *(uint2*)(out + (size_t)(row0 + r) * K9_INNER + c) = *(const uint2*)o;
  }
  if (blockIdx.z == 0) {
    // New window = x_{span-3..span-1}; every old tap this thread needs it
    // already read above, so the writes race with nothing.
    float nt[3][K9_VEC];
    for (int t = 0; t < 3; ++t) k9_input(partial, win, s, c, span - 3 + t, nt[t]);
    for (int t = 0; t < 3; ++t) {
      bf16 o[K9_VEC];
      k9_pack(nt[t], o);
      *(uint2*)(win + (size_t)t * K9_INNER + c) = *(const uint2*)o;
    }
  }
}

// The packed call's KDA prologue and epilogue around FlashKDA: rows
// [cu_seqlens[j], cu_seqlens[j + 1]) are sequence j's, its KDA line
// line_index[j]. FlashKDA's descriptors are fixed at load, so its inputs land
// in the span buffers and its f32 state is staged in a flat buffer
// ([nseq][HEADS][128][128]) on the way in and out.
//
// `kern_k3_span_gather_packed`: K9's conv + SiLU of every row (each sequence
// reads its own window, left untouched here: `kern_k3_kda_rec` advances it),
// beta / flow transposed, and FlashKDA's tile prefix (16-row tiles per
// sequence, upstream's `_flash_kda_build_tile_prefix`).
//
//   kern_k3_span_gather_packed(partial, cw, kda_base, line_index, line_bytes, wsm_partial,
//                              span_q, span_k, span_v, span_beta, span_flow,
//                              const long long* cu_seqlens, int nseq, int span, int* tile_prefix,
//                              int* progress);
//   grid (37 * ceil(span / 48) + 1)   block 128
//   block b < 37 * ceil(span / 48): rows 48 * (b / 37) ..; b % 37 < 36 is stream (b % 37) / 12,
//   columns 1024 * (b % 12) + 8 * thread; b % 37 == 36 beta / flow. The last block writes
//   the tile prefix and zeroes k3_kda_rec's progress counters ([nseq][HEADS], one a 128-byte line).
#ifdef PARTIAL_BF16
#define K9_PROWS 48
#define K9_PUNROLL 12
#define K9_PVEC 8
#define K9_PCOLS (K9_BLOCK * K9_PVEC)
#define K9_PCOL_BLOCKS (K9_INNER / K9_PCOLS)
#define K9_PCONV (3 * K9_PCOL_BLOCKS + 1)
#define K9_TILE 16
#define K9_PROGRESS_STRIDE 32  // k3_kda_rec.cu K12_PROGRESS_STRIDE

__device__ __forceinline__ int k9_seq_of(const long long* cu, int nseq, int i) {
  int j = 0;
  while (j + 1 < nseq && cu[j + 1] <= i) ++j;
  return j;
}

__device__ __forceinline__ bf16* k9_win(void* kda_base, const int* line_index, long long line_bytes, int j, int s) {
  return (bf16*)((char*)kda_base + (long long)line_index[j] * line_bytes + K9_REC_BYTES + (long long)s * K9_WIN_BYTES);
}

__device__ __forceinline__ void k9_unpack8(uint4 raw, float* x) {
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    x[2 * k] = __uint_as_float(w[k] << 16);
    x[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
  }
}


// x_i of stream s, 8 columns from c: the window before the sequence's first row bos.
__device__ __forceinline__ uint4 k9_tap(const part_t* __restrict__ partial, const bf16* win, int bos, int s, int c,
                                        int i) {
  return i < bos ? *(const uint4*)(win + (size_t)(i - bos + 3) * K9_INNER + c)
                 : *(const uint4*)(partial + (size_t)i * K9_LDP + (size_t)s * K9_INNER + c);
}

__device__ __forceinline__ void k9_cp_async(unsigned smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n cp.async.commit_group;" ::"r"(smem), "l"(gmem));
}

// K9's SiLU of two columns, the same value without the IEEE division's per-element
// slow-path branch (its reconvergence serializes a thread's columns):
// exp(-sb) by ex2.ftz (__expf's value wherever it is not subnormal, and then
// 1 + it is 1 either way), the quotient by one reciprocal and two Newton
// steps over a denominator scaled by 2^-64, correctly rounded, scaled back
// exactly while it stays normal (down to sb = -88, or an f32-subnormal sb).
// The negated residual keeps a zero's sign; an infinite denominator is -0. The f32 operations
// run two at a time (f32x2: each lane rounds as the scalar operation does).
__device__ __forceinline__ unsigned k9_silu2(float2 y) {
  const __nv_bfloat162 yb = __float22bfloat162_rn(y);
  const float2 sb = __bfloat1622float2(yb);
  const float2 a = __fmul2_rn(sb, make_float2(-1.4426950408889634f, -1.4426950408889634f));
  float2 e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e.x) : "f"(a.x));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e.y) : "f"(a.y));
  const float2 d = __fadd2_rn(make_float2(1.0f, 1.0f), e), ds = __fmul2_rn(d, make_float2(0x1p-64f, 0x1p-64f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r.x) : "f"(ds.x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r.y) : "f"(ds.y));
  const float2 nds = make_float2(-ds.x, -ds.y);
  r = __ffma2_rn(r, __ffma2_rn(nds, r, make_float2(1.0f, 1.0f)), r);
  const float2 q = __fmul2_rn(sb, r);
  const float2 res = __ffma2_rn(make_float2(-r.x, -r.y), __ffma2_rn(ds, q, make_float2(-sb.x, -sb.y)), q);
  const float2 qs = __fmul2_rn(res, make_float2(0x1p-64f, 0x1p-64f));
  const __nv_bfloat162 o =
      __floats2bfloat162_rn(d.x == INFINITY ? -0.0f : qs.x, d.y == INFINITY ? -0.0f : qs.y);
  return *(const unsigned*)&o;
}

// One row of 8 columns: conv over the taps and x (K9's contraction, x w3 + (t2 w2 + (t0 w0 +
// t1 w1))), SiLU, landed; x becomes the newest tap.
__device__ __forceinline__ uint4 k9_conv_row(uint4 raw, const float (&wt)[4][K9_PVEC], float* t0, float* t1,
                                             float* t2) {
  float x[K9_PVEC];
  k9_unpack8(raw, x);
  uint4 o;
  unsigned* ow = (unsigned*)&o;
  const auto f2 = [](const float* v, int k) { return make_float2(v[k], v[k + 1]); };
#pragma unroll
  for (int k = 0; k < K9_PVEC; k += 2) {
    const float2 y = __ffma2_rn(f2(x, k), f2(wt[3], k),
                                __ffma2_rn(f2(t2, k), f2(wt[2], k),
                                           __ffma2_rn(f2(t0, k), f2(wt[0], k), __fmul2_rn(f2(t1, k), f2(wt[1], k)))));
    ow[k / 2] = k9_silu2(y);
  }
#pragma unroll
  for (int k = 0; k < K9_PVEC; ++k) {
    t0[k] = t1[k];
    t1[k] = t2[k];
    t2[k] = x[k];
  }
  return o;
}

extern "C" __global__ __launch_bounds__(K9_BLOCK) void kern_k3_span_gather_packed(
    const part_t* __restrict__ partial,
    const float* __restrict__ cw,
    void* __restrict__ kda_base,
    const int* __restrict__ line_index,
    long long line_bytes,
    const part_t* __restrict__ wsm_partial,
    bf16* __restrict__ span_q, bf16* __restrict__ span_k, bf16* __restrict__ span_v,
    bf16* __restrict__ span_beta,
    bf16* __restrict__ span_flow,
    const long long* __restrict__ cu_seqlens,
    int nseq,
    int span,
    int* __restrict__ tile_prefix,
    int* __restrict__ progress) {
  const int nconv = (span + K9_PROWS - 1) / K9_PROWS * K9_PCONV;
  const int b = blockIdx.x;
  if (b >= nconv) {
    for (int i = threadIdx.x; i < nseq * HEADS; i += K9_BLOCK) progress[K9_PROGRESS_STRIDE * i] = 0;
    if (threadIdx.x == 0) {
      int acc = 0;
      tile_prefix[0] = 0;
      for (int j = 0; j < nseq; ++j) {
        acc += (int(cu_seqlens[j + 1] - cu_seqlens[j]) + K9_TILE - 1) / K9_TILE;
        tile_prefix[j + 1] = acc;
      }
    }
    return;
  }
  const int row0 = b / K9_PCONV * K9_PROWS, role = b % K9_PCONV;
  const int rows = min(K9_PROWS, span - row0);
  if (role == 3 * K9_PCOL_BLOCKS) {
    // beta: 32 rows of a head per warp pass; flow: a row's 128 columns per 16 threads
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int i = lane; i < rows; i += 32)
      for (int h = w; h < HEADS; h += 4)
        span_beta[(size_t)h * span + row0 + i] = k9_land(wsm_partial[(size_t)(row0 + i) * K9_LDW + h]);
    for (int r = threadIdx.x >> 4; r < rows; r += K9_BLOCK / 16) {
      const int k = threadIdx.x & 15;
      const part_t* row = wsm_partial + (size_t)(row0 + r) * K9_LDW + K9_WSM_FA + k * 8;
      bf16 f[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) f[e] = k9_land(row[e]);
      *(uint4*)(span_flow + (size_t)(row0 + r) * 128 + k * 8) = *(const uint4*)f;
    }
    return;
  }
  const int s = role / K9_PCOL_BLOCKS;
  const int c = (role % K9_PCOL_BLOCKS) * K9_PCOLS + threadIdx.x * K9_PVEC;
  bf16* __restrict__ out = s == 0 ? span_q : s == 1 ? span_k : span_v;
  const float* __restrict__ w = cw + (size_t)s * 4 * K9_INNER + c;
  float wt[4][K9_PVEC];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const float4 lo = *(const float4*)(w + t * K9_INNER), hi = *(const float4*)(w + t * K9_INNER + 4);
    wt[t][0] = lo.x; wt[t][1] = lo.y; wt[t][2] = lo.z; wt[t][3] = lo.w;
    wt[t][4] = hi.x; wt[t][5] = hi.y; wt[t][6] = hi.z; wt[t][7] = hi.w;
  }
  int j = k9_seq_of(cu_seqlens, nseq, row0);
  int bos = (int)cu_seqlens[j];
  int next = j + 1 < nseq ? (int)cu_seqlens[j + 1] : INT_MAX;
  const bf16* win = k9_win(kda_base, line_index, line_bytes, j, s);
  float t0[K9_PVEC], t1[K9_PVEC], t2[K9_PVEC];
  k9_unpack8(k9_tap(partial, win, bos, s, c, row0 - 3), t0);
  k9_unpack8(k9_tap(partial, win, bos, s, c, row0 - 2), t1);
  k9_unpack8(k9_tap(partial, win, bos, s, c, row0 - 1), t2);
  const part_t* src = partial + (size_t)row0 * K9_LDP + (size_t)s * K9_INNER + c;
  bf16* dst = out + (size_t)row0 * K9_INNER + c;
  if (rows == K9_PROWS && row0 + K9_PROWS <= next) {
    // The common block: whole, inside one sequence. Each thread streams its
    // rows through its own ring of K9_PUNROLL shared slots (cp.async, one
    // group a row), so that many loads stay in flight; the taps rotate by
    // register renaming.
    __shared__ uint4 ring[K9_PUNROLL][K9_BLOCK];
    const unsigned slot0 = (unsigned)__cvta_generic_to_shared(&ring[0][threadIdx.x]);
#pragma unroll
    for (int r = 0; r < K9_PUNROLL; ++r) k9_cp_async(slot0 + r * K9_BLOCK * 16, src + (size_t)r * K9_LDP);
    for (int r0 = 0; r0 < K9_PROWS; r0 += K9_PUNROLL) {
#pragma unroll
      for (int r = 0; r < K9_PUNROLL; ++r) {
        asm volatile("cp.async.wait_group %0;" ::"n"(K9_PUNROLL - 1));
        const uint4 x = ring[r][threadIdx.x];
        if (r0 + K9_PUNROLL + r < K9_PROWS)
          k9_cp_async(slot0 + r * K9_BLOCK * 16, src + (size_t)(r0 + K9_PUNROLL + r) * K9_LDP);
        else
          asm volatile("cp.async.commit_group;");
        *(uint4*)(dst + (size_t)(r0 + r) * K9_INNER) = k9_conv_row(x, wt, t0, t1, t2);
      }
    }
    return;
  }
  for (int r = 0; r < rows; ++r) {
    const int i = row0 + r;
    if (i >= next) {
      while (j + 1 < nseq && i >= cu_seqlens[j + 1]) ++j;
      bos = i;
      next = j + 1 < nseq ? (int)cu_seqlens[j + 1] : INT_MAX;
      win = k9_win(kda_base, line_index, line_bytes, j, s);
      k9_unpack8(k9_tap(partial, win, bos, s, c, i - 3), t0);
      k9_unpack8(k9_tap(partial, win, bos, s, c, i - 2), t1);
      k9_unpack8(k9_tap(partial, win, bos, s, c, i - 1), t2);
    }
    *(uint4*)(dst + (size_t)r * K9_INNER) = k9_conv_row(*(const uint4*)(src + (size_t)r * K9_LDP), wt, t0, t1, t2);
  }
}
#endif
