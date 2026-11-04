// `k3_kda_rec` — the packed prefill's KDA recurrence and output gate in one
// launch: FlashKDA's kernel 2 (source/flash-kda, `_flash_kda_fwd_recurrence`)
// over its kernel 1's workspace, operation for operation, with the state read
// from and written to each sequence's KDA line, then K11 (k3_kda_out_gate.cu)
// on every output row, and the conv windows advanced (K9's last step).
//
//   extern "C" __global__ void kern_k3_kda_rec(
//       CUtensorMap tm_kd, tm_qd, tm_kr,  // [HEADS * tiles * 16 rows][128] bf16 as (64, rows, 2 halves), box
//                                         // 64 x 16 x 2, 128B swizzle
//       CUtensorMap tm_v,                 // span_v [rows][INNER] as (64, rows, HEADS, 2), box 64 x 16 x 1 x 2
//       CUtensorMap tm_inv, tm_mqk,       // [HEADS * tiles * 16 rows][16] bf16, box 16 x 16
//       const bf16* v,          // [rows, INNER]  conv'd v (span_v), a partial tile's rows
//       const bf16* beta,       // [HEADS, span]  beta logits (span_beta)
//       const float* ws_gt,     // [HEADS * tiles][128]
//       void* kda_base, const int* line_index, long long line_bytes,  // sequence j's line line_index[j]
//       const bf16* partial,    // [rows, K12_LD]  q | k | v | gate projections (then wsm's, if K12_LD > 4 * INNER)
//       const float* gamma_o,   // [128]
//       bf16* raw,              // [rows, INNER]  the ungated output (q's buffer, which kernel 1 has read)
//       bf16* gated,            // [rows, INNER]  rows span_at[0] + i
//       const int* span_at,
//       const long long* cu_seqlens, const int* tile_prefix,  // sequence j: rows cu[j]..cu[j+1], tiles from prefix[j]
//       int* progress,          // [nseq][HEADS][32]  element 0 zero (the gather zeroes it)
//       int tiles,              // the workspace's head stride (kernel 1's total_tiles)
//       int span,               // rows of the call (beta's row stride)
//       int nseq);
//
//   grid (nseq * HEADS + K12_GATE_CTAS)   block 512   dynamic smem sizeof(K12Smem) = 182272
//
// CTA j * HEADS + h runs sequence j's head h. Warps 0-7 run the recurrence,
// warp w the 16 value columns 16w..16w+16 of the head's state, held as f32 in
// registers and rounded to bf16 as the A operand of mma.m16n8k16: FlashKDA's products transposed
// (S [dv][dk] times kd^T instead of kd times S^T), which sums the same products
// in the same k order per output element. Warps 8 and 10 stream each 16-row
// tile's inputs into an eight-stage ring (TMA, 128-byte swizzle so ldmatrix is
// conflict-free); warp 9 advances the windows, then copies every output tile
// to `raw` and publishes them four at a time (`progress` = tiles done, st.release). The last
// K12_GATE_CTAS CTAs, on the SMs one sequence's heads leave idle, gate the
// rows (K11 to the bit) as they are published (ld.acquire); the recurrence
// never waits for them, so any residency order makes progress. (Gate CTAs
// first in the grid puts the recurrence on other SMs and costs 2 ms an item.)
// FlashKDA's roundings are kept, but for the state's: it stays f32 between
// tiles (FlashKDA rounds it to bf16 after each). u and the output in bf16
// (HADD2 / HMUL2), beta = bf16(sigmoid by tanh.approx).
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef HEADS
#define HEADS 96
#endif
#define K12_INNER (HEADS * 128)
#define K12_FUSED (4 * K12_INNER)
#ifndef K12_LD  // partial's row stride: q | k | v | gate, then wsm when one projection made both
#define K12_LD K12_FUSED
#endif
// -DKDA_STATE_BF16=1: the recurrent state is stored bf16 (vLLM's mamba_ssm_cache_dtype bfloat16); it is
// f32 in registers between tiles either way.
#ifndef KDA_STATE_BF16
#define KDA_STATE_BF16 0
#endif
#if KDA_STATE_BF16
typedef __nv_bfloat16 rec_t;
__device__ __forceinline__ float2 rec_ld(const rec_t* p) { return __bfloat1622float2(*(const __nv_bfloat162*)p); }
__device__ __forceinline__ void rec_st(rec_t* p, float2 v) { *(__nv_bfloat162*)p = __float22bfloat162_rn(v); }
#else
typedef float rec_t;
__device__ __forceinline__ float2 rec_ld(const rec_t* p) { return *(const float2*)p; }
__device__ __forceinline__ void rec_st(rec_t* p, float2 v) { *(float2*)p = v; }
#endif
#define K12_REC_BYTES ((long long)HEADS * 128 * 128 * sizeof(rec_t))
#define K12_WIN_BYTES ((long long)3 * K12_INNER * 2)
#define K12_STAGES 8
#define K12_OSTAGES 4
#define K12_ROWS_MAX 8192 // a sequence's rows: its betas stay in shared memory
#define K12_ROW 272   // a 128-column bf16 row, padded
#define K12_MMA_WARPS 8
#define K12_THREADS 512  // a recurrence CTA's 8 MMA warps, loader and store; a gate CTA's 16 gate warps
#define K12_GATE_CTAS 56 // the GB300's 152 SMs less one sequence's 96 heads
#define K12_RMS_EPS 1e-5f
#define K12_PROGRESS_STRIDE 32  // ints: each (sequence, head)'s counter on its own 128-byte line

typedef __nv_bfloat16 bf16;

// A 16-row tile of 128 bf16 columns as one TMA box lands it with the 128-byte
// swizzle: two halves of 64 columns, 16-byte chunk c of row r at c ^ (r % 8).
struct K12Stage {
  alignas(1024) unsigned char kd[4096];
  alignas(1024) unsigned char qd[4096];
  alignas(1024) unsigned char kr[4096];
  alignas(1024) unsigned char v[4096];
  alignas(128) unsigned char inv[16 * 32];
  alignas(128) unsigned char mqk[16 * 32];
  alignas(128) float gt[128];
};

struct K12Smem {
  K12Stage in[K12_STAGES];
  alignas(128) unsigned char out[K12_OSTAGES][16 * K12_ROW];
  bf16 beta[K12_ROWS_MAX];  // bf16(sigmoid(logit)), the sequence's rows
  uint64_t full[K12_STAGES], empty[K12_STAGES], ofull[K12_OSTAGES], oempty[K12_OSTAGES];
};

// the launch's dynamic shared memory (scripts/varlen_abi.py REC_SMEM), 1024-aligned by the kernel
static_assert(sizeof(K12Smem) == 182272, "K12Smem");

// A CUtensorMap by value (the manifest's bytes<128> tensormap field).
struct alignas(64) K12Tmap {
  unsigned char bytes[128];
};

__device__ __forceinline__ unsigned k12_smem(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void k12_bar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(k12_smem(bar)), "r"(count));
}

__device__ __forceinline__ void k12_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(k12_smem(bar)) : "memory");
}

__device__ __forceinline__ void k12_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(k12_smem(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void k12_wait(uint64_t* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n"
      "wait_%=:\n"
      " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra wait_%=;\n}" ::"r"(k12_smem(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void k12_bulk(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   k12_smem(dst)),
               "l"(src), "r"(bytes), "r"(k12_smem(bar))
               : "memory");
}

__device__ __forceinline__ void k12_tma(void* dst, const K12Tmap& map, int x, int y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(k12_smem(dst)),
      "l"(&map), "r"(x), "r"(y), "r"(k12_smem(bar))
      : "memory");
}

// byte offset of row `row`, 16-byte chunk `chunk` (of 16) in a swizzled tile
__device__ __forceinline__ unsigned k12_sw(int row, int chunk) {
  return (chunk >> 3) * 2048 + row * 128 + (((chunk & 7) ^ (row & 7)) << 4);
}

__device__ __forceinline__ void k12_tma3(void* dst, const K12Tmap& map, int x, int y, int z, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];" ::"r"(k12_smem(dst)),
      "l"(&map), "r"(x), "r"(y), "r"(z), "r"(k12_smem(bar))
      : "memory");
}

__device__ __forceinline__ void k12_tma4(void* dst, const K12Tmap& map, int x, int y, int z, int w, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
      "%5}], [%6];" ::"r"(k12_smem(dst)),
      "l"(&map), "r"(x), "r"(y), "r"(z), "r"(w), "r"(k12_smem(bar))
      : "memory");
}

__device__ __forceinline__ void k12_ldsm(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void k12_ldsm_t(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void k12_stsm_t(unsigned addr, const unsigned (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3])
               : "memory");
}

// d = a * b (+ d), m16n8k16, bf16 in, f32 accumulate
__device__ __forceinline__ void k12_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned k12_pack(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *(const unsigned*)&p;
}

__device__ __forceinline__ __nv_bfloat162 k12_b2(unsigned x) { return *(const __nv_bfloat162*)&x; }
__device__ __forceinline__ unsigned k12_u(__nv_bfloat162 x) { return *(const unsigned*)&x; }
__device__ __forceinline__ float k12_lo(unsigned x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float k12_hi(unsigned x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ float k12_fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// FlashKDA's sigmoid_tanh_approx_f32 as its fast-math build runs it
__device__ __forceinline__ float k12_sigmoid(float x) {
  float h, th;
  asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(x));
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
  return k12_fma_ftz(th, 0.5f, 0.5f);
}

// 1 / x as IEEE division computes it (its fast path: reciprocal, two Newton
// steps) for x in [1, 2^126); the caller takes a larger x by `/`.
__device__ __forceinline__ float k12_recip(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  r = fmaf(r, fmaf(-x, r, 1.0f), r);
  return fmaf(r, fmaf(-x, r, 1.0f), r);
}

// The state, every value column of this warp: a[kb] the A fragment of dk 16kb..16kb+16.
struct K12State {
  float2 a[8][4];
};

__device__ __forceinline__ void k12_recurrence(K12Smem& sm, const rec_t* rec_in, rec_t* rec_out, int len) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int m = lane >> 3, r = lane & 7;
  K12State S;
  {
    const rec_t* row0 = rec_in + (size_t)(16 * w + g) * 128 + 2 * t;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const float2 a0 = rec_ld(row0 + 16 * kb), a1 = rec_ld(row0 + 8 * 128 + 16 * kb);
      const float2 a2 = rec_ld(row0 + 16 * kb + 8), a3 = rec_ld(row0 + 8 * 128 + 16 * kb + 8);
      S.a[kb][0] = a0;
      S.a[kb][1] = a1;
      S.a[kb][2] = a2;
      S.a[kb][3] = a3;
    }
  }
  // per-lane ldmatrix rows: 8(m/2)+r for kd / qd / v / inv / mqk / out, 8(m%2)+r for kr (transposed)
  const int row_a = 8 * (m >> 1) + r, row_t = 8 * (m & 1) + r;
  const unsigned off_sq = row_a * 32 + 16 * (m & 1);
  const unsigned off_out = row_a * K12_ROW + 16 * (m & 1) + 32 * w;
  const int tiles = (len + 15) >> 4;
  for (int c = 0; c < tiles; ++c) {
    const int s = c % K12_STAGES, o = c % K12_OSTAGES;
    K12Stage& st = sm.in[s];
    k12_wait(&sm.full[s], (c / K12_STAGES) & 1);
    // Phase 1: u^T = S kd^T, out^T = S qd^T over dk in FlashKDA's k order
    float u[2][4] = {}, q[2][4] = {};
    const unsigned kd = k12_smem(st.kd), qd = k12_smem(st.qd);
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      unsigned bk[4], bq[4];
      const unsigned off = k12_sw(row_a, 2 * kb + (m & 1));
      k12_ldsm(bk, kd + off);
      k12_ldsm(bq, qd + off);
      const unsigned sa[4] = {k12_pack(S.a[kb][0].x, S.a[kb][0].y), k12_pack(S.a[kb][1].x, S.a[kb][1].y),
                              k12_pack(S.a[kb][2].x, S.a[kb][2].y), k12_pack(S.a[kb][3].x, S.a[kb][3].y)};
      k12_mma(u[0], sa, bk[0], bk[1]);
      k12_mma(u[1], sa, bk[2], bk[3]);
      k12_mma(q[0], sa, bq[0], bq[1]);
      k12_mma(q[1], sa, bq[2], bq[3]);
    }
    // Phase 2: u = (v - bf16(u)) * beta in bf16; the pairs run over two rows i at one dv
    unsigned vv[4];
    k12_ldsm_t(vv, k12_smem(st.v) + k12_sw(row_a, 2 * w + (m & 1)));
    unsigned ua[4];
    {
      const unsigned b0 = *(const unsigned*)&sm.beta[16 * c + 2 * t], b1 = *(const unsigned*)&sm.beta[16 * c + 8 + 2 * t];
      const unsigned uc[4] = {k12_pack(u[0][0], u[0][1]), k12_pack(u[0][2], u[0][3]), k12_pack(u[1][0], u[1][1]),
                              k12_pack(u[1][2], u[1][3])};
#pragma unroll
      for (int k = 0; k < 4; ++k)
        ua[k] = k12_u(__hmul2(__hsub2(k12_b2(vv[k]), k12_b2(uc[k])), k12_b2(k < 2 ? b0 : b1)));
    }
    // Phase 3: U^T = u^T INV^T
    unsigned U[4];
    {
      unsigned bi[4];
      k12_ldsm(bi, k12_smem(st.inv) + off_sq);
      float x[2][4] = {};
      k12_mma(x[0], ua, bi[0], bi[1]);
      k12_mma(x[1], ua, bi[2], bi[3]);
      U[0] = k12_pack(x[0][0], x[0][1]);
      U[1] = k12_pack(x[0][2], x[0][3]);
      U[2] = k12_pack(x[1][0], x[1][1]);
      U[3] = k12_pack(x[1][2], x[1][3]);
    }
    // Phase 4: out = bf16(Mqk U) + bf16(q S)
    unsigned out[4];
    {
      unsigned bm[4];
      k12_ldsm(bm, k12_smem(st.mqk) + off_sq);
      float x[2][4] = {};
      k12_mma(x[0], U, bm[0], bm[1]);
      k12_mma(x[1], U, bm[2], bm[3]);
      out[0] = k12_u(__hadd2(k12_b2(k12_pack(x[0][0], x[0][1])), k12_b2(k12_pack(q[0][0], q[0][1]))));
      out[1] = k12_u(__hadd2(k12_b2(k12_pack(x[0][2], x[0][3])), k12_b2(k12_pack(q[0][2], q[0][3]))));
      out[2] = k12_u(__hadd2(k12_b2(k12_pack(x[1][0], x[1][1])), k12_b2(k12_pack(q[1][0], q[1][1]))));
      out[3] = k12_u(__hadd2(k12_b2(k12_pack(x[1][2], x[1][3])), k12_b2(k12_pack(q[1][2], q[1][3]))));
    }
    // Phase 6: S = S g_total[dk] + U^T kr
    {
      const unsigned kr = k12_smem(st.kr);
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        unsigned bk[4];
        k12_ldsm_t(bk, kr + k12_sw(row_t, 2 * kb + (m >> 1)));
        float x0[4] = {}, x1[4] = {};
        k12_mma(x0, U, bk[0], bk[1]);
        k12_mma(x1, U, bk[2], bk[3]);
        const float2 g0 = *(const float2*)&st.gt[16 * kb + 2 * t], g1 = *(const float2*)&st.gt[16 * kb + 8 + 2 * t];
        float2* a = S.a[kb];
        a[0] = __ffma2_rn(a[0], g0, make_float2(x0[0], x0[1]));
        a[1] = __ffma2_rn(a[1], g0, make_float2(x0[2], x0[3]));
        a[2] = __ffma2_rn(a[2], g1, make_float2(x1[0], x1[1]));
        a[3] = __ffma2_rn(a[3], g1, make_float2(x1[2], x1[3]));
      }
    }
    __syncwarp();
    if (lane == 0) k12_arrive(&sm.empty[s]);
    // the output tile, rows i, this warp's 16 columns
    k12_wait(&sm.oempty[o], ((c / K12_OSTAGES) & 1) ^ 1);
    k12_stsm_t(k12_smem(sm.out[o]) + off_out, out);
    __syncwarp();
    if (lane == 0) k12_arrive(&sm.ofull[o]);
  }
  rec_t* row0 = rec_out + (size_t)(16 * w + g) * 128 + 2 * t;
#pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const float2* a = S.a[kb];
    rec_st(row0 + 16 * kb, a[0]);
    rec_st(row0 + 8 * 128 + 16 * kb, a[1]);
    rec_st(row0 + 16 * kb + 8, a[2]);
    rec_st(row0 + 8 * 128 + 16 * kb + 8, a[3]);
  }
}

// Tile c's inputs into its stage, by two warps issuing side by side (a TMA
// instruction takes ~100 cycles to issue): warp 0 kd / qd / kr and g_total,
// warp 1 v, inv and mqk; a sequence's last, partial tile has its v rows stored
// by hand (zero past the sequence). Each arrives on the stage's full barrier.
__device__ __forceinline__ void k12_load(K12Smem& sm, int c, int which, int lane, int len, long long bos,
                                         long long ws0, int h, const K12Tmap& tm_kd, const K12Tmap& tm_qd,
                                         const K12Tmap& tm_kr, const K12Tmap& tm_v, const K12Tmap& tm_inv,
                                         const K12Tmap& tm_mqk, const bf16* v, const float* ws_gt) {
  const int s = c % K12_STAGES;
  K12Stage& st = sm.in[s];
  const int valid = min(16, len - 16 * c);
  const int ws = (int)(ws0 + c) * 16, row0 = (int)(bos + 16 * c);
  if (lane == 0) {
    k12_wait(&sm.empty[s], ((c / K12_STAGES) & 1) ^ 1);
    if (which == 0) {
      k12_expect_tx(&sm.full[s], 3 * 4096 + 512);
      k12_tma3(st.kd, tm_kd, 0, ws, 0, &sm.full[s]);
      k12_tma3(st.qd, tm_qd, 0, ws, 0, &sm.full[s]);
      k12_tma3(st.kr, tm_kr, 0, ws, 0, &sm.full[s]);
      k12_bulk(st.gt, ws_gt + (long long)ws * 8, 512, &sm.full[s]);
    } else {
      k12_expect_tx(&sm.full[s], 2 * 512 + (valid == 16 ? 4096 : 0));
      if (valid == 16) k12_tma4(st.v, tm_v, 0, row0, h, 0, &sm.full[s]);
      k12_tma(st.inv, tm_inv, 0, ws, &sm.full[s]);
      k12_tma(st.mqk, tm_mqk, 0, ws, &sm.full[s]);
    }
  }
  __syncwarp();
  if (which == 1 && valid < 16 && lane < 16) {
    const uint4* src = (const uint4*)(v + (long long)(row0 + lane) * K12_INNER + h * 128);
#pragma unroll
    for (int k = 0; k < 16; ++k) *(uint4*)(st.v + k12_sw(lane, k)) = lane < valid ? src[k] : make_uint4(0, 0, 0, 0);
  }
  __syncwarp();
  if (lane == 0) k12_arrive(&sm.full[s]);
}



// ld.acquire / st.release of a progress counter (gpu scope)
__device__ __forceinline__ int k12_ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void k12_st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void k12_cp_async(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(k12_smem(dst)), "l"(src) : "memory");
}

// A gate CTA: warp pair p takes the units e + K12_GATE_CTAS (p + 8m), unit 4 item + quarter the
// rows 4 quarter .. +4 of every tile of (sequence, head) item (a CTA gates at most 7/4 heads,
// not 2): four tiles at a time as the recurrence publishes them, warp w stages rows
// 4 quarter + 2 (w % 2) + half (the ungated output and the gate's projection) in its own shared
// memory by cp.async, a group's loads in flight while it gates the group before, 16 lanes a row.
// Lane k sums the squares of K11's lane group (columns 32(k/4) + 4l + k%4, l < 8) in K11's
// butterfly order, then finishes columns 8k..8k+8 with K11's operations.
__device__ __forceinline__ void k12_gate(int e, int nseq, const long long* cu_seqlens, const bf16* raw,
                                         const bf16* partial, const float* gamma_o, bf16* gated, long long at,
                                         const int* progress, unsigned char* smem) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = lane >> 4, k = lane & 15, grp = 64 * (k >> 2);
  // this warp's two buffers of [tile 0..4][half 0..2] rows of 256 bytes, the output then the gate's projection
  unsigned char* ro = smem + warp * 8192;
  float gam[8];
  {
    const float4 g0 = *(const float4*)(gamma_o + 8 * k), g1 = *(const float4*)(gamma_o + 8 * k + 4);
    gam[0] = g0.x, gam[1] = g0.y, gam[2] = g0.z, gam[3] = g0.w, gam[4] = g1.x, gam[5] = g1.y, gam[6] = g1.z,
    gam[7] = g1.w;
  }
  for (int u = e + K12_GATE_CTAS * (warp >> 1); u < 4 * nseq * HEADS; u += 8 * K12_GATE_CTAS) {
    const int item = u >> 2, h = item % HEADS, i = 4 * (u & 3) + 2 * (warp & 1) + half;
    const long long bos = cu_seqlens[item / HEADS];
    const int len = (int)(cu_seqlens[item / HEADS + 1] - bos), ntiles = (len + 15) >> 4;
    const int* flag = progress + K12_PROGRESS_STRIDE * item;
    const auto gate = [&](int c0, int buf) {
      for (int c = c0; c < min(c0 + 4, ntiles); ++c) {
        const int r = 16 * c + i;
        const unsigned char* row = ro + buf * 4096 + ((c - c0) * 2 + half) * 256;
        // staged: nothing reads the ungated row again, so its L2 lines go without a write-back
        if (r < len && (k & 7) == 0)
          asm volatile("discard.global.L2 [%0], 128;" ::"l"(raw + (bos + r) * K12_INNER + h * 128 + 8 * k) : "memory");
        float y[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          // columns 32G + 8q .. +8: l = 2q at +0..3, l = 2q + 1 at +4..7
          const uint4 rq = *(const uint4*)(row + grp + 16 * q);
          const unsigned w0 = (k & 2) ? rq.y : rq.x, w1 = (k & 2) ? rq.w : rq.z;
          const float x0 = (k & 1) ? k12_hi(w0) : k12_lo(w0), x1 = (k & 1) ? k12_hi(w1) : k12_lo(w1);
          y[2 * q] = x0 * x0;
          y[2 * q + 1] = x1 * x1;
        }
        const float sj = ((y[0] + y[4]) + (y[2] + y[6])) + ((y[1] + y[5]) + (y[3] + y[7]));
        const float t1 = sj + __shfl_xor_sync(0xffffffffu, sj, 2);
        const float wsum = t1 + __shfl_xor_sync(0xffffffffu, t1, 1);
        const int base = 16 * half;
        const float tot = __shfl_sync(0xffffffffu, wsum, base) + __shfl_sync(0xffffffffu, wsum, base + 4) +
                          __shfl_sync(0xffffffffu, wsum, base + 8) + __shfl_sync(0xffffffffu, wsum, base + 12);
        const float rr = rsqrtf(tot * (1.0f / 128.0f) + K12_RMS_EPS);
        const uint4 araw = *(const uint4*)(row + 16 * k);
        const uint4 graw = *(const uint4*)(ro + buf * 4096 + 2048 + ((c - c0) * 2 + half) * 256 + 16 * k);
        const unsigned aw[4] = {araw.x, araw.y, araw.z, araw.w}, gwd[4] = {graw.x, graw.y, graw.z, graw.w};
        float ov[8], den[8], sig[8];
        bool far = false;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          const float a = (x & 1) ? k12_hi(aw[x / 2]) : k12_lo(aw[x / 2]);
          const float g = (x & 1) ? k12_hi(gwd[x / 2]) : k12_lo(gwd[x / 2]);
          ov[x] = __bfloat162float(__float2bfloat16(a * rr * gam[x]));
          // K11's sigmoid: __expf (its ex2 wherever the result is not subnormal, and then 1 + it is 1 either way)
          float ex;
          asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(-g * 1.4426950408889634f));
          den[x] = 1.0f + ex;
          sig[x] = k12_recip(den[x]);
          far |= den[x] >= 0x1p126f;
        }
        if (__any_sync(0xffffffffu, far)) {
#pragma unroll
          for (int x = 0; x < 8; ++x)
            if (den[x] >= 0x1p126f) sig[x] = 1.0f / den[x];
        }
        uint4 res;
        unsigned* rw = (unsigned*)&res;
#pragma unroll
        for (int x = 0; x < 8; x += 2) {
          const unsigned gt = k12_pack(sig[x], sig[x + 1]);
          rw[x / 2] = k12_pack(ov[x] * k12_lo(gt), ov[x + 1] * k12_hi(gt));
        }
        if (r < len) *(uint4*)(gated + (at + bos + r) * K12_INNER + h * 128 + 8 * k) = res;
      }
      __syncwarp();
    };
    for (int c0 = 0; c0 < ntiles; c0 += 4) {
      const int buf = (c0 >> 2) & 1, cend = min(c0 + 4, ntiles);
      if (lane == 0)
        while (k12_ld_acquire(flag) < cend) __nanosleep(256);
      __syncwarp();
      for (int c = c0; c < cend; ++c) {
        const int r = 16 * c + i;
        if (r < len) {
          k12_cp_async(ro + buf * 4096 + ((c - c0) * 2 + half) * 256 + 16 * k,
                       raw + (bos + r) * K12_INNER + h * 128 + 8 * k);
          k12_cp_async(ro + buf * 4096 + 2048 + ((c - c0) * 2 + half) * 256 + 16 * k,
                       partial + (at + bos + r) * K12_LD + 3 * K12_INNER + h * 128 + 8 * k);
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      if (c0 > 0) {
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncwarp();
        gate(c0 - 4, buf ^ 1);
      }
    }
    if (ntiles > 0) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      gate((ntiles - 1) & ~3, ((ntiles - 1) >> 2) & 1);
    }
  }
}

extern "C" __global__ void __launch_bounds__(K12_THREADS, 1) kern_k3_kda_rec(
    const __grid_constant__ K12Tmap tm_kd, const __grid_constant__ K12Tmap tm_qd,
    const __grid_constant__ K12Tmap tm_kr, const __grid_constant__ K12Tmap tm_v,
    const __grid_constant__ K12Tmap tm_inv, const __grid_constant__ K12Tmap tm_mqk, const bf16* __restrict__ v,
    const bf16* __restrict__ beta, const float* __restrict__ ws_gt, void* __restrict__ kda_base,
    const int* __restrict__ line_index, long long line_bytes, const bf16* __restrict__ partial,
    const float* __restrict__ gamma_o, bf16* __restrict__ raw, bf16* __restrict__ gated,
    const int* __restrict__ span_at, const long long* __restrict__ cu_seqlens, const int* __restrict__ tile_prefix,
    int* __restrict__ progress, int tiles, int span, int nseq) {
  extern __shared__ __align__(1024) unsigned char k12_raw[];
  K12Smem& sm = *(K12Smem*)k12_raw;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if ((int)blockIdx.x >= nseq * HEADS) {
    k12_gate(blockIdx.x - nseq * HEADS, nseq, cu_seqlens, raw, partial, gamma_o, gated, span_at[0], progress,
             k12_raw);
    return;
  }
  const int j = blockIdx.x / HEADS, h = blockIdx.x % HEADS;
  const long long bos = cu_seqlens[j];
  const int len = (int)(cu_seqlens[j + 1] - bos);
  const int ntiles = (len + 15) >> 4;
  const long long ws0 = (long long)h * tiles + tile_prefix[j];
  char* line = (char*)kda_base + (long long)line_index[j] * line_bytes;
  if (threadIdx.x == 0) {
    for (int s = 0; s < K12_STAGES; ++s) {
      k12_bar_init(&sm.full[s], 2);
      k12_bar_init(&sm.empty[s], K12_MMA_WARPS);
    }
    for (int o = 0; o < K12_OSTAGES; ++o) {
      k12_bar_init(&sm.ofull[o], K12_MMA_WARPS);
      k12_bar_init(&sm.oempty[o], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // beta as FlashKDA's recurrence takes it, zero past the sequence's last tile row
  for (int i = threadIdx.x; i < 16 * ntiles; i += K12_THREADS)
    sm.beta[i] = __float2bfloat16(i < len ? k12_sigmoid(__bfloat162float(beta[(long long)h * span + bos + i])) : 0.0f);
  __syncthreads();

  if (warp < K12_MMA_WARPS) {
    rec_t* rec = (rec_t*)(line + (long long)h * 128 * 128 * sizeof(rec_t));
    k12_recurrence(sm, rec, rec, len);
  } else if (warp == K12_MMA_WARPS || warp == K12_MMA_WARPS + 2) {
    const int which = warp == K12_MMA_WARPS ? 0 : 1;
    for (int c = 0; c < ntiles; ++c)
      k12_load(sm, c, which, lane, len, bos, ws0, h, tm_kd, tm_qd, tm_kr, tm_v, tm_inv, tm_mqk, v, ws_gt);
  } else if (warp == K12_MMA_WARPS + 1) {
    // The conv windows first (the gather read them, nothing after it does), lane l columns 4l..4l+4
    if (len > 0) {
#pragma unroll
      for (int s = 0; s < 3; ++s) {
        bf16* win = (bf16*)(line + K12_REC_BYTES + s * K12_WIN_BYTES) + h * 128 + 4 * lane;
        uint2 nt[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          const int i = len - 3 + k;
          nt[k] = i >= 0 ? *(const uint2*)(partial + (bos + i) * K12_LD + s * K12_INNER + h * 128 + 4 * lane)
                         : *(const uint2*)(win + (i + 3) * K12_INNER);
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) *(uint2*)(win + k * K12_INNER) = nt[k];
      }
    }
    // then every output tile to `raw`, published to the gate CTAs by the item's counter
    int* flag = progress + K12_PROGRESS_STRIDE * (j * HEADS + h);
    for (int c = 0; c < ntiles; ++c) {
      const int o = c % K12_OSTAGES, valid = min(16, len - 16 * c);
      k12_wait(&sm.ofull[o], (c / K12_OSTAGES) & 1);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int i = 2 * q + (lane >> 4), x = lane & 15;
        if (i < valid)
          *(uint4*)(raw + (bos + 16 * c + i) * K12_INNER + h * 128 + 8 * x) =
              *(const uint4*)(sm.out[o] + i * K12_ROW + 16 * x);
      }
      __syncwarp();
      if (lane == 0) k12_arrive(&sm.oempty[o]);
      // every fourth tile (the release waits for the writes before it): the warp's row writes
      // ordered before lane 0's cumulative st.release by bar.warp.sync
      if ((c & 3) == 3 || c == ntiles - 1) {
        __syncwarp();
        if (lane == 0) k12_st_release(flag, c + 1);
      }
    }
  }
}
