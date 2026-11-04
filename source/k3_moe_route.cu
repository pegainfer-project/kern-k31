// The front of a prefill chunk's MoE in one launch: the router top-k, the
// latent in mxfp8, the routing tables of TRT-LLM gen's batched GEMMs and the
// shared expert's activation, where they were six launches (router_topk,
// moe_quant, FlashInfer's init / histogram / offsets, situ).
//
//   kern_k3g_moe_route(S, bias, rs, x, ldx, gu, ldgu, q, sf, idx, wts, act, ldact, sync, cta_batch, cta_limit,
//                      num_non_exiting, total_padded, route_map, exp2perm, tile, B)
//     S f32 [B, EXPERTS] router logits; bias f32 [EXPERTS]; rs bf16 [1];
//     x bf16 [B, ldx] the latent; gu bf16 [B, ldgu] the shared expert's gate | up;
//     q u8 [B, LATENT], sf u8 [B, LATENT / 32]: k3_moe_prefill.cu's kern_k3_moe_quant;
//     idx i32 / wts f32 [B, 16]: k3_prefill_glue.cu's router top-k, bit for bit;
//     act bf16 [B, ldact]: the shared expert's activation, k3_prefill_glue.cu's kern_k3g_situ;
//     sync u32 [SYNC_WORDS]: the grid barrier (arrivals, generation), two routing tables
//       (Table below) and two piece counters, the generation's parity picking this call's; zeroed by
//       kern_k3g_moe_route_init before a stage's first call, every call of a stage's B rows
//       leaves the barrier and the next call's table and counter zero;
//     the tables FlashInfer's routing writes for a rank holding every expert, `tile` rows a CTA.
//   grid (64, 1, 1) up to 256 rows, then (288, 1, 1)   block (512, 1, 1): every block resident
//   (two an SM) for the grid barrier
//
//   kern_k3g_moe_route_init(sync)
//   grid (any, 1, 1)   block (1024, 1, 1)
//
//   kern_k3g_moe_finalize_rms(fc2, exp2perm, wts, gamma, out, ldo, T)
//     k3_prefill_glue.cu's kern_k3g_finalize_rms (the top-k combine, its row normed by gamma)
//     into the first LATENT columns of out's ldo-wide rows.
//   grid (T, 1, 1)   block (LATENT / 8, 1, 1)
//
//   kern_k3_concat_cols(a, b, out, rows, ka, kb)
//     out [rows, ka + kb] = a [rows, ka] | b [rows, kb], bf16; a load-time weight concat.
//   grid (ceil(rows * (ka + kb) / 8 / 256), 1, 1)   block (256, 1, 1)
//
// Phase 1: a warp a row's top-k (a grid-stride over rows when B passes 16 *
// gridDim; from 2048 rows over 4 warps a block), its picks marked in this call's table (each expert's count, its
// count per 256-row chunk, its rows' bitmap: fire-and-forget atomics); then
// the warps take grabs of the rows' 256-column pieces of the latent to quantise
// and of the shared activation off a counter; the next call's table and counter
// zeroed. A grid barrier: every
// block is resident, the generation counts the calls. Phase 2: every block
// scans the counts into the experts' CTA offsets and writes its share of the
// CTA tables; a pick's place among its expert's rows is the picks in the
// chunks and the words before its row's, so every expert's rows are in row
// order (the batched GEMM gathers them faster so than in an atomic's order)
// and the tables are the same bytes every run.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#define TOPK 16
#define LATENT 3584
#define SHARED 6144
#define WARPS 16
#define THREADS (WARPS * 32)
#define TK_PER (EXPERTS / 32)
static_assert(EXPERTS % 128 == 0 && EXPERTS <= 2 * THREADS, "float4s of experts a lane, two experts a thread in the scan");

typedef __nv_bfloat16 bf16_t;
typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  const u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ u32 ld_acquire(const u32* p) {
  u32 v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(u32* p, u32 v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// 1 / d rounded as `1.0f / d` and __frcp_rn round it, without their slow path's branch:
// equal to both for every d in [2^-126, 2^126) (checked over all of them); the callers
// take it only when no lane of the warp holds a d outside.
__device__ __forceinline__ float rcp_fast(float d) {
  float q;
  asm("{\n .reg .f32 r, e;\n rcp.approx.ftz.f32 r, %1;\n fma.rn.f32 e, %1, r, 0fBF800000;\n neg.f32 e, e;\n"
      " fma.rn.f32 %0, r, e, r;\n}"
      : "=f"(q)
      : "f"(d));
  return q;
}
__device__ __forceinline__ bool rcp_ok(float d) { return d >= 0x1p-126f && d < 0x1p126f; }

// a / 448.f without its slow path's branch: equal for every bf16 magnitude but infinity.
__device__ __forceinline__ float div448(float a) {
  float q;
  asm("{\n .reg .f32 r, t, q0, m;\n mov.f32 r, 0f3B124925;\n fma.rn.f32 t, r, 0fC3E00000, 0f3F800000;\n"
      " fma.rn.f32 r, t, r, r;\n fma.rn.f32 q0, %1, r, 0f00000000;\n fma.rn.f32 m, q0, 0fC3E00000, %1;\n"
      " fma.rn.f32 %0, r, m, q0;\n}"
      : "=f"(q)
      : "f"(a));
  return a == __int_as_float(0x7f800000) ? a : q;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// kern_k3g_router_topk's picks: the 16 largest keys sigmoid(S) + bias, ties to
// the smaller expert, lane r < 16 ending with the r-th and its sigmoid. Lane l
// holds experts l * TK_PER + j. Every pick is at least the 16th largest of the
// lanes' maxima (16 lanes hold a key that large), so the keys that clear it
// are the candidates (16 or more; past CAND, the original 16 rounds), ranked by
// key then expert into their places.
#define CAND 64
#ifndef QBATCH
#define QBATCH 8  // quant pieces a warp loads at once
#endif
#ifndef SBATCH
#define SBATCH 4  // situ pieces a warp loads at once
#endif
__device__ __forceinline__ void topk_row(const float* __restrict__ row, const float* __restrict__ bias, int lane,
                                         u32* __restrict__ ck, int* __restrict__ ce, int& pick_e, float& pick_w) {
  u32 o[TK_PER];
  bool ok = true;
#pragma unroll
  for (int j = 0; j < TK_PER; j += 4) {
    const float4 sv = *reinterpret_cast<const float4*>(row + lane * TK_PER + j);
    const float s4[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float d = 1.0f + expf(-s4[i]);
      ok &= rcp_ok(d);
      o[j + i] = __float_as_uint(d);
    }
  }
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) o[j] = __float_as_uint(rcp_fast(__uint_as_float(o[j])));
  } else {
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) o[j] = __float_as_uint(1.0f / __uint_as_float(o[j]));
  }
#pragma unroll
  for (int j = 0; j < TK_PER; j += 4) {
    const float4 bv = *reinterpret_cast<const float4*>(bias + lane * TK_PER + j);
    const float b4[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) o[j + i] = ord_f32(__uint_as_float(o[j + i]) + b4[i]);
  }
  u32 v = o[0];
#pragma unroll
  for (int j = 1; j < TK_PER; ++j) v = max(v, o[j]);
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const u32 p = __shfl_xor_sync(0xffffffffu, v, j);
      v = (((lane & j) == 0) == ((lane & k) == 0)) ? min(v, p) : max(v, p);
    }
  const u32 tau = __shfl_sync(0xffffffffu, v, 16);
  int c = 0;
#pragma unroll
  for (int j = 0; j < TK_PER; ++j) c += o[j] >= tau;
  int inc = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, d);
    if (lane >= d) inc += y;
  }
  const int n = __shfl_sync(0xffffffffu, inc, 31);
  pick_e = 0;
  if (n <= CAND) {
    int at = inc - c;
#pragma unroll
    for (int j = 0; j < TK_PER; ++j)
      if (o[j] >= tau) {
        ck[at] = o[j];
        ce[at] = lane * TK_PER + j;
        ++at;
      }
    __syncwarp();
    int rk[CAND / 32], ex[CAND / 32];
#pragma unroll
    for (int s = 0; s < CAND / 32; ++s) {
      const int i = lane + 32 * s;
      rk[s] = TOPK;
      if (i < n) {
        const u32 ki = ck[i];
        const int ei = ce[i];
        int r = 0;
        for (int j = 0; j < n; ++j) {
          const u32 kj = ck[j];
          r += (kj > ki) | ((kj == ki) & (ce[j] < ei));
        }
        rk[s] = r;
        ex[s] = ei;
      }
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < CAND / 32; ++s)
      if (rk[s] < TOPK) ce[rk[s]] = ex[s];
    __syncwarp();
    if (lane < TOPK) pick_e = ce[lane];
    __syncwarp();
  } else {
    for (int r = 0; r < TOPK; ++r) {
      u32 m = o[0];
#pragma unroll
      for (int j = 1; j < TK_PER; ++j) m = max(m, o[j]);
      m = __reduce_max_sync(0xffffffffu, m);
      u32 my = 0xffffffffu;
#pragma unroll
      for (int j = TK_PER - 1; j >= 0; --j)
        if (o[j] == m) my = (u32)(lane * TK_PER + j);
      const u32 ew = __reduce_min_sync(0xffffffffu, my);
      if (my == ew)
#pragma unroll
        for (int j = 0; j < TK_PER; ++j)
          if (lane * TK_PER + j == (int)ew) o[j] = 0u;
      if (lane == r) pick_e = (int)ew;
    }
  }
  pick_w = lane < TOPK ? sigmoid(row[pick_e]) : 0.f;
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// bf16 pairs as plain bits: a bf16 is the top half of its f32; cvt.rn.bf16x2.f32 packs two
__device__ __forceinline__ float2 unpack2(unsigned u) {
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ unsigned pack2(float2 f) {
  unsigned r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(f.y), "f"(f.x));
  return r;
}
// two IEEE multiplies / adds in one f32x2 instruction, each lane rounded as the scalar one
__device__ __forceinline__ unsigned long long b64(float2 a) { return *reinterpret_cast<unsigned long long*>(&a); }
__device__ __forceinline__ float2 f2(unsigned long long d) { return *reinterpret_cast<float2*>(&d); }
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  unsigned long long d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(b64(a)), "l"(b64(b)));
  return f2(d);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  unsigned long long d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(b64(a)), "l"(b64(b)));
  return f2(d);
}
__device__ __forceinline__ float ex2_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// k3_prefill_glue.cu's kern_k3g_situ on a pair of columns: (4 tanh(g / 4) * sigmoid(g)) * 25 tanh(u / 25),
// e^-g as __expf's ex2 of -g log2(e) (flushed below 2^-126, where 1 + e^-g is 1 either way); `fast`
// when every lane's g >= -87, so 1 + e^-g < 2^126 is in rcp_fast's range.
__device__ __forceinline__ unsigned situ2(unsigned gw, unsigned uw, bool fast) {
  const float2 g = unpack2(gw), u = unpack2(uw);
  const float2 tg = mul2(g, make_float2(0.25f, 0.25f)), tu = mul2(u, make_float2(0.04f, 0.04f));
  const float2 x = mul2(g, make_float2(-1.44269502f, -1.44269502f));
  const float2 d = add2(make_float2(ex2_ftz(x.x), ex2_ftz(x.y)), make_float2(1.0f, 1.0f));
  const float2 a = mul2(make_float2(tanh_approx(tg.x), tanh_approx(tg.y)), make_float2(4.0f, 4.0f));
  const float2 c = mul2(make_float2(tanh_approx(tu.x), tanh_approx(tu.y)), make_float2(25.0f, 25.0f));
  const float2 sg = fast ? make_float2(rcp_fast(d.x), rcp_fast(d.y)) : make_float2(__frcp_rn(d.x), __frcp_rn(d.y));
  return pack2(mul2(mul2(a, sg), c));
}

__device__ __forceinline__ void situ8(const uint4 g, const uint4 u, bf16_t* __restrict__ act) {
  const unsigned* gp = reinterpret_cast<const unsigned*>(&g);
  const unsigned* upp = reinterpret_cast<const unsigned*>(&u);
  uint4 o;
  unsigned* op = reinterpret_cast<unsigned*>(&o);
  bool ok = true;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const float2 gf = unpack2(gp[l]);
    ok = ok && gf.x >= -87.0f && gf.y >= -87.0f;
  }
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int l = 0; l < 4; ++l) op[l] = situ2(gp[l], upp[l], true);
  } else {
#pragma unroll
    for (int l = 0; l < 4; ++l) op[l] = situ2(gp[l], upp[l], false);
  }
  *reinterpret_cast<uint4*>(act) = o;
}

// kern_k3_moe_quant on 8 elements, four lanes a 32-element group.
__device__ __forceinline__ void quant8(const uint4 u, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int lane) {
  const unsigned* p = reinterpret_cast<const unsigned*>(&u);
  float2 v[4];
  float amax = 0.f;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    v[l] = unpack2(p[l]);
    amax = fmaxf(amax, fmaxf(fabsf(v[l].x), fabsf(v[l].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  int e = amax > 0.f ? (int)ceilf(log2f(div448(amax))) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  if ((lane & 3) == 0) *sf = (uint8_t)(e + 127);
  uint2 w;
  unsigned short* h = reinterpret_cast<unsigned short*>(&w);
#pragma unroll
  for (int l = 0; l < 4; ++l) h[l] = __nv_cvt_float2_to_fp8x2(mul2(v[l], make_float2(s, s)), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint2*>(q) = w;
}

// The routing state, one table per parity of the generation: every expert's
// pick count, its picks per 256-row chunk, and its rows as a bitmap.
#define MAX_ROWS 8192
#define CHUNK 256
#define CHUNKS (MAX_ROWS / CHUNK)
#define WORDS (MAX_ROWS / 32)
#define TABLE (EXPERTS * (1 + CHUNKS + WORDS))
// the barrier's arrival count at word 0 and the generation its blocks poll at word 32 (own 128-byte
// lines), the two tables from word 64, then the two piece counters
#define GEN_AT 32
#define TABLES_AT 64
#define SYNC_WORDS (TABLES_AT + 2 * TABLE + 2)
static_assert(CHUNKS <= 32 && CHUNK / 32 <= 32, "a chunk prefix is one warp sum");
struct Table {
  int* tot;   // [EXPERTS]
  int* chunk;  // [EXPERTS][CHUNKS]
  u32* bits;   // [EXPERTS][WORDS]
};
__device__ __forceinline__ Table table(u32* sync, u32 parity) {
  u32* t = sync + TABLES_AT + parity * TABLE;
  return {reinterpret_cast<int*>(t), reinterpret_cast<int*>(t + EXPERTS), t + EXPERTS * (1 + CHUNKS)};
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) kern_k3g_moe_route(
    const float* __restrict__ S, const float* __restrict__ bias, const bf16_t* __restrict__ rs,
    const bf16_t* __restrict__ x, int ldx, const bf16_t* __restrict__ gu, int ldgu, uint8_t* __restrict__ q,
    uint8_t* __restrict__ sf, int* __restrict__ idx, float* __restrict__ wts, bf16_t* __restrict__ act, int ldact,
    u32* __restrict__ sync, int* __restrict__ cta_batch, int* __restrict__ cta_limit,
    int* __restrict__ num_non_exiting, int* __restrict__ total_padded, int* __restrict__ route_map,
    int* __restrict__ exp2perm, int tile, int B) {
  __shared__ u32 gen_s;
  __shared__ int off[EXPERTS], cnt[EXPERTS];
  __shared__ int wsum[WARPS];
  __shared__ u32 ck[WARPS][CAND];
  __shared__ int ce[WARPS][CAND];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int nthreads = gridDim.x * THREADS, gt = blockIdx.x * THREADS + t;
  const int rstride = gridDim.x * WARPS, r0 = blockIdx.x * WARPS + warp;

  if (t == 0) gen_s = ld_acquire(sync + GEN_AT);
  __syncthreads();
  const u32 gen = gen_s;
  const Table cur = table(sync, gen & 1), next = table(sync, ~gen & 1);
  // the next call's table: what a call of these B rows marks
  {
    const int nc = (B + CHUNK - 1) / CHUNK, nw = (B + 31) / 32;
    for (int i = gt; i < EXPERTS * (1 + nc + nw); i += nthreads) {
      const int e = i % EXPERTS, j = i / EXPERTS;
      if (j == 0)
        next.tot[e] = 0;
      else if (j <= nc)
        next.chunk[e * CHUNKS + j - 1] = 0;
      else
        next.bits[e * WORDS + j - 1 - nc] = 0;
    }
  }

  const float rsv = __bfloat162float(rs[0]);
  // from 2048 rows a quarter of a block's warps take the top-k rows and the rest start on the pieces at
  // once (an all-warp top-k leaves DRAM idle); the top-k warps join the grabs when their rows are done
  const int tkw = B >= 2048 ? WARPS / 4 : WARPS;
  for (int b = warp < tkw ? blockIdx.x * tkw + warp : B; b < B; b += gridDim.x * tkw) {
    int pick_e;
    float pick_w;
    topk_row(S + (long long)b * EXPERTS, bias, lane, ck[warp], ce[warp], pick_e, pick_w);
    float den = pick_w;
#pragma unroll
    for (int o = TOPK / 2; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane < TOPK) {
      wts[b * TOPK + lane] = (pick_w / (den + 1e-20f)) * rsv;
      idx[b * TOPK + lane] = pick_e;
      atomicAdd(cur.tot + pick_e, 1);
      atomicAdd(cur.chunk + pick_e * CHUNKS + b / CHUNK, 1);
      atomicOr(cur.bits + pick_e * WORDS + b / 32, 1u << (b & 31));
    }
  }
  // the rows' latent and shared activation in grabs of 256-column pieces, each warp taking the next
  // grab off this call's counter when it is done with the last (and with its top-k rows): the warps
  // with more top-k rows take fewer, and every warp reaches the grid barrier at about the same time
  const int nq = B * (LATENT / 256), ns = B * (SHARED / 256);
  const int gq = B >= 1024 ? 2 * QBATCH : QBATCH, gs = B >= 1024 ? 2 * SBATCH : SBATCH;  // fewer grabs, less contention
  const int nqb = (nq + gq - 1) / gq, nsb = (ns + gs - 1) / gs;
  u32* const ctr = sync + SYNC_WORDS - 2;
  if (gt == 0) ctr[~gen & 1] = 0;
  for (;;) {
    int g = 0;
    if (lane == 0) g = (int)atomicAdd(ctr + (gen & 1), 1u);
    g = __shfl_sync(0xffffffffu, g, 0);
    if (g >= nqb + nsb) break;
    if (g < nqb) {
      for (int i0 = g * gq; i0 < min(nq, (g + 1) * gq); i0 += QBATCH) {
        uint4 u[QBATCH];
#pragma unroll
        for (int k = 0; k < QBATCH; ++k) {
          const int ik = min(i0 + k, nq - 1);
          const int b = ik / (LATENT / 256), c = ik % (LATENT / 256) * 256 + lane * 8;
          u[k] = *reinterpret_cast<const uint4*>(x + (long long)b * ldx + c);
        }
#pragma unroll
        for (int k = 0; k < QBATCH; ++k) {
          const int ik = i0 + k;
          if (ik >= nq) break;
          const int b = ik / (LATENT / 256), c = ik % (LATENT / 256) * 256 + lane * 8;
          quant8(u[k], q + (long long)b * LATENT + c, sf + (long long)b * (LATENT / 32) + c / 32, lane);
        }
      }
    } else {
      for (int i0 = (g - nqb) * gs; i0 < min(ns, (g - nqb + 1) * gs); i0 += SBATCH) {
        uint4 gg[SBATCH], u[SBATCH];
#pragma unroll
        for (int k = 0; k < SBATCH; ++k) {
          const int ik = min(i0 + k, ns - 1);
          const int b = ik / (SHARED / 256), c = ik % (SHARED / 256) * 256 + lane * 8;
          const bf16_t* gate = gu + (long long)b * ldgu + c;
          gg[k] = *reinterpret_cast<const uint4*>(gate);
          u[k] = *reinterpret_cast<const uint4*>(gate + SHARED);
        }
#pragma unroll
        for (int k = 0; k < SBATCH; ++k) {
          const int ik = i0 + k;
          if (ik >= ns) break;
          const int b = ik / (SHARED / 256), c = ik % (SHARED / 256) * 256 + lane * 8;
          situ8(gg[k], u[k], act + (long long)b * ldact + c);
        }
      }
    }
  }

  // grid barrier
  __syncthreads();
  if (t == 0) {
    __threadfence();
    if (atomicAdd(sync, 1u) == gridDim.x - 1) {
      sync[0] = 0;
      st_release(sync + GEN_AT, gen + 1);
    } else {
      while (ld_acquire(sync + GEN_AT) == gen) __nanosleep(64);
    }
    __threadfence();
  }
  __syncthreads();

  // the experts' CTA counts, exclusive-scanned: thread t holds experts 2t, 2t + 1
  int c0 = 0, c1 = 0;
  if (2 * t < EXPERTS) {
    cnt[2 * t] = __ldcg(cur.tot + 2 * t);
    cnt[2 * t + 1] = __ldcg(cur.tot + 2 * t + 1);
    c0 = (cnt[2 * t] + tile - 1) / tile;
    c1 = (cnt[2 * t + 1] + tile - 1) / tile;
  }
  int incl = c0 + c1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  int base = 0, total = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    const int s = wsum[w];
    base += w < warp ? s : 0;
    total += s;
  }
  if (2 * t < EXPERTS) {
    off[2 * t] = base + incl - c0 - c1;
    off[2 * t + 1] = base + incl - c1;
  }
  __syncthreads();
  if (gt == 0) {
    num_non_exiting[0] = total;
    total_padded[0] = total * tile;
  }
  for (int e = gt; e < EXPERTS; e += nthreads) {
    const int n = (cnt[e] + tile - 1) / tile, o = off[e];
    for (int c = 0; c < n; ++c) {
      cta_batch[o + c] = e;
      cta_limit[o + c] = min((o + c + 1) * tile, o * tile + cnt[e]);
    }
  }
  // row b's place among its expert's rows: the picks of the chunks before b's, then of b's chunk's words
  // before b's, then of b's word below b
  for (int b = r0; b < B; b += rstride) {
    const int mine = lane < TOPK ? idx[b * TOPK + lane] : 0;
    const int bc = b / CHUNK, bw = b / 32;
    int place = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int e = __shfl_sync(0xffffffffu, mine, k);
      int v = lane < bc ? __ldcg(cur.chunk + e * CHUNKS + lane) : 0;
      const int w = bc * (CHUNK / 32) + lane;
      if (w <= bw) {
        const u32 word = __ldcg(cur.bits + e * WORDS + w);
        v += __popc(w < bw ? word : word & ((1u << (b & 31)) - 1));
      }
      const int r = __reduce_add_sync(0xffffffffu, v);
      if (lane == k) place = r;
    }
    if (lane < TOPK) {
      const int p = off[mine] * tile + place;
      exp2perm[b * TOPK + lane] = p;
      route_map[p] = b;
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3g_moe_route_init(u32* __restrict__ sync) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SYNC_WORDS; i += gridDim.x * blockDim.x) sync[i] = 0;
}

// The row's 16 (permuted row, weight) pairs staged in shared memory, read back by every thread (a
// broadcast) instead of 32 global loads a thread; each pair of columns one packed FMA (two IEEE FMAs).
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;"
      : "=l"(d)
      : "l"(*reinterpret_cast<unsigned long long*>(&a)), "l"(*reinterpret_cast<unsigned long long*>(&b)),
        "l"(*reinterpret_cast<unsigned long long*>(&c)));
  return *reinterpret_cast<float2*>(&d);
}

// four blocks an SM (32 registers): the more rows' gathers in flight, the less of the DRAM latency shows
extern "C" __global__ void __launch_bounds__(LATENT / 8, 4) kern_k3g_moe_finalize_rms(
    const bf16_t* __restrict__ fc2, const int* __restrict__ exp2perm, const float* __restrict__ wts,
    const bf16_t* __restrict__ gamma, bf16_t* __restrict__ out, int ldo, int T) {
  __shared__ float sm[33];
  __shared__ int sp[TOPK];
  __shared__ float spw[TOPK];
  const int t = blockIdx.x, h = threadIdx.x;
  const int lane = h & 31, warp = h >> 5;
  if (h < TOPK && t < T) {
    sp[h] = exp2perm[t * TOPK + h];
    spw[h] = wts[t * TOPK + h];
  }
  __syncthreads();
  float2 acc[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
  if (t < T) {
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int p = sp[k];
      if (p < 0) continue;
      const float w = spw[k];
      const uint4 u = reinterpret_cast<const uint4*>(fc2 + (long long)p * LATENT)[h];
      const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int l = 0; l < 4; ++l) acc[l] = fma2(make_float2(w, w), __bfloat1622float2(v[l]), acc[l]);
    }
  }
  __nv_bfloat162 x[4];
  float sum = 0.0f;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    x[l] = __floats2bfloat162_rn(acc[l].x, acc[l].y);
    const float2 f = __bfloat1622float2(x[l]);
    sum += f.x * f.x;
    sum += f.y * f.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) sm[warp] = sum;
  if (h < 32 && h >= (int)(blockDim.x >> 5)) sm[h] = 0.f;
  __syncthreads();
  if (warp == 0) {
    float v = sm[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sm[32] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(sm[32] / (float)LATENT + 1e-5f);
  const uint4 gw = reinterpret_cast<const uint4*>(gamma)[h];
  const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&gw);
  uint4 o;
  __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const float2 f = __bfloat1622float2(x[l]);
    ov[l] = __hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), g[l]);
  }
  reinterpret_cast<uint4*>(out + (long long)t * ldo)[h] = o;
}

extern "C" __global__ void kern_k3_concat_cols(const bf16_t* __restrict__ a, const bf16_t* __restrict__ b,
                                               bf16_t* __restrict__ out, int rows, int ka, int kb) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int k = ka + kb;
  if (i >= (long long)rows * k / 8) return;
  const long long r = i / (k / 8);
  const int c = (int)(i % (k / 8)) * 8;
  reinterpret_cast<uint4*>(out)[i] = c < ka ? *reinterpret_cast<const uint4*>(a + r * ka + c)
                                            : *reinterpret_cast<const uint4*>(b + r * kb + c - ka);
}
