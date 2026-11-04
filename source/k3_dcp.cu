// K3 MLA decode under decode context parallelism (DCP): a sequence's KV is
// dealt over the NRANKS members of the tp group position by position
// (position p on member p % NRANKS), every member runs the MLA decode over its
// own positions for every row and all HEADS heads, and these three kernels
// turn the members' partials into this member's HL = HEADS / NRANKS heads of
// the whole attention. Between pack and combine sits one all-to-all of equal
// chunks (NCCL's, `count` = chunk elements per peer), so a chunk's layout is
// the whole contract:
//
//   chunk(R) = R * HL * (LAT + 2) bf16 elements, chunk q of a buffer at
//   q * chunk(R): o [R][HL][LAT] bf16, then lse [R][HL] f32.
//   Head h of a partial travels to member h / HL as its local head h % HL.
//
// lse is the natural-log log-sum-exp the FlashInfer DSL decode returns
// (its user-facing lse, already divided by log2 e), over the member's
// positions only.
//
//   extern "C" __global__ void kern_k3_dcp_fixup(
//       bf16* o_lat,          // [R, HEADS, LAT]   in place
//       float* lse,           // [R, HEADS]        in place
//       const int* seq_lens,  // [R]  this member's positions of the row
//       int R);
//   grid (R, 1, 1)   block 256
//   A row with seq_lens[b] == 0 (the member holds none of its positions yet,
//   or a padding row) gets o = 0 and lse = -inf; other rows are untouched.
//   The decode leaves lse = +inf and an undefined o there.
//
//   extern "C" __global__ void kern_k3_dcp_pack(
//       const bf16* o_lat,    // [R, HEADS, LAT]
//       const float* lse,     // [R, HEADS]
//       bf16* send,           // [NRANKS, chunk(R)]
//       int R);
//   grid (R, HEADS / 4, 1)   block 256: four heads per block, one 16-byte
//   vector of o per thread.
//
//   extern "C" __global__ void kern_k3_dcp_combine(
//       const bf16* recv,     // [NRANKS, chunk(R)]  chunk q from member q
//       bf16* o,              // [R, HL, LAT]
//       int R);
//   grid (R, HL / 4, 1)   block 256: four local heads per block, one 16-byte
//   vector of o per thread.
//   Per (b, j), over the NRANKS partials q (all f32):
//     l_q = lse_q, with NaN and +inf read as -inf
//     m   = max_q l_q;  if m == -inf the row has no position anywhere: o = 0
//     w_q = exp(l_q - m)
//     o[b, j] = bf16(sum_q w_q * f32(o_q) / sum_q w_q), a term with w_q == 0
//               never reading o_q, so whatever an empty partial left there
//               (NaN included) cannot reach the sum.
//   Summation order is q = 0 .. NRANKS - 1, so the result is the same bytes on
//   every run.
//
// kern_k3_dcp_exchange is the DSL decode's split reduction, the four steps
// above and the v-up + gate of this member's heads (k3_mla_vup_gate.cu) in
// one launch: fixup, pack, the all-to-all and combine, with the group's
// peer-mapped Lamport stages for the wire instead of NCCL (TensorRT-LLM's
// one-shot protocol, as peer_allreduce_bf16.cu runs it).
//
//   extern "C" __global__ void kern_k3_dcp_exchange(
//       const float* acc_o,                    // [R][M_TILE][split_max][LAT]  the splits' normalized o
//       const float* acc_lse,                  // [R][M_TILE][split_max]       their log2-sum-exp
//       const int* seq_lens, const int* bsk,   // [R]  positions held, splits asked for
//       int split_max,
//       const bf16* w_kv_b,                    // [HL*256, LAT]  W_UV = rows j*256+128..256
//       const bf16* mla_gate,                  // [R, HL*128]
//       bf16* gated,                           // [R, HL*128]
//       uint8_t* lamport,                      // 3 stages of `stage_bytes`
//       const unsigned long long* peers,       // [NRANKS] every member's `lamport`
//       int* state, int* err,                  // [8] zeroed carry; 1 + a member that never showed
//       int rank, int R, long long stage_bytes, long long timeout_ns);
//   grid: any grid whose blocks are all resident at once   block 256
//   dynamic smem 64 KB: a merge item's half of W_UV, staged by cp.async while it sends and waits
//
//   The splits merge as the DSL's reduction kernel does: row b ran
//   S = ceil(t / ceil(t / bsk[b])) splits of its t = ceil(seq_lens[b] / 128)
//   tiles; g = m + log2(sum_s exp2(l_s - m)) over them (m their max, 0 if
//   -inf), o = bf16(sum_s acc_o[s] * exp2(l_s - g)) and lse = g / log2(e),
//   ex2 / lg2 approximate.
//
//   A stage holds NRANKS slots, slot q member q's partials of this member's
//   heads: R * HL records of REC 16-byte vectors, a head's o (LANES vectors,
//   -0.0 sent as +0.0: a bf16 -0.0 in a 16-byte vector means "not arrived")
//   and then its lse, one byte per 32-bit word (low half 0x01XX, never the
//   poison). The empty partial (seq_lens[b] == 0, or an lse of NaN / +inf)
//   travels as o = 0, lse = -inf, so the receiver merges exactly as combine
//   does. Every block sends, re-poisons its share of the stage the previous
//   call used, then merges; the stages rotate through state[2] as the
//   all-reduce's do. A merge item is half of one local head's 128 output
//   columns for RB rows: the rows' o land in bf16 (as combine stored them)
//   in shared memory, then
//   gated[b, j*128 + dv] = bf16(sum_c W_UV_j[dv, c] * o[c]) * bf16(sigmoid(gate)),
//   the sum f32 over a warp's 32 lanes, 16 columns each.
//
//   -DCAKE: the partials are the Cake wide decode's (FlashInfer's Kimi-K3 MLA
//   over the fp8 cache, source/flashinfer-cake-k3-mla), every row run as the
//   same `split` splits, the first five parameters
//       const bf16* part_o,     // [R][HEADS][split][LAT]  the splits' normalized o
//       const float* part_max,  // [R][HEADS][split]       their max, log2 units
//       const float* part_sum,  // [R][HEADS][split]       their sum below it
//       const int* seq_lens, int split
//   A split's log2-sum-exp is max + log2(sum), -inf for an empty split
//   (max -inf, or sum 0), off the true one by a constant of the kernel, the
//   same on every split and member, so the merges never see it. The splits
//   merge as Cake's reduction does: m their largest, w = exp2(l_s - m),
//   o = bf16(sum_s w o_s / sum_s w), lse = (m + log2(sum_s w)) / log2(e).
//
//   nvcc -cubin -arch=sm_103a source/k3_dcp.cu
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

#ifndef HEADS
#define HEADS 96
#endif
#ifndef NRANKS
#define NRANKS 8
#endif
#define LAT 512
#define HL (HEADS / NRANKS)
#define VEC 8                 // bf16 per 16-byte vector
#define LANES (LAT / VEC)     // 64 threads cover one head's row
#define HPB (256 / LANES)     // heads per block

static_assert(HEADS % NRANKS == 0, "every member takes the same number of heads");
static_assert(HEADS % HPB == 0 && HL % HPB == 0, "a block's four heads never straddle two members");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ long long chunk(int R) { return (long long)R * HL * (LAT + 2); }

__device__ __forceinline__ float* chunk_lse(bf16* buf, int q, int R) {
  return reinterpret_cast<float*>(buf + q * chunk(R) + (long long)R * HL * LAT);
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_fixup(bf16* __restrict__ o_lat, float* __restrict__ lse,
                                                                   const int* __restrict__ seq_lens, int R) {
  const int b = blockIdx.x;
  if (seq_lens[b] != 0) return;
  uint4* o = reinterpret_cast<uint4*>(o_lat + (long long)b * HEADS * LAT);
  for (int i = threadIdx.x; i < HEADS * LANES; i += blockDim.x) o[i] = make_uint4(0, 0, 0, 0);
  for (int h = threadIdx.x; h < HEADS; h += blockDim.x) lse[(long long)b * HEADS + h] = -CUDART_INF_F;
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_pack(const bf16* __restrict__ o_lat,
                                                                  const float* __restrict__ lse,
                                                                  bf16* __restrict__ send, int R) {
  const int b = blockIdx.x, h = blockIdx.y * HPB + threadIdx.x / LANES, v = threadIdx.x % LANES;
  const int q = h / HL, j = h % HL;
  const uint4 x = reinterpret_cast<const uint4*>(o_lat + ((long long)b * HEADS + h) * LAT)[v];
  reinterpret_cast<uint4*>(send + q * chunk(R) + ((long long)b * HL + j) * LAT)[v] = x;
  if (v == 0) chunk_lse(send, q, R)[(long long)b * HL + j] = lse[(long long)b * HEADS + h];
}

extern "C" __global__ void __launch_bounds__(256) kern_k3_dcp_combine(const bf16* __restrict__ recv,
                                                                     bf16* __restrict__ o, int R) {
  const int b = blockIdx.x, j = blockIdx.y * HPB + threadIdx.x / LANES, v = threadIdx.x % LANES;
  float l[NRANKS];
  float m = -CUDART_INF_F;
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) {
    const float x = chunk_lse(const_cast<bf16*>(recv), q, R)[(long long)b * HL + j];
    l[q] = (x != x || x == CUDART_INF_F) ? -CUDART_INF_F : x;
    m = fmaxf(m, l[q]);
  }
  float acc[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float den = 0.f;
  if (m != -CUDART_INF_F) {
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      const float w = expf(l[q] - m);
      if (w == 0.f) continue;
      den += w;
      const uint4 x = reinterpret_cast<const uint4*>(recv + q * chunk(R) + ((long long)b * HL + j) * LAT)[v];
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        const float2 f = __bfloat1622float2(x2[k]);
        acc[2 * k] = fmaf(w, f.x, acc[2 * k]);
        acc[2 * k + 1] = fmaf(w, f.y, acc[2 * k + 1]);
      }
    }
  }
  const float r = den > 0.f ? 1.0f / den : 0.f;
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int k = 0; k < VEC / 2; ++k) o2[k] = __floats2bfloat162_rn(acc[2 * k] * r, acc[2 * k + 1] * r);
  reinterpret_cast<uint4*>(o + ((long long)b * HL + j) * LAT)[v] = out;
}

#define REC (LANES + 1)
#define SPLITS_AT_ONCE 16
#define NOPE 128
#define RB (256 / LANES)   // rows of a merge item
#define M_TILE 128
#define LOG2E 1.4426950408889634f

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

#ifdef CAKE
// A Cake split's log2-sum-exp from its (max, sum); an empty split (no tile of the row) is -inf.
__device__ __forceinline__ float split_lse(float m, float sum) {
  return m == -CUDART_INF_F || !(sum > 0.f) ? -CUDART_INF_F : m + lg2(sum);
}
#endif

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}


__device__ __forceinline__ bool poisoned(uint32_t w) { return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u; }

__device__ __forceinline__ bool arrived(uint4 v) {
  return !(poisoned(v.x) || poisoned(v.y) || poisoned(v.z) || poisoned(v.w));
}

__device__ __forceinline__ uint32_t unpoison(uint32_t w) {
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}

__device__ __forceinline__ uint4 lse_word(float l) {
  const uint32_t u = __float_as_uint(l);
  return make_uint4((u & 0xffu) | 0x100u, ((u >> 8) & 0xffu) | 0x100u, ((u >> 16) & 0xffu) | 0x100u, (u >> 24) | 0x100u);
}

__device__ __forceinline__ float word_lse(uint4 v) {
  return __uint_as_float((v.x & 0xffu) | (v.y & 0xffu) << 8 | (v.z & 0xffu) << 16 | (v.w & 0xffu) << 24);
}

#define QROWS (NOPE / 4)

// Quarter q of head j's W_UV (rows QROWS * q.., [QROWS][LAT] bf16, 32 KB) into buffer q % 2 of `sw`, one
// cp.async group in flight until wait_w.
__device__ __forceinline__ void stage_w(uint4* sw, const bf16* w_kv_b, int j, int q) {
  const uint4* g = reinterpret_cast<const uint4*>(w_kv_b + ((long long)j * 256 + NOPE + q * QROWS) * LAT);
  uint4* d = sw + (q % 2) * (QROWS * LAT / 8);
  for (int i = threadIdx.x; i < QROWS * LAT / 8; i += blockDim.x)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(d + i)),
                 "l"(g + i));
  asm volatile("cp.async.commit_group;");
}

// All but the latest `pending` groups of this thread have landed.
template <int pending>
__device__ __forceinline__ void wait_w() { asm volatile("cp.async.wait_group %0;" ::"n"(pending) : "memory"); }

// v[q] = the vector at p + q * stride for every member q, all loads in flight at once, re-polling the
// ones that have not arrived; `fail` takes 1 + a late member past the deadline.
__device__ __forceinline__ void await_all(uint4 (&v)[NRANKS], const uint4* p, long long stride, long long timeout_ns,
                                          int& fail) {
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) v[q] = ld_volatile(p + q * stride);
  // The deadline counts SM cycles (~2 per ns): %globaltimer is resynced to the host clock early in a
  // run and its jumps fired the timeout with nothing late.
  long long t0 = 0;
  while (true) {
    int missing = 0;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q)
      if (!arrived(v[q])) {
        v[q] = ld_volatile(p + q * stride);
        missing = 1 + q;
      }
    if (!missing || fail) return;
    const long long now = clock64();
    if (t0 == 0) {
      t0 = now;
    } else if (now - t0 > 2 * timeout_ns) {
      fail = missing;
      return;
    }
  }
}

// await_all of two vectors of every member's record at once (the lse word and a vector of o): both
// loads in flight together instead of one wait after the other.
__device__ __forceinline__ void await_all2(uint4 (&u)[NRANKS], const uint4* pu, uint4 (&v)[NRANKS], const uint4* pv,
                                           long long stride, long long timeout_ns, int& fail) {
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) u[q] = ld_volatile(pu + q * stride), v[q] = ld_volatile(pv + q * stride);
  long long t0 = 0;
  while (true) {
    int missing = 0;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      if (!arrived(u[q])) u[q] = ld_volatile(pu + q * stride), missing = 1 + q;
      if (!arrived(v[q])) v[q] = ld_volatile(pv + q * stride), missing = 1 + q;
    }
    if (!missing || fail) return;
    const long long now = clock64();
    if (t0 == 0) {
      t0 = now;
    } else if (now - t0 > 2 * timeout_ns) {
      fail = missing;
      return;
    }
  }
}

// The sums over the warp of 16 values a lane, the butterfly of xor 1, 2, .. 16 (own + partner's) as a
// reduce-scatter: a stage halves the values a lane carries, 16 shuffles for all instead of 80, each
// sum bit for bit the butterfly's. Lane l ends with value (l & 1) * 8 + (l >> 1 & 1) * 4 +
// (l >> 2 & 1) * 2 + (l >> 3 & 1); lane l ^ 16 holds the same.
__device__ __forceinline__ float warp_sums16(float (&a)[16], int lane) {
#pragma unroll
  for (int o = 1, n = 16; o < 32; o <<= 1, n = n > 1 ? n / 2 : 1) {
    if (n > 1) {
      const bool up = lane & o;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (k < n / 2) {
          const float keep = up ? a[k + n / 2] : a[k], send = up ? a[k] : a[k + n / 2];
          a[k] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    } else {
      a[0] += __shfl_xor_sync(0xffffffffu, a[0], o);
    }
  }
  return a[0];
}

extern "C" __global__ void __launch_bounds__(256, 2) kern_k3_dcp_exchange(
#ifdef CAKE
    const bf16* __restrict__ part_o, const float* __restrict__ part_max, const float* __restrict__ part_sum,
    const int* __restrict__ seq_lens, int split,
#else
    const float* __restrict__ acc_o, const float* __restrict__ acc_lse, const int* __restrict__ seq_lens,
    const int* __restrict__ bsk, int split_max,
#endif
    const bf16* __restrict__ w_kv_b,
    const bf16* __restrict__ mla_gate, bf16* __restrict__ gated, uint8_t* lamport,
    const unsigned long long* __restrict__ peers, int* state, int* err,
    int rank, int R, long long stage_bytes, long long timeout_ns) {
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
  const long long slot = (long long)R * HL * REC;
  const long long stage = (flag % 3) * stage_bytes;
  const uint4* mine = reinterpret_cast<const uint4*>(lamport + stage);
  uint4* clear_buf = reinterpret_cast<uint4*>(lamport + ((flag + 2) % 3) * stage_bytes);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(state, 1);
  // o_proj's GEMM behind this kernel launches as a programmatic dependent and waits on its own; the
  // trigger fires once every block has run it, so every block is resident before a GEMM block lands.
  asm volatile("griddepcontrol.launch_dependents;");
  extern __shared__ uint4 sw[];  // [2][QROWS][LAT / 8]
  // a merge item is (row group, local head j, half of j's dv): `it` = (rows * HL + j) * 2 + half
  const int items = (R + RB - 1) / RB * HL * 2;
  if (blockIdx.x < items) stage_w(sw, w_kv_b, blockIdx.x / 2 % HL, blockIdx.x % 2 * 2), stage_w(sw, w_kv_b, blockIdx.x / 2 % HL, blockIdx.x % 2 * 2 + 1);

  const int lane = threadIdx.x % LANES, sub = threadIdx.x / LANES;
  for (int it = blockIdx.x; it < R * (HEADS / HPB); it += gridDim.x) {
    const int b = it / (HEADS / HPB), h = it % (HEADS / HPB) * HPB + sub;
#ifdef CAKE
    const long long rec = (long long)b * HEADS + h;
    const float* ms = part_max + rec * split;
    const float* ss = part_sum + rec * split;
    const uint4* src = reinterpret_cast<const uint4*>(part_o + rec * split * LAT) + lane;
    float l = -CUDART_INF_F;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (seq_lens[b] != 0) {
      float m = -CUDART_INF_F;
#pragma unroll 1
      for (int i = 0; i < split; ++i) m = fmaxf(m, split_lse(ms[i], ss[i]));
      if (m != -CUDART_INF_F) {
        float a[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float den = 0.f;
#pragma unroll 4
        for (int i = 0; i < split; ++i) {
          const float li = split_lse(ms[i], ss[i]);
          if (li == -CUDART_INF_F) continue;
          const float w = ex2(li - m);
          const uint4 x = src[i * LANES];
          const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
          den += w;
#pragma unroll
          for (int k = 0; k < VEC / 2; ++k) {
            const float2 f = __bfloat1622float2(x2[k]);
            a[2 * k] = fmaf(w, f.x, a[2 * k]), a[2 * k + 1] = fmaf(w, f.y, a[2 * k + 1]);
          }
        }
        const float r = 1.0f / den;
        __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) v2[k] = __floats2bfloat162_rn(a[2 * k] * r, a[2 * k + 1] * r);
        v = make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
        l = (m + lg2(den)) * (1.0f / LOG2E);
      }
    }
#else
    // split 0 is loaded with the row's length, whether or not the row has any position here
    const long long rec = (long long)b * M_TILE + h;
    const float* ls = acc_lse + rec * split_max;
    const float4* src = reinterpret_cast<const float4*>(acc_o + rec * split_max * LAT) + 2 * lane;
    const float4 x0 = src[0], y0 = src[1];
    const int n = seq_lens[b], asked = bsk[b];
    // the first SPLITS_AT_ONCE splits' lse in registers, every load issued before the first use
    float lv[SPLITS_AT_ONCE];
#pragma unroll
    for (int i = 0; i < SPLITS_AT_ONCE; ++i) lv[i] = i == 0 || i < asked ? ls[i] : 0.f;
    float l = -CUDART_INF_F;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (n != 0) {
      const int tiles = (n + M_TILE - 1) / M_TILE, per = (tiles + asked - 1) / asked, splits = (tiles + per - 1) / per;
      float m = lv[0];
#pragma unroll
      for (int i = 1; i < SPLITS_AT_ONCE; ++i)
        if (i < splits) m = fmaxf(m, lv[i]);
#pragma unroll 1
      for (int i = SPLITS_AT_ONCE; i < splits; ++i) m = fmaxf(m, ls[i]);
      if (m == -CUDART_INF_F) m = 0.f;
      float sum = ex2(lv[0] - m);
#pragma unroll
      for (int i = 1; i < SPLITS_AT_ONCE; ++i)
        if (i < splits) sum += ex2(lv[i] - m);
#pragma unroll 1
      for (int i = SPLITS_AT_ONCE; i < splits; ++i) sum += ex2(ls[i] - m);
      l = sum != 0.f ? m + lg2(sum) : CUDART_INF_F;
      float a[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      // the first SPLITS_AT_ONCE splits' o four at a time, then any more one by one, in split order
#pragma unroll
      for (int g = 0; g < SPLITS_AT_ONCE; g += 4) {
        if (g >= splits) break;
        float4 x[4], y[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int i = g + k;
          x[k] = i == 0 ? x0 : i < splits ? src[i * (LAT / 4)] : x0;
          y[k] = i == 0 ? y0 : i < splits ? src[i * (LAT / 4) + 1] : y0;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if (g + k >= splits) break;
          const float w = ex2(lv[g + k] - l);
          a[0] = fmaf(x[k].x, w, a[0]), a[1] = fmaf(x[k].y, w, a[1]), a[2] = fmaf(x[k].z, w, a[2]), a[3] = fmaf(x[k].w, w, a[3]);
          a[4] = fmaf(y[k].x, w, a[4]), a[5] = fmaf(y[k].y, w, a[5]), a[6] = fmaf(y[k].z, w, a[6]), a[7] = fmaf(y[k].w, w, a[7]);
        }
      }
#pragma unroll 1
      for (int i = SPLITS_AT_ONCE; i < splits; ++i) {
        const float w = ex2(ls[i] - l);
        const float4 x = src[i * (LAT / 4)], y = src[i * (LAT / 4) + 1];
        a[0] = fmaf(x.x, w, a[0]), a[1] = fmaf(x.y, w, a[1]), a[2] = fmaf(x.z, w, a[2]), a[3] = fmaf(x.w, w, a[3]);
        a[4] = fmaf(y.x, w, a[4]), a[5] = fmaf(y.y, w, a[5]), a[6] = fmaf(y.z, w, a[6]), a[7] = fmaf(y.w, w, a[7]);
      }
      __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) v2[k] = __floats2bfloat162_rn(a[2 * k], a[2 * k + 1]);
      v = make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
      l *= 1.0f / LOG2E;
    }
#endif
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[h / HL]) + stage) + rank * slot +
                 ((long long)b * HL + h % HL) * REC;
    dst[lane] = v;
    if (lane == 0) dst[LANES] = lse_word(l != l || l == CUDART_INF_F ? -CUDART_INF_F : l);
  }
  const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < clear_count; i += (long long)gridDim.x * blockDim.x)
    clear_buf[i] = poison;

  __shared__ __align__(16) bf16 lat[RB][LAT];
  __shared__ __align__(16) bf16 gate[RB][NOPE];
  int fail = 0;
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int half = it % 2, j = it / 2 % HL, b0 = it / 2 / HL * RB, b = b0 + sub;
    // the item's gate rows, one vector per thread of the first RB * NOPE / 8, landed in smem after the merge
    const bool gl = threadIdx.x < RB * NOPE / 8 && b0 + threadIdx.x / (NOPE / 8) < R;
    const uint4 gv = gl ? reinterpret_cast<const uint4*>(mla_gate + (long long)(b0 + threadIdx.x / (NOPE / 8)) * (HL * NOPE) +
                                                         j * NOPE)[threadIdx.x % (NOPE / 8)]
                        : make_uint4(0, 0, 0, 0);
    if (b >= R) {
      reinterpret_cast<uint4*>(lat[sub])[lane] = make_uint4(0, 0, 0, 0);
    } else {
    const uint4* rec = mine + ((long long)b * HL + j) * REC;
    uint4 v[NRANKS], u[NRANKS];
    await_all2(u, rec + LANES, v, rec + lane, slot, timeout_ns, fail);
    float l[NRANKS];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      l[q] = word_lse(u[q]);
      m = fmaxf(m, l[q]);
    }
    float acc[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float den = 0.f;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      const uint4 x = v[q];
      const float w = m == -CUDART_INF_F ? 0.f : expf(l[q] - m);
      if (w == 0.f) continue;
      den += w;
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        const float2 f = __bfloat1622float2(x2[k]);
        acc[2 * k] = fmaf(w, f.x, acc[2 * k]);
        acc[2 * k + 1] = fmaf(w, f.y, acc[2 * k + 1]);
      }
    }
    const float r = den > 0.f ? 1.0f / den : 0.f;
    uint4 out;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) o2[k] = __floats2bfloat162_rn(acc[2 * k] * r, acc[2 * k + 1] * r);
    reinterpret_cast<uint4*>(lat[sub])[lane] = out;
    }
    if (threadIdx.x < RB * NOPE / 8) reinterpret_cast<uint4*>(&gate[0][0])[threadIdx.x] = gv;
    __syncthreads();
    const int dl = threadIdx.x / 32, kl = threadIdx.x % 32;
    float x[RB][16];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 v = reinterpret_cast<const uint4*>(lat[r])[32 * i + kl];
        const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __bfloat1622float2(v2[k]);
          x[r][8 * i + 2 * k] = f.x;
          x[r][8 * i + 2 * k + 1] = f.y;
        }
      }
    wait_w<0>();
    __syncthreads();
    // warp dl's dv are half * 64 + dl + 8 k: four at a time, the four rows' sums of each in one reduce-scatter
    const int vi = (kl & 1) * 8 + (kl >> 1 & 1) * 4 + (kl >> 2 & 1) * 2 + (kl >> 3 & 1);
#pragma unroll 1
    for (int k0 = 0; k0 < 8; k0 += 4) {
      float a[16];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int dv = half * NOPE / 2 + dl + 8 * (k0 + k);
        const uint4* w = sw + (dv - half * NOPE / 2) * (LAT / 8) + kl;
        const uint4 wv[2] = {w[0], w[32]};
#pragma unroll
        for (int r = 0; r < RB; ++r) a[r * 4 + k] = 0.f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[i]);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const float2 g = __bfloat1622float2(w2[c]);
#pragma unroll
            for (int r = 0; r < RB; ++r) {
              a[r * 4 + k] = fmaf(g.x, x[r][8 * i + 2 * c], a[r * 4 + k]);
              a[r * 4 + k] = fmaf(g.y, x[r][8 * i + 2 * c + 1], a[r * 4 + k]);
            }
          }
        }
      }
      const float sum = warp_sums16(a, kl);
      const int r = vi / 4, dv = half * NOPE / 2 + dl + 8 * (k0 + vi % 4);
      if (kl < 16 && b0 + r < R) {
        const long long oi = (long long)(b0 + r) * (HL * NOPE) + j * NOPE + dv;
        const float gf = __bfloat162float(gate[r][dv]);
        gated[oi] = __hmul(__float2bfloat16_rn(sum), __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
      }
    }
    __syncthreads();
    const int next = it + gridDim.x;
    if (next < items) stage_w(sw, w_kv_b, next / 2 % HL, next % 2 * 2), stage_w(sw, w_kv_b, next / 2 % HL, next % 2 * 2 + 1);
  }
  if (fail) {
    // A member's data never came: whatever this step computes from here is wrong, so stop it loudly
    // (`err` names the member; a printf here would give the kernel a stack frame).
    atomicMax(err, fail);
    __threadfence_system();
    __trap();
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
    }
    state[2] = (flag + 1) % 3;
    *clear_ptr = NRANKS * slot;
    state[0] = 0;
  }
}
