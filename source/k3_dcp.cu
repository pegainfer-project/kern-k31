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
//   -DSPARSE: the partial is K3.1's sparse MLA over this member's selected positions (TRT-LLM gen's
//   sparse kernel, scripts/k31_sparse_mla_abi.py), one split a row, its heads padded to HEADS_PAD, and
//   the v-up moves before the wire. W_UV is linear, so W_UV (sum_q w_q o_q) = sum_q w_q (W_UV o_q):
//   every member expands its own partial of every head, 512 latent columns to 128, and a record carries
//   128 values instead of 512, a quarter of the bytes for eight times the v-up's FLOPs (on tcgen05:
//   alone on one GPU 4.5 us at 64 rows, 7.5 us at 256). The parameters
//       CUtensorMap o_map,      // part_o [rows, HEADS_PAD, LAT] bf16 normalized o: dims (LAT, HEADS_PAD,
//                               //   rows), box (64, 1, 128), 128 B swizzle
//       const float2* md,       // [R][HEADS_PAD]  (max of the scaled logits, natural units; sum below it)
//       const int* live,        // [R]  selected positions this member holds; 0: an empty partial
//       CUtensorMap w_map,      // w_kv_b [HEADS*256, LAT] bf16, every head's: box (64, 128), 128 B swizzle
//       then mla_gate .. timeout_ns as above
//   grid: any grid whose blocks are all resident at once (one an SM)   block 256
//   dynamic smem SMEM_BYTES: four stages of a chunk of the partial and of W_UV, the result tile
//
//   A partial's lse is (log2(e) * max + log2(sum)) / log2(e), as FlashInfer's ComputeLSEFromMD has
//   it; a row with nothing here ran one dummy position, and travels as the empty partial (o = 0,
//   lse = -inf). A push item is 128 rows of one head: warp 0 streams its eight 64-column chunks of
//   the partial and of W_UV_h through the stages by TMA (W's first stages before griddepcontrol.wait:
//   a weight), warp 1 runs M 128 x N 128 x K 16 MMAs into one TMEM accumulator, warps 4..7 drain it a
//   row a thread into a tile of records,
//     v[b, h, dv] = bf16(sum_c W_UV_h[dv, c] * o[b, h, c]),  f32 accumulate, the tensor core's order
//   (-0.0 sent as +0.0), and push the tile to the owner 16-byte vector by vector. The owner merges as
//   combine does, each thread one vector of a record:
//     gated[b, j*128 + dv] = bf16(sum_q w_q v_q / sum_q w_q) * bf16(sigmoid(gate)).
//   On the wire 17 vectors a record; bf16 is kept there because it is already closer to the exact
//   result than the latent merge was: relative RMS against f64 2.98e-3 (f32 on the wire: 2.70e-3;
//   the latent merge's 3.17e-3; bf16 of the exact result alone 1.66e-3).
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
#ifndef HEADS_PAD
#define HEADS_PAD 128
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

#ifndef SPARSE
#define REC (LANES + 1)

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
  // o_proj's GEMM behind this kernel launches as a programmatic dependent and waits on its own; the
  // trigger fires once every block has run it, so every block is resident before a GEMM block lands.
  asm volatile("griddepcontrol.launch_dependents;");
  extern __shared__ uint4 sw[];  // [2][QROWS][LAT / 8]
  // a merge item is (row group, local head j, half of j's dv): `it` = (rows * HL + j) * 2 + half
  const int items = (R + RB - 1) / RB * HL * 2;
  if (blockIdx.x < items) stage_w(sw, w_kv_b, blockIdx.x / 2 % HL, blockIdx.x % 2 * 2), stage_w(sw, w_kv_b, blockIdx.x / 2 % HL, blockIdx.x % 2 * 2 + 1);
  // D launches this kernel as a programmatic dependent of the attention: only the weight is read
  // before the wait, the state the previous call left included after it.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
  const long long slot = (long long)R * HL * REC;
  const long long stage = (flag % 3) * stage_bytes;
  const uint4* mine = reinterpret_cast<const uint4*>(lamport + stage);
  uint4* clear_buf = reinterpret_cast<uint4*>(lamport + ((flag + 2) % 3) * stage_bytes);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(state, 1);

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
#else
#include <cuda.h>

#define VL (NOPE / VEC)              // vectors of a record's values
#define REC (VL + 1)
#define BM 128                       // rows of a push item: the MMA's M, a TMEM lane each
#define KC 64                        // latent columns of a chunk: one 128-byte swizzle atom a row
#define CHUNKS (LAT / KC)
#define STAGES 4
#define TILE_BYTES (BM * KC * 2)     // 16 KB
#define STAGE_BYTES (2 * TILE_BYTES) // the partial's rows, then W_UV's 128
#define OPITCH (REC * VEC)           // the result tile's rows, records as they travel: 272 bytes, a warp's 16-byte stores in 4 wavefronts
#define SMEM_BYTES (1024 + STAGES * STAGE_BYTES + BM * OPITCH * 2 + 128)

// K-major, 128 B swizzle: start >> 4 | LBO 16 >> 4 << 16 | SBO 1024 >> 4 << 32 | version 1 << 46 | 2 << 61
__device__ __forceinline__ uint64_t umma_desc(uint32_t addr) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void mwait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(bar),
               "r"(parity)
               : "memory");
}

__device__ __forceinline__ void expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// chunk kc of head h: the partial's rows r0 .. r0 + BM (past the buffer: zero), W_UV_h's 128 rows
__device__ __forceinline__ void load_a(uint32_t dst, const CUtensorMap* m, int kc, int h, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(m), "r"(kc * KC), "r"(h), "r"(r0), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void load_w(uint32_t dst, const CUtensorMap* m, int kc, int h, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(m), "r"(kc * KC), "r"(h * 256 + NOPE), "r"(bar)
               : "memory");
}

// A partial with nothing in it: no selected position on this member, or a head without mass.
__device__ __forceinline__ bool empty(const float2* md, const int* live, int b, int h) {
  if (live[b] == 0) return true;
  const float2 s = md[(long long)b * HEADS_PAD + h];
  return s.x == -CUDART_INF_F || !(s.y > 0.f);
}

extern "C" __global__ void __launch_bounds__(256, 1) kern_k3_dcp_exchange(
    const __grid_constant__ CUtensorMap o_map, const float2* __restrict__ md, const int* __restrict__ live,
    const __grid_constant__ CUtensorMap w_map, const bf16* __restrict__ mla_gate, bf16* __restrict__ gated,
    uint8_t* lamport, const unsigned long long* __restrict__ peers, int* state, int* err,
    int rank, int R, long long stage_bytes, long long timeout_ns) {
  // o_proj's GEMM behind this kernel launches as a programmatic dependent and waits on its own; the
  // trigger fires once every block has run it, so every block is resident before a GEMM block lands.
  asm volatile("griddepcontrol.launch_dependents;");
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  const uint32_t tile_s = base + STAGES * STAGE_BYTES, bars = tile_s + BM * OPITCH * 2;
  const uint32_t full = bars, emptied = full + 8 * STAGES, tfull = emptied + 8 * STAGES, tempty = tfull + 8,
                 hold = tempty + 8;
  bf16* const tile = reinterpret_cast<bf16*>(smem_raw + (tile_s - raw));
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  // a push item is head it % HEADS of the rows BM * (it / HEADS) ..
  const int items = (R + BM - 1) / BM * HEADS;
  const bool pushes = blockIdx.x < items;
  // D launches this kernel as a programmatic dependent of the attention: only the first item's W_UV is
  // read before the wait (its stages armed for the partial's bytes too), the state the previous call
  // left included after it.
  if (t == 0) {
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(full + 8 * s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(emptied + 8 * s));
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tfull));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(tempty));
    asm volatile("fence.mbarrier_init.release.cluster;");
    if (pushes)
      for (int s = 0; s < STAGES; ++s) {
        expect(full + 8 * s, STAGE_BYTES);
        load_w(base + s * STAGE_BYTES + TILE_BYTES, &w_map, s, blockIdx.x % HEADS, full + 8 * s);
      }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int flag = state[2];
  long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
  const long long clear_count = *clear_ptr;
  const long long slot = (long long)R * HL * REC;
  const long long stage = (flag % 3) * stage_bytes;
  const uint4* mine = reinterpret_cast<const uint4*>(lamport + stage);
  uint4* clear_buf = reinterpret_cast<uint4*>(lamport + ((flag + 2) % 3) * stage_bytes);
  if (pushes && warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(hold), "r"(NOPE));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(smem_raw + (hold - raw));
  if (t == 0) atomicAdd(state, 1);

  // Warp 0 streams each item's eight chunks through the stages (TMA), warp 1 issues the M 128 x N 128 x
  // K 16 MMAs into the TMEM accumulator, warps 4..7 drain it a row a thread into the result tile and
  // push the tile's records to the head's owner, 16 threads to a record's 256 bytes.
  if (warp == 0) {
    if (lane == 0) {
      uint32_t s = 0, ph = 0;
      for (int it = blockIdx.x; it < items; it += gridDim.x) {
        const int h = it % HEADS, r0 = it / HEADS * BM;
        for (int kc = 0; kc < CHUNKS; ++kc) {
          const uint32_t a = base + s * STAGE_BYTES, f = full + 8 * s;
          if (it == blockIdx.x && kc < STAGES) {
            load_a(a, &o_map, kc, h, r0, f);
          } else {
            mwait(emptied + 8 * s, ph ^ 1);
            expect(f, STAGE_BYTES);
            load_a(a, &o_map, kc, h, r0, f);
            load_w(a + TILE_BYTES, &w_map, kc, h, f);
          }
          if (++s == STAGES) s = 0, ph ^= 1;
        }
      }
    }
  } else if (warp == 1) {
    // bf16 A and B, both K-major, f32 D, M 128, N 128
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(NOPE >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);
    uint32_t s = 0, ph = 0, n = 0;
    for (int it = blockIdx.x; it < items; it += gridDim.x, ++n) {
      mwait(tempty, (n & 1) ^ 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for (int kc = 0; kc < CHUNKS; ++kc) {
        mwait(full + 8 * s, ph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t a = base + s * STAGE_BYTES;
#pragma unroll
        for (int k = 0; k < KC / 16; ++k)
          asm volatile("{\n.reg .pred p, e;\nsetp.ne.b32 p, %4, 0;\nelect.sync _|e, 0xffffffff;\n"
                       "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem),
                       "l"(umma_desc(a + k * 32)), "l"(umma_desc(a + TILE_BYTES + k * 32)), "r"(idesc), "r"(kc | k));
        asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
                     "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(emptied + 8 * s));
        if (++s == STAGES) s = 0, ph ^= 1;
      }
      asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
                   "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(tfull));
    }
  } else if (warp >= 4) {
    // warp 4 + q drains TMEM lanes 32 q .. +32 into the tile, a row (a record: the head's values, then its lse
    // word) a thread
    const int e = t - 128, r = 32 * (warp - 4) + lane;
    uint4* const row = reinterpret_cast<uint4*>(tile + r * OPITCH);
    uint32_t n = 0;
    for (int it = blockIdx.x; it < items; it += gridDim.x, ++n) {
      const int h = it % HEADS, r0 = it / HEADS * BM, b = r0 + r;
      // the row's lse, its loads in flight while the MMAs run; the empty partial travels as o = 0, lse = -inf
      const bool full_row = b < R && !empty(md, live, b, h);
      float l = -CUDART_INF_F;
      if (full_row) {
        const float2 s = md[(long long)b * HEADS_PAD + h];
        l = (LOG2E * s.x + lg2(s.y)) * (1.0f / LOG2E);
      }
      mwait(tfull, n & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int c = 0; c < NOPE / 32; ++c) {
        uint32_t v[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
              "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
              "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
              "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
            : "r"(tmem + ((32 * (warp - 4)) << 16) + 32 * c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          uint32_t w[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const __nv_bfloat162 x =
                __floats2bfloat162_rn(__uint_as_float(v[8 * u + 2 * k]), __uint_as_float(v[8 * u + 2 * k + 1]));
            w[k] = full_row ? unpoison(*reinterpret_cast<const uint32_t*>(&x)) : 0u;
          }
          row[4 * c + u] = make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
      row[VL] = lse_word(l != l || l == CUDART_INF_F ? -CUDART_INF_F : l);
      // the accumulator is read: the next item's MMAs may overwrite it
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tempty) : "memory");
      asm volatile("bar.sync 1, 128;" ::: "memory");
      uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[h / HL]) + stage) + rank * slot +
                   (long long)h % HL * REC;
#pragma unroll
      for (int k = 0; k < REC; ++k) {
        const int i = e + 128 * k, rr = i / REC, v = i % REC;
        if (r0 + rr < R)
          dst[(long long)(r0 + rr) * HL * REC + v] = reinterpret_cast<const uint4*>(tile + rr * OPITCH)[v];
      }
      // the tile is read before the next item's rows land on it
      asm volatile("bar.sync 1, 128;" ::: "memory");
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (pushes && warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(NOPE));

  const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (long long i = (long long)blockIdx.x * blockDim.x + t; i < clear_count; i += (long long)gridDim.x * blockDim.x)
    clear_buf[i] = poison;

  // a merge thread is one vector of one record: row b, local head j, VEC of its dv
  int fail = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + t; i < (long long)R * HL * VL;
       i += (long long)gridDim.x * blockDim.x) {
    const int v = i % VL;
    const long long rec = i / VL;
    const uint4* p = mine + rec * REC;
    uint4 u[NRANKS], x[NRANKS];
    await_all2(u, p + VL, x, p + v, slot, timeout_ns, fail);
    float l[NRANKS];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      l[q] = word_lse(u[q]);
      m = fmaxf(m, l[q]);
    }
    float acc[VEC] = {};
    float den = 0.f;
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
      const float w = m == -CUDART_INF_F ? 0.f : expf(l[q] - m);
      if (w == 0.f) continue;
      den += w;
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x[q]);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        const float2 f = __bfloat1622float2(x2[k]);
        acc[2 * k] = fmaf(w, f.x, acc[2 * k]);
        acc[2 * k + 1] = fmaf(w, f.y, acc[2 * k + 1]);
      }
    }
    const float rd = den > 0.f ? 1.0f / den : 0.f;
    // rec = b * HL + j, so the record's columns of `gated` (and of the gate) start at rec * NOPE
    const long long oi = rec * NOPE + v * VEC;
    const uint4 gv = *reinterpret_cast<const uint4*>(mla_gate + oi);
    const bf16* g = reinterpret_cast<const bf16*>(&gv);
    uint4 out;
    bf16* o = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int k = 0; k < VEC; ++k)
      o[k] = __hmul(__float2bfloat16_rn(acc[k] * rd), __float2bfloat16_rn(1.0f / (1.0f + expf(-__bfloat162float(g[k])))));
    *reinterpret_cast<uint4*>(gated + oi) = out;
  }
  if (fail) {
    // A member's data never came: whatever this step computes from here is wrong, so stop it loudly
    // (`err` names the member; a printf here would give the kernel a stack frame).
    atomicMax(err, fail);
    __threadfence_system();
    __trap();
  }

  if (blockIdx.x == 0 && t == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
    }
    state[2] = (flag + 1) % 3;
    *clear_ptr = NRANKS * slot;
    state[0] = 0;
  }
}
#endif
