// A DCP step's two all-reduces with the work on either side of them, on the
// Lamport one-shot of peer_allreduce_bf16.cu (TensorRT-LLM's allreduce_fusion
// protocol): every rank pushes its bf16 partial into slot `rank` of every
// peer's stage, a 16-byte word holding a bf16 -0.0 has not arrived, every
// rank sums the slots in rank order in f32 and rounds once. Three stages
// rotate by call, each block keeping the rotation itself (`state`, below).
//
// The exchange runs over a row's sixteen-byte vectors. Every block pushes an
// equal run of all rows' vectors (the two-shot: the first blocks, a vector a
// thread) and re-poisons a share of the previous call's stage; block b < B
// then waits for row b and runs the epilogue on it,
// a row per 1024 threads. From TWO_SHOT_ROWS rows a row goes to its owner
// (row % NRANKS) only, whose row block sums it as the one-shot does; the
// blocks of the rows a rank does not own first gather its rows' sums to the
// others (gather_share). The blocks past B load the next GEMM's weight
// `pf` (`pf_bytes`) into L2 while a row block still waits (k3_l2_prefetch.cuh;
// state[2 * phase] stamps a row's data in, state[7] is the loads' sink);
// kern_k3_ar_finalize_rms's then the next layer's o_proj
// weight `pf2` at evict_last, which the kern_k3_ar_attnres_rms after that
// o_proj puts back to evict_normal (`dm`).
//
//   kern_k3_ar_attnres_rms_nb<NB>(const bf16* x, u8* lamport, const u64* lamport_peers, i32* state,
//       i32* err, int rank, const bf16* prefix, const bf16* blocks, const f32* sw,
//       const bf16* gamma, bf16* prefix2, bf16* normed, int snapshot, int B,
//       i64 stage_bytes, i64 timeout_ns, const u8* pf, i64 pf_bytes, const u8* dm, i64 dm_bytes)
//     o_proj's partials x [B, H] summed, then k3_residual.cu's K1b (-DLAND_BF16) on the sum
//     with NB snapshots (0..KNB_MAX).
//
//   kern_k3_ar_finalize_rms(const bf16* fc2, const i32* exp2perm, const f32* wts,
//       const bf16* shared, u8* lamport, const u64* lamport_peers, i32* state, i32* err,
//       int rank, const bf16* gamma, bf16* latent_norm, bf16* shared_sum, int B, int early,
//       i64 stage_bytes, i64 timeout_ns, const u8* pf, i64 pf_bytes, const u8* pf2, i64 pf2_bytes)
//     A row's routed latent partial (k3_moe_prefill.cu's kern_k3_moe_finalize over this
//     rank's experts, bf16) and its shared expert's down partial `shared` [B, H], summed;
//     latent_norm = k3_land.cu's kern_k3_rms(latent sum, gamma), shared_sum the shared sum.
//     `early`: fc2 is done at entry (a kernel that is no programmatic dependent ran between
//     them), so its partial is weighted before griddepcontrol.wait; else after it.
//
//   grid (KGRID, 1, 1) on every call of a group   block (1024, 1, 1)
//   kern_k3_ar_attnres_rms_nb<NB>: dynamic smem KCAND_BYTES (the row's snapshots, below)
//   `err` gets 1 + the rank whose data did not show within `timeout_ns`, then the kernel traps.
//   `lamport` holds the three stages, then NRANKS handshake words (the first call's, below).
//   `state` [8 + 2 * KGRID] starts zeroed: [6] the handshake done, [8 + 2b] block b's stage,
//   [9 + 2b] the vectors it re-poisons next call.
//
// The residual rows around them:
//   kern_k3_land_add2_attnres_rms_nb<NB>   a layer's closing add, then the next layer's mix and norm:
//                                   a row a cluster of LAND_CL blocks of 1024 / LAND_CL threads, dynamic
//                                   smem KCAND_BYTES / LAND_CL; the blocks past LAND_CL * B prefetch `pf`
// A snapshot count is an entry, not a switch in one: a call runs its own code from the entry on. In
// the step every call fetches its instructions cold, and one kernel holding all nine counts (390 KB of
// code) ran the counts far from its entry 5-6 us slower than nb 1.
//   kern_k3_embed_rms(_plan)        the step's embedding gather and its first layer's norm
//                                   (_plan: and the MLA split plan)
// Every kernel here triggers its dependents at entry (griddepcontrol.launch_dependents): the
// cuBLAS GEMM after it launches as a programmatic dependent and waits for this grid itself.
// The all-reduces and the closing add may launch as programmatic dependents themselves: they
// read the kernel before them's output only past griddepcontrol.wait.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace cg = cooperative_groups;

// the loads' bytes in flight (k3_l2_prefetch.cuh's 12 MB): past the window's end the grid waits for the ones
// issued to return, and the exchange's own traffic shares HBM with them
#define L2_INFLIGHT (8 << 20)
#include "k3_l2_prefetch.cuh"

// how long the loads of the next weight go on past a row's arrival: past that the grid's end waits for them,
// not for the row epilogue (kern_k3_ar_attnres_rms's attention residual ~1.5 us, the latent norm's ~1 us)
#ifndef AR_GRACE_NS
#define AR_GRACE_NS 500
#endif
#ifndef FIN_GRACE_NS
#define FIN_GRACE_NS 500
#endif

#ifndef NRANKS
#define NRANKS 8
#endif
#define KH 7168
#define KVEC (KH / 8)
#define KLAT 3584
#define KLATV (KLAT / 8)
#define KNB_MAX 8
#define KEPS 1e-5f
#define KTHREADS 1024
#define TOPK 16
#define KGRID 144  // the exchanges' grid (gen_stage.py's TP_AR_GRID): one 1024-thread block an SM, all resident on 148 SMs
#ifndef TWO_SHOT_ROWS
#define TWO_SHOT_ROWS 14
#endif

typedef __nv_bfloat16 bf16_t;
typedef __nv_bfloat162 bf162_t;

// ---------------------------------------------------------------- protocol
// A wait's deadline counts SM cycles, `timeout_ns` at 2 a ns: %globaltimer is resynced to the host clock
// early in a process, and its jump fires a deadline with nothing late (k3_dcp.cu's await_all).
__device__ __forceinline__ long long dl_now() { return clock64(); }
__device__ __forceinline__ bool dl_past(long long t0, long long timeout_ns) { return dl_now() - t0 > 2 * timeout_ns; }

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
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t round_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) { return round_bf16(a) | (round_bf16(b) << 16); }

// A launch with `pdl` starts while the kernel before it finishes: what does not depend on that kernel
// (the stage, the snapshots, the weights) is read first, its output only past this wait.
__device__ __forceinline__ void grid_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

struct Lamport {
  uint4* const* slot;  // this rank's slot in every peer's stage (shared memory)
  const uint4* mine;   // this rank's stage
  uint4* clear_buf;    // the previous call's stage
  long long clear_count, tot;
  int phase;
};

// A deadline that passes is a broken group, never a slow one: the kernel records the late
// rank in `err` and traps, so the run fails instead of summing a slot that never arrived.
__device__ __forceinline__ void give_up(int* err, int missing) {
  atomicMax(err, missing);
  __threadfence_system();
  __trap();
}

// The first call of a process waits for every peer to have run its once-program (the
// poisoning of its stages, kern_peer_lamport_init_bf16) before pushing into them: nothing
// orders one rank's once-program after another's first step, and a push that lands before
// the peer's poison is lost. Each rank's block 0 sets `ready[rank]` in every peer's
// handshake words (past the three stages, which the init never touches and nothing
// rewrites), every block waits for all of its own, and state[6] marks the handshake done.
__device__ __forceinline__ void lamport_handshake(uint8_t* lamport, const unsigned long long* peers, int* state,
                                                  int* err, int rank, long long stage_bytes, long long timeout_ns) {
  if (threadIdx.x >= NRANKS || *reinterpret_cast<volatile int*>(state + 6)) return;
  const long long at = 3 * stage_bytes;
  if (blockIdx.x == 0)
    reinterpret_cast<volatile int*>(reinterpret_cast<uint8_t*>(peers[threadIdx.x]) + at)[rank] = 1;
  if (threadIdx.x == 0) {
    const volatile int* ready = reinterpret_cast<const volatile int*>(lamport + at);
    const long long t0 = dl_now();
    for (int q = 0; q < NRANKS; ++q)
      while (!ready[q])
        if (dl_past(t0, timeout_ns)) give_up(err, 1 + q);
  }
}

// Three stages rotate by call: a call pushes into stage `phase`, and re-poisons the stage of
// the call before it (whose sums are done everywhere: this call's pushes are in) for the call
// after next. Every block keeps the rotation itself (state[8 + 2 * block]: the phase, then
// the vectors to re-poison) and moves it on as it leaves: every call of the group runs the
// same grid, so no block waits for another to start.
__device__ __forceinline__ Lamport lamport_open(uint8_t* lamport, const unsigned long long* peers, int* state, int* err,
                                                int rank, long long tot, long long stage_bytes, long long timeout_ns) {
  __shared__ uint4* s_slot[NRANKS];
  asm volatile("griddepcontrol.launch_dependents;");
  Lamport l;
  l.slot = s_slot;
  l.phase = state[8 + 2 * blockIdx.x];
  l.clear_count = state[9 + 2 * blockIdx.x];
  l.tot = tot;
  if (threadIdx.x < NRANKS)
    s_slot[threadIdx.x] = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(peers[threadIdx.x]) +
                                                   l.phase * stage_bytes) + (long long)rank * tot;
  l.mine = reinterpret_cast<const uint4*>(lamport + l.phase * stage_bytes);
  l.clear_buf = reinterpret_cast<uint4*>(lamport + (l.phase + 2) % 3 * stage_bytes);
  lamport_handshake(lamport, peers, state, err, rank, stage_bytes, timeout_ns);
  __syncthreads();
  return l;
}

__device__ __forceinline__ void lamport_clear(const Lamport& l) {
  const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < l.clear_count;
       i += (long long)gridDim.x * blockDim.x)
    l.clear_buf[i] = poison;
}

// Vector i of a call's `tot` this thread pushes, or -1: runs of PUSH_RUN vectors dealt to the
// KGRID blocks in order, so a row's pushes leave from several SMs rather than from the one block
// a row would fill (an SM pushes ~55 GB/s). The grid is a constant: a division by gridDim.x
// costs the push 0.3 us. The two-shot (from TWO_SHOT_ROWS rows) pushes from the first blocks a
// vector a thread instead, so the blocks past them start loading the next weight (l2_load_while) at once.
#ifndef PUSH_RUN
#define PUSH_RUN 128
#endif
__device__ __forceinline__ int share_per(long long tot) {
  return (((int)tot + PUSH_RUN - 1) / PUSH_RUN + KGRID - 1) / KGRID * PUSH_RUN;
}
__device__ __forceinline__ long long share(long long tot, int B) {
  if (B >= TWO_SHOT_ROWS) {
    const int i = (int)blockIdx.x * KTHREADS + (int)threadIdx.x;
    return i < (int)tot ? i : -1;
  }
  const int per = share_per(tot);
  const int i = per * (int)blockIdx.x + (int)threadIdx.x;
  return (int)threadIdx.x < per && i < (int)tot ? i : -1;
}

__device__ __forceinline__ uint4 unpoison4(uint4 v) {
  return make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
}

__device__ __forceinline__ void lamport_push(const Lamport& l, long long i, uint4 v) {
  v = unpoison4(v);
#pragma unroll
  for (int q = 0; q < NRANKS; ++q) l.slot[q][i] = v;
}

// The rank-order f32 sum of vector i over the slots, rounded once.
__device__ __forceinline__ uint4 lamport_sum(const Lamport& l, long long i, long long timeout_ns, int& fail) {
  uint4 vals[NRANKS];
  long long t0 = 0;
  while (true) {
    int missing = 0;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      vals[r] = ld_volatile(l.mine + (long long)r * l.tot + i);
      if (missing == 0 && !arrived(vals[r])) missing = 1 + r;
    }
    if (missing == 0) break;
    if (t0 == 0) {
      t0 = dl_now();
    } else if (dl_past(t0, timeout_ns)) {
      fail = missing;
      break;
    }
  }
  float acc[8];
  acc[0] = lo(vals[0].x), acc[1] = hi(vals[0].x), acc[2] = lo(vals[0].y), acc[3] = hi(vals[0].y);
  acc[4] = lo(vals[0].z), acc[5] = hi(vals[0].z), acc[6] = lo(vals[0].w), acc[7] = hi(vals[0].w);
#pragma unroll
  for (int r = 1; r < NRANKS; ++r) {
    acc[0] += lo(vals[r].x), acc[1] += hi(vals[r].x), acc[2] += lo(vals[r].y), acc[3] += hi(vals[r].y);
    acc[4] += lo(vals[r].z), acc[5] += hi(vals[r].z), acc[6] += lo(vals[r].w), acc[7] += hi(vals[r].w);
  }
  return make_uint4(pack2(acc[0], acc[1]), pack2(acc[2], acc[3]), pack2(acc[4], acc[5]), pack2(acc[6], acc[7]));
}

// Vector i of slot q alone: a two-shot's gathered sum.
__device__ __forceinline__ uint4 lamport_one(const Lamport& l, int q, long long i, long long timeout_ns, int& fail) {
  long long t0 = 0;
  const uint4* p = l.mine + (long long)q * l.tot + i;
  while (true) {
    const uint4 v = ld_volatile(p);
    if (arrived(v)) return v;
    if (t0 == 0) {
      t0 = dl_now();
    } else if (dl_past(t0, timeout_ns)) {
      fail = 1 + q;
      return v;
    }
  }
}

// Vector i of a row's sum: the one-shot's sum of the slots, or a two-shot's (`owner` of the row): the
// owner sums the slots in rank order as the one-shot does, keeps -0.0 as +0.0 (as it is sent) and gathers
// the sum into every peer's slot `rank`; the others read the owner's slot.
__device__ __forceinline__ uint4 lamport_get(const Lamport& l, bool two, int owner, int rank, long long i,
                                             long long timeout_ns, int& fail) {
  if (two && owner != rank) return lamport_one(l, owner, i, timeout_ns, fail);
  const uint4 v = lamport_sum(l, i, timeout_ns, fail);
  return two ? unpoison4(v) : v;
}

// Two-shot: the sums of this rank's rows (row % NRANKS == rank) gathered to the other ranks by the blocks of the
// rows it does not own, an equal run of the owned vectors each (RW a row), before they wait for their own row: the
// gather leaves from ~7B/8 SMs instead of B/8. The owner's row block sums its row again for itself.
template <int RW>
__device__ __forceinline__ void gather_share(const Lamport& l, int rank, int B, long long timeout_ns, int& fail) {
  const int b = blockIdx.x, nown = (B - rank + NRANKS - 1) / NRANKS, m = B - nown;
  const int per = (nown * RW + m - 1) / m, g = (b - (b - rank + NRANKS - 1) / NRANKS) * per + (int)threadIdx.x;
  if ((int)threadIdx.x >= per || g >= nown * RW) return;
  const long long i = (long long)(rank + NRANKS * (g / RW)) * RW + g % RW;
  const uint4 v = unpoison4(lamport_sum(l, i, timeout_ns, fail));
#pragma unroll 1
  for (int q = 0; q < NRANKS; ++q)
    if (q != rank) l.slot[q][i] = v;
}

// The word l2_load_while stops on (the first row block's arrival stamp): one a phase, state[2 * phase]; the
// call of phase p zeroes phase p + 1's for the call after it, so no grid-wide count clears it at the end.
__device__ __forceinline__ unsigned* l2_stamp(int* state, const Lamport& l) {
  if (blockIdx.x == 0 && threadIdx.x == 0) state[2 * ((l.phase + 1) % 3)] = 0;
  return reinterpret_cast<unsigned*>(state) + 2 * l.phase;
}

__device__ __forceinline__ void lamport_close(const Lamport& l, int* state, int* err, int fail) {
  if (fail) give_up(err, fail);
  if (threadIdx.x == 0) {
    state[8 + 2 * blockIdx.x] = (l.phase + 1) % 3;
    state[9 + 2 * blockIdx.x] = (int)(NRANKS * l.tot);
    if (blockIdx.x == 0) state[6] = 1;
  }
}

// ---------------------------------------------------------------- row math (k3_residual.cu)
struct V8 {
  unsigned w[4];
};
__device__ __forceinline__ V8 ldv(const void* p) {
  const uint4 t = *(const uint4*)p;
  V8 v;
  v.w[0] = t.x, v.w[1] = t.y, v.w[2] = t.z, v.w[3] = t.w;
  return v;
}
__device__ __forceinline__ V8 asv(uint4 t) {
  V8 v;
  v.w[0] = t.x, v.w[1] = t.y, v.w[2] = t.z, v.w[3] = t.w;
  return v;
}
__device__ __forceinline__ void stv(void* p, const V8& v) { *(uint4*)p = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]); }
__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
  return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                            __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ unsigned from_bf162(bf162_t h) {
  return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
         ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}
__device__ __forceinline__ float2 bf2f(unsigned u) { return __bfloat1622float2(as_bf162(u)); }
__device__ __forceinline__ unsigned f2bf(float2 f) { return from_bf162(__float22bfloat162_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// k3_residual.cu's attnres_rms_row, as a short chain: thread t < KVEC holds vector t of the
// row and starts the copies of its vector of candidates 0..NB-1 into shared memory (`cand`,
// [KNB_MAX][KVEC] sixteen-byte vectors) and its sw / gamma into registers at entry; every
// candidate's sum of squares and dot with sw goes through one block reduction (the
// snapshots' before the all-reduce's poll, which they do not need), each thread runs the
// softmax over the totals, mixes in f32 and rounds to bf16 as attnres_rms_row does, and the
// mix's sum of squares is one more reduction. The candidate count is a template argument.
#define KCAND_BYTES (KNB_MAX * KVEC * 16)
#define RED_LD 33

struct RowSmem {
  float red[3][32 * RED_LD];  // one per reduction of a row: no barrier between them
};

struct RowRegs {
  float swv[8];
  V8 g;
};

template <int NB>
__device__ __forceinline__ RowRegs row_prefetch(const bf16_t* __restrict__ blk_row, const float* __restrict__ sw,
                                                const bf16_t* __restrict__ gamma, uint4* cand, int t) {
  RowRegs r;
#pragma unroll
  for (int j = 0; j < 8; ++j) r.swv[j] = 0.f;
  r.g.w[0] = 0u, r.g.w[1] = 0u, r.g.w[2] = 0u, r.g.w[3] = 0u;
  if (t < KVEC) {
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(cand + c * KVEC + t);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(blk_row + (size_t)c * KH + t * 8));
    }
    if (NB > 0) {
      const float4* sp4 = (const float4*)(sw + t * 8);
      const float4 a = sp4[0], b = sp4[1];
      r.swv[0] = a.x; r.swv[1] = a.y; r.swv[2] = a.z; r.swv[3] = a.w;
      r.swv[4] = b.x; r.swv[5] = b.y; r.swv[6] = b.z; r.swv[7] = b.w;
    }
    r.g = ldv(gamma + t * 8);
  }
  asm volatile("cp.async.commit_group;");
  return r;
}

__host__ __device__ constexpr int pow2_at_least(int n) { return n <= 1 ? 1 : 2 * pow2_at_least((n + 1) / 2); }
__host__ __device__ constexpr int log2_of(int n) { return n <= 1 ? 0 : 1 + log2_of(n / 2); }

// Recursive halving over the warp: N values in, lane l out with the warp's total of value
// l >> (5 - log2 N) in v[0] (N <= 32), N - 1 + 5 - log2 N shuffles.
template <int N, int O>
__device__ __forceinline__ void halve(float* v, int lane) {
  if constexpr (N > 1) {
    const bool up = lane & O;
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
      const float send = up ? v[j] : v[j + N / 2], keep = up ? v[j + N / 2] : v[j];
      v[j] = keep + __shfl_xor_sync(0xffffffffu, send, O);
    }
  } else {
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], O);
  }
  if constexpr (O > 1) halve<(N > 1 ? N / 2 : 1), O / 2>(v, lane);
}

#ifndef BS_SMALL
#define BS_SMALL 8
#endif
// Every thread of the 32-warp block gets the block's totals of its N values: the warps'
// totals meet in `red` behind one barrier, and every warp sums them itself.
template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N], float* red, int t) {
  constexpr int SH = 5 - log2_of(N);
  const int lane = t & 31, warp = t >> 5;
  halve<N, 16>(v, lane);
  if ((lane & ((1 << SH) - 1)) == 0) red[(lane >> SH) * RED_LD + warp] = v[0];
  __syncthreads();
  if constexpr (N <= BS_SMALL) {
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = red[j * RED_LD + lane];
    halve<N, 16>(v, lane);
    const float mine = v[0];
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = __shfl_sync(0xffffffffu, mine, j << SH);
  } else {
    // warp w sums value w, the totals behind one more barrier: N x 5 shuffles a block, not 32 x 3N
    if (warp < N) {
      float x = red[warp * RED_LD + lane];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
      if (lane == 0) red[warp * RED_LD + 32] = x;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = red[j * RED_LD + 32];
  }
}

// block_sum over a row split into CL slices of SW warps on a cluster of CL blocks (the slices aligned to
// 32 vectors): each warp's first stage goes into its row-wide warp slot in every block's `red` (the
// slots past the row from the blocks' spare warps, zero), and once its own `red` holds all of them
// every block runs block_sum's second stage over the row's 32 slots: the same sums in the same order
// as one 1024-thread block. With a `bar`, the slots go out by st.async, each completing its bytes on the
// receiving block's mbarrier (armed with cluster_sum_bytes before a cluster barrier, phase 0): a block
// waits for its own bytes, not for a cluster barrier's round trip. Without, DSMEM stores and the
// cluster barrier: st.async a value is slower once a warp sends more than a few (32 values: +0.8 us).
template <int N, int CL, int SW>
constexpr unsigned cluster_sum_bytes = CL * (SW + 1) * N * 4;

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void bar_arm(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(bar)));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

template <int N, int CL, int SW, int W>
__device__ __forceinline__ void cluster_block_sum(float (&v)[N], float* red, unsigned long long* bar, int t) {
  static_assert(CL * (SW + 1) == 32, "a row's 32 warp slots: SW a block, one zero slot a block");
  constexpr int SH = 5 - log2_of(N);
  cg::cluster_group cl = cg::this_cluster();
  const int lane = t & 31, warp = t >> 5, me = cl.block_rank();
  halve<N, 16>(v, lane);
  const int slot = warp < SW ? me * SW + warp : CL * SW + me;
  if (bar) {
    if (warp <= SW && (lane & ((1 << SH) - 1)) == 0) {
      const unsigned at = smem_u32(red + (lane >> SH) * RED_LD + slot), b = smem_u32(bar);
      const unsigned x = __float_as_uint(warp < SW ? v[0] : 0.f);
#pragma unroll
      for (int q = 0; q < CL; ++q) {
        unsigned ra, rb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(at), "r"(q));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(q));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(ra), "r"(x),
                     "r"(rb) : "memory");
      }
    }
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra WAIT_%=;\n}" ::"r"(smem_u32(bar))
        : "memory");
  } else {
    if (warp <= SW && (lane & ((1 << SH) - 1)) == 0) {
#pragma unroll
      for (int q = 0; q < CL; ++q) cl.map_shared_rank(red, q)[(lane >> SH) * RED_LD + slot] = warp < SW ? v[0] : 0.f;
    }
    cl.sync();
  }
  if constexpr (N <= BS_SMALL) {
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = red[j * RED_LD + lane];
    halve<N, 16>(v, lane);
    const float mine = v[0];
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = __shfl_sync(0xffffffffu, mine, j << SH);
  } else {
    for (int j = warp; j < N; j += W) {
      float x = red[j * RED_LD + lane];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
      if (lane == 0) red[j * RED_LD + 32] = x;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = red[j * RED_LD + 32];
  }
}

// candidate x's sum of squares and dot with sw over this thread's eight
__device__ __forceinline__ void partial(const V8& x, const float* swv, float& sq, float& dp) {
  sq = 0.f, dp = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = bf2f(x.w[j]);
    sq += f.x * f.x; sq += f.y * f.y;
    dp += f.x * swv[2 * j]; dp += f.y * swv[2 * j + 1];
  }
}

__device__ __forceinline__ V8 cand_at(const uint4* cand, int c, int t) {
  V8 x;
  x.w[0] = 0u, x.w[1] = 0u, x.w[2] = 0u, x.w[3] = 0u;
  if (t < KVEC) x = asv(cand[c * KVEC + t]);
  return x;
}

// Candidates 0..C-1 of the row (the snapshots, then the prefix `pv` when C = NB + 1): their
// totals, sq in tot[2c], dot in tot[2c + 1].
template <int NB, int C>
__device__ __forceinline__ void row_scores(const uint4* cand, const RowRegs& r, const V8& pv, float (&tot)[2 * C],
                                           float* red, int t) {
  constexpr int N = pow2_at_least(2 * C);
  float v[N];
#pragma unroll
  for (int j = 0; j < N; ++j) v[j] = 0.f;
  if (NB > 0) asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
  for (int c = 0; c < C; ++c) partial(c < NB ? cand_at(cand, c, t) : pv, r.swv, v[2 * c], v[2 * c + 1]);
  block_sum<N>(v, red, t);
#pragma unroll
  for (int j = 0; j < 2 * C; ++j) tot[j] = v[j];
}

__device__ __forceinline__ float score_of(float sq, float dp) { return dp * rsqrtf(sq * (1.0f / (float)KH) + KEPS); }

// The softmax's p_c = e_c / den: lane c divides, the warp broadcasts (k3_residual.cu's attnres_rms_row), not
// C divisions a thread: a row's epilogue on one SM of 1024 threads is issue-bound past a few snapshots.
template <int C>
__device__ __forceinline__ void softmax_div(float (&e)[C], float den, int lane) {
  static_assert(C <= 32, "a candidate a lane");
  float mine = e[0];
#pragma unroll
  for (int c = 1; c < C; ++c) mine = lane == c ? e[c] : mine;
  mine = mine / den;
#pragma unroll
  for (int c = 0; c < C; ++c) e[c] = __shfl_sync(0xffffffffu, mine, c);
}

// The softmax over the candidates' scores, the mix, its rms, the output.
template <int NB>
__device__ __forceinline__ void row_mix_norm(const uint4* cand, const RowRegs& r, const V8& pv, float (&sc)[NB + 1],
                                             bf16_t* __restrict__ normed_row, int t, RowSmem& s) {
  const bool act = (t < KVEC);
  V8 mixed = pv;
  if constexpr (NB > 0) {
    float mx = -3.0e38f, den = 0.f;
#pragma unroll
    for (int c = 0; c <= NB; ++c) mx = fmaxf(mx, sc[c]);
#pragma unroll
    for (int c = 0; c <= NB; ++c) den += (sc[c] = __expf(sc[c] - mx));
    softmax_div(sc, den, t & 31);
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
    for (int c = 0; c <= NB; ++c) {
      const float p = sc[c];
      const V8 x = c < NB ? cand_at(cand, c, t) : pv;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(x.w[j]);
        acc[2 * j] += p * f.x;
        acc[2 * j + 1] += p * f.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) mixed.w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
  }
  float sq[1] = {0.f};
  if (act) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf2f(mixed.w[j]);
      sq[0] += f.x * f.x; sq[0] += f.y * f.y;
    }
  }
  block_sum<1>(sq, s.red[2], t);
  const float rr = rsqrtf(sq[0] * (1.0f / (float)KH) + KEPS);
  if (act) {
    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = bf2f(mixed.w[j]);
      f.x *= rr; f.y *= rr;
      o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(r.g.w[j])));
    }
    stv(normed_row + t * 8, o);
  }
}

// The snapshots' scores (sc[0..NB)), which need no prefix row: before the poll or the grid wait.
template <int NB>
__device__ __forceinline__ void row_snap_scores(const uint4* cand, const RowRegs& r, float (&sc)[NB + 1], RowSmem& s,
                                                int t) {
  if constexpr (NB > 0) {
    float snap[2 * NB];
    V8 none;
    none.w[0] = 0u, none.w[1] = 0u, none.w[2] = 0u, none.w[3] = 0u;
    row_scores<NB, NB>(cand, r, none, snap, s.red[0], t);
#pragma unroll
    for (int c = 0; c < NB; ++c) sc[c] = score_of(snap[2 * c], snap[2 * c + 1]);
  }
}

// The prefix row's score (sc[NB]), the mix, the norm.
template <int NB>
__device__ __forceinline__ void row_prefix_finish(const uint4* cand, const RowRegs& r, const V8& pv, float (&sc)[NB + 1],
                                                  bf16_t* __restrict__ normed_row, int t, RowSmem& s) {
  if constexpr (NB > 0) {
    float pvs[2] = {0.f, 0.f};
    if (t < KVEC) partial(pv, r.swv, pvs[0], pvs[1]);
    block_sum<2>(pvs, s.red[1], t);
    sc[NB] = score_of(pvs[0], pvs[1]);
  }
  row_mix_norm<NB>(cand, r, pv, sc, normed_row, t, s);
}

// The whole routine once the prefix row `pv` is known: every candidate scored in one reduction.
template <int NB>
__device__ __forceinline__ void row_finish(const uint4* cand, const RowRegs& r, const V8& pv,
                                           bf16_t* __restrict__ normed_row, int t, RowSmem& s) {
  float tot[2 * NB + 2], sc[NB + 1];
  row_scores<NB, NB + 1>(cand, r, pv, tot, s.red[0], t);
#pragma unroll
  for (int c = 0; c <= NB; ++c) sc[c] = score_of(tot[2 * c], tot[2 * c + 1]);
  row_mix_norm<NB>(cand, r, pv, sc, normed_row, t, s);
}

#define NB_ENTRIES(entry) entry(0) entry(1) entry(2) entry(3) entry(4) entry(5) entry(6) entry(7) entry(8)

// ---------------------------------------------------------------- attention all-reduce + K1b
template <int NB>
__device__ __forceinline__ void ar_attnres_rms(const uint4* __restrict__ x, uint8_t* lamport,
                                               const unsigned long long* lamport_peers, int* state, int* err, int rank,
                                               const bf16_t* __restrict__ prefix, const bf16_t* __restrict__ blocks,
                                               const float* __restrict__ sw, const bf16_t* __restrict__ gamma,
                                               bf16_t* __restrict__ prefix2, bf16_t* __restrict__ normed, int snapshot,
                                               int B, long long stage_bytes, long long timeout_ns,
                                               const uint8_t* __restrict__ pf, long long pf_bytes,
                                               const uint8_t* __restrict__ dm, long long dm_bytes, uint4* cand,
                                               RowSmem& s) {
  const int t = threadIdx.x, b = blockIdx.x;
  const bool row = b < B, act = t < KVEC;
  const size_t off = (size_t)b * KH + t * 8;
  const long long tot = (long long)B * KVEC, i = share(tot, B);
  RowRegs r;
  V8 pr;
  pr.w[0] = 0u, pr.w[1] = 0u, pr.w[2] = 0u, pr.w[3] = 0u;
  if (row) {
    r = row_prefetch<NB>(blocks + (size_t)b * KNB_MAX * KH, sw, gamma, cand, t);
    if (act && !snapshot) pr = ldv(prefix + off);
  }
  const Lamport l = lamport_open(lamport, lamport_peers, state, err, rank, tot, stage_bytes, timeout_ns);
  grid_wait();
  const uint4 v = i >= 0 ? x[i] : make_uint4(0u, 0u, 0u, 0u);
  const bool two = B >= TWO_SHOT_ROWS;  // as kern_k3_ar_finalize_rms's
  if (i >= 0) {
    if (two)
      l.slot[(i / KVEC) % NRANKS][i] = unpoison4(v);
    else
      lamport_push(l, i, v);
  }
  lamport_clear(l);
  unsigned* const stamp = l2_stamp(state, l);
  l2_load_while(pf, pf_bytes, B, stamp, AR_GRACE_NS, state + 7);
  l2_demote(dm, dm_bytes, B);

  int fail = 0;
  if (row) {
    if (two && b % NRANKS != rank) gather_share<KVEC>(l, rank, B, timeout_ns, fail);
    float sc[NB + 1];
    row_snap_scores<NB>(cand, r, sc, s, t);
    V8 pv;
    pv.w[0] = 0u, pv.w[1] = 0u, pv.w[2] = 0u, pv.w[3] = 0u;
    if (act) {
      const long long i = (long long)b * KVEC + t;
      const int owner = b % NRANKS;
      const V8 lp = asv(lamport_get(l, two, owner, rank, i, timeout_ns, fail));
      l2_arrive(stamp);
      if (snapshot) {
        pv = lp;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 rf = bf2f(pr.w[j]), lf = bf2f(lp.w[j]);
          pv.w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
        }
      }
      stv(prefix2 + off, pv);
    }
    row_prefix_finish<NB>(cand, r, pv, sc, normed + (size_t)b * KH, t, s);
  }
  lamport_close(l, state, err, fail);
}

#define AR_ATTNRES_ENTRY(NB)                                                                                      \
  extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_ar_attnres_rms_nb##NB(                         \
      const uint4* __restrict__ x, uint8_t* lamport, const unsigned long long* lamport_peers, int* state, int* err, \
      int rank, const bf16_t* __restrict__ prefix, const bf16_t* __restrict__ blocks, const float* __restrict__ sw, \
      const bf16_t* __restrict__ gamma, bf16_t* __restrict__ prefix2, bf16_t* __restrict__ normed, int snapshot,   \
      int B, long long stage_bytes, long long timeout_ns, const uint8_t* __restrict__ pf, long long pf_bytes,       \
      const uint8_t* __restrict__ dm, long long dm_bytes) {                                                         \
    extern __shared__ uint4 cand[];                                                                                 \
    __shared__ RowSmem s;                                                                                           \
    ar_attnres_rms<NB>(x, lamport, lamport_peers, state, err, rank, prefix, blocks, sw, gamma, prefix2, normed,     \
                       snapshot, B, stage_bytes, timeout_ns, pf, pf_bytes, dm, dm_bytes, cand, s);                  \
  }
NB_ENTRIES(AR_ATTNRES_ENTRY)

// ---------------------------------------------------------------- K1d
// k3_residual.cu's K1c on bf16 partials then K1a: hidden = bf16(prefix2 + p1 + (two ? p2 : 0))
// written (and snapshotted into blocks[b, nb] when `snapshot`), then normed = rms(attnres(
// blocks, hidden, nb), gamma): a layer's closing add fused into the next layer's mix.
// A row over a cluster of LAND_CL blocks of 1024 / LAND_CL threads, each a slice of LV vectors, its
// reductions the 1024-thread block's over DSMEM (a row on one SM is bound by its arithmetic and
// barriers, ~0.4 us a snapshot).
#define LAND_CL 4
#define LV (KVEC / LAND_CL)
#define LT (KTHREADS / LAND_CL)

struct LandSmem {
  float red[2][32 * RED_LD];
  unsigned long long bar;  // the norm's (cluster_block_sum)
};

template <int NB>
__device__ __forceinline__ void land_add2_attnres_rms(const bf16_t* __restrict__ p1, const bf16_t* __restrict__ p2,
                                                      const bf16_t* __restrict__ prefix2, bf16_t* __restrict__ hidden,
                                                      int two, bf16_t* __restrict__ blocks, const float* __restrict__ sw,
                                                      const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed,
                                                      int snapshot, uint4* cand, LandSmem& s) {
  const int b = blockIdx.x / LAND_CL, t = threadIdx.x, v0 = (blockIdx.x % LAND_CL) * LV;
  const bool act = t < LV;
  const size_t off = (size_t)b * KH + (v0 + t) * 8;
  const bf16_t* blk_row = blocks + (size_t)b * KNB_MAX * KH;
  float swv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  V8 g, pr;
  g.w[0] = g.w[1] = g.w[2] = g.w[3] = 0u;
  pr = g;
  if (act) {
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(cand + c * LV + t);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(blk_row + (size_t)c * KH + (v0 + t) * 8));
    }
    if (NB > 0) {
      const float4* sp4 = (const float4*)(sw + (v0 + t) * 8);
      const float4 a = sp4[0], c = sp4[1];
      swv[0] = a.x; swv[1] = a.y; swv[2] = a.z; swv[3] = a.w;
      swv[4] = c.x; swv[5] = c.y; swv[6] = c.z; swv[7] = c.w;
    }
    g = ldv(gamma + (v0 + t) * 8);
    pr = ldv(prefix2 + off);
  }
  asm volatile("cp.async.commit_group;");
  // the norm's mbarrier, armed before the scores' cluster barrier: no slot of the norm is sent before it
  if (NB > 0 && t == 0) {
    bar_arm(&s.bar, cluster_sum_bytes<1, LAND_CL, LV / 32>);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  grid_wait();
  V8 pv = pr;
  if (act) {
    const V8 a = ldv(p1 + off);
    V8 c;
    if (two) c = ldv(p2 + off);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 rf = bf2f(pr.w[j]), l1 = bf2f(a.w[j]);
      float x = rf.x + l1.x, y = rf.y + l1.y;
      if (two) {
        const float2 l2 = bf2f(c.w[j]);
        x += l2.x;
        y += l2.y;
      }
      pv.w[j] = f2bf(make_float2(x, y));
    }
    stv(hidden + off, pv);
    if (snapshot && NB < KNB_MAX) stv(blocks + ((size_t)b * KNB_MAX + NB) * KH + (v0 + t) * 8, pv);
  }
  const auto cand_v = [&](int c) {
    V8 x;
    x.w[0] = x.w[1] = x.w[2] = x.w[3] = 0u;
    if (act) x = asv(cand[c * LV + t]);
    return x;
  };
  V8 mixed = pv;
  if constexpr (NB > 0) {
    constexpr int N = pow2_at_least(2 * NB + 2);
    float v[N];
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] = 0.f;
    asm volatile("cp.async.wait_all;" ::: "memory");
    if (act) {
#pragma unroll
      for (int c = 0; c <= NB; ++c) partial(c < NB ? cand_v(c) : pv, swv, v[2 * c], v[2 * c + 1]);
    }
    cluster_block_sum<N, LAND_CL, LV / 32, LT / 32>(v, s.red[0], nullptr, t);
    float sc[NB + 1], mx = -3.0e38f, den = 0.f;
#pragma unroll
    for (int c = 0; c <= NB; ++c) mx = fmaxf(mx, sc[c] = score_of(v[2 * c], v[2 * c + 1]));
#pragma unroll
    for (int c = 0; c <= NB; ++c) den += (sc[c] = __expf(sc[c] - mx));
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c <= NB; ++c) {
      const float p = sc[c] / den;
      const V8 x = c < NB ? cand_v(c) : pv;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(x.w[j]);
        acc[2 * j] += p * f.x;
        acc[2 * j + 1] += p * f.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) mixed.w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
  }
  float sq[1] = {0.f};
  if (act) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf2f(mixed.w[j]);
      sq[0] += f.x * f.x; sq[0] += f.y * f.y;
    }
  }
  cluster_block_sum<1, LAND_CL, LV / 32, LT / 32>(sq, s.red[1], NB > 0 ? &s.bar : nullptr, t);
  const float rr = rsqrtf(sq[0] * (1.0f / (float)KH) + KEPS);
  if (act) {
    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = bf2f(mixed.w[j]);
      f.x *= rr; f.y *= rr;
      o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(g.w[j])));
    }
    stv(normed + off, o);
  }
}

// the idle blocks' hints after lat_up (the kernel before) is done: not on top of its own weight stream
#define LAND_ENTRY(NB)                                                                                           \
  extern "C" __global__ void __launch_bounds__(LT, 1) kern_k3_land_add2_attnres_rms_nb##NB(                    \
      const bf16_t* __restrict__ p1, const bf16_t* __restrict__ p2, const bf16_t* __restrict__ prefix2,        \
      bf16_t* __restrict__ hidden, int two, bf16_t* __restrict__ blocks, const float* __restrict__ sw,         \
      const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int snapshot, int B,                      \
      const uint8_t* __restrict__ pf, long long pf_bytes) {                                                    \
    extern __shared__ uint4 cand[];                                                                            \
    __shared__ LandSmem s;                                                                                     \
    asm volatile("griddepcontrol.launch_dependents;");                                                         \
    if ((int)blockIdx.x >= LAND_CL * B) return grid_wait(), l2_prefetch(pf, pf_bytes, LAND_CL * B);            \
    land_add2_attnres_rms<NB>(p1, p2, prefix2, hidden, two, blocks, sw, gamma, normed, snapshot, cand, s);     \
  }
NB_ENTRIES(LAND_ENTRY)

// ---------------------------------------------------------------- MoE finalize + all-reduce + latent rms
// A row's exchange: the latent's KLATV vectors, then the shared expert's KVEC.
#define ROWV (KLATV + KVEC)
// a block's push share is one vector a thread up to the manifests' 64 rows
static_assert(((64 * ROWV + PUSH_RUN - 1) / PUSH_RUN + KGRID - 1) / KGRID * PUSH_RUN <= KTHREADS,
              "a push share past KTHREADS");

// kern_k3_moe_finalize's column vector h of a row, its top-k (exp2perm, wts) in shared memory
__device__ __forceinline__ uint4 finalize(const bf16_t* __restrict__ fc2, const int* e2p, const float* w16, int h) {
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const int p = e2p[k];
    if (p < 0) continue;
    const float w = w16[k];
    const uint4 u = reinterpret_cast<const uint4*>(fc2 + (long long)p * KLAT)[h];
    const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(v[j]);
      acc[2 * j] += w * f.x;
      acc[2 * j + 1] += w * f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int j = 0; j < 4; ++j) ov[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  return o;
}

extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_ar_finalize_rms(
    const bf16_t* __restrict__ fc2, const int* __restrict__ exp2perm, const float* __restrict__ wts,
    const uint4* __restrict__ shared, uint8_t* lamport, const unsigned long long* lamport_peers, int* state, int* err,
    int rank, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ latent_norm, uint4* __restrict__ shared_sum, int B,
    int early, long long stage_bytes, long long timeout_ns, const uint8_t* __restrict__ pf, long long pf_bytes,
    const uint8_t* __restrict__ pf2, long long pf2_bytes) {
  __shared__ float red[32 * RED_LD];
  const int t = threadIdx.x, b = blockIdx.x;
  const long long tot = (long long)B * ROWV, i = share(tot, B);
  const int row = (int)(i / ROWV), c = (int)(i % ROWV);
  // the top-k of the (at most two) rows this block's share spans, written by moe_front
  __shared__ int s_e2p[2][TOPK];
  __shared__ float s_w[2][TOPK];
  const int row0 = (B >= TWO_SHOT_ROWS ? b * KTHREADS : share_per(tot) * b) / ROWV;
  if (t < 2 * TOPK && row0 + t / TOPK < B) {
    s_e2p[t / TOPK][t % TOPK] = exp2perm[(row0 + t / TOPK) * TOPK + t % TOPK];
    s_w[t / TOPK][t % TOPK] = wts[(row0 + t / TOPK) * TOPK + t % TOPK];
  }
  const Lamport l = lamport_open(lamport, lamport_peers, state, err, rank, tot, stage_bytes, timeout_ns);
  // griddepcontrol.wait returns at once the second time
  if (!early) grid_wait();
  uint4 v = i >= 0 && c < KLATV ? finalize(fc2, s_e2p[row - row0], s_w[row - row0], c) : make_uint4(0u, 0u, 0u, 0u);
  grid_wait();
  if (i >= 0 && c >= KLATV) v = shared[(long long)row * KVEC + c - KLATV];
  // From TWO_SHOT_ROWS rows the sum runs as two shots: every row goes to its owner (row % NRANKS) only, the
  // owner sums it and gathers the sum to the others (gather_share), 2 x 7/8 of a partial out of each rank
  // instead of 7.
  const bool two = B >= TWO_SHOT_ROWS;
  if (i >= 0) {
    if (two)
      l.slot[row % NRANKS][i] = unpoison4(v);
    else
      lamport_push(l, i, v);
  }
  lamport_clear(l);
  unsigned* const stamp = l2_stamp(state, l);
  l2_load_while(pf, pf_bytes, B, stamp, FIN_GRACE_NS, state + 7);
  l2_load_while<true>(pf2, pf2_bytes, B, stamp, FIN_GRACE_NS, state + 7);

  int fail = 0;
  if (b < B) {
    const long long base = (long long)b * ROWV;
    const int owner = b % NRANKS;
    // Two-shot, the block waits for its gather, then for the row's shared half, before any thread polls the
    // next thing: warps that run ahead spin on slots still empty, and their polls slow everyone's (10.5 -> 9.0 us
    // at 36 rows; one-shot the second wait costs 0.2-0.4, so not there)
    if (two && owner != rank) gather_share<ROWV>(l, rank, B, timeout_ns, fail);
    if (two) __syncthreads();
    const auto summed = [&](long long i) { return lamport_get(l, two, owner, rank, i, timeout_ns, fail); };
    if (t < KVEC) shared_sum[(long long)b * KVEC + t] = summed(base + KLATV + t);
    if (two) __syncthreads();
    l2_arrive(stamp);
    // kern_k3_rms at h = KLAT: vector t of the row, its squares serially, a warp
    // butterfly, then the 32 warp slots (zero past the row) by another, in every warp
    uint4 x = make_uint4(0u, 0u, 0u, 0u), g;
    float sum = 0.f;
    if (t < KLATV) {
      x = summed(base + t);
      g = reinterpret_cast<const uint4*>(gamma)[t];
      const V8 v = asv(x);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(v.w[j]);
        sum += f.x * f.x;
        sum += f.y * f.y;
      }
    }
    float tot[1] = {sum};
    block_sum<1>(tot, red, t);
    const float rs = rsqrtf(tot[0] / (float)KLAT + 1e-5f);
    if (t < KLATV) {
      const V8 v = asv(x), gv = asv(g);
      V8 o;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf2f(v.w[j]);
        o.w[j] = from_bf162(__hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), as_bf162(gv.w[j])));
      }
      stv(latent_norm + (long long)b * KLAT + t * 8, o);
    }
  }
  lamport_close(l, state, err, fail);
}

// ---------------------------------------------------------------- embedding + K1a of the first layer
// hidden = embed[ids[b]] (written, and snapshot 0 of blocks), normed = rms(hidden, gamma): the
// gather and the first layer's attnres_rms with no snapshot to mix (nb = 0, the mix is the row).
// kern_k3_embed_rms_plan also writes row b's MLA split count: kern's k3_mla_split_plan.cu sizes
// splits to fill one wave of clusters; here the batch fills the 1 to MLA_WAVES waves of the least
// modelled time (one wave of 48 rows held one split a row on 96 of 152 SMs). The plan needs only
// every row's tile count: block B works it out for every row while the row blocks gather and norm.
//   kern_k3_embed_rms(const i64* ids, const bf16* table, bf16* hidden, bf16* blocks,
//       const bf16* gamma, bf16* normed, int B)   grid (B, 1, 1)
//   kern_k3_embed_rms_plan(..., const i32* seq_lens, i32* block_split_kvs, int split_max, int B)
//       grid (B + 1, 1, 1)
//   block (1024, 1, 1)
#define MLA_TILE 128
// A wave of n 2-CTA clusters runs a tile in ~1.46 us a cluster up to ~53 clusters; past that the GPU runs ~36
// tiles a us (96 heads, bf16 MMA padded to 128 rows: the tensor pipes and HBM both near their limit); a
// split also writes and the exchange reads back its f32 partial, ~2 tiles of traffic. So a wave of n
// clusters at `longest` tiles a split costs (longest + 2) * max(53, n). A batch fills up to MLA_WAVES.
#define MLA_SATURATE 53
#define MLA_SPLIT_TILES 2
#define MLA_WAVES 3

__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ int warp_max_i(int v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// kern's k3_mla_split_plan.cu: `per` tiles a split over `budget` clusters, a row's splits from it (32-bit: a
// batch's tiles are at most 64 x 2560)
__device__ __forceinline__ int split_per(int total, int budget) {
  budget = max(budget, 1);
  return max((total + budget - 1) / budget, 1);
}
__device__ __forceinline__ int row_splits(int tiles, int per, int split_max) {
  return min(max((tiles + per - 1) / per, 1), split_max);
}

// the gathered row, its stores, then the norm (nb = 0)
__device__ __forceinline__ V8 embed_row(const long long* __restrict__ ids, const bf16_t* __restrict__ table, int b,
                                        int t) {
  V8 pv;
  pv.w[0] = 0u, pv.w[1] = 0u, pv.w[2] = 0u, pv.w[3] = 0u;
  if (t < KVEC) pv = ldv(table + ids[b] * KH + t * 8);
  return pv;
}
__device__ __forceinline__ void embed_rms(const V8& pv, bf16_t* __restrict__ hidden, bf16_t* __restrict__ blocks,
                                          const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int b,
                                          int t) {
  __shared__ RowSmem s;
  const RowRegs r = row_prefetch<0>(blocks, nullptr, gamma, nullptr, t);
  if (t < KVEC) {
    stv(hidden + (size_t)b * KH + t * 8, pv);
    stv(blocks + (size_t)b * KNB_MAX * KH + t * 8, pv);
  }
  row_finish<0>(nullptr, r, pv, normed + (size_t)b * KH, t, s);
}

extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_embed_rms(
    const long long* __restrict__ ids, const bf16_t* __restrict__ table, bf16_t* __restrict__ hidden,
    bf16_t* __restrict__ blocks, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed, int B) {
  asm volatile("griddepcontrol.launch_dependents;");
  if (blockIdx.x >= B) return;
  embed_rms(embed_row(ids, table, blockIdx.x, threadIdx.x), hidden, blocks, gamma, normed, blockIdx.x, threadIdx.x);
}

extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_embed_rms_plan(
    const long long* __restrict__ ids, const bf16_t* __restrict__ table, bf16_t* __restrict__ hidden,
    bf16_t* __restrict__ blocks, const bf16_t* __restrict__ gamma, bf16_t* __restrict__ normed,
    const int* __restrict__ seq_lens, int* __restrict__ block_split_kvs, int split_max, int B) {
  asm volatile("griddepcontrol.launch_dependents;");
  const int b = blockIdx.x, t = threadIdx.x;
  if (b < B) return embed_rms(embed_row(ids, table, b, t), hidden, blocks, gamma, normed, b, t);
  if (t < 32) {  // block B: every row's plan, its tile count read once (B <= 64: two a lane)
    int tiles[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) tiles[k] = t + 32 * k < B ? (seq_lens[t + 32 * k] + MLA_TILE - 1) / MLA_TILE : 0;
    const int total = warp_sum_i(tiles[0] + tiles[1]);
    unsigned nsm;
    asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    const int clusters = nsm / 2;
    // `per` tiles a split such that the batch fills W waves of clusters, the W of the least modelled time
    long long best_cost = 0;
    int best = 0;
    for (int w = 1; w <= MLA_WAVES; ++w) {
      const int per = split_per(total, w * clusters - B);
      int splits = 0, longest = 0;
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if (t + 32 * k < B) {
          const int sp = row_splits(tiles[k], per, split_max);
          splits += sp;
          longest = max(longest, (tiles[k] + sp - 1) / sp);
        }
      splits = warp_sum_i(splits);
      longest = warp_max_i(longest);
      long long cost = 0;
      for (int n = splits; n > 0; n -= clusters) cost += (long long)(longest + MLA_SPLIT_TILES) * max(MLA_SATURATE, min(n, clusters));
      if (w == 1 || cost < best_cost) best = per, best_cost = cost;
    }
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (t + 32 * k < B) block_split_kvs[t + 32 * k] = row_splits(tiles[k], best, split_max);
  }
}
