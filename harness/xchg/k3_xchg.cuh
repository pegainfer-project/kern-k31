// The D group's Lamport exchange over peer memory, for kernels that fuse work around it.
//
// The wire is peer_allreduce_bf16.cu's (TensorRT-LLM's one-shot): this rank's 16-byte vector i of a
// call lands in slot `rank` of a member's current stage, a word holding a bf16 -0.0 (0x8000) has not
// arrived, three stages of `stage_bytes` rotate and the previous call's is re-poisoned during this
// one. `lamport` holds the three stages, then NRANKS handshake words (never poisoned).
//
// The bookkeeping is per block: `recs` holds one 16-byte record a block of the grid, {calls, -,
// vectors to re-poison (i64)}, zeroed once. A block reads its record with its inputs at entry and
// writes it back at exit; no block waits for another, nothing is atomic. Every kernel on one
// `recs` runs the same grid (its blocks stay in step), any kernel the block partitions agree on.
//
// The call's vectors go out in 32-vector chunks dealt round-robin over the blocks, a warp storing
// its chunk to every member (or to the row's owner, two-shot): no SM issues more than a share. The
// two-shot's owner work is spread the same way, a vector a thread over the grid; a row block of the
// owner sums its row from the slots again (same order, same bytes) instead of waiting for it.
//
// Every sum is the rank-order f32 sum of the slots, rounded once to bf16: bit-identical to
// peer_allreduce_bf16.cu and k3_ar_fused.cu.
//
// A kernel on this header:
//   const Rec r = xchg_rec(recs);                         // issue its own input loads next to this
//   const Peers p = xchg_peers(peers);
//   const Lamport l = xchg_open(lamport, p, r, rank, tot, stage_bytes);
//   xchg_handshake(l, r, err, stage_bytes, timeout_ns);   // the first call only
//   xchg_push(l, vec, dest); xchg_clear(l, r);
//   [two-shot: xchg_gather(l, rows, rv, ...)]
//   ... xchg_get / xchg_sum / xchg_one ...
//   xchg_close(l, recs, r, err, fail);
#pragma once
#include <cstdint>

#ifndef NRANKS
#define NRANKS 8
#endif

namespace xchg {

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
__device__ __forceinline__ uint4 unpoison(uint4 v) { return make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w)); }
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t round_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) { return round_bf16(a) | (round_bf16(b) << 16); }

constexpr uint4 POISON = {0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u};

// Whether a wait that started at its first check (t0 = 0 then) has outlived `timeout_ns`. It counts
// SM cycles (~2 a ns): %globaltimer is resynced to the host clock early in a process, and its jump
// fires a deadline with nothing late (k3_dcp.cu's await_all).
__device__ __forceinline__ bool late(long long& t0, long long timeout_ns) {
  const long long now = clock64();
  if (t0 == 0) t0 = now;
  return now - t0 > 2 * timeout_ns;
}

// A passed deadline is a broken group: record 1 + the late rank and trap, never sum a missing slot.
__device__ __forceinline__ void give_up(int* err, int missing) {
#ifdef XCHG_DEBUG
  printf("xchg: block %d thread %d gave up on %d\n", blockIdx.x, threadIdx.x, missing);
#endif
  atomicMax(err, missing);
  __threadfence_system();
  __trap();
}

struct Rec {
  unsigned calls;
  long long clear;  // vectors of the previous call's stage to re-poison
};

__device__ __forceinline__ Rec xchg_rec(const int* recs) {
  const uint4 v = reinterpret_cast<const uint4*>(recs)[blockIdx.x];
  return {v.x, (long long)((unsigned long long)v.w << 32 | v.z)};
}

// The members' `lamport` addresses in registers, loaded at entry with the record and the inputs.
struct Peers {
  unsigned long long a[NRANKS];
};
__device__ __forceinline__ Peers xchg_peers(const unsigned long long* peers) {
  Peers p;
#pragma unroll
  for (int q = 0; q < NRANKS; q += 2) {
    const ulonglong2 v = reinterpret_cast<const ulonglong2*>(peers)[q / 2];
    p.a[q] = v.x, p.a[q + 1] = v.y;
  }
  return p;
}

struct Lamport {
  uint8_t* lamport;
  Peers peers;
  long long stage_off;  // byte offset of this call's stage
  const uint4* mine;    // this call's stage here
  uint4* clear_buf;     // the previous call's stage here
  long long tot;
  int rank;
};

__device__ __forceinline__ Lamport xchg_open(uint8_t* lamport, const Peers& peers, const Rec& r, int rank,
                                             long long tot, long long stage_bytes) {
  Lamport l;
  l.lamport = lamport;
  l.peers = peers;
  l.tot = tot;
  l.rank = rank;
  l.stage_off = (long long)(r.calls % 3) * stage_bytes;
  l.mine = reinterpret_cast<const uint4*>(lamport + l.stage_off);
  l.clear_buf = reinterpret_cast<uint4*>(lamport + ((r.calls + 2) % 3) * stage_bytes);
  return l;
}

// Slot q's stage at member p: this rank's vector i goes to slot(l, p, rank) + i.
__device__ __forceinline__ uint4* slot(const Lamport& l, int p, int q) {
  unsigned long long a = l.peers.a[0];
#pragma unroll
  for (int k = 1; k < NRANKS; ++k) a = p == k ? l.peers.a[k] : a;  // a select chain: no local memory
  return reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a) + l.stage_off) + (long long)q * l.tot;
}

// The first call of a process waits for every peer's once-program to have poisoned its stages before
// pushing into them (k3_ar_fused.cu's handshake, the same words): block 0 marks `ready[rank]` at
// every peer, every block's thread 0 waits for all of its own.
__device__ __forceinline__ void xchg_handshake(const Lamport& l, const Rec& r, int* err, long long stage_bytes,
                                               long long timeout_ns) {
  if (r.calls) return;
  const long long at = 3 * stage_bytes;
  if (blockIdx.x == 0 && threadIdx.x < NRANKS)
#pragma unroll
    for (int q = 0; q < NRANKS; ++q)
      if (q == (int)threadIdx.x) reinterpret_cast<volatile int*>(reinterpret_cast<uint8_t*>(l.peers.a[q]) + at)[l.rank] = 1;
  if (threadIdx.x == 0) {
    const volatile int* ready = reinterpret_cast<const volatile int*>(l.lamport + at);
    long long t0 = 0;
    for (int q = 0; q < NRANKS; ++q)
      while (!ready[q])
        if (late(t0, timeout_ns)) give_up(err, 1 + q);
  }
  __syncthreads();
}

// Re-poison the previous call's stage, the grid's threads in turn.
__device__ __forceinline__ void xchg_clear(const Lamport& l, const Rec& r) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < r.clear; i += (long long)gridDim.x * blockDim.x)
    l.clear_buf[i] = POISON;
}

// Chunk c of the call's 32-vector chunks is block c % G's, warp c / G % W's: the first chunk of this
// thread (-1: none), then every further one.
__device__ __forceinline__ long long first_chunk() { return blockIdx.x + (long long)gridDim.x * (threadIdx.x / 32); }
__device__ __forceinline__ long long chunk_step() { return (long long)gridDim.x * (blockDim.x / 32); }

// Store vector i (already loaded: v) to every member (d < 0) or to member d.
__device__ __forceinline__ void xchg_store(const Lamport& l, long long i, uint4 v, int d) {
  v = unpoison(v);
  if (d < 0) {
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) slot(l, q, l.rank)[i] = v;
  } else {
    slot(l, d, l.rank)[i] = v;
  }
}

// The spread push of every vector: `vec(i)` this rank's vector i, `dest(i)` -1 (one-shot) or the
// member it goes to (two-shot).
template <class Vec, class Dest>
__device__ __forceinline__ void xchg_push(const Lamport& l, Vec&& vec, Dest&& dest) {
  const int lane = threadIdx.x % 32;
  for (long long c = first_chunk(); c * 32 < l.tot; c += chunk_step()) {
    const long long i = c * 32 + lane;
    if (i < l.tot) xchg_store(l, i, vec(i), dest(i));
  }
}

__device__ __forceinline__ uint4 rank_sum(const uint4 (&vals)[NRANKS]) {
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

// Wait for vector i of every slot; the rank-order f32 sum, rounded once.
__device__ __forceinline__ uint4 xchg_sum(const Lamport& l, long long i, long long timeout_ns, int& fail) {
  uint4 vals[NRANKS];
#pragma unroll
  for (int r = 0; r < NRANKS; ++r) vals[r] = ld_volatile(l.mine + (long long)r * l.tot + i);
  long long t0 = 0;
  while (true) {
    int missing = 0;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r)
      if (!arrived(vals[r])) {
        vals[r] = ld_volatile(l.mine + (long long)r * l.tot + i);
        missing = 1 + r;
      }
    if (!missing) break;
    if (late(t0, timeout_ns)) {
      fail = missing;
      break;
    }
  }
  return rank_sum(vals);
}

// Wait for vector i of slot q alone.
__device__ __forceinline__ uint4 xchg_one(const Lamport& l, int q, long long i, long long timeout_ns, int& fail) {
  long long t0 = 0;
  while (true) {
    const uint4 v = ld_volatile(l.mine + (long long)q * l.tot + i);
    if (arrived(v)) return v;
    if (late(t0, timeout_ns)) {
      fail = 1 + q;
      return v;
    }
  }
}

// The two-shot's middle: this rank's owned rows (row % NRANKS == rank, `rv` vectors a row, `rows`
// rows), a vector a thread over the grid, summed and stored into slot `rank` of every other member.
__device__ __forceinline__ void xchg_gather(const Lamport& l, int rows, long long rv, long long timeout_ns, int& fail) {
  const int owned = rows > l.rank ? (rows - 1 - l.rank) / NRANKS + 1 : 0;
  const long long n = owned * rv;
  for (long long c = first_chunk(); c * 32 < n; c += chunk_step()) {
    const long long j = c * 32 + threadIdx.x % 32;
    if (j >= n) continue;
    const long long i = (long long)(l.rank + NRANKS * (int)(j / rv)) * rv + j % rv;
    const uint4 v = unpoison(xchg_sum(l, i, timeout_ns, fail));
#pragma unroll
    for (int q = 1; q < NRANKS; ++q) slot(l, (l.rank + q) % NRANKS, l.rank)[i] = v;
  }
}

// Vector i of a row's sum: the one-shot's, or a two-shot's (`owner` of the row): the owner sums the
// slots itself, the others read the owner's gather.
__device__ __forceinline__ uint4 xchg_get(const Lamport& l, bool two, int owner, long long i, long long timeout_ns,
                                          int& fail) {
  return two && owner != l.rank ? xchg_one(l, owner, i, timeout_ns, fail) : xchg_sum(l, i, timeout_ns, fail);
}

// The vectors no row block needs whole: of every row, vectors [from, rv) summed by the grid in
// 32-vector chunks dealt over the blocks (as the push), vector c of row b to `out(b, c, sum)`.
template <class Out>
__device__ __forceinline__ void xchg_spread(const Lamport& l, bool two, int rows, int rv, int from, Out&& out,
                                            long long timeout_ns, int& fail) {
  const int w = rv - from;
  const long long n = (long long)rows * w;
  for (long long c = first_chunk(); c * 32 < n; c += chunk_step()) {
    const long long j = c * 32 + threadIdx.x % 32;
    if (j >= n) continue;
    const int b = (int)(j / w), k = from + (int)(j % w);
    out(b, k, xchg_get(l, two, b % NRANKS, (long long)b * rv + k, timeout_ns, fail));
  }
}

// The end of every block: a deadline that passed traps; the block's record moves to the next call.
__device__ __forceinline__ void xchg_close(const Lamport& l, int* recs, const Rec& r, int* err, int fail) {
  if (fail) give_up(err, fail);
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long c = (unsigned long long)NRANKS * l.tot;
    reinterpret_cast<uint4*>(recs)[blockIdx.x] = make_uint4(r.calls + 1, 0u, (unsigned)c, (unsigned)(c >> 32));
  }
}

}  // namespace xchg
