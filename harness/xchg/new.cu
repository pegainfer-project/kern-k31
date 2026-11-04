// The exchange on k3_xchg.cuh as a bare all-reduce: rows x rv vectors, row b's output by block b.
#include <cstdio>
#include "k3_xchg.cuh"
#include <cuda_bf16.h>
using namespace xchg;
typedef unsigned long long u64;

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// stamps[call % 16][block][event], thread 0 of each block: entry, opened, pushed, gathered, got, end
__device__ __forceinline__ void stamp(u64* st, int call, int e) {
  if (st && threadIdx.x == 0) st[((call % 16) * gridDim.x + blockIdx.x) * 8 + e] = gtimer();
}

template <bool PDL>
__device__ __forceinline__ void xar(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* recs, int* err,
                                    int rank, int rows, int rv, long long stage_bytes, long long timeout_ns, int two_from,
                                    u64* st, int* calls) {
  const bool gate = two_from < 0;
  two_from = gate ? -two_from : two_from;
  const int call = st ? *calls : 0;
  stamp(st, call, 0);
  const bool two = rows >= two_from;
  const long long tot = (long long)rows * rv;
  const auto dest = [&](long long i) { return two ? (int)(i / rv % NRANKS) : -1; };
  const long long i0 = first_chunk() * 32 + threadIdx.x % 32;
  if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint4 v0 = i0 < tot ? x[i0] : make_uint4(0, 0, 0, 0);
  const Rec r = xchg_rec(recs);
  const Peers pe = xchg_peers(peers);
  const Lamport l = xchg_open(lamport, pe, r, rank, tot, stage_bytes);
  xchg_handshake(l, r, err, stage_bytes, timeout_ns);
  stamp(st, call, 1);
  if (i0 < tot) xchg_store(l, i0, v0, dest(i0));
  for (long long c = first_chunk() + chunk_step(); c * 32 < tot; c += chunk_step()) {
    const long long i = c * 32 + threadIdx.x % 32;
    if (i < tot) xchg_store(l, i, x[i], dest(i));
  }
  xchg_clear(l, r);
  stamp(st, call, 2);
  int fail = 0;
  if (two) xchg_gather(l, rows, rv, timeout_ns, fail);
  stamp(st, call, 3);
  // a row block holds `hold` vectors of its row (the MoE's latent: 448 of 1344), the grid the rest
  const int hold = rv > 1024 ? rv / 3 : rv;
  if (st == nullptr && calls != nullptr) {
    // the rows' sums spread: row b's four quarters by blocks 4 b .. 4 b + 3
    for (int q = blockIdx.x; q < rows * 4; q += gridDim.x) {
      const int b = q / 4, w = (rv + 3) / 4;
      for (int c = (q % 4) * w + threadIdx.x; c < min(rv, (q % 4 + 1) * w); c += blockDim.x) {
        const long long i = (long long)b * rv + c;
        y[i] = xchg_get(l, two, b % NRANKS, i, timeout_ns, fail);
      }
    }
  } else
  for (int b = blockIdx.x; b < rows; b += gridDim.x) {
    for (int c = threadIdx.x; c < hold; c += blockDim.x) {
      const long long i = (long long)b * rv + c;
      y[i] = xchg_get(l, two, b % NRANKS, i, timeout_ns, fail);
    }
  }
  if (hold < rv && !(st == nullptr && calls != nullptr))
    xchg_spread(l, two, rows, rv, hold, [&](int b, int c, uint4 v) { y[(long long)b * rv + c] = v; }, timeout_ns, fail);
  stamp(st, call, 4);
  xchg_close(l, recs, r, err, fail);
  stamp(st, call, 5);
  if (PDL) {
    // back to back the next call is this kernel: it may read the records only once this call is done
    __threadfence();
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

extern "C" __global__ void __launch_bounds__(1024) kern_xar(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers,
                                                          int* state, int* err, int rank, int rows, int rv,
                                                          long long stage_bytes, long long timeout_ns, int two_from,
                                                          u64* st, int* calls) {
  xar<false>(x, y, lamport, peers, state, err, rank, rows, rv, stage_bytes, timeout_ns, two_from, st, calls);
}
extern "C" __global__ void __launch_bounds__(1024) kern_xar_pdl(const uint4* x, uint4* y, uint8_t* lamport,
                                                              const u64* peers, int* state, int* err, int rank, int rows,
                                                              int rv, long long stage_bytes, long long timeout_ns,
                                                              int two_from, u64* st, int* calls) {
  xar<true>(x, y, lamport, peers, state, err, rank, rows, rv, stage_bytes, timeout_ns, two_from, st, calls);
}
extern "C" __global__ void kern_bump(int* calls) { ++*calls; }

// The same all-reduce with every store all members receive as one multimem.st to the stage's multicast
// alias `mc`: the one-shot's push of a vector into slot `rank` everywhere, and the two-shot owner's sum,
// into slot NRANKS (its own slot of a gathered sum, so the owner's partial in slot `rank` stays). A
// stage needs NRANKS + 1 slots; `recs` here hold the clear count of NRANKS + 1 slots.
__device__ __forceinline__ void st_mc(uint4* p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
extern "C" __global__ void __launch_bounds__(1024) kern_xar_mc(const uint4* x, uint4* y, uint8_t* lamport, uint8_t* mc,
                                                             const u64* peers, int* recs, int* err, int rank, int rows,
                                                             int rv, long long stage_bytes, long long timeout_ns,
                                                             int two_from) {
  const bool two = rows >= two_from;
  const long long tot = (long long)rows * rv;
  const Rec r = xchg_rec(recs);
  const Peers pe = xchg_peers(peers);
  const Lamport l = xchg_open(lamport, pe, r, rank, tot, stage_bytes);
  uint4* mslot = reinterpret_cast<uint4*>(mc + l.stage_off) + (long long)rank * tot;
  for (long long c = first_chunk(); c * 32 < tot; c += chunk_step()) {
    const long long i = c * 32 + threadIdx.x % 32;
    if (i >= tot) continue;
    const uint4 v = unpoison(x[i]);
    if (two) slot(l, (int)(i / rv % NRANKS), rank)[i] = v;
    else st_mc(mslot + i, v);
  }
  xchg_clear(l, r);
  int fail = 0;
  uint4* gslot = reinterpret_cast<uint4*>(mc + l.stage_off) + (long long)NRANKS * tot;
  if (two) {
    const int owned = rows > rank ? (rows - 1 - rank) / NRANKS + 1 : 0;
    const long long n = owned * (long long)rv;
    for (long long c = first_chunk(); c * 32 < n; c += chunk_step()) {
      const long long j = c * 32 + threadIdx.x % 32;
      if (j >= n) continue;
      const long long i = (long long)(rank + NRANKS * (int)(j / rv)) * rv + j % rv;
      st_mc(gslot + i, unpoison(xchg_sum(l, i, timeout_ns, fail)));
    }
  }
  for (int b = blockIdx.x; b < rows; b += gridDim.x)
    for (int c = threadIdx.x; c < rv; c += blockDim.x) {
      const long long i = (long long)b * rv + c;
      y[i] = two ? xchg_one(l, NRANKS, i, timeout_ns, fail) : xchg_sum(l, i, timeout_ns, fail);
    }
  if (fail) give_up(err, fail);
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long cl = (unsigned long long)(NRANKS + 1) * tot;
    reinterpret_cast<uint4*>(recs)[blockIdx.x] = make_uint4(r.calls + 1, 0u, (unsigned)cl, (unsigned)(cl >> 32));
  }
}

// FlashInfer's one-shot shape on k3_xchg.cuh: a thread owns vector i (rows x rv, row-major over the grid),
// pushes it to every member, then polls slot by slot for the sum. Grid = rows * rv / blockDim.
__device__ __forceinline__ void xar_local(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* recs,
                                          int* err, int rank, int rows, int rv, long long stage_bytes,
                                          long long timeout_ns) {
  const long long tot = (long long)rows * rv, i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint4 v = i < tot ? x[i] : make_uint4(0, 0, 0, 0);
  const Rec r = xchg_rec(recs);
  const Peers pe = xchg_peers(peers);
  const Lamport l = xchg_open(lamport, pe, r, rank, tot, stage_bytes);
  int fail = 0;
  if (i < tot) xchg_store(l, i, v, -1);
  xchg_clear(l, r);
  if (i < tot) y[i] = xchg_sum(l, i, timeout_ns, fail);
  xchg_close(l, recs, r, err, fail);
}
extern "C" __global__ void kern_xar_local(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* recs,
                                          int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns) {
  xar_local(x, y, lamport, peers, recs, err, rank, rows, rv, stage_bytes, timeout_ns);
}
extern "C" __global__ void __launch_bounds__(1024, 1) kern_xar_local1024(const uint4* x, uint4* y, uint8_t* lamport,
    const u64* peers, int* recs, int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns) {
  xar_local(x, y, lamport, peers, recs, err, rank, rows, rv, stage_bytes, timeout_ns);
}
