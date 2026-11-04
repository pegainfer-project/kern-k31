// The next GEMM's weight pulled into L2 by a kernel's idle blocks (blockIdx.x >= B, the blocks that hold no
// row), so the cuBLAS GEMM after it reads its weight from L2 instead of HBM.
//
// l2_prefetch: a prefetch.global.L2 a 128-byte line. It never holds the block, but the GEMM after it runs as if
//   only about half the bytes were in L2 (qkvg after 16 us of hints: 13.5 us, against 10.1 after real loads of
//   the same bytes and 17.2 cold; work/h/pf_pair.cu). For a kernel that is over in a few microseconds.
// l2_load_while: real loads (ld.global.L2::128B, one 128-byte line a lane) until `grace_ns` after a row block's
//   data is in, then stop: for a kernel whose row blocks wait on peers. The first row block in stamps
//   `arrived` (l2_arrive: the global timer's low 32 bits, odd, so never 0), so the loads go on under the row's epilogue for `grace_ns` and
//   drain in what is left of it; the member that arrives last, which waits the least, loads the least. The
//   warps that load keep ~L2_INFLIGHT bytes in flight.
#pragma once
#include <cstdint>

#ifndef L2_INFLIGHT
#define L2_INFLIGHT (12 << 20)
#endif

__device__ __forceinline__ void l2_prefetch(const uint8_t* w, long long bytes, int B) {
  if ((int)blockIdx.x < B) return;
  const long long step = (long long)(gridDim.x - B) * blockDim.x * 128;
  for (long long o = ((long long)(blockIdx.x - B) * blockDim.x + threadIdx.x) * 128; o < bytes; o += step)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(w + o));
}

// `sink` keeps the loads alive (ptxas drops a load nobody reads): written only if the xor of the words
// loaded is a constant, so in practice never.
__device__ __forceinline__ unsigned l2_now() {
  unsigned t;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t));
  return t | 1u;
}

// `last`: the lines go in at evict_last, so a weight read a few kernels later outlives the traffic between
// (l2_demote puts them back once their GEMM has run; an evict_last line nobody demotes holds L2 for good).
template <bool last = false>
__device__ __forceinline__ void l2_load_while(const uint8_t* w, long long bytes, int B,
                                              const unsigned* arrived, int grace_ns, int* sink) {
  if ((int)blockIdx.x < B) return;
  const int nb = gridDim.x - B, lane = threadIdx.x & 31;
  const int warps = min((int)(blockDim.x >> 5), (L2_INFLIGHT / (32 * 128) + nb - 1) / nb);
  if ((int)(threadIdx.x >> 5) >= warps) return;
  const long long lines = bytes >> 7, round = (long long)nb * warps * 32;
  unsigned long long pol = 0;
  if (last) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  unsigned acc = 0;
  for (long long l = ((long long)(blockIdx.x - B) * warps + (threadIdx.x >> 5)) * 32 + lane; l < lines; l += round) {
    int done = 0;
    if (lane == 0) {
      const unsigned at = *reinterpret_cast<const volatile unsigned*>(arrived);
      done = at && (int)(l2_now() - at) > grace_ns;
    }
    if (__shfl_sync(0xffffffffu, done, 0)) break;
    unsigned v;
    if (last)
      asm volatile("ld.global.L2::cache_hint.L2::128B.b32 %0, [%1], %2;" : "=r"(v) : "l"(w + (l << 7)), "l"(pol));
    else
      asm volatile("ld.global.L2::128B.b32 %0, [%1];" : "=r"(v) : "l"(w + (l << 7)));
    acc ^= v;
  }
  if (acc == 0x5f3759dfu) *sink = (int)acc;
}

__device__ __forceinline__ void l2_demote(const uint8_t* w, long long bytes, int B) {
  if ((int)blockIdx.x < B) return;
  const long long step = (long long)(gridDim.x - B) * blockDim.x * 128;
  for (long long o = ((long long)(blockIdx.x - B) * blockDim.x + threadIdx.x) * 128; o < bytes; o += step)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(w + o));
}

__device__ __forceinline__ void l2_arrive(unsigned* arrived) {
  if (threadIdx.x == 0 && !*reinterpret_cast<volatile unsigned*>(arrived)) atomicCAS(arrived, 0u, l2_now());
}

// `blocks_done` counts the blocks through; the grid's last zeroes it and `arrived` for the next call.
__device__ __forceinline__ void l2_close(unsigned* arrived, int* blocks_done) {
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(blocks_done, 1) == (int)gridDim.x - 1) *arrived = 0, *blocks_done = 0;
}
