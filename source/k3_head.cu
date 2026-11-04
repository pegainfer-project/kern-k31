// A DCP step's head over a vocab-parallel lm_head: rank r holds vocab rows
// [r * VS, r * VS + VS) of the head and its GEMM writes `part` [B, VS] f32. One
// launch then picks the step's tokens and lands the logits rank 0 reads:
//
//   kern_k3_head_argmax(const f32* part, f32* logits, const u64* logits_peers, i64* out, i32* done,
//       u64* pkey, u8* keys, const u64* keys_peers, i32* state, i32* err, int rank, int VS,
//       i64 timeout_ns, int B)
//   grid (B, PARTS, 1)   block (256, 1, 1)   every block resident at once
//
// Block (b, p) takes the p-th of PARTS chunks of row b's slice: copies it into rank 0's
// `logits` [rows, NRANKS * VS] at column rank * VS (the full row the step's caller reads
// on rank 0), and takes its (value, global index) argmax key, the stage-1 key of
// k3_router_argmax.cu: larger value, then smaller index, so the pick over any partition of
// the vocab is the whole row's. The row's last block (`done[b]`, a zeroed carry it
// re-zeroes) takes the slice's best key and pushes it into slot [rank][b] of every peer's
// `keys` stage; every rank waits for the row's NRANKS keys and writes the largest's token.
// A rank pushes its key only after its slice's copies (system fences between), so rank 0
// holds the whole row once it has every key.
//
// `keys` holds 3 stages of NRANKS * KEY_ROWS u64, zero meaning "not arrived" (a key never
// is: ord() > 0); `state` [8] starts zeroed: [0] block counter, [2] stage. Block (0, 0)
// re-zeroes the stage the previous call used, then the last block to arrive flips the
// stage, as the Lamport all-reduce does. A deadline that passes records 1 + the late rank
// in `err` and traps.
#include <cstdint>

#ifndef NRANKS
#define NRANKS 8
#endif
#define PARTS 16
#define THREADS 256
#define KEY_ROWS 64
#define IDX_TOP 0x7fffffff

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ u64 key_of(float v, int i) { return ((u64)ord_f32(v) << 32) | (u32)(IDX_TOP - i); }


extern "C" __global__ void __launch_bounds__(THREADS) kern_k3_head_argmax(
    const float* __restrict__ part, float* logits, const unsigned long long* __restrict__ logits_peers,
    long long* __restrict__ out,
    int* __restrict__ done, u64* __restrict__ pkey, uint8_t* keys_bytes, const unsigned long long* __restrict__ keys_peers,
    int* state, int* err, int rank, int VS, long long timeout_ns, int B) {
  __shared__ u64 sm[THREADS / 32];
  __shared__ int s_last;
  const int b = blockIdx.x, p = blockIdx.y, t = threadIdx.x;
  const int flag = state[2];
  u64* keys = reinterpret_cast<u64*>(keys_bytes);
  if (b == 0 && p == 0) {
    u64* prev = keys + ((flag + 2) % 3) * NRANKS * KEY_ROWS;
    for (int i = t; i < NRANKS * KEY_ROWS; i += THREADS) prev[i] = 0ull;
  }
  __syncthreads();
  if (t == 0) atomicAdd(state, 1);

  // the chunk: copy to rank 0's row, key over it
  const int chunk = VS / PARTS, lo = p * chunk;
  const float4* src = reinterpret_cast<const float4*>(part + (long long)b * VS + lo);
  float* row0 = rank == 0 ? logits : reinterpret_cast<float*>(logits_peers[0]);
  float4* dst = reinterpret_cast<float4*>(row0 + (long long)b * NRANKS * VS + (long long)rank * VS + lo);
  const int base = rank * VS + lo;
  u64 best = 0ull;
  for (int u = t; u < chunk / 4; u += THREADS) {
    const float4 v = src[u];
    dst[u] = v;
    const int i = base + 4 * u;
    best = max(best, max(max(key_of(v.x, i), key_of(v.y, i + 1)), max(key_of(v.z, i + 2), key_of(v.w, i + 3))));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) best = max(best, __shfl_xor_sync(0xffffffffu, best, o));
  if ((t & 31) == 0) sm[t >> 5] = best;
  __syncthreads();
  if (t == 0) {
#pragma unroll
    for (int w = 1; w < THREADS / 32; ++w) best = max(best, sm[w]);
    pkey[b * PARTS + p] = best;
    __threadfence_system();  // the copies and the key before the count
    s_last = atomicAdd(done + b, 1) == PARTS - 1;
  }
  __syncthreads();

  if (s_last && t < 32) {
    // the row's other blocks' keys, read past their count: an acquire at GPU scope (every block's own system
    // fence already put its copies out before it counted, so the key pushed below follows them anyway)
    __threadfence();
    u64 k = t < PARTS ? __ldcg(pkey + b * PARTS + t) : 0ull;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) k = max(k, __shfl_xor_sync(0xffffffffu, k, o));
    const long long stage = (long long)(flag % 3) * NRANKS * KEY_ROWS;
    if (t < NRANKS)
      reinterpret_cast<volatile u64*>(reinterpret_cast<u64*>(keys_peers[t]) + stage)[rank * KEY_ROWS + b] = k;
    u64 g = 0ull;
    if (t < NRANKS) {
      const volatile u64* slot = keys + stage + t * KEY_ROWS + b;
      // the deadline counts SM cycles (~2 a ns): %globaltimer jumps early in a process (k3_dcp.cu)
      const long long t0 = clock64();
      while ((g = *slot) == 0ull)
        if (clock64() - t0 > 2 * timeout_ns) {
          atomicMax(err, 1 + t);
          __threadfence_system();
          __trap();
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) g = max(g, __shfl_xor_sync(0xffffffffu, g, o));
    if (t == 0) {
      out[b] = (long long)(IDX_TOP - (int)(u32)(g & 0xffffffffull));
      done[b] = 0;
    }
  }

  if (b == 0 && p == 0 && t == 0) {
    while (*reinterpret_cast<volatile int*>(state) != (int)(gridDim.x * gridDim.y)) {
    }
    state[2] = (flag + 1) % 3;
    state[0] = 0;
  }
}
