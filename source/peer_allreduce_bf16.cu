// A DCP group's bf16 all-reduce of a flat buffer: TensorRT-LLM's Lamport
// one-shot on kern's peer ABI, every rank holding the same elements in the
// same order (a replicated decode batch), so there is no row structure and
// no rotation. Derived from kern's tools/kernels-src/peer_allreduce.cu, itself
// a port of tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
// (NVIDIA, Apache-2.0).
//
//   Every rank pushes its whole partial into slot `rank` of every peer's
//   Lamport stage; a 16-byte word holding a bf16 -0.0 (0x8000) has not
//   arrived, so the poison is the flag and the wire carries payload only.
//   Every rank sums the N slots in rank order in f32 and rounds once, so all
//   ranks write identical bytes. Three stages rotate: the one used last call
//   is re-poisoned while this one is in flight, so no barrier surrounds the
//   exchange. An input -0.0 is sent as +0.0 (a sum's sign of zero is lost).
//
//   kern_peer_allreduce_bf16(in bf16 x[], out bf16 y[], inout u8 lamport,
//                            in u64 lamport_peers[N], inout i32 state[8],
//                            out i32 err[1], i32 rank, i64 n, i64 at2, i64 n2,
//                            i64 stage_bytes, i64 timeout_ns)
//     Sums elements [0, n) and [at2, at2 + n2) of x (at2 >= n) into the same
//     places of y, one exchange for both: a MoE layer's routed rows and its
//     shared expert's, laid out apart, without the rows between them. N is the
//     compile-time NRANKS (default 8); n and n2 multiples of 8. `lamport`
//     holds 3 stages of `stage_bytes` >= N * (n + n2)_max * 2, poisoned once by
//     kern_peer_lamport_init_bf16, then N handshake words the first call sets
//     in every peer and waits for in its own (no push before the peers' init).
//     `state` starts zeroed: [0] block counter, [2] stage, [4..5] i64 16-byte
//     words to re-poison next call, [6] handshake done. Any grid whose blocks
//     are all resident at once (block 0 waits for every block to arrive).
//     `err` gets 1 + the rank whose data did not show within `timeout_ns`,
//     then the kernel traps.
//
//   kern_peer_lamport_init_bf16(inout u8 lamport, i64 bytes)

#include <cstdint>

#ifndef NRANKS
#define NRANKS 8
#endif

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A passed deadline is a broken group: record the late rank and trap, never sum a missing slot.
__device__ __forceinline__ void give_up(int* err, int missing) {
    atomicMax(err, missing);
    __threadfence_system();
    __trap();
}

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ bool poisoned(uint32_t w) {
    return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u;
}

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

__device__ __forceinline__ uint32_t pack(float a, float b) {
    return round_bf16(a) | (round_bf16(b) << 16);
}

extern "C" __global__ void __launch_bounds__(1024)
kern_peer_allreduce_bf16(const uint4* x, uint4* y, uint8_t* lamport, const unsigned long long* lamport_peers, int* state,
                         int* err, int rank, long long n, long long at2, long long n2, long long stage_bytes,
                         long long timeout_ns) {
    const long long first_n = n >> 3, gap = (at2 - n) >> 3;
    const long long tot = (n + n2) >> 3;
    // the i-th vector of the exchange is x's vector at
    #define AT(i) ((i) < first_n ? (i) : (i) + gap)
    const int flag = state[2];
    long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
    const long long clear_count = *clear_ptr;
    uint4* slot[NRANKS];
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
        uint8_t* base = reinterpret_cast<uint8_t*>(lamport_peers[q]) + (flag % 3) * stage_bytes;
        slot[q] = reinterpret_cast<uint4*>(base) + (long long)rank * tot;
    }
    const uint4* mine = reinterpret_cast<const uint4*>(lamport + (flag % 3) * stage_bytes);
    uint4* clear_buf = reinterpret_cast<uint4*>(lamport + ((flag + 2) % 3) * stage_bytes);
    // k3_ar_fused.cu's first-call handshake: no push before every peer's stages are poisoned
    if (threadIdx.x < NRANKS && !*reinterpret_cast<volatile int*>(state + 6)) {
        if (blockIdx.x == 0)
            reinterpret_cast<volatile int*>(reinterpret_cast<uint8_t*>(lamport_peers[threadIdx.x]) + 3 * stage_bytes)[rank] = 1;
        if (threadIdx.x == 0) {
            const volatile int* ready = reinterpret_cast<const volatile int*>(lamport + 3 * stage_bytes);
            const unsigned long long t0 = gtimer();
            for (int q = 0; q < NRANKS; ++q)
                while (!ready[q])
                    if ((long long)(gtimer() - t0) > timeout_ns) give_up(err, 1 + q);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(state, 1);

    const long long first = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = first; i < tot; i += stride) {
        uint4 v = x[AT(i)];
        v = make_uint4(unpoison(v.x), unpoison(v.y), unpoison(v.z), unpoison(v.w));
#pragma unroll
        for (int q = 0; q < NRANKS; ++q) slot[q][i] = v;
    }
    const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    for (long long i = first; i < clear_count; i += stride) clear_buf[i] = poison;

    int fail = 0;
    for (long long i = first; i < tot; i += stride) {
        uint4 vals[NRANKS];
        unsigned long long t0 = 0;
        int missing;
        while (true) {
            missing = 0;
#pragma unroll
            for (int r = 0; r < NRANKS; ++r) {
                vals[r] = ld_volatile(mine + (long long)r * tot + i);
                if (missing == 0 && !arrived(vals[r])) missing = 1 + r;
            }
            if (missing == 0) break;
            const unsigned long long now = gtimer();
            if (t0 == 0) {
                t0 = now;
            } else if ((long long)(now - t0) > timeout_ns) {
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
        y[AT(i)] = make_uint4(pack(acc[0], acc[1]), pack(acc[2], acc[3]), pack(acc[4], acc[5]), pack(acc[6], acc[7]));
    }
    if (fail) give_up(err, fail);
    #undef AT

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (*reinterpret_cast<volatile int*>(state) != (int)gridDim.x) {
        }
        state[2] = (flag + 1) % 3;
        *clear_ptr = (long long)NRANKS * tot;
        state[6] = 1;
        state[0] = 0;
    }
}

extern "C" __global__ void kern_peer_lamport_init_bf16(uint8_t* lamport, long long bytes) {
    uint4* p = reinterpret_cast<uint4*>(lamport);
    const uint4 poison = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16; i += (long long)gridDim.x * blockDim.x) {
        p[i] = poison;
    }
}
