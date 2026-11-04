// The harness's tests. Every member runs the same sequence; the Lamport chain keeps them in step.
#pragma once

__device__ __forceinline__ u64 gt() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void k_fill(bf16* p, long long n, unsigned seed, float scale) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed * 40503u;
    h ^= h >> 13, h *= 0x5bd1e995u, h ^= h >> 15;
    p[i] = __float2bfloat16(((h & 0xffffu) / 65535.f - 0.5f) * scale);
  }
}

// One thread bounces a counter off a peer `iters` times: total time / (2 iters) is the one-way flag.
__global__ void k_pingpong(uint32_t* mine, uint32_t* peer, int iters, uint32_t base, int leader, u64* out) {
  const u64 t0 = gt();
  for (uint32_t i = base + 1; i <= base + iters; ++i) {
    if (leader) {
      asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(peer), "r"(i) : "memory");
      uint32_t v;
      do asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(mine) : "memory");
      while (v != i);
    } else {
      uint32_t v;
      do asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(mine) : "memory");
      while (v != i);
      asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(peer), "r"(i) : "memory");
    }
  }
  *out = gt() - t0;
}

// The empty exchange at its simplest: block 0's thread q stores the call's number into slot `rank` of
// member q's flags and waits for slot q of its own. Other blocks only exist.
__global__ void k_flag_xchg(uint32_t* mine, const u64* peers, int* ctr, int rank) {
  if (blockIdx.x) return;
  const int q = threadIdx.x;
  const uint32_t seq = (uint32_t)ctr[0] + 1;
  __syncthreads();
  if (q < 8) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(peers[q]) + rank;
    asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(dst), "r"(seq) : "memory");
    uint32_t v;
    do asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(mine + q) : "memory");
    while (v != seq);
  }
  __syncthreads();
  if (q == 0) ctr[0] = seq;
}

__global__ void k_empty() {}
__global__ void k_empty_pdl() {
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// k_flag_xchg launched as a programmatic dependent: everything before the wait overlaps the
// previous call's tail
__global__ void k_flag_xchg_pdl(uint32_t* mine, const u64* peers, int* ctr, int rank) {
  asm volatile("griddepcontrol.launch_dependents;");
  if (blockIdx.x) return;
  const int q = threadIdx.x;
  uint32_t* dst = q < 8 ? reinterpret_cast<uint32_t*>(peers[q]) + rank : nullptr;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t seq = (uint32_t)ctr[0] + 1;
  __syncthreads();
  if (q < 8) {
    asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(dst), "r"(seq) : "memory");
    uint32_t v;
    do asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(mine + q) : "memory");
    while (v != seq);
  }
  __syncthreads();
  if (q == 0) ctr[0] = seq;
}

// One warp's issue time (cycles) of K 16-byte-a-lane stores: mode 0 all to one peer `p0`, 1 round-robin
// over the 8 members, 2 local. The value is in registers; out[0] = issue cycles, out[1] = cycles to a
// following __threadfence_system (the stores acknowledged).
__global__ void k_store_issue(const u64* peers, int rank, int K, int mode, int p0, int width, long long* out) {
  const int lane = threadIdx.x;
  const uint4 v = make_uint4(lane, 1, 2, 3);
  uint4* dst[8];
  for (int q = 0; q < 8; ++q) dst[q] = reinterpret_cast<uint4*>(peers[mode == 2 ? rank : mode == 0 ? p0 : q]) + 4096;
  __syncwarp();
  const long long t0 = clock64();
  for (int k = 0; k < K; ++k)
    if (lane < width) dst[k % 8][(k / 8) * 32 + lane + (mode == 0 ? k * 32 : 0)] = v;
  const long long t1 = clock64();
  __threadfence_system();
  const long long t2 = clock64();
  if (lane == 0) out[0] = t1 - t0, out[1] = t2 - t0;
}

// One block of 1024 reading `slots` slots of a `rv`-vector row (already there) with ld.volatile, the
// way a row block polls: cycles of one pass, the median of `reps` passes. out[blockIdx.x].
__global__ void k_readrow(const uint4* buf, long long tot, int slots, int rv, int reps, long long* out) {
  __shared__ long long best;
  long long acc = 0, tmin = 1ll << 60;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const long long t0 = clock64();
    for (int c = threadIdx.x; c < rv; c += blockDim.x)
      for (int q = 0; q < slots; ++q) {
        uint4 v;
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(buf + q * tot + (long long)blockIdx.x * rv + c + (long long)rep * 8 * rv));
        acc += v.x;
      }
    __syncthreads();
    tmin = min(tmin, clock64() - t0);
  }
  if (threadIdx.x == 0) out[blockIdx.x] = tmin + (acc == 12345);
}

// The empty exchange in steps from k_flag_xchg toward a Lamport one-shot of one vector, block 0 only.
// PAR: thread q stores to member q and polls slot q; else thread 0 does all eight. LAMPORT: a 16-byte
// payload poll (-0.0 = not arrived) over three stages with the previous one re-poisoned, the stage
// from a per-block record; else a sequence-number flag. Deadline-free (harness only).
template <bool PAR, bool LAMPORT>
__global__ void k_lx(uint8_t* lam, const u64* peers, int* recs, int rank, long long stage_bytes) {
  if (blockIdx.x) return;
  const int t = threadIdx.x;
  const uint4 rec = reinterpret_cast<const uint4*>(recs)[0];
  const unsigned call = rec.x;
  const int nq = PAR ? 1 : 8;
  if (PAR ? t < 8 : t == 0) {
    for (int k = 0; k < nq; ++k) {
      const int q = PAR ? t : k;
      uint8_t* base = reinterpret_cast<uint8_t*>(peers[q]);
      if (LAMPORT) {
        uint4* dst = reinterpret_cast<uint4*>(base + 64 * 1024 + (call % 3) * stage_bytes) + rank;
        *dst = make_uint4(call + 1, 1, 1, 1);
      } else {
        uint32_t* dst = reinterpret_cast<uint32_t*>(base) + rank;
        asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(dst), "r"(call + 1) : "memory");
      }
    }
    for (int k = 0; k < nq; ++k) {
      const int q = PAR ? t : k;
      if (LAMPORT) {
        const uint4* src = reinterpret_cast<const uint4*>(lam + (call % 3) * stage_bytes) + q;
        while (!xchg::arrived(xchg::ld_volatile(src))) {}
        reinterpret_cast<uint4*>(lam + ((call + 2) % 3) * stage_bytes)[q] = xchg::POISON;
      } else {
        uint32_t v;
        do asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(reinterpret_cast<uint32_t*>(lam) + q) : "memory");
        while (v != call + 1);
      }
    }
  }
  __syncthreads();
  if (t == 0) reinterpret_cast<uint4*>(recs)[0] = make_uint4(call + 1, 0, 0, 0);
}

// k_lx<SEQ, LAMPORT> with thread 0's poll as xchg_sum's: all eight loads in flight, the missing re-issued.
// STORES_PAR: the eight stores by threads 0..7 (then a __syncwarp) instead of thread 0.
template <bool STORES_PAR>
__global__ void k_lx2(uint8_t* lam, const u64* peers, int* recs, int rank, long long stage_bytes) {
  if (blockIdx.x) return;
  const int t = threadIdx.x;
  const uint4 rec = reinterpret_cast<const uint4*>(recs)[0];
  const unsigned call = rec.x;
  if (STORES_PAR ? t < 8 : t == 0)
    for (int k = 0; k < (STORES_PAR ? 1 : 8); ++k) {
      const int q = STORES_PAR ? t : k;
      uint8_t* base = reinterpret_cast<uint8_t*>(peers[q]);
      *(reinterpret_cast<uint4*>(base + 64 * 1024 + (call % 3) * stage_bytes) + rank) = make_uint4(call + 1, 1, 1, 1);
    }
  if (t == 0) {
    xchg::Lamport l;
    l.mine = reinterpret_cast<const uint4*>(lam + (call % 3) * stage_bytes);
    l.tot = 1;
    int fail = 0;
    const uint4 v = xchg::xchg_sum(l, 0, 1ll << 40, fail);
    for (int q = 0; q < 8; ++q) reinterpret_cast<uint4*>(lam + ((call + 2) % 3) * stage_bytes)[q] = xchg::POISON;
    if (v.x == 12345) recs[100] = 1;
  }
  __syncthreads();
  if (t == 0) reinterpret_cast<uint4*>(recs)[0] = make_uint4(call + 1, 0, 0, 0);
}

__global__ void k_fill_f32(float* p, long long n, unsigned seed, float base, float scale) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed * 40503u;
    h ^= h >> 13, h *= 0x5bd1e995u, h ^= h >> 15;
    p[i] = base + ((h & 0xffffu) / 65535.f - 0.5f) * scale;
  }
}

// embed_rms_plan's split plan (k3_ar_fused.cu): every row's splits for `seq` positions a member
static std::vector<int> mla_plan(int B, int seq, int split_max) {
  const long long tiles = (seq + 127) / 128, total = tiles * B, clusters = 76;
  auto rs = [&](long long per) { return std::min(std::max((tiles + per - 1) / per, 1ll), (long long)split_max); };
  long long best = 0, best_cost = 0;
  for (int w = 1; w <= 3; ++w) {
    const long long budget = std::max(w * clusters - B, 1ll), per = std::max((total + budget - 1) / budget, 1ll);
    const long long sp = rs(per), splits = sp * B, longest = (tiles + sp - 1) / sp;
    long long cost = 0;
    for (long long n = splits; n > 0; n -= clusters) cost += (longest + 2) * std::max(53ll, std::min(n, clusters));
    if (w == 1 || cost < best_cost) best = per, best_cost = cost;
  }
  return std::vector<int>(B, (int)rs(best));
}

// A one-block Lamport ping-pong of n 16-byte vectors: the leader stores them into the follower's
// buffer of the round, the follower's threads wait for all, re-poison, store back. out = ns / round.
// mine / peer: 2 buffers of n vectors each, poisoned. WIDE: every block of the grid takes a share.
__global__ void k_pp_data(uint4* mine, uint4* peer, int n, int rounds, int leader, u64* out) {
  const u64 t0 = gt();
  for (int r = 0; r < rounds; ++r) {
    uint4* in = mine + (r & 1) * n;
    uint4* to = peer + (r & 1) * n;
    if (leader) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) to[i] = make_uint4(r + 1, 1, 1, 1);
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        while (!xchg::arrived(xchg::ld_volatile(in + i))) {}
        in[i] = xchg::POISON;
      }
      __syncthreads();
    } else {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        while (!xchg::arrived(xchg::ld_volatile(in + i))) {}
        in[i] = xchg::POISON;
      }
      __syncthreads();
      for (int i = threadIdx.x; i < n; i += blockDim.x) to[i] = make_uint4(r + 1, 1, 1, 1);
    }
  }
  if (threadIdx.x == 0) *out = (gt() - t0) / rounds;
}

// Every member but 0 stores `n` float4 into its own slice of member 0's buffer (grid-stride), then
// the group meets in a flag exchange: per call, rank 0's ingress of 7 n float4.
__global__ void k_ingress(float4* dst0, long long n, int rank) {
  if (rank == 0) return;
  float4* d = dst0 + (long long)rank * n;
  const float4 v = make_float4(rank, 1, 2, 3);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) d[i] = v;
}

// The closing add's skeleton: a row a 1024-thread block, three vectors loaded, added, stored (STAGES 0),
// then STAGES block reductions (warp butterfly, smem, warp butterfly, barrier), then a store.
template <int STAGES>
__global__ void __launch_bounds__(1024, 1) k_land_skel(const bf16* p1, const bf16* p2, const bf16* pre, bf16* out, bf16* out2, int B) {
  __shared__ float red[32];
  const int b = blockIdx.x, t = threadIdx.x;
  if (b >= B) return;
  float acc = 0.f;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (t < 896) {
    const size_t off = (size_t)b * 7168 + t * 8;
    const uint4 a = *(const uint4*)(p1 + off), c = *(const uint4*)(p2 + off), d = *(const uint4*)(pre + off);
    v = make_uint4(a.x + c.x + d.x, a.y + c.y + d.y, a.z + c.z + d.z, a.w + c.w + d.w);
    *(uint4*)(out + off) = v;
    acc = __uint_as_float(v.x) * 1e-30f;
  }
#pragma unroll
  for (int st = 0; st < STAGES; ++st) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((t & 31) == 0) red[t >> 5] = acc;
    __syncthreads();
    acc = red[t & 31];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __syncthreads();
  }
  if (t < 896) *(uint4*)(out2 + (size_t)b * 7168 + t * 8) = make_uint4(v.x ^ __float_as_uint(acc), v.y, v.z, v.w);
}

struct Bench {
  Group& g;
  cudaStream_t s;
  static constexpr int MAXR = 64, H = 7168, HM = 10752, LAT = 3584;
  static long long timeout;
  long long stage = 8ll * MAXR * HM * 2;
  Peer L, F;
  int *state, *err, *ctr, *recs, *recs2;
  u64* dout;
  bf16 *x, *y, *prefix, *blocks, *gamma, *prefix2, *normed, *fc2, *lat_norm, *shared_sum;
  float *sw, *wts;
  int* perm;

  template <class T>
  T* dalloc(long long n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
  }

  Bench(Group& g_) : g(g_) {
    timeout = getenv("X_DEADLINE_NS") ? atoll(getenv("X_DEADLINE_NS")) : 30000000000ll;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    L = alloc_peer(g, "lamport", 3 * stage + 64);
    F = alloc_peer(g, "flags", 1 << 20);
    recs = dalloc<int>(4 * 1024), recs2 = dalloc<int>(4 * 1024), state = dalloc<int>(8), err = dalloc<int>(8), ctr = dalloc<int>(8);
    dout = dalloc<u64>(64);
    x = dalloc<bf16>((long long)MAXR * HM), y = dalloc<bf16>((long long)MAXR * HM);
    prefix = dalloc<bf16>((long long)MAXR * H), prefix2 = dalloc<bf16>((long long)MAXR * H);
    normed = dalloc<bf16>((long long)MAXR * H), blocks = dalloc<bf16>((long long)MAXR * 8 * H);
    gamma = dalloc<bf16>(HM), sw = dalloc<float>(H);
    fc2 = dalloc<bf16>((long long)MAXR * 2 * LAT), lat_norm = dalloc<bf16>((long long)MAXR * LAT);
    shared_sum = dalloc<bf16>((long long)MAXR * H), wts = dalloc<float>(MAXR * 16), perm = dalloc<int>(MAXR * 16);
    k_fill<<<512, 256, 0, s>>>(x, (long long)MAXR * HM, 1 + g.rank, 2.f);
    k_fill<<<512, 256, 0, s>>>(prefix, (long long)MAXR * H, 100, 2.f);
    k_fill<<<512, 256, 0, s>>>(blocks, (long long)MAXR * 8 * H, 200, 2.f);
    k_fill<<<512, 256, 0, s>>>(gamma, HM, 300, 2.f);
    k_fill<<<512, 256, 0, s>>>(fc2, (long long)MAXR * 2 * LAT, 400 + g.rank, 2.f);
    std::vector<int> hp(MAXR * 16, -1);
    std::vector<float> hw(MAXR * 16, 1.f / 16), hs(H, 0.01f);
    for (int t = 0; t < MAXR; ++t) hp[t * 16] = 2 * t, hp[t * 16 + 5] = 2 * t + 1;
    CK(cudaMemcpy(perm, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sw, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
    CK(cudaStreamSynchronize(s));
    barrier(g, "init");
  }

  void res(const char* name, int rows, int width, double us) {
    printf("RES %s %d %d %.3f\n", name, rows, width, us);
    fflush(stdout);
  }

  double timed(int calls, std::function<void()> f) {
    barrier(g, "t");
    return per_call_us(s, calls, 5, f);
  }

  void pingpong() {
    for (int q : {1, 4, 5}) {
      barrier(g, "pp");
      if (g.rank == 0 || g.rank == q) {
        uint32_t* mine = reinterpret_cast<uint32_t*>(F.local) + 64;
        uint32_t* peer = reinterpret_cast<uint32_t*>(F.peers[g.rank == 0 ? q : 0]) + 64;
        CK(cudaMemset(mine, 0, 4));
        CK(cudaDeviceSynchronize());
        barrier(g, "pp0");
        const int iters = 2000;
        k_pingpong<<<1, 1, 0, s>>>(mine, peer, iters, 0, g.rank == 0, dout);
        u64 ns;
        CK(cudaMemcpyAsync(&ns, dout, 8, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        if (g.rank == 0) res(q < 4 ? "oneway_tray" : "oneway_cross", 0, q, ns / 1e3 / (2.0 * iters));
      } else {
        barrier(g, "pp0");
      }
    }
  }

  void run(const std::string& t) {
    const std::vector<int> rows = {1, 8, 16, 24, 32, 48, 64};
    if (t == "pingpong") return pingpong();
    if (t == "empty") {
      for (int grid : {1, 152, 304})
        for (int pdl : {0, 1})
          res(pdl ? "empty_pdl" : "empty", grid, 0, timed(200, [&] {
                if (pdl) launch_ex(s, k_empty_pdl, grid, grid == 1 ? 32 : 1024 / (grid / 152), 0, true);
                else k_empty<<<grid, grid == 1 ? 32 : 1024 / (grid / 152), 0, s>>>();
              }));
      return;
    }
    if (t == "flagpdl") {
      for (int grid : {1, 152})
        res("flag_xchg_pdl", grid, 0, timed(200, [&] {
              launch_ex(s, k_flag_xchg_pdl, grid, grid == 1 ? 32 : 1024, 0, true, reinterpret_cast<uint32_t*>(F.local),
                        (const u64*)F.d_peers, ctr, g.rank);
            }));
      return;
    }
    if (t == "xar" || t == "xar2" || t == "xarp" || t == "xar2p") {
      // k3_xchg.cuh's all-reduce: one-shot (xar), two-shot (xar2), PDL launches (..p)
      const bool pdl = t.back() == 'p', two = t.rfind("xar2", 0) == 0, gate = false;
      for (int w : {H, HM})
        for (int r : {0, 1, 8, 16, 24, 32, 48, 64}) {
          if (two && r < 8) continue;
          const int rows = r ? r : 1, rv = r ? w / 8 : 1;
          res(t.c_str(), r, w, timed(100, [&] {
                launch_ex(s, pdl ? kern_xar_pdl : kern_xar, 152, 1024, 0, pdl, (const uint4*)x, (uint4*)y, L.local,
                          (const u64*)L.d_peers, recs, err, g.rank, rows, rv, stage, timeout, (gate ? -1 : 1) * (two ? 1 : 1 << 30),
                          (u64*)nullptr, (int*)nullptr);
              }));
        }
      return;
    }
    if (t == "issue") {
      if (g.rank == 0) {
        for (int mode : {2, 0, 1})
          for (int p0 : {1, 4})
            for (int K : {1, 2, 4, 8, 16, 64}) {
              if (mode != 0 && p0 == 4) continue;
              long long h[2] = {0, 0}, best[2] = {1ll << 60, 1ll << 60};
              for (int rep = 0; rep < 20; ++rep) {
                k_store_issue<<<1, 32, 0, s>>>(F.d_peers, g.rank, K, mode, p0, 32, (long long*)dout);
                CK(cudaMemcpy(h, dout, 16, cudaMemcpyDeviceToHost));
                best[0] = std::min(best[0], h[0]), best[1] = std::min(best[1], h[1]);
              }
              printf("ISSUE mode %s peer %d K %d: issue %lld cyc, acked %lld cyc\n",
                     mode == 2 ? "local" : mode == 0 ? "one-peer" : "8-members", p0, K, best[0], best[1]);
            }
        fflush(stdout);
      }
      barrier(g, "issue");
      return;
    }
    if (t == "readrow") {
      if (g.rank == 0) {
        long long* o = dalloc<long long>(256);
        for (int rv : {896, 1344})
          for (int slots : {1, 8}) {
            k_readrow<<<1, 1024, 0, s>>>((const uint4*)L.local, (long long)MAXR * HM / 8, slots, rv, 20, o);
            long long h;
            CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
            printf("READROW rv %d slots %d: %lld cycles (one block)\n", rv, slots, h);
            k_readrow<<<64, 1024, 0, s>>>((const uint4*)L.local, (long long)MAXR * HM / 8, slots, rv, 20, o);
            CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
            printf("READROW rv %d slots %d: %lld cycles (block 0 of 64)\n", rv, slots, h);
          }
        fflush(stdout);
      }
      barrier(g, "rr");
      return;
    }
    if (t == "lx") {
      // fresh stages a variant: the flags variant uses the stage area's start as flags
      auto go = [&](const char* name, void (*k)(uint8_t*, const u64*, int*, int, long long), int threads, int grid) {
        CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
        CK(cudaMemsetAsync(L.local, 0, 64, s));
        kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local + 64 * 1024, 3 * stage);
        CK(cudaStreamSynchronize(s));
        res(name, grid, threads, timed(200, [&] { k<<<grid, threads, 0, s>>>(L.local + 64 * 1024, L.d_peers, recs, g.rank, stage); }));
      };
      // the flag variant writes at offset 0 of each peer's L: pass the base for it
      auto go_flag = [&](const char* name, void (*k)(uint8_t*, const u64*, int*, int, long long), int threads, int grid) {
        CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
        CK(cudaMemsetAsync(L.local, 0, 64, s));
        CK(cudaStreamSynchronize(s));
        res(name, grid, threads, timed(200, [&] { k<<<grid, threads, 0, s>>>(L.local, L.d_peers, recs, g.rank, stage); }));
      };
      go_flag("lx_par_flag", k_lx<true, false>, 32, 1);
      go_flag("lx_seq_flag", k_lx<false, false>, 32, 1);
      go("lx_par_lamport", k_lx<true, true>, 32, 1);
      go("lx_seq_lamport", k_lx<false, true>, 32, 1);
      go("lx_seq_lamport", k_lx<false, true>, 1024, 152);
      go("lx2_seqstore_parpoll", k_lx2<false>, 32, 1);
      go("lx2_parstore_parpoll", k_lx2<true>, 32, 1);
      return;
    }
    if (t == "verify") {
      // main's fused kernels (main_fused.cu, own lamport and state) against this tree's (k3_xchg.cuh, recs):
      // the same outputs byte for byte at every row count, one-shot and two-shot
      static Peer L2 = alloc_peer(g, "lamport_main", 3 * stage + 64);
      static int* st2 = dalloc<int>(4096);
      static bool init = false;
      if (!init) {
        kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L2.local, 3 * stage);
        CK(cudaStreamSynchronize(s));
        barrier(g, "vinit");
        init = true;
      }
      CK(cudaFuncSetAttribute(kern_k3_ar_attnres_rms, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 896 * 16));
      CK(cudaFuncSetAttribute(kern_main_ar_attnres_rms, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 896 * 16));
      bf16 *pa = dalloc<bf16>((long long)MAXR * H), *na = dalloc<bf16>((long long)MAXR * H);
      bf16 *ln = dalloc<bf16>((long long)MAXR * LAT), *ss = dalloc<bf16>((long long)MAXR * H);
      auto same = [&](const bf16* a, const bf16* b, long long n) {
        std::vector<uint16_t> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 2, cudaMemcpyDeviceToHost));
        long long d = 0;
        for (long long i = 0; i < n; ++i) d += ha[i] != hb[i];
        return d;
      };
      for (int nbv : {1, 3, 4, 8})
      for (int r : {1, 4, 8, 11, 12, 16, 19, 20, 24, 32, 33, 36, 48, 64})
        for (int rep = 0; rep < 3; ++rep) {
          // three calls each so the stages rotate
          CK(cudaMemsetAsync(prefix2, 0, (long long)MAXR * H * 2, s));
          kern_main_ar_attnres_rms<<<152, 1024, 8 * 896 * 16, s>>>((const uint4*)x, L2.local, L2.d_peers, st2, err, g.rank,
                                                                   prefix, blocks, sw, gamma, pa, na, nbv, 0, r, stage,
                                                                   timeout, nullptr, 0, nullptr, 0);
          kern_k3_ar_attnres_rms<<<152, 1024, 8 * 896 * 16, s>>>((const uint4*)x, L.local, L.d_peers, recs, err, g.rank,
                                                                 prefix, blocks, sw, gamma, prefix2, normed, nbv, 0, r,
                                                                 stage, timeout, nullptr, 0, nullptr, 0);
          kern_main_ar_finalize_rms<<<152, 1024, 0, s>>>(fc2, perm, wts, (const uint4*)x, L2.local, L2.d_peers, st2, err,
                                                         g.rank, gamma, ln, (uint4*)ss, r, stage, timeout, nullptr, 0, nullptr, 0);
          kern_k3_ar_finalize_rms<<<152, 1024, 0, s>>>(fc2, perm, wts, (const uint4*)x, L.local, L.d_peers, recs, err,
                                                       g.rank, gamma, lat_norm, (uint4*)shared_sum, r, stage, timeout,
                                                       nullptr, 0, nullptr, 0);
          CK(cudaStreamSynchronize(s));
          if (rep == 2)
            printf("VERIFY nb %d rows %d: attn prefix2 %lld normed %lld | moe latent_norm %lld shared_sum %lld differ\n", nbv, r,
                   same(pa, prefix2, (long long)r * H), same(na, normed, (long long)r * H),
                   same(ln, lat_norm, (long long)r * LAT), same(ss, shared_sum, (long long)r * H));
        }
      fflush(stdout);
      return;
    }
    if (t == "fused") {
      // main's fused kernels and this tree's, interleaved in one lease: us per call
      static Peer L2 = alloc_peer(g, "lamport_main2", 3 * stage + 64);
      static int* st2 = dalloc<int>(4096);
      kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L2.local, 3 * stage);
      CK(cudaStreamSynchronize(s));
      barrier(g, "finit");
      CK(cudaFuncSetAttribute(kern_k3_ar_attnres_rms, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 896 * 16));
      CK(cudaFuncSetAttribute(kern_main_ar_attnres_rms, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 896 * 16));
      for (int r : {1, 8, 16, 24, 36, 48, 64}) {
        res("attn_main", r, H, timed(100, [&] {
              kern_main_ar_attnres_rms<<<152, 1024, 8 * 896 * 16, s>>>((const uint4*)x, L2.local, L2.d_peers, st2, err,
                  g.rank, prefix, blocks, sw, gamma, prefix2, normed, 4, 0, r, stage, timeout, nullptr, 0, nullptr, 0);
            }));
        res("attn_new", r, H, timed(100, [&] {
              kern_k3_ar_attnres_rms<<<152, 1024, 8 * 896 * 16, s>>>((const uint4*)x, L.local, L.d_peers, recs, err,
                  g.rank, prefix, blocks, sw, gamma, prefix2, normed, 4, 0, r, stage, timeout, nullptr, 0, nullptr, 0);
            }));
        res("moe_main", r, HM, timed(100, [&] {
              kern_main_ar_finalize_rms<<<152, 1024, 0, s>>>(fc2, perm, wts, (const uint4*)x, L2.local, L2.d_peers, st2,
                  err, g.rank, gamma, lat_norm, (uint4*)shared_sum, r, stage, timeout, nullptr, 0, nullptr, 0);
            }));
        res("moe_new", r, HM, timed(100, [&] {
              kern_k3_ar_finalize_rms<<<152, 1024, 0, s>>>(fc2, perm, wts, (const uint4*)x, L.local, L.d_peers, recs,
                  err, g.rank, gamma, lat_norm, (uint4*)shared_sum, r, stage, timeout, nullptr, 0, nullptr, 0);
            }));
      }
      return;
    }
    if (t == "dcp") {
      // kern_k3_dcp_exchange as the step runs it (grid 256 x 256, 64 KB smem), 8192 positions a member (64k)
      const int SM = 16, HLc = 12;
      static Peer D = alloc_peer(g, "dcp", 3ll * 8 * 64 * HLc * 65 * 16 + 4096);
      const long long dstage = 8ll * 64 * HLc * 65 * 16;
      static float* acc_o = nullptr;
      static float *acc_lse;
      static int *sl, *bk, *dst;
      static bf16 *wkv, *gate, *gated;
      if (!acc_o) {
        acc_o = dalloc<float>(64ll * 128 * SM * 512), acc_lse = dalloc<float>(64ll * 128 * SM);
        sl = dalloc<int>(64), bk = dalloc<int>(64), dst = dalloc<int>(8);
        wkv = dalloc<bf16>(HLc * 256ll * 512), gate = dalloc<bf16>(64ll * HLc * 128), gated = dalloc<bf16>(64ll * HLc * 128);
        k_fill_f32<<<1024, 256, 0, s>>>(acc_o, 64ll * 128 * SM * 512, 7 + g.rank, 0.f, 1.f);
        k_fill_f32<<<256, 256, 0, s>>>(acc_lse, 64ll * 128 * SM, 9 + g.rank, 8.f, 2.f);
        k_fill<<<512, 256, 0, s>>>(wkv, HLc * 256ll * 512, 11, 0.1f);
        k_fill<<<512, 256, 0, s>>>(gate, 64ll * HLc * 128, 13, 2.f);
        kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(D.local, 3 * dstage);
        CK(cudaStreamSynchronize(s));
        CK(cudaFuncSetAttribute(kern_k3_dcp_exchange, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
        barrier(g, "dinit");
      }
      if (t == "dcp") {
        // main's kernel against this tree's, own stages each: gated byte for byte at every shape
        static Peer D2 = alloc_peer(g, "dcp_main", 3 * dstage + 4096);
        static int* dst2 = dalloc<int>(8);
        static bf16* gated2 = dalloc<bf16>(64ll * HLc * 128);
        static bool i2 = false;
        if (!i2) {
          kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(D2.local, 3 * dstage);
          CK(cudaFuncSetAttribute(kern_maindcp_exchange, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
          CK(cudaStreamSynchronize(s));
          barrier(g, "d2init");
          i2 = true;
        }
        for (int seq : {8192, 100, 3000})
          for (int r : {1, 3, 8, 16, 24, 36, 48, 64}) {
            const auto plan = mla_plan(r, seq, SM);
            std::vector<int> hs(r, seq);
            if (r > 1) hs[1] = 0;  // a row with no position on this member
            CK(cudaMemcpy(sl, hs.data(), r * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(bk, plan.data(), r * 4, cudaMemcpyHostToDevice));
            for (int rep = 0; rep < 3; ++rep) {
              kern_maindcp_exchange<<<256, 256, 65536, s>>>(acc_o, acc_lse, sl, bk, SM, wkv, gate, gated2, D2.local, D2.d_peers,
                                                           dst2, err, g.rank, r, dstage, timeout);
              kern_k3_dcp_exchange<<<256, 256, 65536, s>>>(acc_o, acc_lse, sl, bk, SM, wkv, gate, gated, D.local, D.d_peers,
                                                          dst, err, g.rank, r, dstage, timeout);
            }
            CK(cudaStreamSynchronize(s));
            std::vector<uint16_t> a(r * HLc * 128), b2(r * HLc * 128);
            CK(cudaMemcpy(a.data(), gated2, a.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b2.data(), gated, b2.size() * 2, cudaMemcpyDeviceToHost));
            long long d = 0;
            for (size_t i = 0; i < a.size(); ++i) d += a[i] != b2[i];
            printf("VERIFY dcp seq %d rows %d splits %d: gated %lld of %zu differ\n", seq, r, plan[0], d, a.size());
          }
        fflush(stdout);
      }
      for (int seq : {8192, 16384})
        for (int r : {1, 8, 16, 24, 36, 48, 64}) {
          const auto plan = mla_plan(r, seq, SM);
          std::vector<int> hs(r, seq);
          CK(cudaMemcpy(sl, hs.data(), r * 4, cudaMemcpyHostToDevice));
          CK(cudaMemcpy(bk, plan.data(), r * 4, cudaMemcpyHostToDevice));
          char name[64];
          snprintf(name, sizeof name, "dcp_ctx%dk_splits%d", seq * 8 / 1024, plan[0]);
          res(name, r, 0, timed(100, [&] {
                kern_k3_dcp_exchange<<<256, 256, 65536, s>>>(acc_o, acc_lse, sl, bk, SM, wkv, gate, gated, D.local,
                                                            D.d_peers, dst, err, g.rank, r, dstage, timeout);
              }));
        }
      return;
    }
    if (t == "ppdata") {
      for (int q : {1, 4})
        for (int n : {1, 32, 256, 896, 2048, 4096}) {
          barrier(g, "ppd");
          if (g.rank == 0 || g.rank == q) {
            uint4* mine = reinterpret_cast<uint4*>(F.local + 4096);
            uint4* peer = reinterpret_cast<uint4*>(F.peers[g.rank == 0 ? q : 0] + 4096);
            kern_peer_lamport_init_bf16<<<64, 256, 0, s>>>((uint8_t*)mine, 2ll * n * 16);
            CK(cudaStreamSynchronize(s));
            barrier(g, "ppd0");
            k_pp_data<<<1, 1024, 0, s>>>(mine, peer, n, 500, g.rank == 0, dout);
            u64 ns;
            CK(cudaMemcpyAsync(&ns, dout, 8, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            if (g.rank == 0) res(q < 4 ? "ppdata_tray_oneway" : "ppdata_cross_oneway", n, n * 16, ns / 2e3);
          } else {
            barrier(g, "ppd0");
          }
        }
      return;
    }
    if (t == "mc") {
      // multicast: kern_xar_mc (one multimem.st a vector every member receives) against kern_xar, one lease
      const long long mstage = 9ll * MAXR * HM * 2;
      static Peer M = alloc_peer(g, "lamport_mc", 3 * mstage + 64, true);
      static int* mrecs = dalloc<int>(4 * 1024);
      kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(M.local, 3 * mstage);
      CK(cudaStreamSynchronize(s));
      barrier(g, "mcinit");
      for (int w : {H, HM})
        for (int r : {1, 8, 16, 24, 32, 48, 64})
          for (int two : {0, 1}) {
            if (two && r < 8) continue;
            const int rv = w / 8;
            res(two ? "mc_two" : "mc_one", r, w, timed(100, [&] {
                  kern_xar_mc<<<152, 1024, 0, s>>>((const uint4*)x, (uint4*)y, M.local, M.mc, (const u64*)M.d_peers, mrecs,
                                                   err, g.rank, r, rv, mstage, timeout, two ? 0 : 1 << 30);
                }));
            res(two ? "uc_two" : "uc_one", r, w, timed(100, [&] {
                  launch_ex(s, kern_xar, 152, 1024, 0, false, (const uint4*)x, (uint4*)y, L.local, (const u64*)L.d_peers,
                            recs, err, g.rank, r, rv, stage, timeout, two ? 1 : 1 << 30, (u64*)nullptr, (int*)nullptr);
                }));
          }
      return;
    }
    if (t == "head") {
      // k3_head as the step runs it: grid (B, 16) x 256, every rank's [B, 20480] f32 slice to rank 0
      const int VS = 20480;
      static Peer LG = alloc_peer(g, "logits", 64ll * 8 * VS * 4);
      static Peer KY = alloc_peer(g, "keys", 3 * 8 * 64 * 8 + 4096);
      static float* part = nullptr;
      static long long* out;
      static int *done, *hst;
      static u64* pkey;
      if (!part) {
        part = dalloc<float>(64ll * VS), out = dalloc<long long>(64), done = dalloc<int>(64), hst = dalloc<int>(8);
        pkey = dalloc<u64>(64 * 16);
        k_fill_f32<<<512, 256, 0, s>>>(part, 64ll * VS, 31 + g.rank, 0.f, 10.f);
        CK(cudaStreamSynchronize(s));
        barrier(g, "hinit");
      }
      for (int r : {1, 8, 16, 24, 36, 48, 64}) {
        res("head", r, VS, timed(50, [&] {
              kern_k3_head_argmax<<<dim3(r, 16), 256, 0, s>>>(part, (float*)LG.local, LG.d_peers, out, done, pkey, KY.local,
                                                              KY.d_peers, hst, err, g.rank, VS, timeout, r);
            }));
      }
      return;
    }
    if (t == "ingress") {
      static Peer IG = alloc_peer(g, "ingress", 8ll * 8 * 1024 * 1024);
      const double fx = timed(100, [&] {
        k_flag_xchg<<<1, 32, 0, s>>>(reinterpret_cast<uint32_t*>(F.local), F.d_peers, ctr, g.rank);
      });
      for (long long mb : {1, 2, 4, 8}) {
        const long long n = mb * 1024 * 1024 / 16;
        for (int grid : {152, 304, 608}) {
          const double us = timed(50, [&] {
            k_ingress<<<grid, 512, 0, s>>>(reinterpret_cast<float4*>(IG.peers[0]), n, g.rank);
            k_flag_xchg<<<1, 32, 0, s>>>(reinterpret_cast<uint32_t*>(F.local), F.d_peers, ctr, g.rank);
          });
          if (g.rank == 0)
            printf("RES ingress_%lldMB_per_rank grid %d: %.2f us (flag %.2f) -> %.0f GB/s into rank 0\n", mb, grid, us, fx,
                   7.0 * mb * 1048576 / ((us - fx) * 1e3));
        }
      }
      fflush(stdout);
      return;
    }
    if (t == "fi") {
      // FlashInfer's TRT-LLM allreduce fusion as written (kAllReduce pattern, bf16, f32 accumulation off as
      // its default for bf16), on a workspace laid out as trtllm_create_ipc_workspace_for_all_reduce_fusion's
      const long long buf = 8ll * MAXR * HM * 2, flags = 8ll * 256 * 4, lcomm = 8ll * 128 * HM * 2;
      static Peer CB = alloc_peer(g, "fi_comm", buf), BF = alloc_peer(g, "fi_flags", flags),
                  LB = alloc_peer(g, "fi_lamport", 3 * lcomm);
      static void** ws = nullptr;
      if (!ws) {
        int* ctl = dalloc<int>(16);
        const int h5[5] = {0, 0, 0, (int)lcomm, 0};
        CK(cudaMemcpy(ctl, h5, sizeof h5, cudaMemcpyHostToDevice));
        std::vector<void*> hw(3 * 8 + 1);
        for (int q = 0; q < 8; ++q) hw[q] = (void*)CB.peers[q], hw[8 + q] = (void*)BF.peers[q], hw[16 + q] = (void*)LB.peers[q];
        hw[24] = ctl;
        CK(cudaMalloc(&ws, hw.size() * 8));
        CK(cudaMemcpy(ws, hw.data(), hw.size() * 8, cudaMemcpyHostToDevice));
        kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(LB.local, 3 * lcomm);
        CK(cudaStreamSynchronize(s));
        barrier(g, "fiinit");
      }
      for (int w : {H, HM})
        for (int r : {1, 8, 16, 24, 32, 48, 64})
          for (int one : {1, 0}) {
            res(one ? "fi_oneshot" : "fi_twoshot", r, w, timed(100, [&] {
                  const int e = fi_allreduce_bf16(ws, g.rank, r, w, x, y, one, false, s);
                  if (e) { fprintf(stderr, "fi error %d\n", e); exit(1); }
                }));
          }
      return;
    }
    if (t == "local") {
      // kern_xar_local at FlashInfer's grids (7168: 4 CTAs x 224 a row; 10752: 8 x 168) against kern_xar
      for (int w : {H, HM})
        for (int r : {1, 8, 16, 24}) {
          const int rv = w / 8, bs = w == H ? 224 : 168, grid = r * rv / bs;
          CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
          kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
          CK(cudaStreamSynchronize(s));
          barrier(g, "lreset");
          res("xar_local", r, w, timed(100, [&] {
                kern_xar_local<<<grid, bs, 0, s>>>((const uint4*)x, (uint4*)y, L.local, L.d_peers, recs, err, g.rank, r, rv,
                                                  stage, timeout);
              }));
          CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
          kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
          CK(cudaStreamSynchronize(s));
          barrier(g, "lreset2");
          res("xar", r, w, timed(100, [&] {
                launch_ex(s, kern_xar, 152, 1024, 0, false, (const uint4*)x, (uint4*)y, L.local, (const u64*)L.d_peers,
                          recs, err, g.rank, r, rv, stage, timeout, 1 << 30, (u64*)nullptr, (int*)nullptr);
              }));
          {
            CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
            kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
            CK(cudaStreamSynchronize(s));
            barrier(g, "lreset4");
            res("xar_local1024", r, w, timed(100, [&] {
                  kern_xar_local1024<<<(r * rv + 1023) / 1024, 1024, 0, s>>>((const uint4*)x, (uint4*)y, L.local, L.d_peers,
                                                                             recs, err, g.rank, r, rv, stage, timeout);
                }));
          }
          {
            static int* dummy = dalloc<int>(1);
            CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
            kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
            CK(cudaStreamSynchronize(s));
            barrier(g, "lreset5");
            res("xar_spreadpoll", r, w, timed(100, [&] {
                  launch_ex(s, kern_xar, 152, 1024, 0, false, (const uint4*)x, (uint4*)y, L.local, (const u64*)L.d_peers,
                            recs, err, g.rank, r, rv, stage, timeout, 1 << 30, (u64*)nullptr, dummy);
                }));
          }
          for (int gb : {r, 4 * r}) {
            CK(cudaMemsetAsync(recs, 0, 16 * 1024, s));
            kern_peer_lamport_init_bf16<<<256, 256, 0, s>>>(L.local, 3 * stage);
            CK(cudaStreamSynchronize(s));
            barrier(g, "lreset3");
            res(gb == r ? "xar_gridB" : "xar_grid4B", r, w, timed(100, [&] {
                  launch_ex(s, kern_xar, std::min(gb, 152), 1024, 0, false, (const uint4*)x, (uint4*)y, L.local, (const u64*)L.d_peers,
                            recs, err, g.rank, r, rv, stage, timeout, 1 << 30, (u64*)nullptr, (int*)nullptr);
                }));
          }
        }
      return;
    }
    if (t == "skel") {
      for (int r : {1, 24})
        for (int grid : {0, 152}) {
          const int G = grid ? grid : r;
          res(grid ? "skel0_g152" : "skel0_gB", r, H, per_call_us(s, 64, 5, [&] { k_land_skel<0><<<G, 1024, 0, s>>>(x, y, prefix, prefix2, normed, r); }));
          res(grid ? "skel1_g152" : "skel1_gB", r, H, per_call_us(s, 64, 5, [&] { k_land_skel<1><<<G, 1024, 0, s>>>(x, y, prefix, prefix2, normed, r); }));
          res(grid ? "skel2_g152" : "skel2_gB", r, H, per_call_us(s, 64, 5, [&] { k_land_skel<2><<<G, 1024, 0, s>>>(x, y, prefix, prefix2, normed, r); }));
          res(grid ? "skel4_g152" : "skel4_gB", r, H, per_call_us(s, 64, 5, [&] { k_land_skel<4><<<G, 1024, 0, s>>>(x, y, prefix, prefix2, normed, r); }));
        }
      return;
    }
    if (t == "flag") {
      for (int grid : {1, 152})
        res("flag_xchg", grid, 0, timed(200, [&] {
              k_flag_xchg<<<grid, grid == 1 ? 32 : 1024, 0, s>>>(reinterpret_cast<uint32_t*>(F.local), F.d_peers, ctr, g.rank);
            }));
      return;
    }
    if (t == "ar") {
      // kern_peer_allreduce_bf16 as --peer-ar launches it (2 x 152 blocks of 512); rows 0 is one vector
      for (int w : {H, HM})
        for (int r : {0, 1, 8, 16, 24, 32, 48, 64}) {
          const long long n = r ? (long long)r * w : 8;
          res("ar_oneshot", r, w, timed(100, [&] {
                kern_peer_allreduce_bf16<<<304, 512, 0, s>>>((const uint4*)x, (uint4*)y, L.local, L.d_peers, state, err,
                                                             g.rank, n, n, 0, stage, timeout);
              }));
        }
      return;
    }
    fprintf(stderr, "unknown test %s\n", t.c_str());
    exit(2);
  }
};
long long Bench::timeout;
