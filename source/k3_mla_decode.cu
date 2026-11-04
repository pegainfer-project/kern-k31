// K3 MLA decode attention over a DCP member's latent pages, bf16: the DSL decode's split kernel and its contract
// (gen_stage.py mla_attn_op), a CTA pair a (row, split), heads as M.
//
//   extern "C" __global__ void kern_k3_mla_decode(
//       CUtensorMap q_map,     // q_abs [B][96][576] bf16: box 64 dims x 64 heads x 1, 128 B swizzle
//       CUtensorMap kv_map,    // latent pages [pages][64][576] bf16: box 64 dims x 64 positions x 1, 128 B swizzle
//       const int* table,      // [B][max_pages]
//       const int* seq_lens,   // [B]  positions this member holds
//       const int* bsk,        // [B]  splits asked for
//       float* acc_o,          // [B][128][split_max][512]  the splits' normalized o
//       float* acc_lse,        // [B][128][split_max]       their log2-sum-exp
//       int split_max, int max_pages, float scale_log2);
//
//   grid (2, B, split_max)   cluster (2, 1, 1)   block THREADS   smem SMEM_BYTES dynamic
//
// Row b's t = ceil(n / 128) tiles of 128 positions run as S = ceil(t / per) splits of per = ceil(t / bsk) tiles;
// clusters past S, and rows with n = 0, exit. CTA c holds heads 48c .. 48c + 63 as its 64 MMA rows (heads 48-63 twice,
// CTA 1's last 16 the map's zero fill) and writes its first 48; it holds page 2i + c of tile i. Per tile:
//   S = Q K^T    M 128 (pair) x N 128 positions (each CTA its page) x K 576, into TMEM (three buffers: Q K^T
//                of tile i + 2 goes before P V of tile i, so the K stream does not wait for the softmax)
//   P = exp2(S scale - m)   softmax warps: warps w and w ^ 2 share a head, positions 0-63 / 64-127
//   O += P V     M 128 x N 256 (each CTA 128 latent dims of each 256) x K 128, two MMAs into TMEM
// m is lazily raised: O and the sum are rescaled only when the tile's max passes m + 8.
// Two rings, each in its MMA's order and freed as soon as its MMAs complete: K, 8 KB boxes of each tile's own page
// (nine: 8 latent, rope) from HBM; V, 16 KB pair slots (2g + p: page 2i + p's dims 256g + 128c .. +128 in CTA c,
// the P V operand of 128 dims) from L2 once the tile's Q K^T is done.
#include <cuda.h>
#include <cstdint>

#define HEADS 96
#define CTA_HEADS (HEADS / 2)       // the heads a CTA writes
#define ROW 576
#define LAT 512
#define TILE 128
#define PAGE 64
#define M_TILE 128
#define THREADS 224
#define SOFTMAX_WARPS 4
#define LOAD_WARP 4
#define MMA_WARP 5
#define V_WARP 6
#define BOX 8192                 // 64 dims x 64 rows
#define PAIR (2 * BOX)
#ifndef NK
#define NK 9                     // K ring: 8 KB boxes, a tile's own page
#endif
#ifndef NVP
#define NVP 4                    // V ring: pair slots, a tile's V
#endif
#define RESCALE 8.0f
#define STAGE_ROW (LAT * 4 + 16)        // the epilogue's rows in the ring

#define Q_OFF 0
#define P_OFF (Q_OFF + 9 * BOX)
#define RING_OFF (P_OFF + 2 * BOX)
#define V_OFF (RING_OFF + NK * BOX)
#define XCH_OFF (V_OFF + NVP * PAIR)              // [3][2][64] f32: the halves' max (two tiles), then sum
#define BAR_OFF (XCH_OFF + 3 * 2 * 64 * 4)
#define SMEM_USED (BAR_OFF + 384)
#define SMEM_BYTES (SMEM_USED + 1024)

// barriers, 8 B each; the pair's full ones live in CTA 0 and count both CTAs' bytes
#define B_QFULL 0
#define B_FULL (B_QFULL + 1)        // [NK]
#define B_EMPTY (B_FULL + NK)       // [NK]
#define B_VFULL (B_EMPTY + NK)      // [NVP]
#define B_VEMPTY (B_VFULL + NVP)    // [NVP]
#define NSB 3                       // S buffers
#define B_QKDONE (B_VEMPTY + NVP)   // [NSB]
#define B_SFULL (B_QKDONE + NSB)    // [NSB]
#define B_SEMPTY (B_SFULL + NSB)    // [NSB]
#define B_PFULL (B_SEMPTY + NSB)
#define B_ODONE (B_PFULL + 1)
#define B_COUNT (B_ODONE + 1)

#define TM_O 0
#define TM_S 256

__device__ __forceinline__ uint64_t desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 | (uint64_t)((sbo >> 4) & 0x3FFF) << 32 |
         (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
#ifdef MLA_DEBUG
__device__ __forceinline__ void wait(uint32_t bar, uint32_t parity) {
  for (long long i = 0;; ++i) {
    uint32_t ok;
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    if (ok) return;
    if (i == (1ll << 24)) {
      printf("hang: block %d,%d,%d thread %d bar %d parity %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, (bar & 1023) / 8, parity);
      asm volatile("trap;");
    }
  }
}
#define wait_cluster wait
#else
__device__ __forceinline__ void wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(bar), "r"(parity)
               : "memory");
}
__device__ __forceinline__ void wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}"
               ::"r"(bar), "r"(parity) : "memory");
}
#endif
__device__ __forceinline__ uint32_t peer0(uint32_t addr) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(addr));
  return r;
}
__device__ __forceinline__ void expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void arrive_remote(uint32_t cluster_bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar) : "memory");
}
__device__ __forceinline__ void tma(uint32_t dst, const CUtensorMap* map, int x, int y, int z, uint32_t cluster_bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(z), "r"(cluster_bar) : "memory");
}
#ifdef MLA_PROF
__device__ unsigned long long mla_prof[32];
#define PT0 const long long _pt = clock64();
#define PT1(i) pacc[i] += clock64() - _pt;
#define PTA(i) pacc[i] = clock64() - t_entry;
#define PFLUSH if (blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x % 32) == 0) for (int _i = 0; _i < 16; ++_i) if (pacc[_i]) { if (_i < 11) atomicAdd(&mla_prof[_i + 16 * blockIdx.x], pacc[_i]); else atomicMax(&mla_prof[_i + 16 * blockIdx.x], pacc[_i]); }
#else
#define PT0
#define PT1(i) ;
#define PTA(i) ;
#define PFLUSH
#endif
__device__ __forceinline__ void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"((uint16_t)3) : "memory");
}
__device__ __forceinline__ void tld32(uint32_t taddr, float* v) {
  uint32_t* u = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
               : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]), "=r"(u[4]), "=r"(u[5]), "=r"(u[6]), "=r"(u[7]), "=r"(u[8]),
                 "=r"(u[9]), "=r"(u[10]), "=r"(u[11]), "=r"(u[12]), "=r"(u[13]), "=r"(u[14]), "=r"(u[15]), "=r"(u[16]),
                 "=r"(u[17]), "=r"(u[18]), "=r"(u[19]), "=r"(u[20]), "=r"(u[21]), "=r"(u[22]), "=r"(u[23]), "=r"(u[24]),
                 "=r"(u[25]), "=r"(u[26]), "=r"(u[27]), "=r"(u[28]), "=r"(u[29]), "=r"(u[30]), "=r"(u[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tst32(uint32_t taddr, const float* v) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(v);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
               ::"r"(taddr), "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]), "r"(u[8]),
                 "r"(u[9]), "r"(u[10]), "r"(u[11]), "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]), "r"(u[16]), "r"(u[17]),
                 "r"(u[18]), "r"(u[19]), "r"(u[20]), "r"(u[21]), "r"(u[22]), "r"(u[23]), "r"(u[24]), "r"(u[25]), "r"(u[26]),
                 "r"(u[27]), "r"(u[28]), "r"(u[29]), "r"(u[30]), "r"(u[31]));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ bool elect_one() {
  uint32_t p;
  asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\nselp.u32 %0, 1, 0, e;\n}" : "=r"(p));
  return p;
}
__device__ __forceinline__ void pair_sync(int id) { asm volatile("bar.sync %0, 64;" ::"r"(id) : "memory"); }

extern "C" __global__ void __launch_bounds__(THREADS, 1) __cluster_dims__(2, 1, 1)
kern_k3_mla_decode(const __grid_constant__ CUtensorMap q_map, const __grid_constant__ CUtensorMap kv_map,
                   const int* __restrict__ table, const int* __restrict__ seq_lens, const int* __restrict__ bsk,
                   float* __restrict__ acc_o, float* __restrict__ acc_lse, int split_max, int max_pages, float scale_log2) {
#ifdef MLA_PROF
  const long long t_entry = clock64();
#endif
  const int b = blockIdx.y, split = blockIdx.z;
  const int n = seq_lens[b], asked = bsk[b];
  if (n == 0) return;
  const int tiles = (n + TILE - 1) / TILE, per = (tiles + asked - 1) / asked;
  const int t0 = split * per, nt = min(tiles, t0 + per) - t0;
  if (nt <= 0) return;
  const int npages = (n + PAGE - 1) / PAGE;
  uint32_t c;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(c));
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  // the loaders' first 32 tiles' page ids, in flight during the setup
  const int* const rt = table + (size_t)b * max_pages;
  auto page = [&](int i) { return rt[min(i, npages - 1)]; };
  int id0 = 0, id1 = 0;
  if (warp == LOAD_WARP || warp == V_WARP) id0 = page(2 * (t0 + lane) + c), id1 = page(2 * (t0 + lane) + 1 - c);

  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  uint8_t* const gen = smem_raw + (base - raw);
  const uint32_t bars = base + BAR_OFF;
  auto bar = [&](int i) { return bars + 8 * i; };
#ifdef MLA_PROF
  unsigned long long pacc[16] = {};
#endif

  // the barriers, then the cluster's arrival (the other CTA signals ours), the loaders' wait, the TMEM
  // allocation beside it: the TMEM users meet warp 0 at a named barrier
  if (t == 0) {
    for (int i = 0; i < B_COUNT; ++i) init(bar(i), 1);
    for (int i = 0; i < NSB; ++i) init(bar(B_SEMPTY + i), 2 * SOFTMAX_WARPS);
    init(bar(B_PFULL), 2 * SOFTMAX_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncwarp();
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  const bool loader = warp == LOAD_WARP || warp == V_WARP;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(bars + 8 * B_COUNT));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  uint32_t tmem = 0;
  if (!loader) {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("bar.sync 4, %0;" ::"n"(THREADS - 64) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tmem = *reinterpret_cast<volatile uint32_t*>(gen + BAR_OFF + 8 * B_COUNT);
  }
  PTA(11)

  if (loader) {
    // lane l holds the page ids of tile i + l for the next 32 tiles; lane 0 issues
    const bool kw = warp == LOAD_WARP;
    if (kw && lane == 0) {
      if (c == 0) expect(bar(B_QFULL), 2 * 9 * BOX);
      for (int k = 0; k < 9; ++k) tma(base + Q_OFF + k * BOX, &q_map, 64 * k, CTA_HEADS * c, b, peer0(bar(B_QFULL)));
    }
    int s = 0, ph = 1;
    const uint32_t full = peer0(bar(kw ? B_FULL : B_VFULL));
    for (int i = 0; i < nt; ++i) {
      if (i % 32 == 0 && i > 0) {
        const int p = 2 * (t0 + i + lane);
        id0 = page(p + c), id1 = page(p + 1 - c);
      }
      const int own = __shfl_sync(0xffffffff, id0, i % 32), other = __shfl_sync(0xffffffff, id1, i % 32);
      if (kw) {
        // the tile's own page from HBM
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          wait(bar(B_EMPTY + s), ph);
          if (lane == 0) {
            if (c == 0) expect(bar(B_FULL + s), 2 * BOX);
            tma(base + RING_OFF + s * BOX, &kv_map, 64 * j, 0, own, full + 8 * s);
          }
          if (++s == NK) s = 0, ph ^= 1;
        }
      } else {
        // once the tile's Q K^T is done its pages are in L2: both pages' dims 256g + 128c .. +128, slot 2g + p
        wait(bar(B_QKDONE + i % NSB), (i / NSB) & 1);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          wait(bar(B_VEMPTY + s), ph);
          if (lane == 0) {
            const int pg = (v & 1) == c ? own : other, x = 256 * (v >> 1) + 128 * c;
            if (c == 0) expect(bar(B_VFULL + s), 2 * PAIR);
            tma(base + V_OFF + s * PAIR, &kv_map, x, 0, pg, full + 8 * s);
            tma(base + V_OFF + s * PAIR + BOX, &kv_map, x + 64, 0, pg, full + 8 * s);
          }
          if (++s == NVP) s = 0, ph ^= 1;
        }
      }
    }
  }
  if (warp == MMA_WARP && c == 0) {
    // f32 D, bf16 A and B, K-major A, B K-major (K) / MN-major (V); M 128 (the pair) x N 128 / 256
    const uint32_t id_qk = (1u << 4) | (1u << 7) | (1u << 10) | ((TILE >> 3) << 17) | ((M_TILE >> 4) << 24);
    const uint32_t id_pv = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((256 >> 3) << 17) | ((M_TILE >> 4) << 24);
    const bool e = elect_one();
    wait(bar(B_QFULL), 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    // descriptors: a base each, offsets added in the address field (shared addresses < 256 KB fit its 14 bits)
    const uint64_t dq = desc(base + Q_OFF, 16, 1024), dk = desc(base + RING_OFF, 16, 1024),
                   dp = desc(base + P_OFF, 16, 1024), dv = desc(base + V_OFF, BOX, 1024);
    int ks = 0, kph = 0, vs = 0, vph = 0;
    auto qk = [&](int i) {
      const int sb = i % NSB;
      { PT0 wait(bar(B_SEMPTY + sb), ((i / NSB) & 1) ^ 1); PT1(0) }
      const uint32_t d = tmem + TM_S + 64 * sb;
#pragma unroll
      for (int j = 0; j < 9; ++j) {
        { PT0 wait(bar(B_FULL + ks), kph); PT1(1) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (e) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
            mma(d, dq + ((j * BOX + q * 32) >> 4), dk + ((ks * BOX + q * 32) >> 4), id_qk, j | q);
          commit(bar(B_EMPTY + ks));
        }
        __syncwarp();
        if (++ks == NK) ks = 0, kph ^= 1;
      }
      if (e) {
        commit(bar(B_QKDONE + sb));
        commit(bar(B_SFULL + sb));
      }
      __syncwarp();
    };
    auto pv = [&](int i) {
      { PT0 wait(bar(B_PFULL), i & 1); PT1(2) }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int g = v >> 1, p = v & 1;
        { PT0 wait(bar(B_VFULL + vs), vph); PT1(3) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (e) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
            mma(tmem + TM_O + 128 * g, dp + ((p * BOX + q * 32) >> 4), dv + ((vs * PAIR + q * 2048) >> 4), id_pv,
                i | p | q);
          commit(bar(B_VEMPTY + vs));
        }
        __syncwarp();
        if (++vs == NVP) vs = 0, vph ^= 1;
      }
      if (e) commit(bar(B_ODONE));
      __syncwarp();
    };
    PT0
    for (int i = 0; i < NSB - 1 && i < nt; ++i) qk(i);
    for (int i = 0; i < nt; ++i) {
      if (i + NSB - 1 < nt) qk(i + NSB - 1);
      pv(i);
    }
    PT1(7)
  }
  if (warp < SOFTMAX_WARPS) {
    // lane quarter w: head row 32 (w & 1) + lane, positions / O dims half w >> 1
    const int row = 32 * (warp & 1) + lane, half = warp >> 1;
    const uint32_t lanes = (uint32_t)(32 * warp) << 16;
    float* const xch = reinterpret_cast<float*>(gen + XCH_OFF);
    uint8_t* const prow = gen + P_OFF + half * BOX + row * 128;
    const uint32_t sempty = peer0(bar(B_SEMPTY)), pfull = peer0(bar(B_PFULL));
    float m = -INFINITY, sum = 0.f;
    for (int i = 0; i < nt; ++i) {
      const int sb = i % NSB, xb = i & 1;
      float x[64];
      { PT0 wait(bar(B_SFULL + sb), (i / NSB) & 1); if (warp == 0) PT1(4) }
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      PT0
      tld32(tmem + lanes + TM_S + 64 * sb, x);
      tld32(tmem + lanes + TM_S + 64 * sb + 32, x + 32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) arrive_remote(sempty + 8 * sb);
      if (warp == 0) PT1(6)
      { PT0
      const int valid = n - (TILE * (t0 + i) + PAGE * half);
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < 64; ++k) {
        x[k] = k < valid ? x[k] * scale_log2 : -INFINITY;
        mx = fmaxf(mx, x[k]);
      }
      xch[(xb * 2 + half) * 64 + row] = mx;
      pair_sync(1 + (warp & 1));
      mx = fmaxf(mx, xch[(xb * 2 + (half ^ 1)) * 64 + row]);
      const bool raise = mx > m + RESCALE;
      const float scale = raise ? ex2(m - mx) : 1.f;
      if (raise) m = mx;
      float s = 0.f;
      uint32_t pk[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        const float a = ex2(x[2 * k] - m), bb = ex2(x[2 * k + 1] - m);
        s += a + bb;
        pk[k] = bf2(a, bb);
      }
      sum = sum * scale + s;
      if (warp == 0) PT1(8)
      if (i > 0) {
        { PT0 wait(bar(B_ODONE), (i - 1) & 1); if (warp == 0) PT1(5) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (__any_sync(0xffffffff, raise)) {
#ifdef MLA_PROF
          pacc[10] += 1940;
#endif
#pragma unroll 1
          for (int q = 0; q < 8; ++q) {
            float o[32];
            tld32(tmem + lanes + TM_O + 32 * q, o);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
            for (int k = 0; k < 32; ++k) o[k] *= scale;
            tst32(tmem + lanes + TM_O + 32 * q, o);
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u)
        *reinterpret_cast<uint4*>(prow + ((u ^ (row & 7)) << 4)) = make_uint4(pk[4 * u], pk[4 * u + 1], pk[4 * u + 2], pk[4 * u + 3]);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) arrive_remote(pfull);
      if (warp == 0) PT1(9) }
    }
    wait(bar(B_ODONE), (nt - 1) & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    xch[(4 + half) * 64 + row] = sum;
    pair_sync(1 + (warp & 1));
    const float total = sum + xch[(4 + (half ^ 1)) * 64 + row], inv = 1.f / total;
    const int head = CTA_HEADS * c + row;
    // TMEM column 32q: dims 256 (q / 4) + 128 half + 32 (q % 4). The rows go through the free ring (2064 B apart:
    // a warp's float4 stores 4 wavefronts), then out a row at a time.
    float* const stage = reinterpret_cast<float*>(gen + RING_OFF + row * STAGE_ROW) + 128 * half;
#pragma unroll 1
    for (int q = 0; q < 8; ++q) {
      float v[32];
      tld32(tmem + lanes + TM_O + 32 * q, v);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      float4* dst = reinterpret_cast<float4*>(stage + 256 * (q / 4) + 32 * (q % 4));
      if (row < CTA_HEADS)
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = make_float4(v[4 * k] * inv, v[4 * k + 1] * inv, v[4 * k + 2] * inv, v[4 * k + 3] * inv);
    }
    if (warp == 0) PTA(13)
    // the CTA's rows back, a warp a row at a time: 512 B a store
    asm volatile("bar.sync 3, %0;" ::"n"(SOFTMAX_WARPS * 32) : "memory");
    for (int r = warp; r < CTA_HEADS; r += SOFTMAX_WARPS) {
      const float4* src = reinterpret_cast<const float4*>(gen + RING_OFF + r * STAGE_ROW);
      float4* dst = reinterpret_cast<float4*>(acc_o + (((size_t)b * M_TILE + CTA_HEADS * c + r) * split_max + split) * LAT);
#pragma unroll
      for (int k = 0; k < LAT / 4 / 32; ++k) dst[lane + 32 * k] = src[lane + 32 * k];
    }
    if (half == 0 && row < CTA_HEADS) acc_lse[((size_t)b * M_TILE + head) * split_max + split] = m + __log2f(total);
    if (warp == 0) PTA(15)
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  PFLUSH
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}
