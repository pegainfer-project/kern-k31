// xbench --rank R --world W --gpu G --dir D TEST...: the exchanges of the D step on a real group,
// us per call in a CUDA graph; every member prints "RES <test> <rows> <width> <us>".
#include <cuda_bf16.h>

#include "host.cuh"

typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

extern "C" __global__ void kern_peer_allreduce_bf16(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* state,
                                                    int* err, int rank, long long n, long long at2, long long n2,
                                                    long long stage_bytes, long long timeout_ns);
extern "C" __global__ void kern_peer_lamport_init_bf16(uint8_t* lamport, long long bytes);
extern "C" __global__ void kern_k3_ar_attnres_rms(const uint4* x, uint8_t* lamport, const u64* peers, int* state, int* err,
                                                  int rank, const bf16* prefix, const bf16* blocks, const float* sw,
                                                  const bf16* gamma, bf16* prefix2, bf16* normed, int nb, int snapshot,
                                                  int B, long long stage_bytes, long long timeout_ns, const uint8_t* pf,
                                                  long long pf_bytes, const uint8_t* dm, long long dm_bytes);
extern "C" __global__ void kern_k3_ar_finalize_rms(const bf16* fc2, const int* exp2perm, const float* wts,
                                                   const uint4* shared, uint8_t* lamport, const u64* peers, int* state,
                                                   int* err, int rank, const bf16* gamma, bf16* latent_norm,
                                                   uint4* shared_sum, int B, long long stage_bytes, long long timeout_ns,
                                                   const uint8_t* pf, long long pf_bytes, const uint8_t* pf2, long long pf2_bytes);

extern "C" __global__ void kern_xar(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* state, int* err,
                                    int rank, int rows, int rv, long long stage_bytes, long long timeout_ns, int two_from,
                                    u64* st, int* calls);
extern "C" __global__ void kern_xar_pdl(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* state,
                                        int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns,
                                        int two_from, u64* st, int* calls);
extern "C" __global__ void kern_bump(int* calls);

extern "C" __global__ void kern_main_ar_attnres_rms(const uint4* x, uint8_t* lamport, const u64* peers, int* state, int* err,
                                                  int rank, const bf16* prefix, const bf16* blocks, const float* sw,
                                                  const bf16* gamma, bf16* prefix2, bf16* normed, int nb, int snapshot,
                                                  int B, long long stage_bytes, long long timeout_ns, const uint8_t* pf,
                                                  long long pf_bytes, const uint8_t* dm, long long dm_bytes);
extern "C" __global__ void kern_main_ar_finalize_rms(const bf16* fc2, const int* exp2perm, const float* wts,
                                                   const uint4* shared, uint8_t* lamport, const u64* peers, int* state,
                                                   int* err, int rank, const bf16* gamma, bf16* latent_norm,
                                                   uint4* shared_sum, int B, long long stage_bytes, long long timeout_ns,
                                                   const uint8_t* pf, long long pf_bytes, const uint8_t* pf2, long long pf2_bytes);
extern "C" __global__ void kern_k3_dcp_exchange(const float* acc_o, const float* acc_lse, const int* seq_lens, const int* bsk,
                                                int split_max, const bf16* w_kv_b, const bf16* mla_gate, bf16* gated,
                                                uint8_t* lamport, const u64* peers, int* state, int* err, int rank, int R,
                                                long long stage_bytes, long long timeout_ns);
extern "C" __global__ void kern_maindcp_exchange(const float* acc_o, const float* acc_lse, const int* seq_lens, const int* bsk,
                                                int split_max, const bf16* w_kv_b, const bf16* mla_gate, bf16* gated,
                                                uint8_t* lamport, const u64* peers, int* state, int* err, int rank, int R,
                                                long long stage_bytes, long long timeout_ns);

extern "C" __global__ void kern_xar_mc(const uint4* x, uint4* y, uint8_t* lamport, uint8_t* mc, const u64* peers, int* recs,
                                       int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns,
                                       int two_from);
extern "C" __global__ void kern_k3_head_argmax(const float* part, float* logits, const u64* logits_peers, long long* out,
                                               int* done, u64* pkey, uint8_t* keys, const u64* keys_peers, int* state,
                                               int* err, int rank, int VS, long long timeout_ns, int B);
extern "C" int fi_allreduce_bf16(void** workspace, int rank, int tokens, int hidden, void* in, void* out, bool oneshot,
                                 bool pdl, cudaStream_t s);
extern "C" __global__ void kern_xar_local(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* recs,
                                          int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns);
extern "C" __global__ void kern_xar_local1024(const uint4* x, uint4* y, uint8_t* lamport, const u64* peers, int* recs,
                                          int* err, int rank, int rows, int rv, long long stage_bytes, long long timeout_ns);
#include "k3_xchg.cuh"
#include "tests.cuh"

int main(int argc, char** argv) {
  Group g;
  std::vector<std::string> tests;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rank") g.rank = atoi(argv[++i]);
    else if (a == "--world") g.world = atoi(argv[++i]);
    else if (a == "--gpu") g.gpu = atoi(argv[++i]);
    else if (a == "--dir") g.dir = argv[++i];
    else tests.push_back(a);
  }
  CK(cudaSetDevice(g.gpu));
  CK(cudaFree(0));
  CK(cuDeviceGet(&g.dev, g.gpu));
  Bench b(g);
  for (const auto& t : tests) b.run(t);
  barrier(g, "end");
  return 0;
}
