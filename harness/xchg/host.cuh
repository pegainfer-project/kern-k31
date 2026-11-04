// The harness's group: NRANKS processes over the lease, every peer buffer a fabric allocation
// exported through files in a shared directory, mapped by every member (and optionally bound to a
// multicast object), as kern's peer buffers are.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#define CK(x) ck_((x), #x, __LINE__)
static void ck_(CUresult r, const char* w, int line) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    fprintf(stderr, "line %d: %s: %s\n", line, w, s ? s : "?");
    exit(1);
  }
}
static void ck_(cudaError_t r, const char* w, int line) {
  if (r != cudaSuccess) {
    fprintf(stderr, "line %d: %s: %s\n", line, w, cudaGetErrorString(r));
    exit(1);
  }
}

struct Group {
  int rank, world, gpu;
  std::string dir;
  int seq = 0;
  CUdevice dev;
};

static bool exists(const std::string& p) { return access(p.c_str(), F_OK) == 0; }

static void barrier(Group& g, const char* tag) {
  const std::string base = g.dir + "/b" + std::to_string(g.seq++) + "." + tag + ".";
  std::ofstream(base + std::to_string(g.rank)) << 1;
  for (int r = 0; r < g.world; ++r)
    while (!exists(base + std::to_string(r))) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

struct Peer {
  size_t bytes = 0;
  uint8_t* local = nullptr;
  std::vector<unsigned long long> peers;  // every member's mapping, rank order
  unsigned long long* d_peers = nullptr;  // the same on the device
  uint8_t* mc = nullptr;                  // the multicast alias, if any
};

static Peer alloc_peer(Group& g, const char* name, size_t want, bool multicast = false) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = g.dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  size_t gran = 0;
  CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  CUmulticastObjectProp mprop = {};
  if (multicast) {
    mprop.numDevices = g.world;
    mprop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t mg = 0;
    mprop.size = want;
    CK(cuMulticastGetGranularity(&mg, &mprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    gran = std::max(gran, mg);
  }
  Peer p;
  p.bytes = (want + gran - 1) / gran * gran;
  mprop.size = p.bytes;
  CUmemGenericAllocationHandle h;
  CK(cuMemCreate(&h, p.bytes, &prop, 0));
  CUmemFabricHandle fh;
  CK(cuMemExportToShareableHandle(&fh, h, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  const std::string base = g.dir + "/h." + name + ".";
  {
    std::ofstream(base + std::to_string(g.rank) + ".tmp", std::ios::binary).write((const char*)&fh, sizeof fh);
    rename((base + std::to_string(g.rank) + ".tmp").c_str(), (base + std::to_string(g.rank)).c_str());
  }
  barrier(g, name);
  CUmemAccessDesc acc = {};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = g.dev;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  p.peers.resize(g.world);
  for (int q = 0; q < g.world; ++q) {
    CUmemGenericAllocationHandle hq = h;
    if (q != g.rank) {
      // the shared filesystem can show the file before its bytes: re-read until the import takes
      CUresult rc = CUDA_ERROR_INVALID_HANDLE;
      for (int tries = 0; tries < 200 && rc != CUDA_SUCCESS; ++tries) {
        CUmemFabricHandle f = {};
        std::ifstream(base + std::to_string(q), std::ios::binary).read((char*)&f, sizeof f);
        rc = cuMemImportFromShareableHandle(&hq, &f, CU_MEM_HANDLE_TYPE_FABRIC);
        if (rc != CUDA_SUCCESS) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      CK(rc);
    }
    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, p.bytes, gran, 0, 0));
    CK(cuMemMap(va, p.bytes, 0, hq, 0));
    CK(cuMemSetAccess(va, p.bytes, &acc, 1));
    p.peers[q] = va;
  }
  p.local = (uint8_t*)p.peers[g.rank];
  CK(cudaMemset(p.local, 0, p.bytes));
  CK(cudaMalloc(&p.d_peers, g.world * 8));
  CK(cudaMemcpy(p.d_peers, p.peers.data(), g.world * 8, cudaMemcpyHostToDevice));
  if (multicast) {
    CUmemGenericAllocationHandle mc;
    const std::string mf = g.dir + "/mc." + name;
    CUmemFabricHandle mfh;
    if (g.rank == 0) {
      CK(cuMulticastCreate(&mc, &mprop));
      CK(cuMemExportToShareableHandle(&mfh, mc, CU_MEM_HANDLE_TYPE_FABRIC, 0));
      std::ofstream(mf + ".tmp", std::ios::binary).write((const char*)&mfh, sizeof mfh);
      rename((mf + ".tmp").c_str(), mf.c_str());
    } else {
      while (!exists(mf)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
      std::ifstream(mf, std::ios::binary).read((char*)&mfh, sizeof mfh);
      CK(cuMemImportFromShareableHandle(&mc, &mfh, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    CK(cuMulticastAddDevice(mc, g.dev));
    barrier(g, "mcadd");
    CK(cuMulticastBindMem(mc, 0, h, 0, p.bytes, 0));
    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, p.bytes, gran, 0, 0));
    CK(cuMemMap(va, p.bytes, 0, mc, 0));
    CK(cuMemSetAccess(va, p.bytes, &acc, 1));
    p.mc = (uint8_t*)va;
    barrier(g, "mcbound");
  }
  CK(cudaDeviceSynchronize());
  barrier(g, "mapped");
  return p;
}

// us per call of `launch` (called `calls` times in one graph), the median of `reps` replays after a
// warm replay that absorbs the members' start skew.
template <class F>
static double per_call_us(cudaStream_t s, int calls, int reps, F&& launch) {
  cudaGraph_t gr;
  cudaGraphExec_t ex;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < calls; ++i) launch();
  CK(cudaStreamEndCapture(s, &gr));
  CK(cudaGraphInstantiate(&ex, gr, 0));
  std::vector<cudaEvent_t> ev(reps + 1);
  for (auto& e : ev) CK(cudaEventCreate(&e));
  CK(cudaGraphLaunch(ex, s));
  CK(cudaGraphLaunch(ex, s));
  CK(cudaEventRecord(ev[0], s));
  for (int r = 0; r < reps; ++r) {
    CK(cudaGraphLaunch(ex, s));
    CK(cudaEventRecord(ev[r + 1], s));
  }
  CK(cudaStreamSynchronize(s));
  std::vector<double> t(reps);
  for (int r = 0; r < reps; ++r) {
    float ms;
    CK(cudaEventElapsedTime(&ms, ev[r], ev[r + 1]));
    t[r] = ms * 1e3 / calls;
  }
  for (auto& e : ev) CK(cudaEventDestroy(e));
  CK(cudaGraphExecDestroy(ex));
  CK(cudaGraphDestroy(gr));
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}

// cudaLaunchKernelEx with the step's attributes: PDL (programmatic stream serialization) and dynamic smem
template <class... A>
static void launch_ex(cudaStream_t s, void (*k)(A...), dim3 grid, dim3 block, size_t smem, bool pdl,
                      typename std::type_identity<A>::type... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  CK(cudaLaunchKernelEx(&cfg, k, args...));
}
