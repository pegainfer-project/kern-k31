extern "C" __global__ void flush_read(const float4* p, long long n, float* sink) {
  float4 a = {0, 0, 0, 0};
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const float4 v = __ldcg(p + i);
    a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  }
  if (a.x + a.y + a.z + a.w == 1234.5f) *sink = a.x;
}
