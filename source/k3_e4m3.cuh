// e4m3 casts with cuda_fp8's no-saturation result (aten's cast: overflow is NaN) at hardware speed.
// cuda_fp8 does __NV_NOSAT in software, ~40 integer instructions a pair; this is cvt.rn.satfinite with the
// overflow put back: past 464 (448 and half its ulp; 464 itself rounds to 448 by ties-to-even) the result is
// NaN with x's sign, every NaN gives 0x7f either way. Equal to __nv_cvt_float_to_fp8(x, __NV_NOSAT,
// __NV_E4M3) for all 2^32 floats (checked exhaustively on GB300).
#pragma once

__device__ __forceinline__ unsigned e4m3_fix(unsigned byte, float x) {
  return fabsf(x) > 464.f ? (__float_as_uint(x) >> 24 & 0x80u) | 0x7fu : byte;
}

// lo in the low byte, as __nv_cvt_float2_to_fp8x2
__device__ __forceinline__ unsigned short e4m3x2_nosat(float lo, float hi) {
  unsigned short d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return (unsigned short)(e4m3_fix(d & 0xffu, lo) | e4m3_fix(d >> 8, hi) << 8);
}
