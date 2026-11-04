#pragma once
// The packed prefill's FMHA tables (kern_k3_fmha_lens_varlen in k3_prefill.cu, and block 0 of
// kern_k3g_embed_rms in k3_prefill_glue.cu where a lone chunk's stage 0 makes them in its first launch).
//
// A call's rows attend their own rows through the causal FMHA and their sequences' cached context
// through its dense sibling, in rounds: the sequences' cached keys back to back (sequence j's at
// [C_j, C_j + P_j), C the prefix sums) cut into pieces of `window` rows, round w the keys
// [w window, (w + 1) window). The pieces are merged by the kernels' softmax stats (k3_mla_glue.cu).
//
// fmha_plan_set writes set k of the tables: k = 0 the own pass, k = 1 + w round w. A set is four rows
// of `ns` (>= n + 1) words: seq_lens_kv, cum_seq_lens_q and cum_seq_lens_kv over the call's n
// sequences, then aux: in set 0 the cached-key prefix sums C_0 .. C_n, elsewhere unused. A call of at
// most short_max rows attends in the latent space instead: its round sets show every sequence zero
// keys. The kernel aligns its causal mask to cum_kv's spans; the tile past a sequence's end reads the
// next one's real rows, masked, and past the last the gather's zeros.
#define FMHA_SET_ROWS 4
__device__ __forceinline__ void fmha_plan_set(const int* __restrict__ seq_lens, const long long* __restrict__ cu_q,
                                              int* __restrict__ lens, int n, int ns, int k, int window, int short_max) {
  int* const kl = lens + (long long)k * FMHA_SET_ROWS * ns;
  int* const cq = kl + ns;
  int* const ck = cq + ns;
  int* const aux = ck + ns;
  const int T = (int)cu_q[n];
  const bool expand = T > short_max;
  const long long lo = (long long)(k - 1) * window, hi = lo + window;
  long long c = 0;
  for (int j = 0; j < n; ++j) {
    const int lq = (int)(cu_q[j + 1] - cu_q[j]), p = seq_lens[j] - lq;
    cq[j] = (int)cu_q[j];
    if (k == 0) {
      kl[j] = lq, ck[j] = (int)cu_q[j], aux[j] = (int)c;
    } else {
      const long long a = max(c, lo), b = min(c + p, hi);
      kl[j] = expand && b > a ? (int)(b - a) : 0;
      ck[j] = (int)min(max(c - lo, 0ll), (long long)window);
    }
    c += p;
  }
  cq[n] = T;
  if (k == 0) ck[n] = T, aux[n] = (int)c;
  else ck[n] = (int)min(max(c - lo, 0ll), (long long)window);
}
