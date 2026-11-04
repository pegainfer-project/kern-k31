"""inventory.py <capture rank dir> <graphs dir> <out json>: every kernel the decode graphs launch, the cubin
it lives in, and where that cubin comes from.

A symbol is found in a module by its bytes in the cubin's string table. The origin is read off the symbol
(namespace or naming convention of the library that emits it); an unknown origin is reported as such.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

ORIGINS = [
    # (test on the symbol, origin, license)
    (lambda s: s.startswith("nvjet_") or "cublasLt" in s, "cuBLASLt (CUDA 13 toolkit, via torch)", "NVIDIA CUDA EULA, not redistributable as extracted cubins"),
    (lambda s: s.startswith("_Z") and "ncclDevKernel" in s, "NCCL 2.x (via torch)", "BSD-3-Clause (NCCL LICENSE.txt)"),
    (lambda s: s.startswith("_ZN2at6native") or s.startswith("_ZN50_GLOBAL") or "at6native" in s, "PyTorch ATen", "BSD-3-Clause"),
    (lambda s: "_ZN6sglang" in s or s.startswith("kernel_cutlass_kernel_sglang"), "SGLang (sgl-kernel / sglang.kernels JIT)", "Apache-2.0"),
    (lambda s: "flashinfer" in s, "FlashInfer (CuTe DSL JIT)", "Apache-2.0"),
    (lambda s: "_ZN3moe3dev" in s, "FlashInfer / TRT-LLM gen MoE (routing, finalize)", "Apache-2.0"),
    (lambda s: s.startswith("bmm_") or "trtllm" in s.lower(), "TRT-LLM gen (flashinfer-cubin)", "Apache-2.0"),
    (lambda s: s.startswith("kernel_cutlass_kernel_TgvGemm"), "FlashInfer TGV GEMM (CuTe DSL JIT)", "Apache-2.0"),
    (lambda s: not s.startswith("_Z") and not s.startswith("kernel_cutlass"), "Triton JIT (SGLang Python source)", "Apache-2.0 (SGLang source), MIT (Triton)"),
]


def origin(sym):
    for test, o, lic in ORIGINS:
        if test(sym):
            return o, lic
    return "unknown", "unknown"


def main():
    rank, graphs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    counts = collections.defaultdict(collections.Counter)
    first = {}
    for b in (1, 8, 48):
        for line in open(graphs / f"graph-bs{b}.rank0.jsonl"):
            r = json.loads(line)
            counts[r["symbol"]][b] += 1
            first.setdefault(r["symbol"], r)
    cubins = {p: p.read_bytes() for p in sorted(rank.glob("module_*.cubin"))}
    rows = []
    for sym in sorted(counts, key=lambda s: -counts[s][48]):
        key = sym.encode() + b"\0"
        mods = [p for p, data in cubins.items() if key in data]
        o, lic = origin(sym)
        rows.append({
            "symbol": sym,
            "launches": {f"bs{b}": counts[sym][b] for b in (1, 8, 48)},
            "modules": [{"file": p.name, "bytes": len(cubins[p]), "sha256": hashlib.sha256(cubins[p]).hexdigest()} for p in mods],
            "origin": o,
            "license": lic,
            "attributes": first[sym]["attributes"],
        })
    out.write_text(json.dumps(rows, indent=1))
    by = collections.Counter()
    for r in rows:
        by[r["origin"]] += r["launches"]["bs48"]
    for o, n in by.most_common():
        print(f"{n:5d} launches at bs48  {o}")
    print(f"{len(rows)} kernels, {sum(1 for r in rows if not r['modules'])} without a module found")


if __name__ == "__main__":
    main()
