"""assemble.py <capture rank dir> <inventory.json> <graphs dir> <repo decode dir>: the decode capture as the
kern-k3-pp-prefill repo keeps it (prebuilt/, kernels.toml, graphs/*.xz).

A module is copied only when its license allows redistribution and it is small; the others are listed with
their upstream and the reason they are not here.
"""
import hashlib
import json
import lzma
import re
import subprocess
import sys
from pathlib import Path

SGLANG = "sglang@295a53c90ea4adb2588084dc0e3004cd3145e414"
FI = "flashinfer-python 0.7.0.post1"
SOURCES = [
    ("kda_decode_fusion_many_heads_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh"),
    ("attn_res_fused_tma_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh"),
    ("route_quant_fused_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh"),
    ("situ_and_mul_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh"),
    ("tiny_n_gemm_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh"),
    ("tiny_k_gemm_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh"),
    ("add3_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/elementwise/add3.cuh"),
    ("fixup_zero_kv_rows_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh"),
    ("mla_output_gate_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh"),
    ("set_mla_kv_concat_q_fp8_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh"),
    ("fused_a_gemm_kernel", f"{SGLANG}:python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh"),
    ("_dcp_pack_a2a_send_kernel", f"{SGLANG}:python/sglang/kernels/ops/attention/dcp_kernels.py"),
    ("_dcp_lse_combine_kernel", f"{SGLANG}:python/sglang/kernels/ops/attention/dcp_kernels.py"),
    ("_vocab_parallel_embedding_kernel", f"{SGLANG}:python/sglang/kernels/ops/embeddings/vocab_parallel_embedding.py"),
    ("RMSNormKernel", f"{FI}:flashinfer/norm/kernels/rmsnorm.py"),
    ("mla_decode_fp8", f"{FI}:flashinfer/cute_dsl/attention/mla_decode_fp8.py"),
    ("TgvGemmCuteExtKernel", f"{FI}:flashinfer/gemm/kernels/tgv_gemm_cute_ext.py"),
    ("helixAllToAll", f"{FI}:flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/helixAllToAll.cu (built by flashinfer/jit/comm.py)"),
    ("routingIndices", f"{FI}:flashinfer/data/csrc/fused_moe/trtllm_backend/ (flashinfer-jit-cache 0.7.0.post1 trtllm_gen_routing)"),
    ("finalizeKernel", f"{FI}:flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu (flashinfer-jit-cache 0.7.0.post1 fused_moe_trtllm_sm100)"),
    ("bmm_", "flashinfer-cubin 0.7.0.post1 (TRT-LLM gen batched GEMM bundle)"),
    ("ncclDevKernel", "NCCL 2.29.7 as linked by torch 2.13.0+cu130"),
    ("nvjet_", "cuBLASLt 13.1.1.3 (nvidia-cublas wheel)"),
    ("cublasLt", "cuBLASLt 13.1.1.3 (nvidia-cublas wheel)"),
    ("at6native", "PyTorch 2.13.0+cu130"),
]
LICENSES = {
    "Apache-2.0": "licenses/LICENSE.sglang (SGLang), licenses/LICENSE.flashinfer (FlashInfer, incl. the TRT-LLM sources it ships)",
}
INCLUDED = {
    "SGLang (sgl-kernel / sglang.kernels JIT)": "licenses/LICENSE.sglang",
    "FlashInfer (CuTe DSL JIT)": "licenses/LICENSE.flashinfer",
    "FlashInfer TGV GEMM (CuTe DSL JIT)": "licenses/LICENSE.flashinfer",
    "TRT-LLM gen (flashinfer-cubin)": "../../prebuilt/LICENSE.trtllm-gen.txt",
    "TRT-LLM Helix all-to-all (via FlashInfer comm)": "licenses/LICENSE.flashinfer",
    "Triton JIT (SGLang Python source)": "licenses/LICENSE.sglang, licenses/LICENSE.triton",
}
EXCLUDED = {
    "cuBLASLt (CUDA 13 toolkit, via torch)": "NVIDIA proprietary library; extracted cubins are not redistributed",
    "NCCL 2.x (via torch)": "one 136 MB module (BSD-3-Clause); install NCCL instead",
    "PyTorch ATen": "fat modules of 3-8 MB each (BSD-3-Clause) holding generic elementwise kernels",
    "FlashInfer / TRT-LLM gen MoE (routing, finalize)": "fat JIT-cache modules of 2.6-36 MB (Apache-2.0); the routing kernels build from ../../source/flashinfer-moe-routing",
}


def demangle(sym):
    if not sym.startswith("_Z"):
        return sym
    return subprocess.run(["c++filt", sym], capture_output=True, text=True).stdout.strip()


def label(sym):
    d = demangle(sym)
    if d.startswith("kernel_cutlass_"):
        m = re.search(r"(RMSNormKernel|split_kv_kernel|reduction_kernel|TgvGemmCuteExtKernel)", d)
        base = m.group(1) if m else d[:40]
    elif d.startswith(("nvjet_", "bmm_")) or not sym.startswith("_Z"):
        base = d
    else:
        head = re.split(r"[<(]", d.replace("(anonymous namespace)::", ""), 1)[0]
        base = head.split("::")[-1].strip()
    return re.sub(r"[^A-Za-z0-9_]+", "_", base)[:80]


def source(sym):
    return next((s for k, s in SOURCES if k in sym), "unknown")


def main():
    rank, inv, graphs, out = (Path(a) for a in sys.argv[1:5])
    rows = json.loads(inv.read_text())
    (out / "prebuilt").mkdir(parents=True, exist_ok=True)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    modules = {}
    for r in rows:
        for m in r["modules"]:
            e = modules.setdefault(m["sha256"], {"file": m["file"], "bytes": m["bytes"], "symbols": [], "origin": r["origin"]})
            e["symbols"].append(r)
    names = {}
    toml = ["# Every module SGLang's TP8 / DCP8 / EP8 decode graphs launch (bs 1, 8, 48), as captured; see README.md.\n"]
    for sha, e in sorted(modules.items(), key=lambda kv: (kv[1]["origin"], label(kv[1]["symbols"][0]["symbol"]))):
        first = e["symbols"][0]["symbol"]
        name = label(first)
        if name in names:
            name = f"{name}_{sha[:8]}"
        names[name] = sha
        toml.append(f'[kernels."{name}"]')
        toml.append(f'sha256 = "{sha}"')
        toml.append(f"bytes = {e['bytes']}")
        toml.append(f'origin = "{e["origin"]}"')
        toml.append(f'upstream = "{source(first)}"')
        if e["origin"] in INCLUDED:
            data = (rank / e["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == sha
            (out / "prebuilt" / f"{name}.cubin").write_bytes(data)
            toml.append(f'prebuilt = "prebuilt/{name}.cubin"')
            toml.append(f'license = "{INCLUDED[e["origin"]]}"')
        else:
            toml.append(f'not_included = "{EXCLUDED[e["origin"]]}"')
        toml.append("symbols = [")
        for r in e["symbols"]:
            n = r["launches"]
            toml.append(f'  {{ name = "{r["symbol"]}", launches = {{ bs1 = {n["bs1"]}, bs8 = {n["bs8"]}, bs48 = {n["bs48"]} }} }},')
        toml.append("]\n")
    (out / "kernels.toml").write_text("\n".join(toml))
    for f in sorted(graphs.glob("graph-bs*.rank*.jsonl")):
        (out / "graphs" / (f.name + ".xz")).write_bytes(lzma.compress(f.read_bytes(), preset=9))
    shutil_json = graphs / "graphs.json"
    (out / "graphs" / "graphs.json").write_text(shutil_json.read_text())
    inc = sum(1 for e in modules.values() if e["origin"] in INCLUDED)
    print(f"{len(modules)} modules, {inc} included, {len(modules) - inc} listed only")


if __name__ == "__main__":
    main()
