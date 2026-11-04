#!/usr/bin/env python3
"""Replay kern's TRT-LLM gen batched-GEMM port (kern tools/kernels/abi/trtllm_bmm.py)
against SGLang's K3 decode graphs (decode/sglang-tp8-dcp8-ep8/graphs): for every
`bmm_*` launch, build the manifest op for the index variant of the same name at
that graph's batch (B rows, 112 of 896 experts on the rank, top-16) and compare
the parameter block field by field.

Layout facts must match exactly: grid, block, shared memory, the parameter
size, every descriptor's dtype / strides / box / swizzle / L2 promotion, every
scalar, and no set byte outside the port's fields (the routing arrays inside
`KernelParams` are uninitialised host memory and are skipped). Extents that
follow from how big the caller made its buffers (descriptor dims) are listed
apart: they differ when SGLang's workspace is sized differently from kern's,
which is not an ABI difference. Pointers are checked for consistency: every
field naming one interface param holds one address, and FC2 reads what FC1
of the same layer wrote, over the same routing tables.

    KERN=<kern checkout> KERN_INDEX_DIR=<kern master>/kernel_index \
      python3 scripts/check_decode_bmm.py [graphs dir]
"""
import functools
import json
import lzma
import os
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(os.environ["KERN"]) / "tools"))
from kernels import index  # noqa: E402
from kernels.abi import trtllm_bmm  # noqa: E402

H, I, TOPK, LOCAL = 3584, 3072, 16, 112
FAMILIES = ["trtllm_bmm_mxe4m3_mxe2m1_mxe4m3", "trtllm_bmm_bf16_mxe2m1_mxe4m3"]
TMA_DTYPE = {"u8": 0, "u16": 1, "u32": 2, "i32": 3, "u64": 4, "i64": 5, "f16": 6, "f32": 7, "bf16": 9, "tf32": 11,
             "u4packed": 13, "u4": 14}
UNINITIALISED = [(0, 768), (1008, 17392), (17448, 17472)]
LAYOUT_KEYS = ("dtype_enum", "strides", "box", "swizzle", "l2_promotion")


@functools.cache
def variant(name):
    for fam in FAMILIES:
        if any(v["name"] == name for v in index.load(fam).get("variant", [])):
            return index.variant_by_name(fam, name)
    raise KeyError(f"{name} is in no kernel index family")


def addresses(rec, launch, names):
    """{interface param: set of addresses the launch's fields give it}."""
    p = rec["params"][0]
    raw = bytes.fromhex(p["data"])
    maps = {t["at"]: t for t in p.get("tensormaps", [])}
    seen = {}
    for f in launch["args"][0]["pack"]["fields"]:
        if "tensormap" in f and f["at"] in maps:
            seen.setdefault(names[f["tensormap"]["param"]], set()).add(int(maps[f["at"]]["address"], 16))
        elif "param" in f:
            seen.setdefault(names[f["param"]], set()).add(struct.unpack_from("<Q", raw, f["at"])[0])
    return seen


def check(rec, B):
    """(variant name, layout errors, extent differences, {param: address}) of one launch."""
    v = variant(rec["symbol"])
    gated = bool(v.tags["fused_act"])
    m, k = (2 * I, H) if gated else (H, I)
    ctas = trtllm_bmm.max_ctas(B, TOPK, LOCAL, v.tags["tile"][1])
    launch = trtllm_bmm.op(v, m, k, LOCAL, B, B, ctas, ctas)["impl"]["launches"][0]
    names = trtllm_bmm.params(v)
    errs, extents = [], []
    fact = lambda what, got, want: errs.append(f"{what}: got {got}, want {want}") if got != want else None
    fact("grid", rec["grid"], launch["grid"])
    fact("block", rec["block"], launch["block"])
    fact("shared_mem", rec["dynamic_shared_mem_bytes"], launch["shared_mem"])
    p = rec["params"][0]
    fact("param size", p["size"], trtllm_bmm.SIZE)
    raw = bytes.fromhex(p["data"])
    maps = {t["at"]: t for t in p.get("tensormaps", [])}
    covered = bytearray(len(raw))
    for f in launch["args"][0]["pack"]["fields"]:
        at = f["at"]
        if "tensormap" in f:
            t, got = f["tensormap"], maps.get(at)
            covered[at:at + 128] = b"\1" * 128
            if got is None:
                errs.append(f"tensormap at {at} ({names[t['param']]}): not in the capture")
                continue
            want = {"dtype_enum": TMA_DTYPE[t["dtype"]], "strides": t["strides"], "box": t["box"],
                    "swizzle": t["swizzle"], "l2_promotion": t["l2_promotion"]}
            for key in LAYOUT_KEYS:
                fact(f"tensormap at {at} ({names[t['param']]}) {key}", got[key], want[key])
            if got["dims"] != t["dims"]:
                extents.append(f"tensormap at {at} ({names[t['param']]}) dims: SGLang {got['dims']}, kern {t['dims']}")
        elif "param" in f:
            covered[at:at + 8] = b"\1" * 8
            if struct.unpack_from("<Q", raw, at)[0] == 0:
                errs.append(f"pointer at {at} ({names[f['param']]}): null")
        elif "i64" in f:
            covered[at:at + 8] = b"\1" * 8
            fact(f"i64 at {at}", struct.unpack_from("<q", raw, at)[0], f["i64"])
        elif "i32" in f:
            covered[at:at + 4] = b"\1" * 4
            fact(f"i32 at {at}", struct.unpack_from("<i", raw, at)[0], f["i32"])
        elif "var" in f:
            covered[at:at + 4] = b"\1" * 4
            fact(f"tokens at {at}", struct.unpack_from("<i", raw, at)[0], B)
        else:
            errs.append(f"field at {at}: unexpected source {f}")
    for at in maps:
        if not covered[at]:
            errs.append(f"tensormap at {at}: SGLang's launcher built one, the port does not")
    stray = [i for i, (b, c) in enumerate(zip(raw, covered))
             if b and not c and not any(lo <= i < hi for lo, hi in UNINITIALISED)]
    if stray:
        errs.append(f"{len(stray)} set bytes outside the fields, first at {stray[0]} ({raw[stray[0]:stray[0] + 8].hex()})")
    addr = {}
    for name, s in addresses(rec, launch, names).items():
        if len(s) != 1:
            errs.append(f"param {name}: fields disagree on its address {sorted(hex(a) for a in s)}")
        addr[name] = min(s)
    return v.name, gated, errs, extents, addr


def pairs(fc1, fc2):
    """FC2 of a layer reads FC1's output and scales over the same routing tables."""
    want = {"b": "c", "sf_b": "sf_c", "num_non_exiting": "num_non_exiting", "total_padded": "total_padded",
            "cta_batch": "cta_batch", "cta_limit": "cta_limit"}
    return [f"FC2 {n} {hex(fc2[n])} is not FC1 {m} {hex(fc1[m])}" for n, m in want.items() if fc2[n] != fc1[m]]


def main():
    graphs = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "decode/sglang-tp8-dcp8-ep8/graphs"
    bad, total, extents_seen, used = 0, 0, {}, {}
    for path in sorted(graphs.glob("graph-bs*.rank*.jsonl.xz")):
        B = int(path.name.split("bs")[1].split(".")[0])
        recs = [json.loads(l) for l in lzma.open(path, "rt") if '"symbol": "bmm_' in l]
        fc1 = None
        for rec in recs:
            name, gated, errs, extents, addr = check(rec, B)
            used.setdefault(B, set()).add(name)
            for e in extents:
                extents_seen.setdefault((B, e), 0)
                extents_seen[(B, e)] += 1
            if gated:
                fc1 = addr
            else:
                errs += pairs(fc1, addr) if fc1 else ["FC2 without an FC1 before it"]
                fc1 = None
            total += 1
            if errs:
                bad += 1
                print(f"FAIL {path.name} {name}")
                for e in errs:
                    print(f"     {e}")
    for B in sorted(used):
        for name in sorted(used[B]):
            v = variant(name)
            print(f"bs{B}: {name}\n      index sha256 {v.sha256}")
    for (B, e), n in sorted(extents_seen.items()):
        print(f"extent bs{B} ({n} launches): {e}")
    print(f"{total} launches over {len(list(graphs.glob('graph-bs*.rank*.jsonl.xz')))} graphs, {bad} failing")
    sys.exit(1 if bad or not total else 0)


if __name__ == "__main__":
    main()
