"""K3 decode on vLLM's memory.

Rewrites a `gen_stage.py --tp N --peer-ar` decode manifest into one a vLLM
model runner drives (kern's `python/kern_vllm`): `k3_hosted.py`'s program
boundary, and every state a host state, one per layer, laid out the way
vLLM's hybrid allocator lays it out.

vLLM gives every layer of a hybrid model one page per block, the same size
for the attention and the recurrent groups. An MLA layer's token is 576 bf16
(the 512-wide latent, then the 64-wide rope part), 1152 B; vLLM rounds a
hybrid MLA model's block to a multiple of 128 tokens big enough to hold one
KDA line (814,080 B: the recurrent state, then the conv window), so a block
is 768 tokens and a page 884,736 B. An MLA layer sees its page as 12 kernel
blocks of [64 tokens][1 head][576] (vLLM's LBNHC), the 64-token pages the
kernels already read; a KDA layer sees one line at the start of its page.
Only the line stride changes, and only `kda_core` bakes it in.

Each DCP member holds its own positions of an MLA layer (vLLM's slot is -1
elsewhere) and every member a whole KDA line for its own heads, as in the
base: vLLM's `dcp_local_seq_lens` are the base's `seq_lens`.

    python3 scripts/k3_vllm.py --input d.json --output d-vllm.json
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from k3_hosted import hosted  # noqa: E402

KV_LORA, ROPE = 512, 64
ROW = KV_LORA + ROPE
TOKEN_BYTES = ROW * 2
KERNEL_BLOCK = 64
BLOCK_ALIGN = 128


def mla_host(token: int) -> dict:
    """An MLA layer's latent as 64-token kernel blocks: [blocks][1 head][64][576], bf16, or the e4m3
    bytes of SGLang's fp8 cache (`token` 576)."""
    return {"dtype": {TOKEN_BYTES: "bf16", ROW: "u8"}[token], "shape": [0, 1, KERNEL_BLOCK, ROW],
            "strides": [KERNEL_BLOCK * ROW, ROW, ROW, 1]}


MLA_HOST = mla_host(TOKEN_BYTES)


def page_bytes(line: int) -> int:
    """vLLM's page for a hybrid MLA model: whole 128-token blocks of MLA tokens covering a line."""
    return BLOCK_ALIGN * -(-line // (BLOCK_ALIGN * TOKEN_BYTES)) * TOKEN_BYTES


def layer(state: str) -> int:
    return int(re.fullmatch(r"(?:kv|kda)\.(\d+)", state)[1])


def renamed(state: str) -> str:
    return f"{'mla' if state.startswith('kv.') else 'kda'}.l{layer(state)}"


def host_states(m: dict, page: int) -> dict:
    def host(name, s):
        if name.startswith("kv."):
            assert s["bytes_per_token"] == TOKEN_BYTES, f"`{name}` is {s['bytes_per_token']} B a token"
            return {"host": MLA_HOST}
        return {"host": {"dtype": "u8", "shape": [0, s["bytes_per_seq"]], "strides": [page, 1]}}

    return {renamed(n): host(n, s) for n, s in m["states"].items()}


def rebind(call: dict) -> dict:
    return {**call, "args": [{**a, "state": renamed(a["state"])} if "state" in a else a for a in call["args"]]}


def check_mla(name: str, op: dict, token: int = TOKEN_BYTES) -> None:
    """The MLA kernels already read `mla_host(token)`'s kernel blocks: as pages, or as one run of rows."""
    args = [a for l in op["impl"]["launches"] for a in l.get("args", [])]
    if name == "mla_prep":
        assert {"i64": KERNEL_BLOCK * ROW} in args, "`mla_prep` page stride is not `MLA_HOST`'s"
    if name == "mla_attn":
        maps = [f["tensormap"] for a in args if "pack" in a for f in a["pack"]["fields"] if "tensormap" in f]
        pages = [t for t in maps if t["dims"][-1] == 0]
        assert pages and all(t["strides"] in ([token, KERNEL_BLOCK * token], [token]) for t in pages), \
            "`mla_attn` page tensormaps are not `MLA_HOST`'s kernel blocks"


def restride(name: str, op: dict, line: int, page: int) -> dict:
    """`kda_core`'s line stride onto vLLM's page; the MLA kernels' page strides are vLLM's already."""
    check_mla(name, op)
    launches = op["impl"]["launches"]
    if name == "kda_core":
        args = [a for l in launches for a in l.get("args", [])]
        assert args.count({"i64": line}) == 1, f"`kda_core` bakes no line stride {line}"
        launches = [{**l, "args": [{"i64": page} if a == {"i64": line} else a for a in l["args"]]} for l in launches]
    return {**op, "impl": {**op["impl"], "launches": launches}}


def on_vllm(m: dict) -> dict:
    lines = {s["bytes_per_seq"] for s in m["states"].values() if "bytes_per_seq" in s}
    assert len(lines) == 1, f"KDA lines of {sorted(lines)} B"
    (line,) = lines
    page = page_bytes(line)
    mla = next(renamed(n) for n in m["states"] if n.startswith("kv."))
    kda = next(renamed(n) for n in m["states"] if n.startswith("kda."))
    b = m["buffers"]
    ids = {"slot_mapping": {"index_into": mla},
           "block_table": {"index_into": mla, "stride": KERNEL_BLOCK},
           "kda.line_index": {"index_into": kda, "stride": page}}
    return {**m, "model": f"{m['model'].removesuffix('-hosted')}-vllm",
            "states": host_states(m, page),
            "ops": {k: restride(k, v, line, page) for k, v in m["ops"].items()},
            "programs": {k: {**p, "calls": [rebind(c) for c in p["calls"]]} for k, p in m["programs"].items()},
            "buffers": b | {k: {**b[k], "domain": d} for k, d in ids.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(on_vllm(hosted(json.loads(args.input.read_text())))) + "\n")


if __name__ == "__main__":
    main()
