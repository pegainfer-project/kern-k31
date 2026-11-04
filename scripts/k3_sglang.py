"""K3 decode on SGLang's memory.

Rewrites a `gen_stage.py --tp N --peer-ar` decode manifest into one SGLang's
model runner drives (kern's `python/kern_sglang`): `k3_hosted.py`'s program
boundary, and every state a host state over the tensors SGLang allocates for
a hybrid MLA + KDA model, one per layer and kind:

- `mla.l<L>`: an MLA layer's latent, SGLang's `[rows, 1, 576]` bf16 (or the
  uint8 its fp8 cache stores, a manifest made with `--kv fp8`) read as
  64-row pages, the view `k3_vllm.py` binds; no MLA kernel changes.
- `ssm.l<L>`: a KDA layer's recurrent state, SGLang's `[slots, heads,
  128 v, 128 k]` f32: the base line's rec, byte for byte, at a stride of
  its own size.
- `conv.l<L>`: its conv window, SGLang's `[slots, 3 taps, q | k | v]` bf16.
  The base keeps the window after rec in the line as `[q, k, v][tap]`, so
  `kda_core` becomes its CONV_TAP_MAJOR build, which takes the window as a
  state of its own; LINE_GUARD skips the padded rows, whose slot is -1.

SGLang gives a request one slot for both; each state gets its own table
(`kda.line_index`, the base's, and `kda.conv_index`) of the same ids.
Positions stay sharded over the DCP group as in the
base (SGLang's rule is the base's: position p on member p % w at p / w), and
heads over the TP group; the plugin hands each rank its own slots, lengths
and pages.

    python3 scripts/k3_sglang.py --input d.json --output d-sglang.json
"""
import argparse
import json
import pathlib
import re
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from k3_hosted import hosted  # noqa: E402
from k3_vllm import KERNEL_BLOCK, ROW, TOKEN_BYTES, check_mla, mla_host  # noqa: E402

PINNED = tomllib.loads((pathlib.Path(__file__).resolve().parents[1] / "kernels.toml").read_text())["kernels"]
KD, TAPS, STREAMS = 128, 3, 3
REC_HEAD = KD * KD * 4
CONV_HEAD = TAPS * STREAMS * KD * 2


def heads(line: int) -> int:
    """This member's KDA heads, from the base's line: rec, then the conv window."""
    h, rest = divmod(line, REC_HEAD + CONV_HEAD)
    assert rest == 0, f"a {line} B KDA line is not whole heads of rec and conv"
    return h


def host_states(m: dict, h: int) -> dict:
    inner = h * KD
    ssm = {"dtype": "f32", "shape": [0, h, KD, KD], "strides": [h * KD * KD, KD * KD, KD, 1]}
    conv = {"dtype": "bf16", "shape": [0, TAPS, STREAMS * inner], "strides": [TAPS * STREAMS * inner, STREAMS * inner, 1]}

    def hosts(name, s):
        l = layer(name)
        if name.startswith("kv."):
            assert s["bytes_per_token"] in (TOKEN_BYTES, ROW), f"`{name}` is {s['bytes_per_token']} B a token"
            return {f"mla.l{l}": {"host": mla_host(s["bytes_per_token"])}}
        return {f"ssm.l{l}": {"host": ssm}, f"conv.l{l}": {"host": conv}}

    return {k: v for n, s in m["states"].items() for k, v in hosts(n, s).items()}


def layer(state: str) -> int:
    return int(re.fullmatch(r"(?:kv|kda)\.(\d+)", state)[1])


def rebind(call: dict) -> dict:
    """States renamed; `kda_core` takes the conv window after rec and its table after rec's."""
    def arg(a):
        if a.get("buf") == "kda.line_index":
            return [a, {**a, "buf": "kda.conv_index"}]
        if "state" not in a:
            return [a]
        l = layer(a["state"])
        if a["state"].startswith("kv."):
            return [{**a, "state": f"mla.l{l}"}]
        assert call["op"] == "kda_core", f"`{call['label']}` ({call['op']}) touches `{a['state']}`"
        return [{**a, "state": f"ssm.l{l}"}, {**a, "state": f"conv.l{l}"}]

    return {**call, "args": [b for a in call["args"] for b in arg(a)]}


def variant(module: str, modules: dict, **defines) -> tuple[str, dict]:
    """`module` built with `defines` as well: its name and its pinned entry."""
    family, *pairs = module.split("+")
    name = family + "".join(f"+{k}={v}" for k, v in sorted((dict(p.split("=") for p in pairs) | defines).items()))
    assert name in PINNED, f"`{name}` is not pinned in kernels.toml"
    sha = PINNED[name]["sha256"]
    base = modules[module]
    return name, {**base, "source": base["source"].replace(base["sha256"], sha), "sha256": sha}


def kda_core(op: dict, line: int, h: int, modules: dict) -> tuple[dict, dict]:
    """The op with the conv window as a state of its own after rec, its table after rec's, and
    the module it now runs. Params: rec, conv, rec's table, conv's table, then the base's rest."""
    params = op["params"]
    rec = params.index("inout state")
    assert params[rec + 1] == "in buffer<i32>", f"`kda_core` params {params}"
    shift = lambda a: {"param": a["param"] + 2} if a.get("param", -1) > rec + 1 else a
    launches, entries = [], {}
    for l in op["impl"]["launches"]:
        args = l["args"]
        at = args.index({"i64": line})
        assert args[at - 2:at] == [{"param": rec}, {"param": rec + 1}], f"`kda_core` launch args {args}"
        module, entry = variant(l["module"], modules, CONV_TAP_MAJOR=1, LINE_GUARD=1)
        launches.append({**l, "module": module,
                         "params": l["params"][: at + 1] + ["inout state", "in buffer<i32>", "i64"] + l["params"][at + 1:],
                         "args": args[: at - 2] +
                                 [{"param": rec}, {"param": rec + 2}, {"i64": h * REC_HEAD},
                                  {"param": rec + 1}, {"param": rec + 3}, {"i64": h * CONV_HEAD}] +
                                 [shift(a) for a in args[at + 1:]]})
        entries[module] = entry
    params = params[: rec + 1] + ["inout state", "in buffer<i32>", "in buffer<i32>"] + params[rec + 2:]
    return {**op, "params": params, "impl": {**op["impl"], "launches": launches}}, entries


def on_sglang(m: dict) -> dict:
    lines = {s["bytes_per_seq"] for s in m["states"].values() if "bytes_per_seq" in s}
    assert len(lines) == 1, f"KDA lines of {sorted(lines)} B"
    (line,) = lines
    h = heads(line)
    (token,) = {s["bytes_per_token"] for s in m["states"].values() if "bytes_per_token" in s}
    for name in ("mla_prep", "mla_attn"):
        check_mla(name, m["ops"][name], token)
    op, module = kda_core(m["ops"]["kda_core"], line, h, m["modules"])
    mla = next(f"mla.l{layer(n)}" for n in m["states"] if n.startswith("kv."))
    kda = next(layer(n) for n in m["states"] if n.startswith("kda."))
    b = m["buffers"] | {"kda.conv_index": m["buffers"]["kda.line_index"]}
    ids = {"slot_mapping": {"index_into": mla},
           "block_table": {"index_into": mla, "stride": KERNEL_BLOCK},
           "kda.line_index": {"index_into": f"ssm.l{kda}", "stride": h * REC_HEAD},
           "kda.conv_index": {"index_into": f"conv.l{kda}", "stride": h * CONV_HEAD}}
    ops = m["ops"] | {"kda_core": op}
    used = {l["module"] for o in ops.values() for l in o["impl"]["launches"] if "module" in l}
    return {**m, "model": f"{m['model'].removesuffix('-hosted')}-sglang",
            "states": host_states(m, h),
            "ops": ops,
            "modules": {k: v for k, v in (m["modules"] | module).items() if k in used},
            "programs": {k: {**p, "calls": [rebind(c) for c in p["calls"]]} for k, p in m["programs"].items()},
            "buffers": b | {k: {**b[k], "domain": d} for k, d in ids.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(on_sglang(hosted(json.loads(args.input.read_text())))) + "\n")


if __name__ == "__main__":
    main()
