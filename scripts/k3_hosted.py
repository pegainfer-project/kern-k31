"""K3 decode as an engine's model forward: the program boundary.

A serving engine that hosts kern (vLLM, SGLang) keeps sampling for itself,
so the D step it drives stops where the model's own forward stops: after
the final AttnRes mix and norm, at the hidden states. The engine then asks
for logits through a separate `head` program and gathers the vocab itself.

Rewrites a `gen_stage.py --tp N --peer-ar` decode manifest:

- `decode` becomes `decode_batch`, cut after `out.res`, whose normed rows
  land in the new output `hidden` (the residual stream workspace that held
  that name becomes `resid`).
- `head` is the cut tail's GEMM on its own: `head_in [seqs, H]` against
  this member's rows of `w_lm`, into `logits [seqs, V / N]` f32, this
  member's contiguous slice of the vocab. The engine concatenates the
  members' slices in rank order.
- The in-graph argmax goes, and with it everything only it used: the
  carried full-vocab `logits` and its peers, the key exchange, `next_token`.

Memory stays where the base put it; `k3_vllm.py` moves the states onto an
engine's pages. Engine-neutral: no engine's name or layout appears here.

    python3 scripts/k3_hosted.py --input d.json --output d-hosted.json
"""
import argparse
import json
import pathlib

GONE = ("logits", "logits_peers", "logits_part", "head_keys", "head_keys_peers", "head_state", "argmax_done",
        "next_token")


def bufs(calls) -> set[str]:
    return {a["buf"] for c in calls for a in c["args"] if "buf" in a}


def renamed(call: dict, old: str, new: str) -> dict:
    return {**call, "args": [{**a, "buf": new} if a.get("buf") == old else a for a in call["args"]]}


def cut(calls: list[dict]) -> list[dict]:
    """The step up to `out.res`, which writes its normed rows to `hidden`."""
    end = next(i for i, c in enumerate(calls) if c["label"] == "out.res")
    steps = [renamed(c, "hidden", "resid") for c in calls[: end + 1]]
    final = steps[-1]
    assert final["op"].startswith("land_add2_attnres_rms"), f"out.res is `{final['op']}`"
    assert sum(a.get("buf") == "normed" for a in final["args"]) == 1, "out.res writes `normed` once"
    return steps[:-1] + [renamed(final, "normed", "hidden")]


def head(lm_head: dict) -> dict:
    """The base's `out.lm_head` over `head_in`, one row per sequence."""
    a, w, _, _, n, k, ld = lm_head["args"]
    assert lm_head["op"] == "gemm_f32" and n == ld, f"out.lm_head is `{lm_head['op']}` {lm_head['args']}"
    return {"calls": [{"label": "head", "op": "gemm_f32",
                       "args": [{"buf": "head_in"}, w, {"buf": "logits"}, {"var": "seqs"}, n, k, ld]}]}


def hosted(m: dict) -> dict:
    p = m["programs"]
    calls = p["decode"]["calls"]
    lm_head = next(c for c in calls if c["label"] == "out.lm_head")
    _, _, _, _, vocab, width, _ = (a.get("i32") for a in lm_head["args"])
    hidden = {**m["buffers"]["hidden"]}
    assert all("hidden" not in bufs(v["calls"]) for k, v in p.items() if k != "decode"), "only decode names `hidden`"
    programs = {k: v for k, v in p.items() if k != "decode"} | {
        "decode_batch": {**p["decode"], "calls": cut(calls)},
        "head": head(lm_head)}
    buffers = {("resid" if k == "hidden" else k): v for k, v in m["buffers"].items() if k not in GONE} | {
        "hidden": {**hidden, "kind": "output"},
        "head_in": {"dtype": hidden["dtype"], "shape": ["seqs", width], "kind": "input"},
        "logits": {"dtype": "f32", "shape": ["seqs", vocab], "kind": "output"}}
    m = {**m, "model": f"{m['model']}-hosted", "programs": programs, "buffers": buffers}
    return prune(m)


def prune(m: dict) -> dict:
    """Drop the ops and modules no program calls any more; refuse a dangling buffer."""
    calls = [c for p in m["programs"].values() for c in p["calls"]]
    dangling = bufs(calls) - m["buffers"].keys()
    assert not dangling, f"calls still name {sorted(dangling)}"
    ops = {k: v for k, v in m["ops"].items() if k in {c["op"] for c in calls}}
    modules = {l["module"] for o in ops.values() for l in o["impl"]["launches"] if "module" in l}
    return {**m, "ops": ops, "modules": {k: v for k, v in m["modules"].items() if k in modules}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(hosted(json.loads(args.input.read_text()))) + "\n")


if __name__ == "__main__":
    main()
