"""TRT-LLM gen's context FMHA over the packed prefill's sequences at once, at a
batch of `seqs`. It reuses kern's captured ABI (trtllm_fmha_abi); only what
the batch changes is spelled here.
"""
import trtllm_fmha_abi as fmha


def fmha_op(heads, q_max, kv_max, module, q, seqs, entry=fmha.ENTRY, fp8=False):
    """trtllm_fmha_abi.op at a batch of `seqs` (a var): grid z and mBatchSize are the sequences, the
    length tables ([seqs] / [seqs + 1] / [seqs + 1]) are the caller's; the softmax stats (float2 max | sum
    per row and head, extra param 9) written, for the merge of several passes. `entry` picks the
    causal kernel or its dense sibling of the same ABI; `fp8` its e4m3 Q / K / V build (o stays bf16), at
    scale 1: the same parameters over byte-wide rows, the Q and K / V boxes 128 elements (one 128 B
    swizzle span) wide, as FlashInfer's launcher sizes them."""
    op = fmha.op(heads, q_max, kv_max, module, q)
    launch = op["impl"]["launches"][0]
    fields = launch["args"][0]["pack"]["fields"]
    at = {f["at"]: k for k, f in enumerate(fields)}
    fields[at[1132]] = {"at": 1132, "var": seqs}
    fields.append({"at": 1112, "param": 9})  # ptrSoftmaxStats
    launch["entry"] = entry
    launch["grid"] = [launch["grid"][0], launch["grid"][1], seqs]
    op["params"].append("out buffer<f32>")
    if fp8:
        for off in (0, 128, 384):
            t = fields[at[off]]["tensormap"]
            t["dtype"] = "u8"
            t["strides"] = [s // 2 if s > 16 else s for s in t["strides"]]
            t["box"] = [128, *t["box"][1:]]
        launch["shared_mem"] = FMHA_FP8_SMEM
        op["params"][:3] = ["in buffer<fp8e4m3>"] * 3
    return op


FMHA_FP8_SMEM = 117376  # the e4m3 builds' shared memory, from the bundle's flashInferMetaInfo.h
