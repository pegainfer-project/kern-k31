#!/usr/bin/env python3
"""Build FROST's KDA prefill, vendored here from cuDNN frontend 1.32.0, into one sm_103a cubin at
kern-k31's static configuration, and print its two entries.

    python3 source/cudnn-frost-kda/build.py <out.cubin>

Runs on a GB300 (sm_103) with the cuDNN frontend 1.32.0 and nvidia-cutlass-dsl 4.8.0 wheels and torch
installed. The wheel's `cudnn` package supplies only its host-side device queries; every
`cudnn.frost` and `cudnn.linear_attention` module is loaded from this directory, which the build
checks. `forward` launches the prologue and the prefill as cuDNN's uncut forward does without the
d_v split (kda_uncut_forward_f16.uncut_forward_host, prep=False), over placeholders marked the way
its build marks them, so the kernels compile as they do there.

Static: bf16 q / k / v / o / gate / state; safe gate, lower bound -5; q and k L2-normed and beta's
sigmoid in the kernel; i64 cu_seqlens; the state read from and written to a pool row by seed and
final indices; no checkpoints; 152 SMs; the uncut schedule (one work item per sequence and head).
Shapes, strides, pointers, the scale and the sequence count are runtime values.
"""
import importlib.metadata
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
WHEELS = {"nvidia-cudnn-frontend": "1.32.0", "nvidia-cutlass-dsl": "4.8.0"}

for dist, want in WHEELS.items():
    got = importlib.metadata.version(dist)
    if got != want:
        raise SystemExit(f"{dist} {want} expected, got {got}")

# the compile keeps its cubin (CuTe DSL writes the artifacts it keeps to a dump directory as well)
os.environ["CUTE_DSL_KEEP"] = "cubin"
os.environ["CUTE_DSL_DUMP_DIR"] = tempfile.mkdtemp(prefix="cudnn-frost-kda-")

import cudnn  # noqa: E402

assert "cudnn.frost" not in sys.modules, "the wheel's cudnn.frost loaded before the vendored one"
cudnn.__path__.insert(0, str(HERE / "cudnn"))

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import torch  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from cudnn.linear_attention.frost.common.host import tensormap_workspace_bytes  # noqa: E402
from cudnn.linear_attention.frost.common.split_k import WORK_ITEM_FIELDS  # noqa: E402
from cudnn.linear_attention.frost.kernel import kda_prefill_f16 as kp  # noqa: E402

HEADS, DIM, NUM_SM, LOWER_BOUND = 96, 128, 152, -5.0


@cute.jit
def forward(cfg: cutlass.Constexpr, checkpoint_every_n: cutlass.Int32, scale: cutlass.Float32, q: cute.Tensor,
            k: cute.Tensor, v: cute.Tensor, gate: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
            beta: cute.Tensor, o: cute.Tensor, cu_seqlens: cute.Tensor, pool: cute.Tensor, seed: cute.Tensor,
            final: cute.Tensor, work_items: cute.Tensor, work_count: cute.Tensor, scheduler: cute.Tensor,
            workspace: cute.Tensor, stream: cuda.CUstream):
    kp.prologue(cfg.io_dtype, cfg.b_t, cfg.max_active_clusters, q, k, v, gate, o, None, cu_seqlens, work_count,
                work_items, scheduler, workspace, checkpoint_every_n, stream, 1)
    kp.host(cfg, q, k, v, gate, a_log, dt_bias, beta, cu_seqlens, pool, o, pool, seed, final, work_items, work_count,
            scheduler, workspace, checkpoint_every_n, scale, stream)


def placeholders():
    dev = torch.device("cuda")
    rows, seqs, lines = 64, 2, 2
    x = lambda: torch.zeros(rows, HEADS, DIM, device=dev, dtype=torch.bfloat16)
    line = HEADS * DIM * DIM + 4096
    pool = torch.zeros(lines * line, device=dev, dtype=torch.bfloat16).as_strided(
        (lines, HEADS, DIM, DIM), (line, DIM * DIM, DIM, 1))
    rowwise = lambda t: from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    work_items = from_dlpack(torch.zeros(seqs * HEADS, WORK_ITEM_FIELDS, device=dev, dtype=torch.int32),
                             assumed_align=16)
    work_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
    i32 = lambda n: torch.zeros(n, device=dev, dtype=torch.int32)
    return [
        rowwise(x()), rowwise(x()), rowwise(x()), rowwise(x()),
        from_dlpack(torch.zeros(HEADS, device=dev), assumed_align=4).mark_layout_dynamic(),
        from_dlpack(torch.zeros(HEADS, DIM, device=dev), assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(torch.zeros(rows, HEADS, device=dev, dtype=torch.bfloat16), assumed_align=4).mark_layout_dynamic(
            leading_dim=1),
        rowwise(x()),
        from_dlpack(torch.tensor([0, rows // 2, rows], device=dev), assumed_align=8).mark_layout_dynamic(),
        from_dlpack(pool, assumed_align=16).mark_layout_dynamic(leading_dim=3),
        from_dlpack(i32(seqs), assumed_align=4).mark_layout_dynamic(),
        from_dlpack(i32(seqs), assumed_align=4).mark_layout_dynamic(),
        work_items,
        from_dlpack(i32(1), assumed_align=4).mark_layout_dynamic(),
        from_dlpack(i32(2), assumed_align=4).mark_layout_dynamic(),
        from_dlpack(torch.zeros(tensormap_workspace_bytes(kp, seqs) // 8, device=dev, dtype=torch.int64),
                    assumed_align=128).mark_layout_dynamic(),
    ]


def main():
    out = pathlib.Path(sys.argv[1])
    bf16 = cutlass.BFloat16
    cfg = kp.build_cfg(bf16, bf16, bf16, use_initial_state=True, store_final_state=True, enable_checkpoints=False,
                       l2norm=True, safe_gate=True, gate_scale_log2=LOWER_BOUND * kp.LOG2_E, log_gate=True,
                       beta_sigmoid=True, allow_neg_eigval=False, max_active_clusters=NUM_SM, d_k=DIM, d_v=DIM)
    compiled = cute.compile(forward, cfg, cutlass.Int32(0), DIM ** -0.5, *placeholders(),
                            cuda.CUstream(torch.cuda.current_stream().cuda_stream), options="--opt-level 2")
    foreign = sorted(n for n, m in sys.modules.items()
                     if n.startswith(("cudnn.frost", "cudnn.linear_attention"))
                     and not pathlib.Path(m.__file__).resolve().is_relative_to(HERE))
    assert not foreign, f"modules loaded from outside {HERE}: {foreign}"
    out.write_bytes(compiled.__cubin__)
    for sym, attrs in compiled.kernel_info.items():
        print(sym, attrs)


if __name__ == "__main__":
    main()
