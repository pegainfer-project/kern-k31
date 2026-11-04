# xchg: D's exchanges on a real 8-rank group

`xbench` runs the step's exchanges on 8 GPUs over the lease's two hosts, one process a GPU, every
peer buffer a fabric allocation exported through files in a shared directory and mapped by every
member (as kern's peer buffers are; `alloc_peer(..., true)` also binds a multicast object). A test
is a number of launches captured in one CUDA graph, replayed twice to absorb the members' start skew,
then the median of 5 replays: us a call. The Lamport chain keeps the members in step, so a call's
time includes the slowest member's.

    harness/xchg/build.sh
    loop/withgpu -n 4 -H 2 harness/xchg/run.sh OUT TEST...

`build.sh` compiles this tree's kernels (`ours.cu`, a translation unit each), main's
`k3_ar_fused.cu` and `k3_dcp.cu` with their entries renamed (byte-for-byte checks against main),
FlashInfer's TRT-LLM allreduce fusion as written (`fi_ar.cu`, from `$K3_REF`), and the bare
all-reduces on `k3_xchg.cuh` (`new.cu`).

| test | what |
|---|---|
| `pingpong`, `ppdata` | one-way latency of a flag / of 16 B .. 64 KB, same tray and across |
| `empty`, `flag`, `flagpdl`, `lx` | launch floors; the empty 8-rank exchange in steps |
| `issue`, `readrow`, `ingress` | remote store issue and ack; a row's 8-slot read; rank 0's NVLink ingress |
| `ar` | main's `kern_peer_allreduce_bf16` (the pre-round-3 one-shot) |
| `xar`, `xar2`, `local`, `mc` | `k3_xchg.cuh`'s one-shot / two-shot, FlashInfer's shape, multicast |
| `fi` | FlashInfer's one-shot (Lamport) and two-shot as written |
| `verify`, `fused` | main's fused all-reduces against this tree's: bytes, then us a call |
| `dcp`, `head`, `skel` | `k3_dcp` (bytes against main, then time), `k3_head`, the row routine's skeleton |

kern-k3-pp-prefill's `notes/r-comm.md` has the numbers.
