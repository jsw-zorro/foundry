# RL weight updates on Foundry engines

What an RL loop does to a rollout engine besides serving, and how a Foundry LOAD engine behaves under it. The
restored graphs address the parameters, the KV pool and their own buffers by pointer, so weights that sglang
writes in place are picked up by every graph, as with natively captured graphs.

## Measured (2026-10-28)

Upstream main `2c83dc6656`, plugin route, Qwen3-1.7B (BF16, tied embeddings), engine on one H200 and a trainer
process on a second GPU of the same node. Each flow is a fresh engine; afterwards it answers the same requests (a
batch of 8 greedy prompts sent as one call, so decode runs at batch size 8; 4 greedy prompts one at a time; 8
prompts sampled at temperature 1 with one `sampling_seed` each), and output ids and logprobs are compared exactly.
The LOAD engines use one archive, saved with `--load-format dummy`.

| Flow | Native engine | Foundry LOAD engine |
|---|---|---|
| start with the real checkpoint (LOAD of the archive saved with dummy weights) | reference | identical to the reference |
| dummy start, `/update_weights_from_disk` | runs | identical to native |
| dummy start, `/update_weights_from_tensor` (CUDA IPC from the trainer process, same GPU) | runs | identical to native |
| dummy start, 3 rounds of `/update_weights_from_distributed` (NCCL broadcast from the trainer, P2P/IPC) | runs | identical to native, with the trainer settings below; without them the first broadcast hangs |
| dummy start, `/update_weights_from_disk` with `recapture_cuda_graph=true` | the next decode fails with an illegal memory access | the decode graphs are captured again in place; identical to native without the recapture |

The same restart, SAVE with dummy weights and LOAD with the real checkpoint, also gives outputs identical to a native
start with the real checkpoint for Qwen3.5-35B-A3B (hybrid linear attention, MoE, BF16; one H200), with every
allocation checkpoint at the same offset as on SAVE.

The three update paths give the same outputs as each other. They differ from a start with the real checkpoint by
up to 6.1 in logprob, on the native engine as well; the cause is not determined and does not involve Foundry.

## Requirements

- **The trainer's side of a weight-update group** must use the engine's NCCL settings: `NCCL_CUMEM_ENABLE=0` and
  `NCCL_NVLS_ENABLE=0` (as for native sglang) and `NCCL_LOCAL_REGISTER=0` and `NCCL_GRAPH_REGISTER=0` (Foundry's
  pins). With user-buffer registration on in the trainer only, the first broadcast hung (NCCL 2.30.7). NCCL reads
  these once per process, so set them before the trainer's first communicator. The engine logs its values when
  the group is created (`[Foundry] weight-update group: ...`).
- **A recapture after startup** (`recapture_cuda_graph=true`, a PD role switch) captures the graphs in place and
  logs `[Foundry] SGLang decode CUDA graphs recaptured after startup`; the archive keeps the startup graphs, so the
  next LOAD restores those. Before this was handled, the second restore aborted the process.
- **Allocations after startup** are plain driver allocations: the allocation region closes after the last
  startup capture or restore (`[Foundry] SGLang allocation region closed after startup`).
- **A restart with another checkpoint, settings or host** must allocate exactly as SAVE did. SAVE records the
  offset at every `alloc_offset[...]` checkpoint from the scratch boundary on, and LOAD stops at the first
  checkpoint that differs, with both offsets in the error. A loader that allocates device memory while loading a
  checkpoint (not seen in the two models above) would show up there.

## Not supported

- `--enable-memory-saver`, which colocated training and rollout use to release GPU memory between phases: the
  plugin pins it off (torch_memory_saver owns the pools outside Foundry's region).
- An engine created inside the trainer's own process (`sglang.Engine` in-process): Foundry's activation writes its
  environment pins (`TORCHDYNAMO_DISABLE=1`, the NCCL settings above) into that process's environment and adds
  its hook library to `LD_PRELOAD` for the process's later children. Run the engine in its own process.
- Speculative decoding and LoRA are rejected at startup (see [`overview.md`](overview.md)).

Not measured: tensor-parallel engines, FP8 checkpoints (whose weight processing on update differs), flattened-bucket
updates and a trainer on another node.
