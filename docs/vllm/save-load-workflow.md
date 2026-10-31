# vLLM Save / Load Workflow

## Serve scripts

`experimental/single-gpu/serve_qwen_<size>.sh` for 1.7B / 4B / 14B (single-GPU). DP / EP scripts under `experimental/data-parallel/` and `experimental/expert-parallel/`.

Key knobs (1.7B example):

```bash
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.6

FOUNDRY_HOOK="$(cd "$(dirname …)/../../foundry/python/foundry" && pwd)/libcuda_hook.so"
if [[ -f "$FOUNDRY_HOOK" ]]; then
    export LD_PRELOAD="${FOUNDRY_HOOK}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export VLLM_WORKER_MULTIPROC_METHOD=forkserver

vllm serve "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enable-prefix-caching \
    --max-num-seqs 512 \
    --attention-config.backend FLASH_ATTN \
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
    --compilation-config.graph_extension_config_path "$FOUNDRY_TOML"
```

> **Note:** Don't pass `--compilation-config.cudagraph_num_of_warmups N`. vLLM's `VllmConfig.__post_init__` hard-overrides this to `1` whenever cudagraph mode is enabled (`vllm/config/vllm.py:1052`), ignoring whatever you pass. Foundry then stomps it back to `0` inside `install_hooks` because the warmup `_dummy_run(cudagraph_runtime_mode=NONE, force_attention=True)` would invoke `CUDAGraphWrapper.runnable()` outside the capture window and break SAVE/LOAD allocation parity. See [`overview.md`](overview.md#2-cudagraph_num_of_warmups-must-be-0-in-foundry-mode).
>
> `VLLM_USE_AOT_COMPILE` defaults to `False` in `vllm/envs.py` and is not touched by `install_hooks`. If a future vLLM bumps the default to `True`, we'd need to force-set it back (an AOT cache-hit forward would otherwise produce non-deterministic allocations between SAVE and LOAD).

### Why pre-launch `LD_PRELOAD`

`libcuda_hook.so` must be loaded **before** any CUDA call so its interposers (`cuMemAlloc_v2`, `cuModuleLoadData`, …) run in every process. The integration also re-sets `LD_PRELOAD` via `setup_ld_preload_env()` before each `proc.start()`, but pre-loading from the shell protects the parent process and any helper threads.

### Why `VLLM_WORKER_MULTIPROC_METHOD=forkserver`

`forkserver` pre-imports `vllm` once and forks workers from that warmed-up parent — much faster cold start. Works with foundry because:

- `libcuda_hook.so` is in `LD_PRELOAD` so the forkserver itself has it.
- `torch.cuda.get_device_capability` is NVML-backed (foundry's `_quarantine_get_device_capability` patch) so the parent stays CUDA-clean.
- `cudaSetDevice` resolves through `RTLD_DEFAULT+RTLD_NEXT` in forkserver children where libcudart has `RTLD_LOCAL` scope.

## TOML configs

Same shape as SGLang:

```toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive_qwen_1.7b"
scratch_space_size = "1024MB"
```

## Run sequence (single-GPU)

```bash
cd /path/to/foundry-org

# 1. Clean any prior workspace
rm -rf foundry_archive_qwen_1.7b

# 2. SAVE pass 1 (profiles + captures)
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait for "Application startup complete", Ctrl-C / SIGTERM

# 3. SAVE pass 2 (deterministic; reuses saved profile)
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait, SIGTERM

# 4. LOAD
bash experimental/single-gpu/serve_qwen_1.7b.sh --load

# 5. Query
bash experimental/query.sh 12000 Qwen/Qwen3-1.7B
```

Always run two SAVE passes. Pass 1 writes `warmup_state.json` and a non-deterministic set of captured graphs; pass 2 overwrites the graphs with deterministic ones. Skipping pass 2 will produce a LOAD that appears to succeed but fails the cursor check or, worse, succeeds the cursor check and emits garbage.

## DP / EP

`experimental/data-parallel/serve.sh` and `experimental/expert-parallel/serve.sh` extend the same pattern with the per-rank flags vLLM needs (`--data-parallel-size`, `--enable-expert-parallel`, etc.). Each rank writes to its own `rank_{N}/` subdirectory under `workspace_root`; `warmup_state.json` stays at the workspace root and is written by rank 0.

## Expected logs

SAVE (success):

```
[foundry] install_hooks: mode=save workspace=foundry_archive_qwen_1.7b
[foundry TIMING] subprocess spawn → install_hooks: ... ms
[foundry] all patches installed
…
[foundry] start_graph_builds returned in 0.xs (building in background)   ← on LOAD only
[foundry TIMING] init_nvshmem_for_loaded_modules: 0.xs (N modules)       ← on LOAD only
(per-graph replay happens inline during capture_model on LOAD)       ← no separate "finish_graph_loads" log line
…
[foundry] capture_final_alloc_offset = … bytes                       ← on SAVE
INFO:     Application startup complete.
```

LOAD (success):

```
[foundry] install_hooks: mode=load workspace=foundry_archive_qwen_1.7b
…
[foundry] preallocate_for_load_mode = … MB
[foundry] start_graph_builds returned in 0.xs (building in background)
[foundry TIMING] init_nvshmem_for_loaded_modules: 0.xs (N modules)
…
(per-graph replay happens inline during capture_model — no separate log line)
…
INFO:     Application startup complete.
```

## Checks at worker start

Every worker refuses, before the allocation region is set up, the configurations whose graphs Foundry cannot save
or restore: a `cudagraph_mode` with piecewise graphs (only `FULL_DECODE_ONLY` and `FULL`), the V2 model runner,
pipeline parallelism, elastic EP, LoRA and speculative decoding. Custom all-reduce is turned off
(`disable_custom_all_reduce=True`, logged as a pin): its kernels read peer buffers registered per captured graph,
which LOAD does not register. `install_hooks` sets `VLLM_USE_V2_MODEL_RUNNER=0`, `NCCL_GRAPH_REGISTER=0` and
`NCCL_LOCAL_REGISTER=0` when unset and refuses a contrary value.

Each rank's SAVE writes two records next to its graphs:

- `fingerprint.json`: the settings that shape the graphs or size the buffers they address (model, as the
  digest of its `config.json` for a local directory, dtype,
  quantization, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, block size, KV dtype, parallel sizes,
  all-to-all backend, `cudagraph_mode` and capture sizes, the region's base, size and scratch), the vLLM, Foundry,
  torch, CUDA and driver versions and the GPU. LOAD compares it before the archived modules load and stops with the
  differences named, and stops as well when the file is missing or lacks a field the current code records (an
  archive of an older Foundry: re-SAVE); `FOUNDRY_VLLM_SKIP_FINGERPRINT=1` skips the comparison.
- `region_layout.json`: the scratch boundary weight loading starts at (scratch_space_size, or the cursor when
  distributed init allocated past it) and the allocation offset at `after_scratch_skip`, `before_load_model`,
  `after_load_model`, `before_capture_model` and `after_capture_model`. LOAD moves to SAVE's boundary (and stops
  when its own distributed init went past it) and stops at the first checkpoint whose offset differs.

At LOAD the capture loop must ask for the archived graphs in SAVE's order, and every archived graph must be
restored by the end of `capture_model`. A KV block count that differs from `warmup_state.json` stops the engine.

## Validation

A correct LOAD:

1. `final_alloc_offset` from SAVE pass 2 matches the LOAD-side post-replay cursor (foundry validates this internally).
2. `query.sh` returns coherent text. If cursors match but output is garbage, see [`memory-consistency.md`](memory-consistency.md).

### Outputs of LOAD

Measured 2026-10-29 with the vLLM fork at `eed7819a2` (Qwen3-1.7B, 1xH200, `FULL_DECODE_ONLY`; one native
run, SAVE passes 1 and 2 and one LOAD, each answering the same 12 greedy requests of 24 and 32 tokens with
logprobs):

| Compilation in every run | LOAD vs SAVE pass 2 | SAVE pass 2 vs native |
|---|---|---|
| off (`--compilation-config.mode 0`) | texts and logprobs identical | texts identical; logprobs differ by at most 3e-6 |
| default (torch.compile) | 2 of 12 texts differ; first-token logprobs differ by up to 0.078 | texts identical; logprobs differ by at most 3e-6 |

With torch.compile, LOAD does not compile the model (`do_not_compile`, see [`direct-edits.md`](direct-edits.md)):
the decode graphs replay SAVE's compiled kernels, but the prefill runs the uncompiled model, so the outputs differ
from the first token on. LOAD's first-token logprobs also differ from those of a native run without
torch.compile, by up to 0.062 with vLLM's custom ops and up to 0.051 with `custom_ops=["none"]`; which
uncompiled configuration LOAD's prefill equals is not determined. With compilation off in every run, LOAD
reproduces SAVE exactly. The difference between SAVE and native occurs in both settings, from the prefill on;
its cause is not determined.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[HOOK] ERROR: Memory offset mismatch during replay` | An asymmetric allocation between SAVE pass 2 and LOAD. The watermark logs identify which lifecycle phase. |
| `Foundry LOAD: allocation offset at <checkpoint> is ..., SAVE had ...` | LOAD allocated differently from SAVE pass 2 before that checkpoint (other settings, model or software). Serve with SAVE's settings or re-SAVE (both passes). |
| `Foundry LOAD: archive ... was saved under other settings (...)` | The fingerprint differs; the message lists each field with both values. |
| LOAD aborts at first inference with `Called CUDAGraph::replay without a preceding successful capture or load` | On-demand graph lost its template link. Should not happen — `start_graph_builds` loads all paths in a single C++ call; if it does, check `_patch_capture_model` and that `_pending_graph_builds.identifier_to_index` covers every BatchDescriptor visited by the capture loop. |
| `_initialize_kv_caches` complains about missing fields on LOAD | `warmup_state.json` not written by SAVE pass 1. Re-run pass 1, then pass 2. |
| Hangs in NCCL bring-up | `setup_graph_extension` may have set up the VMM region after NCCL already touched memory. Verify `_patch_init_worker_distributed_environment` is wrapping the correct upstream symbol (vLLM renames sometimes). |
| Workers crash with `module not found` | foundry not installed in worker venv; pip install editable in the same env, or use the serve script's PYTHONPATH approach. |

## What to clean between runs

Always `rm -rf workspace_root` before SAVE pass 1.

Pass 2 reads `warmup_state.json` and overwrites graphs — no separate clean needed between passes.

LOAD never writes to the workspace.
