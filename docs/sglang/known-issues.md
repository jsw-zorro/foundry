# Known issues — SGLang integration

## Exec share mode: tokens differ from native on one tested model (cause OPEN, 2026-10-28)

With `FOUNDRY_ENABLE_EXEC_SHARE=1` on LOAD, `tests/sgl/e2e/test_save_load_multigpu.py` failed for Qwen3.5-35B-A3B TP2
(real weights, upstream main `2c83dc6656`, 2xH200): 302 of the 2088 compared outputs differed from native, in 50 of
the 64 decode batch sizes, by up to 6.3 in logprob. In each failing batch of size b replayed on the exec of a template
of size T, the outputs of requests 0 to T-b-1 differ. The default eager mode passes the same case, and share mode
passed the Qwen3.5-35B-A3B EP2 (DP attention) case and the 1-GPU test.

Measured on the same archive (one LOAD per variant, outputs compared with native):

| Share-mode variant | Templates | Probes that differ |
|---|---|---|
| saved groups refined by binary, DAG, cluster dims, cooperative | 22 | 50 |
| members' kernel parameters set again on the exec after the update (all kernels, clustered kernels, or memcpy/memset nodes) | 22 | 50 |
| members split by memcpy extents | 64 | 0 |
| members split by the launch dims of clustered kernels | 64 | 0 |
| members split by every kernel's launch dims | 64 | 0 |

The graphs hold 817 kernel nodes (664 on programmatic edges; clustered allreduce-fusion and cuBLAS kernels), 40
device-to-device memcpy nodes and one memset. A memcpy node, a clustered kernel and a PDL producer/consumer pair each
switch correctly in isolation, with members smaller than their template (`core/test_exec_share.py`). The cause is
not determined. Since 2026-10-28 share mode gives members whose memcpy or memset extents or clustered launch dims
differ from their template their own template, so on this model every graph has its own exec and LOAD matches native;
graphs whose members differ only in kernel arguments and in the launch dims of unclustered kernels are still
switched. Serve with the default eager mode; LOAD logs a warning when share mode is enabled.

## Code audit of 2026-10-28 (FIXED)

A review of the hook, the graph save and restore code, the binary format and both integrations, with every finding
checked against the code by a second reviewer, found the defects below; a review of the fixes (2026-10-29) added
the follow-ups in the same rows. Each is fixed on branch
sglang-upstream-compat; the test that covers it is named where one exists.

| Area | Defect | Effect before the fix | Fix | Test |
|---|---|---|---|---|
| Allocation region | An allocation that did not fit in the rest of the region (64 GiB default) went to the driver's allocator | SAVE finished; the graphs pointed at memory LOAD did not have (crash or wrong tokens) | Refused with out-of-memory and the region size named; default `region_size` 256 GiB | `core/test_allocation_region.py` exhaustion |
| Allocation region | A failed reservation at `base_addr` only printed `[HOOK] ERROR` | The engine ran without the region; the LOAD error named another cause | `set_allocation_region` raises; the schedulers reserve before restoring modules (early reserve) | `core/test_allocation_region.py` reserve failure |
| Allocation region | A large reservation (NVSHMEM heap) placed away from its hint was accepted | DeepEP graphs addressed SAVE's heap | Refused (`FOUNDRY_ALLOW_RELOCATED_RESERVE=1` accepts it) | - |
| Allocation region | `free_preallocated_region` unmapped memory still in use; a second identical `set_allocation_region` moved the cursor back | Use of unmapped memory; two tensors at one address | The span stays mapped while in use (and serves no region set later below it); the cursor is kept | `core/test_allocation_region.py` free live span, kept span |
| Binary archive | `.cugraph` counts and offsets were not checked against the file | A truncated file crashed LOAD (heap over-read) | Checked; a damaged binary is skipped and the graph read from its JSON | `core/test_archive_damage.py` |
| Archive writes | Graph JSON, `.cugraph` and the packed images were written in place and their write errors ignored | A full disk left a truncated archive that SAVE reported as complete | Written to a temporary name and renamed; a failed write raises; a `.cugraph` of an earlier SAVE is removed before the new JSON is written | - |
| Templates | `save_graph_manifest` also matched `graph_manifest.json`; LOAD did not check the manifest's roles (a graph in two groups or in none, a member whose template is not in the same call); the exec-pool prewarm looked dependency node ids up with `.at` | A second manifest write raised `ValueError`; a graph in no group was never built (replay error at first use); the prewarm stopped at a dependency on an unknown node id (logged; LOAD went on without it) | Files matched as `graph_<index>_*.json`; roles checked at LOAD with the graph named; the prewarm skips a graph it cannot build | `unit/test_archive_files.py`, `core/test_archive_damage.py` |
| Templates | A member did not reset kernel-node attributes it does not record (cooperative, mem-sync domain, preferred cluster, ...) | It launched with the template's attributes | Unrecorded attributes reset to zero on each rewrite | `core/test_template_attrs.py` |
| RNG | Restored Philox kernels read SAVE's seed/offset tensors, LOAD filled its own (archive format v3 records the addresses) | Graphs with RNG replayed the same random numbers | LOAD points the tensors at SAVE's addresses when restored memory holds them, otherwise stops. torch < 2.13 keeps one pair per generator, allocated before the recording: there LOAD stops for a graph with RNG whose pair is elsewhere | `core/test_save_load.py` rng_segment (torch >= 2.13) |
| Serial load | `CUDAGraph.load` ignored common kernel attributes and PSS and accepted raw pointers in `extra` | Lost attributes; host memory read | Builds with the JSON template builder; `extra` decoded strictly; parameter count and sizes checked against the loaded function | `core/test_save_load.py` |
| Modules | PTX archived without its NUL; file-loaded images restored through a fixed-name file in the archive; packed-image sizes checked with wrapping arithmetic; module loads of device-link segments passed a null image; an image LOAD read into memory (`FOUNDRY_MMAP_ARCHIVE=0`) or linked was freed after a `cuLibraryLoadData` with `CU_LIBRARY_BINARY_IS_PRESERVED`, which lets the driver read it until unload | Driver over-read; races between LOADs; endless parse loop on a corrupt size; abort; driver reads of freed memory | NUL-terminated copy; per-process file under `$TMPDIR`; checks by subtraction; segments linked at LOAD; such images kept for the process lifetime, linked images loaded without the option | `core/test_ptx_module.py` |
| VMM IPC | The socket name held only the pid | Two hooked processes with one pid in a shared network namespace: exports failed | Name holds the process's random token | - |
| SGLang | Decode graphs archived only the logits | No decode hidden states with a return-hidden-states mode on LOAD | Every tensor field archived (`decode_graphs.json`) | `sgl/test_runtime_contracts.py` |
| SGLang | The plugin entry point imported foundry outside activate()'s guard | An import failure served without Foundry | Exits the process | `sgl/test_runtime_contracts.py` |
| SGLang | The fingerprint did not record `base_addr` and `region_size` | LOAD with another `region_size` placed the NVSHMEM heap (DeepEP) elsewhere than the graphs address it | Recorded; a fingerprint without a field the code records (an older Foundry's) is refused with "re-SAVE" | `sgl/test_runtime_contracts.py` |
| SGLang | Fingerprint checked after the archived modules loaded; environment pins accepted spellings sglang reads differently; `graph_templates = "false"` read as true; embedding models produced empty archives | Abort instead of the refusal; pins not in effect; templates kept; LOAD abort | Checked first; pins rewritten canonically; TOML booleans only; embedding models refused (`--is-embedding` on both routes; a reward or classification model that sglang detects by its architecture on the plugin route only: on the dependency route its SAVE writes no graphs and LOAD stops at the missing `region_layout.json`) | `sgl/test_pins.py`, `sgl/test_runtime_contracts.py` |
| vLLM | No fingerprint, layout record, scratch boundary record, capture-order check or configuration refusals | LOAD under other settings restored graphs at wrong offsets | Ported from the SGLang integration (see [`../vllm/save-load-workflow.md`](../vllm/save-load-workflow.md)) | - |
| Exec share | Members with other cluster dims or cooperative launches shared a template exec | LOAD abort or a rejected update | Such members get their own template | - |
| Exec share | See the entry above (cause open) | Wrong tokens on a TP2 model | Members with other copy extents or clustered launch dims get their own template | `core/test_exec_share.py` |

## Failure-mode audit of 2026-10-29 (FIXED unless marked)

A second review looked at what an engine does after startup, threads and lifetimes, failure paths, multi-rank
consistency and the latest upstream main (`f2d3188`); each finding was checked against the code by a second
reviewer. The configurations of the first three rows were not in the validated set; the e2e cases named in the last
column reproduce them. A third review (the rows from "Host memory" on) covered the entry points, serving features,
child processes of the launcher and the small modules; none of its findings occurred in a validated configuration.

| Area | Defect | Effect before the fix | Fix | Test |
|---|---|---|---|---|
| Lazy model state | SAVE captured each shape without an eager forward, so what a model builds on its first forward went into the first captured graph only: the RoPE cos/sin cache converted to the activation dtype (mrope: Qwen2.5-VL, Qwen3-VL; the fallback rotary kernel for head sizes outside 64/128/256/512, e.g. Phi-3 at 96) and the local expert map of EP without an all-to-all backend | Restored graphs and eager code read a table only the first graph writes, from a source buffer LOAD frees: wrong tokens or an illegal address | One warm-up forward at each capture scope's first shape on SAVE and LOAD (`capture_loop_version` 3); its transient memory is released on both modes | e2e `phi3_rope96`, `q25vl_mrope`, `q35_tp2ep2_std` (each fails without the fix: SAVE tokens differ from native, or an illegal address at the largest batch) |
| Weight loading | sglang's DeepSeek-family loaders copy CPU weights on a thread pool; those threads allocate outside the thread-local region, and post-load processing can place persistent tensors in their cached segments | Weights at driver-chosen addresses that differ on LOAD (W4A16 / AWQ / GPTQ MoE layouts) | `should_async_load` returns False from the end of distributed init to the layout start on SAVE and LOAD (both routes); live tensors outside the region at the layout start stop both modes | `sgl/test_runtime_contracts.py` |
| Collectives | The pre-capture bootstrap connected only the torch.distributed communicators; the captured collectives run on each group's PyNccl communicator, connected per algorithm at first use (inside SAVE's capture, never on LOAD); the DCP and shared-experts TP groups were left out | Hang or fault at the first restored collective of an algorithm the init all-reduce did not connect (TP across nodes, DCP all-to-all) | PyNccl communicators get the same small and large collectives; point-to-point peers connect in the warm-up forward | - |
| Distributed init | Allocations below the scratch boundary (communicators, symmetric-memory buffers, multicast mapping) were not compared | Another NCCL build or setting, cuBLAS workspace or multicast support moved them while every check passed: hang, fault or wrong reductions | Offset trail from `after_setup_graph_ext`; the allocations and carved reservations live below the weights compared at the scratch skip | `sgl/test_runtime_contracts.py` |
| Archive records | Archives without `fingerprint.json`, the scratch boundary or the offset trail (older releases) loaded with those checks off; a LOAD cursor behind the layout start was moved up | Silent loads of archives the current checks would refuse | `capture_loop_version` 3 required; each record required; any cursor difference at the layout start stops LOAD | `sgl/test_runtime_contracts.py`, `unit/test_sglang_runtime_state.py` |
| Fingerprint | Raw (unresolved) server arguments; no deterministic-inference, all-reduce-fusion, DCP, linear-attention, mamba or DSA backend fields; no NCCL, FlashInfer, sgl-kernel, DeepEP, DeepGEMM, NVSHMEM or Triton versions; no NCCL / symmetric-memory / allocator variables; no sglang commit for a git checkout; no rank; model path compared as a path | LOAD under another build or setting passed; one model named two ways was refused | Resolved values, the fields, `package <name>` versions, `nccl version` and the library PyNccl loads, the device-allocator settings, the variables, `sglang source`, `rank` and the model as a digest of its config-class view (AutoConfig, without `transformers_version` and `_name_or_path`; else of config.json, hub ids through the hub cache); a value one side cannot determine (`unknown (...)`) is logged and not compared | `sgl/test_runtime_contracts.py` |
| Archive identity | Nothing tied the files of a rank directory to one SAVE run | Two engines writing one `workspace_root` gave an archive mixing both | A SAVE id in `fingerprint.json` and `region_layout.json`, compared at LOAD | `sgl/test_runtime_contracts.py` |
| Checkpoints | SAVE and LOAD logged different labels around the capture scopes; the per-graph replay moved a lagging cursor up silently | A smaller decode-runner allocation on LOAD went unnoticed | `prefill_scope_end`, `decode_scope_start`, `decode_scope_end` on both modes; each realignment logged with its size | - |
| Setup | No check that the hook is preloaded (a wrong `hook_library_path`, Ray actors) or that torch's allocator is the one the hook sees | Archives without graph memory; failures far from the cause | The scheduler checks that `cuMemAlloc_v2` resolves to the hook; expandable segments and `cudaMallocAsync` refused; a missing `hook_library_path` refused; `--use-ray` refused | `sgl/test_pins.py`, `sgl/test_runtime_contracts.py` |
| NVSHMEM | A restored module whose NVSHMEM init failed or could not run was only logged | DeepEP kernels on unset device state at the first replay | LOAD stops with the modules named | - |
| Exit paths | A failed SAVE packed the images into `./hook_archive` at exit; a thread pool whose thread creation failed called `std::terminate`; a Phase 2 job could write locals already destroyed; a failed prewarm thread creation failed LOAD | Multi-GB writes and aborts after the real error | Pack-on-exit disarmed after each graph save; pools join their started workers and rethrow; locals declared before the worker; prewarm optional | - |
| Dependency route | The warm-up forward needs the attention backend's reset, which the adapter passes only from integration API 2.1 | - | An older adapter runs without the warm-up (warning); API 2.1 adds `post_warmup_hook` to `capture_one` | `sgl/test_runtime_contracts.py` |
| LOAD memory | Memory LOAD mapped on demand inside the preallocated span (allocations SAVE did not keep live, the warm-up forward's among them) stayed mapped after it was freed | LOAD held more memory than SAVE after startup | The hook releases such a segment once no allocation uses it | - |
| Recapture (OPEN, documented) | After `update_weights_from_disk(recapture_cuda_graph=true)` on LOAD, the restored graphs' memory stays mapped while the recaptured graphs take new memory | The recapture needs one more decode-graph footprint of free memory than on a SAVE engine | - | - |
| Exit (OPEN) | The detached exec-pool prewarm and Phase 2 threads keep running when a LOAD fails during startup | The process can abort while exiting, after the error is logged | - | - |
| Per-shape lazy state (OPEN) | State a model creates lazily for one batch size, other than the first captured one, is still created inside that shape's capture | Not observed in the validated configurations | - | - |
| Host memory | HiSparse's host KV pool, DeepSeek-V4.1 Engram tables in host memory (the resolved `--engram-config`; on an sglang without it, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` for a model with Engram layers and no peer shards, both variables as that sglang parses them), the PLE embedding offload (on by default for BF16 Qwen4-Exp on CUDA), `--cpu-offload-gb` and `--offload-group-size` were accepted (the last two refused only at the first graph save) | Restored graphs read the pinned host addresses of the SAVE process | Refused at resolution | `sgl/test_pins.py` |
| Weight cache | `--weight-cache-mode daemon` / `client` was accepted; the parameters are CUDA IPC mappings of the daemon's memory at addresses the driver picks per process | Restored graphs read SAVE's parameter addresses | Refused at resolution; at the layout start every model parameter and buffer must lie in the region, or in a reservation of 2 GiB or more that the hook placed at its fixed address after the region (DeepSeek-V4.1 Engram peer shards) | `sgl/test_pins.py`, `sgl/test_runtime_contracts.py` |
| torch.compile | `--enable-torch-compile` was accepted while Foundry pins `TORCHDYNAMO_DISABLE=1` | The compiled batch sizes captured sglang's uncompiled reference ops | Refused | `sgl/test_pins.py` |
| PD disaggregation | Mooncake's intra-node NVLink transport exports the KV pool (region memory) through CUDA IPC handles that a process without the hook cannot open | Every KV transfer into a Foundry engine fails | Refused with `MC_INTRANODE_NVLINK=1` or `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` | `sgl/test_pins.py` |
| Fingerprint | `disable_radix_cache`, `mamba_radix_cache_strategy`, `mamba_track_interval`, `disable_overlap_schedule`, `weight_cache_mode` and `enable_torch_compile` were not recorded | A LOAD of a hybrid model under another radix-cache setting passed | Recorded | `sgl/test_runtime_contracts.py` |
| Launcher | A record resolved in a process that had not loaded the plugin (sglang's `HttpServerEngineAdapter`, `http_server.launch_server` in a child) reached the schedulers without Foundry's pins and refusals; a second engine in one process kept the first config's mode and preload environment | SAVE with custom all-reduce enabled, then an illegal address at the first LOAD replay; a SAVE engine after a LOAD engine aborted at its first graph save | The spawn site refuses a record without Foundry's resolution declaration, and a config variable that differs from the one the process activated with | `sgl/test_plugin_route.py` |
| Preloads | A library the launcher already preloaded (the documented shell export of the udev-wait shim) stayed after the hook, and the hook's `dlsym` called the real `dlsym` itself, so glibc resolved `RTLD_NEXT` and `RTLD_DEFAULT` from the hook's position | A preloaded wrapper after the hook got its own function back from `dlsym(RTLD_NEXT, ...)` and called itself; no failing run known | Foundry's libraries go first; on x86-64 `dlsym` is an assembly entry point that jumps into the real `dlsym` with the caller's return address, so every lookup the hook does not intercept resolves as without the hook (other architectures keep the C++ forwarding with its two corrections) | `sgl/test_runtime_contracts.py`, `core/test_hook_dlsym.py` |
| Default region | The first `cuCtxCreate*` of a hooked process gave its thread a 2 GiB region at `0x500000000000`, replacing a region the thread already had | In children of a Foundry launcher, a thread with its own context (`NCCL_CREATE_THREAD_CONTEXT=1`, CuTe DSL `initialize_cuda_context`) got memory mapped for its device only; no caller in sglang | Only with `FOUNDRY_CTX_DEFAULT_REGION=1`; never replaces a region | `core/test_allocation_region.py` |
| Archive check | Torch symmetric-memory and multicast pointers (reservations below the layout start) were reported as memory LOAD would not have | False reports of `archive_check` | `region_layout.json` records the reservations; they are covered | - |
| FlashInfer CuTe cache | The cache patch also replaced FlashInfer's SM12x PTX check, and a cache miss under `FLASHINFER_DISABLE_JIT` failed the GDN prefill | Kernel errors off SM90; a prefill failure native sglang does not have | Installed on SM90 only; a cache miss compiles | `unit/test_sglang_flashinfer_cute_cache.py` |
| Preflight | A bare-name `hook_library_path` failed preflight; the LOAD check read one rank and passed archives that LOAD refuses (no layout or fingerprint, another capture-loop version, mixed SAVE ids) | Preflight results that differ from the engine's | A bare name is resolved by ld.so's list mode for the interpreter the schedulers run; every rank directory checked | `sgl/test_preflight.py` |

## Archives from before capture_loop_version 3 (RULE, 2026-10-29)

`capture_loop_version` 3 adds the warm-up forward at each capture scope's first shape, on SAVE and LOAD, which
changes the allocation layout. LOAD refuses version 2 archives at setup: `Foundry archive ... has
capture_loop_version=2, this code needs 3: re-SAVE with the current code`.

## Archives from before capture_loop_version 2 (RULE, 2026-09-30)

SAVE and LOAD now both run sglang's own decode capture loop and substitute only `FullCudaGraphBackend.capture_one`;
the old code ran a FlashInfer metadata pre-pass before the loop, which allocates the per-bs wrappers in another
order, and LOAD cannot replay that layout. `region_layout.json` (per rank) records `capture_loop_version = 2`
(`runtime.CAPTURE_LOOP_VERSION`); LOAD checks it at setup, before the weights load, and refuses an archive without it
or with another value: `Foundry archive ... has capture_loop_version=None, this code needs 2: re-SAVE with the
current code`. Rule: re-SAVE every archive saved by the old code.

## DeepEP v2 (NCCL windows + GIN) not validated on the plugin route (TODO, 2026-09-30)

`--moe-a2a-backend deepep_v2` was validated on the earlier fork route only (H100 EP2 / EP4, all 256 decode graphs,
foundry e24186b). On the plugin route (upstream main) it has not run: the plugin leaves `NCCL_CUMEM_ENABLE` to
sglang for v2 (its NCCL windows need cuMem), and that exception is the one pin without a measurement behind it.

Two attempts on the bare 4xH200 radix hosts failed before any Foundry code ran:

- `sgl-deep-ep` 0.1.2 is built against NCCL 2.30.7 while torch pins 2.29.7 (`NCCL library version is too old`);
  fixed for the run with `uv pip install --no-deps nvidia-nccl-cu13==2.30.7` plus the CUDA 13.3 compat libcuda.
- With NCCL 2.30.7, plain sglang asserts at the first MoE dispatch: `NCCL GIN is unavailable` (`ElasticBuffer.__init__`,
  `csrc/kernels/backend/nccl.cu:87`). v2 requires NCCL GIN even for a single-node EP4, i.e. an RDMA NIC visible to
  the process. These hosts are 4-GPU slices of 8xH200 nodes with IB disabled, so v2 cannot start there at all.

What to check once an RDMA-capable host is available (Qwen3-30B-A3B-FP8 EP4, decode graphs; sglang forces prefill
graphs off for v2): the `[Foundry] pin:` line saying `NCCL_CUMEM_ENABLE` was left to sglang; the ElasticBuffer
bootstrap time on SAVE and LOAD; `Loaded 128 SGLang graphs in`; SAVE `final_alloc_offset` == LOAD
`after_load_all_graphs`; no `[HOOK] WARNING: cuMemAddressReserve returned address ... != hint` (NCCL cuMem
allocations made outside the hook's thread-local region would show up there); greedy parity and TPOT vs native.
Why it is expected to work: v2's windows are library-owned physical memory at a hook-carved VA, registered at the
pre-capture bootstrap on both SAVE and LOAD, the same model as DeepEP v1's NVSHMEM heap; the
`NCCL_GRAPH_REGISTER=0` / `NCCL_LOCAL_REGISTER=0` / `NCCL_NVLS_ENABLE=0` pins stay in force.

## Event-record nodes (TODO, 2026-10-02)

**Why SGLang has an in-graph event.** The overlap scheduler writes batch N+1's shared buffers (`req_to_token`,
`seq_lens`, `positions`, ...) on `schedule_stream` while batch N's forward still reads them on `forward_stream`.
`Scheduler._apply_war_barrier` makes `schedule_stream` wait on an event the runner publishes
(`runner.shared_read_done_event`); with no event it falls back to a coarse `wait_stream(forward_stream)`. Each
attention backend declares where its reads of the shared data end (`SharedReadEnds` in `base_attn_backend.py`):

| value | the reads end ... | event published |
|---|---|---|
| `PRE_REPLAY` | before replay: graph-external metadata prep (`init_forward_metadata_out_graph`) copied everything | recorded before `replay` |
| `IN_REPLAY` | inside the graph, up to a marker | the in-graph marker event |
| `POST_REPLAY` | after the whole replay | recorded after `replay` |
| `UNKNOWN` | not audited | none (coarse `wait_stream`) |

The base backend declares `IN_REPLAY` for decode and target-verify; DeepSeek-V4 inherits it for decode and declares it
for DSPARK verify. The `IN_REPLAY` marker is an event-record node inside the captured graph:
`DecodeCudaGraphRunner._record_in_graph_metadata_prep_done` records an external event (`make_external_event`, i.e.
`torch.cuda.Event(external=True)`) right after `attn_backend.init_forward_metadata_in_graph` in `run_once`, near the
start of the graph. The runner keeps it in `self.in_graph_metadata_prep_done` (one event per runner, created at the
first capture) and `_publish_read_done(in_graph=True)` hands it to the scheduler, which can then overwrite the shared
buffers while the rest of the forward still runs.

**What LOAD loses.** LOAD never runs `run_once` (patch 3 in [`hooks.md`](hooks.md) swaps the capture for the
archived graph), so `runner.in_graph_metadata_prep_done` stays `None`. The archive does not help either. SAVE
serializes event-record / event-wait nodes as `EventRecordNode` / `EventWaitNode` with an `event_id` numbered per
graph in first-seen order (`event_to_id` in `CUDAGraph.cpp`); LOAD creates a fresh private `CUevent` per id
(`cuEventCreate`, kept in `loaded_graph_resources_->created_events`) and adds the node with it. A restored graph
therefore records into an event that only Foundry holds: the link from the node to the host object
(`in_graph_metadata_prep_done`) is not stored, and the id is local to one graph, while SGLang shares one event across
all of a runner's graphs. (Not checked against a SGLang archive for this particular node; the statement is
from the SAVE / LOAD code, which handles every event node this way.)

Upstream's fallback for "`IN_REPLAY` declared, no marker" is `PRE_REPLAY`, which fences too early: the scheduler may
overwrite buffers the graph is still about to read (upstream's own TODO in `_resolve_shared_read_ends` says
`POST_REPLAY` is the sound one). Foundry's `shared_read_ends_override` (`integration/sglang/hooks.py`, called from
`_resolve_shared_read_ends`; patch 3d in [`hooks.md`](hooks.md)) therefore returns `POST_REPLAY` on LOAD in that case
on the plugin route. On the dependency route upstream's own fallback is already `POST_REPLAY`, so no Foundry call is
needed there.
That is correct, but the scheduler's writes for batch N+1 now wait for the whole forward of batch N, so the overlap the
marker buys is lost. The loss was not measurable in our bs 1-128 TPOT runs; it may matter for small-batch, CPU-bound
serving, where the scheduler's next-batch preparation is a large share of the step (not measured).

**TODO (next phase).** Give events a stable identity across the archive instead of a per-graph index: SAVE records,
per event node, an id for the host event (shared by every graph that records it) and, where the integration registers
one, which host object it is. LOAD creates one event per id, adds the nodes with it, and hands the event back to the
integration (SGLang: set `runner.in_graph_metadata_prep_done`). Then the plugin route's `shared_read_ends_override`
patch can go, and upstream's own `IN_REPLAY` path runs unchanged. The generic mechanism
is "graph nodes that reference host objects": events now, conditional handles and child-graph handles later.

## SAVE spends ~30 s in `Init torch distributed` (and ~20 s more elsewhere) copying over-read fatbins (FIXED, 2026-09-25)

SAVE only (LOAD skips fatbin processing). The hook sized each registered fatbin by walking every following
container of the library's `.nv_fatbin` section (`compute_fatbin_size`), so each image was read to the end of the
section: NCCL's 127 MB linked library as 5.95 GB, a 6200-byte cuBLASLt image as 140 MB. SAVE copied and CRC-64'd
7.8 GB per rank during NCCL init (30 s of a 31.5 s dist init) and 14-15 GB / 51-56 s per rank over the run, and
the pack wrote the tails into `fatbin_image_packed.img`. Fixed by sizing a fatbin by its own header
(`header_size + size`). Qwen3-30B-A3B ep4, MAXBS=16, 4xH200: dist init 32 -> 1.6-2.7 s, weight load 4.1 -> 0.26 s,
SAVE health 106 -> 45 s (= native graph), archive 16.7 GB -> 329 MB; parity and every kernel lookup unchanged.
Archives saved before the fix still load (LOAD does not recompute hashes). Details:
`claude-doc/report_coldstart/h200_8gpu_venv/fatbin_overread_fix.md`.

## Bare-host DeepEP engines start ~40 s late: rdma-core's udev wait on blocked verbs devices (FIXED with a shim, 2026-09-25)

Not a Foundry bug, but it hits SAVE and LOAD as hard as the native engines: on a host where
`/dev/infiniband/uverbs*` exist but cannot be opened, DeepEP low-latency's forced IBGDA transport makes
rdma-core wait up to 5 s per HCA for udev in the first DeepEP buffer creation (the first captured shape
natively, the pre-capture bootstrap on SAVE/LOAD). Restore time is unaffected. Cause, evidence and the
`LD_PRELOAD` shim that removes it: [../bare-host-verbs-udev-wait.md](../bare-host-verbs-udev-wait.md)
(`tools/host/no_cdev_wait.c`).

## Restored EP graphs fault at the first decode on multicast-capable hosts (FIXED, 2026-09-23)

**Symptom.** Qwen3-30B-A3B EP2 (DP attention + DeepEP low-latency + DeepGEMM) on an 8xH200 host:
SAVE completes, LOAD restores all 256 graphs with every offset equal to SAVE's, then the first
decode replay dies with `CUDA error: an illegal memory access`. The same code, and even the exact
stack that had passed this row on maui on 2026-09-02, fails identically on this host; the row
passes on the H100 radix hosts.

**Cause.** sglang's `LogitsProcessor` gathers the TP-sharded logits through
`MultimemAllGatherer` (`triton_symm_mem_ag.py`), whose torch symmetric-memory buffer, multicast
mapping and signal pads are built lazily on the first *eager* forward and skipped under capture
(NCCL all-gather fallback). Whether multicast is granted is a host property (IMEX / fabric):
maui and the H100 hosts log `multimem all-gather disabled (no multicast for world_size=2)`, this
H200 grants it. Foundry's SAVE ran an eager warmup pass before capture, so on this host the
gatherer was built during the pass and every graph baked in `_all_gather_kernel_inner(multicast
pointer, signal pads)` and a `memcpy_triton_kernel` reading the symmetric buffer. Torch symmetric
memory bypasses `cudaMalloc`, so the hook never recorded those allocations, they were absent from
the live ranges, and LOAD never recreated them. Verified causally: the same SAVE with
`TORCH_SYMM_MEM_DISABLE_MULTICAST=1` restores and serves correctly (archive then holds the NCCL
all-gather instead).

**Fix.** Two changes, both following the rule that state sglang builds lazily on the first
forward must be created at the same sequence point on SAVE and LOAD: (1) SAVE no longer runs an
eager warmup pass; the only pre-capture work is the two one-time initializations that stream
capture rejects (inductor's lazy init, DeepGEMM's runtime init), run with the allocation region
suspended, so the model's compiles and JIT loads happen inside the captured forward and are
recorded; (2) `bootstrap_logits_gatherer` builds the gatherer's state before capture on both
modes, so the graphs capture the multicast gather exactly as native sglang does and it lands at
the same addresses on LOAD. Validated: identical output, TPOT within 1.3% of native capture.
`FOUNDRY_SGLANG_CHECK_ARCHIVE=1` (or `python -m foundry.integration.sglang.archive_check`)
reports any graph pointer into the region that no recorded range covers, which names the next
such buffer immediately.

## LOAD-mode first-token corruption on the EP / dp-attention path (FIXED)

**Symptom.** On a foundry-LOADed EP server (Qwen3-30B-A3B, EP=2,
fa3 + dp-attention + DeepEP low-latency), some prompts deterministically get
a wrong *first* generated token at temperature 0 — e.g. `" zeroes"` before
`<think>` — after which generation continues coherently and correctly.
Different prompts get different (but per-prompt deterministic) stray tokens;
many prompts are unaffected. The very first request served by a fresh LOAD
process is clean; corruption appears on later requests as a function of the
per-rank request history. Logprobs show this is not a near-tie flip: the
stray token has logprob ≈ −0.0001 with the correct token ~9 nats down and a
garbage tail — the last-position prefill logits are confidently *wrong*, not
perturbed.

**Ruled out (all verified experimentally, 2026-09-01, H200 ×2, GPUs 0/2):**

- *Two-pass vs single-pass save*: byte-identical archives and byte-identical
  (mis)behavior. Single-pass save is sufficient for sglang — the SAVE path
  has no pass-conditional logic (warmup_state.json stores only the memory
  pool config + final alloc offset, both consumed by LOAD only).
- *Foundry-mode environment* (NCCL_CUMEM/NVLS pins, kernel_warmup
  suppression, capture machinery): a foundry SAVE-mode server answers the
  probe prompts byte-identically to a no-foundry baseline.
- *VMM layout divergence*: per-rank labeled alloc offsets and
  final_alloc_offset match SAVE exactly on LOAD (DP0=137308930048,
  DP1=123247525888 on both sides).
- *The restored decode graphs themselves*: decode continues coherently and
  correctly after the corrupted first token; a 6-request run of one prompt
  replays consistently.
- *The server startup warmup request*: with `--skip-server-warmup` the first
  request is clean, but the corruption then appears from the second request
  on — the warmup was merely the "prior request" in default runs.

**Established mechanism (partially pinned).** Some early forward on the
LOAD path (dp-attention idle/companion batches are the prime suspect)
consumes `attn_backend.forward_metadata` *without initializing it first*:
setting it to `None` after the load-time fa3 metadata pre-pass turns the
corruption into a CUDA illegal-memory-access on the first decode batch
(async, surfaces at `copy_done.synchronize()`), proving the leftover is
consumed. On the normal path the leftover is the pre-pass's last-bs decode
metadata, whose contents (built from initial buffer values, no warmup
forward) differ from what upstream capture would have left — wrong
attention state → confidently wrong last-position logits, entangled with
the other rank via the dp-attention gather.

*Update (capture_loop_version 2).* The leftover described above belongs to the
old LOAD path, which ran a post-load fa3 metadata pre-pass after
`load_all_graphs`. LOAD now runs sglang's own decode capture loop (only
`capture_one` is substituted), so `attn_backend.forward_metadata` after LOAD
is whatever that loop leaves, the same as on SAVE and native sglang; the
"pre-pass's last-bs decode metadata" leftover no longer exists on LOAD. This
dp-attention symptom has not been re-tested on the new path (the DeepEP
buffer fix below resolved it on the old one).

The plain-DP (no dp-attention) flashinfer path shows a much milder analog
(token-trajectory divergence from baseline on 1/4 probe prompts, no stray
tokens) — not yet characterized under the same first-vs-later-request lens.

**Note.** This is not new to the sglang-0.5.12 hook port: the previous
integration also populated fa3 metadata post-load without the capture-time
warmup forwards. Earlier validations measured throughput and coherence,
not token-level parity, so this went unnoticed.

**Deep-dive round 2 (2026-09-01, instrumented; all sglang instrumentation
reverted afterwards).** The earlier metadata theory is REFUTED; the full
localization chain, each step measured:

- The serving rank's prefill runs with CORRECT attention metadata
  (max_seq_len_k == prompt length) and per-layer last-token hidden norms are
  IDENTICAL between a clean and a dirty request through all 48 layers.
- The dp-attention hidden gather is correct: right variant (SUM_LEN /
  all-reduce with pre-zeroed buffer), correct offsets (get_dp_local_info),
  and the all-reduce output norm is identical on both ranks and equal to the
  clean value. Logits-shard assembly offsets are correct too.
- The corruption materializes between the LM-head matmul and the sampled
  token: the logits processor already emits argmax=" zeroes" on dirty
  requests. Serving-rank asymmetry: dp_rank 0 requests are always clean;
  dp_rank 1 requests are corrupted (prompt-dependent visibility).
- NOT an execution-order race: persists under CUDA_LAUNCH_BLOCKING=1.
- NOT torch-cache aliasing alone: `torch.cuda.empty_cache()` after graph
  load does not fix it (kept in hooks as hygiene regardless).
- ADDRESS-DISPLACEMENT SENSITIVE (smoking gun): planting a 64 MB tensor at
  the exact region boundary (final_alloc_offset, 0x601cb2200000 on the dirty
  rank) makes ALL requests clean — and the canary itself records ZERO
  corrupted bytes. So nothing writes into that band; displacing runtime
  eager allocations off their default addresses removes the corruption.
  Conclusion: some component holds a STALE INIT-TIME POINTER to an address
  in the early eager zone; when the runtime allocator reuses that address
  for a live tensor (logits-path tensors by default), the stale reference
  corrupts it. One-sided (NVSHMEM/DeepEP) access explains the
  launch-blocking immunity and the sporadic illegal accesses.

**Prime suspect / exact-divergence hypothesis:** the DeepEP buffer address
differs between SAVE and LOAD. On SAVE it is created mid-warmup-pass
(interleaved with activation allocations); on LOAD, `bootstrap_deepep_buffer`
creates it in isolation → different VMM address for the same object, while
`preallocate_for_load_mode` jumps the cursor to final_alloc_offset and MASKS
the order divergence. Restored graphs and the peer rank's one-sided ops
reference the SAVE address; the runtime eager path uses the LOAD address.

**RESOLUTION (confirmed 2026-09-01).** The exact divergence: the DeepEP
buffer was created at different VMM addresses on SAVE (lazily, mid-warmup
pass, after rank-asymmetric JIT/activation allocations) vs LOAD
(bootstrap_deepep_buffer, in isolation). Cursor logs prove it: with the fix,
both modes create the buffer at 120468799488 -> 122398179328 on both ranks;
before the fix, LOAD's bootstrap landed at the 122398179328 point while
SAVE's warmup-lazy creation landed elsewhere (rank-dependent). One-sided
NVSHMEM traffic and graph-referenced addresses used the SAVE-time address
while the runtime used the LOAD-time one; whichever eager tensor later
reused the stale VA got corrupted (logits-path tensors by default).

**Fix** (hooks.py, no dummy forwards): run `bootstrap_deepep_buffer` BEFORE
the SAVE-side warmup pass, so the buffer is created at the same
allocation-sequence point in both modes and its address matches by
construction. Archives must be RE-SAVED (baked addresses change).
`torch.cuda.empty_cache()` after graph load was also added as allocator
hygiene. Validated: the repro sequence is fully clean, first token
`<think>` at logprob ~0.0, DP2 regression clean.

**Follow-up.** Add a token-parity check (fixed prompt set, temp 0,
logprobs) to the validation recipe so LOAD-vs-baseline divergence is caught
routinely.

Repro: `experimental/expert-parallel/serve_sglang_qwen_30b_a3b.sh 2 --load`,
then two chat requests, temp 0: `"What is 15 - 6? Answer briefly."` — the
second (or the first, if the server warmup ran) starts with `" zeroes"`.
