# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state, VMM memory ops, and LD_PRELOAD env setup for the
foundry vLLM integration.

Merged from former ``state.py`` + ``lifecycle.py`` + ``memory.py`` +
``distributed.py``. All process-local globals and the primitives that
touch foundry's C extension live here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import vllm
from vllm.logger import init_logger

from foundry import ops as fops
from foundry.allocation_region import parse_size
from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    _compute_rank_from_parallel_config,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

log = init_logger("vllm.foundry.runtime")


# ---------------------------------------------------------------------------
# Persistent state (WarmupState on disk + process-local globals)
# ---------------------------------------------------------------------------


@dataclass
class WarmupState:
    vllm_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    available_gpu_memory: list[int] = field(default_factory=list)
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0
    gpu_memory_utilization: float = 0.0
    final_alloc_offset: int = 0


_final_alloc_offset: int = 0


def set_final_alloc_offset(v: int) -> None:
    global _final_alloc_offset
    _final_alloc_offset = v


def get_final_alloc_offset() -> int:
    return _final_alloc_offset


def create_warmup_state() -> WarmupState:
    props = torch.cuda.get_device_properties(0)
    return WarmupState(
        vllm_version=vllm.__version__,
        timestamp=datetime.now().isoformat(),
        cuda_version=torch.version.cuda or "unknown",
        gpu_name=props.name,
        gpu_total_memory=props.total_memory,
    )


def save_warmup_state(workspace_root: str, state: WarmupState) -> None:
    path = os.path.join(workspace_root, "warmup_state.json")
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    log.info("Saved warmup state to %s", path)


def load_warmup_state(workspace_root: str) -> WarmupState:
    path = os.path.join(workspace_root, "warmup_state.json")
    if not os.path.exists(path):
        raise RuntimeError(f"Warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Worker-scope state (per-process, set by setup_graph_extension)
# ---------------------------------------------------------------------------


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    # LOAD: strong-reference table keyed by GraphIdentifier. The upstream
    # CUDAGraphWrapper stores `entry.output = weak_ref_tensors(output)`, so
    # without a strong ref here the VMM-backed output tensors returned by
    # `FoundryCUDAGraph.finish_one_graph_load` get GC'd between captures and
    # the next graph replay writes to freed memory → CUDA illegal memory
    # access. Do not remove.
    loaded_graphs: dict = field(default_factory=dict)


_state: CUDAGraphExtensionState | None = None


def get_state() -> CUDAGraphExtensionState | None:
    return _state


# ---------------------------------------------------------------------------
# VMM region setup (called from patched init_worker_distributed_environment)
# ---------------------------------------------------------------------------


def setup_graph_extension(vllm_config: VllmConfig) -> None:
    """VMM region + (LOAD only) CUDA modules. Called BEFORE NCCL."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return

    parallel_config = vllm_config.parallel_config
    t_total = time.perf_counter()
    rank = _compute_rank_from_parallel_config(parallel_config)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    log.info("[foundry] rank=%d workspace_dir=%s", rank, workspace_dir)
    _reset_layout()

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        with open(workspace_dir / _FINGERPRINT_FILE, "w") as f:
            json.dump(archive_fingerprint(vllm_config), f, indent=1)

    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        fops.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            # rank_0 fallback is only safe for single-rank setups.
            # Under TP/PP/DP > 1 each rank has its own device addresses,
            # so loading rank 0's graphs on rank N would produce wrong
            # pointers and likely crash on first replay.
            is_single_rank = (
                parallel_config.tensor_parallel_size == 1
                and parallel_config.pipeline_parallel_size == 1
                and parallel_config.data_parallel_size == 1
                and parallel_config.nnodes == 1
            )
            fallback = Path(cfg.workspace_root) / "rank_0"
            if is_single_rank and fallback.exists():
                log.warning(
                    "Workspace %s missing; falling back to %s (safe only for single-rank setups)",
                    workspace_dir,
                    fallback,
                )
                cfg.workspace_dir = str(fallback)
            else:
                raise RuntimeError(
                    f"Workspace for rank {rank} ({workspace_dir}) does not "
                    f"exist. For multi-rank LOAD, re-run SAVE with the same "
                    f"parallelism topology."
                )
        # Before the archived modules load: an archive from other settings or
        # another GPU is refused with the difference named.
        check_archive(vllm_config)
        log.info("[foundry] LOAD: loading CUDA modules from %s", cfg.workspace_dir)
        t = time.perf_counter()
        fops.load_cuda_modules_and_libraries(cfg.workspace_dir)
        log.info(
            "[foundry TIMING] load_cuda_modules_and_libraries: %.3f s", time.perf_counter() - t
        )

    # set_allocation_region MUST come after load_cuda_modules (which does
    # cuCtxCreate).
    region_size = parse_size(cfg.region_size)
    log.info("[foundry] Setting up VMM region at %s size %s", hex(cfg.base_addr), cfg.region_size)
    fops.set_allocation_region(cfg.base_addr, region_size)
    # NOTE(liuxs): workspace init is in prepare_graph_capture
    # # Bring cuBLAS handle up so its tiny bring-up lands in scratch.
    # _ = torch._C._cuda_getCurrentBlasHandle()

    _state = CUDAGraphExtensionState()
    log.info("[foundry TIMING] setup_graph_extension total: %.3f s", time.perf_counter() - t_total)


def skip_to_scratch_boundary() -> None:
    """Advance allocator past scratch region. Called AFTER NCCL.

    Distributed init allocates a host-dependent amount (NCCL version and
    environment), so weight loading starts at a fixed boundary instead:
    scratch_space_size, or on SAVE the cursor when distributed init went past
    it. SAVE records the boundary it used; LOAD moves to the same one, and
    stops when its own distributed init already went past it."""
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = fops.get_current_alloc_offset()
    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        boundary = max(scratch, current)
        if current > scratch:
            log.warning(
                "[foundry] distributed init allocated %d bytes, more than scratch_space_size "
                "%d: weights start at %d (raise scratch_space_size)",
                current,
                scratch,
                current,
            )
        _layout["scratch_end_offset"] = boundary
    else:
        saved = _saved_layout_value("scratch_end_offset")
        boundary = scratch if saved is None else saved
        if current > boundary:
            raise RuntimeError(
                f"Foundry LOAD: distributed init allocated {current} bytes, past the boundary "
                f"{boundary} where SAVE's weight loading starts (scratch_space_size "
                f"{scratch}): raise scratch_space_size and re-SAVE"
            )
    log.info("[foundry] Skip to scratch boundary: %d bytes (current was %d)", boundary, current)
    fops.set_current_alloc_offset(boundary)
    log_alloc_offset("after_scratch_skip")


# ---------------------------------------------------------------------------
# Archive record: fingerprint (what the archive is valid for) and allocation
# layout (the boundary and the offsets at fixed checkpoints)
# ---------------------------------------------------------------------------

_FINGERPRINT_FILE = "fingerprint.json"
_LAYOUT_FILE = "region_layout.json"
# SAVE: the boundary and the checkpoint offsets of this run; LOAD: SAVE's record.
_layout: dict = {}
_saved_layout: dict | None = None


def _reset_layout() -> None:
    global _layout, _saved_layout
    _layout = {"offsets": {}}
    _saved_layout = None


def _saved_layout_value(key: str):
    return None if _saved_layout is None else _saved_layout.get(key)


def _json_value(value) -> str:
    # An object whose str() is the default repr would differ between runs.
    text = str(value)
    return type(value).__qualname__ if " at 0x" in text else text


def _model_identity(model) -> str | None:
    """The digest of config.json for a local model directory (the path may
    differ between hosts), else the path or hub id as given."""
    if model is None:
        return None
    config = os.path.join(str(model), "config.json")
    if os.path.isfile(config):
        with open(config, "rb") as f:
            return "config.json sha256:" + hashlib.sha256(f.read()).hexdigest()
    return str(model)


def _driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:
        pass
    try:
        return f"driver API {torch._C._cuda_getDriverVersion()}"
    except Exception:
        return None


def archive_fingerprint(vllm_config: VllmConfig) -> dict:
    """What a rank's archive is valid for: the settings that shape the captured
    graphs or size the buffers they address, the software and the GPU."""
    from foundry import __version__ as foundry_version

    cfg = get_config()
    model = vllm_config.model_config
    sched = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    par = vllm_config.parallel_config
    comp = vllm_config.compilation_config
    props = torch.cuda.get_device_properties(torch.cuda.current_device())

    def a(obj, name):
        return getattr(obj, name, None)

    fp = {
        "model": _model_identity(a(model, "model")),
        "revision": a(model, "revision"),
        "dtype": a(model, "dtype"),
        "quantization": a(model, "quantization"),
        "max_model_len": a(model, "max_model_len"),
        "max_num_seqs": a(sched, "max_num_seqs"),
        "max_num_batched_tokens": a(sched, "max_num_batched_tokens"),
        "block_size": a(cache, "block_size"),
        "cache_dtype": a(cache, "cache_dtype"),
        "tensor_parallel_size": a(par, "tensor_parallel_size"),
        "pipeline_parallel_size": a(par, "pipeline_parallel_size"),
        "data_parallel_size": a(par, "data_parallel_size"),
        "enable_expert_parallel": a(par, "enable_expert_parallel"),
        "all2all_backend": a(par, "all2all_backend"),
        "disable_custom_all_reduce": a(par, "disable_custom_all_reduce"),
        "cudagraph_mode": a(comp, "cudagraph_mode"),
        "cudagraph_capture_sizes": a(comp, "cudagraph_capture_sizes"),
        "region base": hex(cfg.base_addr) if cfg is not None else None,
        "region size": parse_size(cfg.region_size) if cfg is not None else None,
        "scratch_space_size": parse_size(cfg.scratch_space_size) if cfg is not None else None,
        "env CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
        "vllm version": vllm.__version__,
        "foundry version": foundry_version,
        "torch version": torch.__version__,
        "cuda version": torch.version.cuda,
        "driver version": _driver_version(),
        "gpu": props.name,
        "gpu compute capability": f"{props.major}.{props.minor}",
        "gpu SMs": props.multi_processor_count,
        "gpu memory": props.total_memory,
    }
    return json.loads(json.dumps(fp, default=_json_value))


def check_archive(vllm_config: VllmConfig) -> None:
    """LOAD, at setup (before the archived modules load): refuse an archive
    saved under other settings, software or GPU (archive_fingerprint), and
    read SAVE's allocation layout. FOUNDRY_VLLM_SKIP_FINGERPRINT=1 skips the
    fingerprint comparison."""
    global _saved_layout
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD or cfg.workspace_dir is None:
        return
    layout_path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(layout_path):
        raise RuntimeError(
            f"Foundry LOAD: {layout_path} not found: the archive was not written by a "
            "completed SAVE (or by an older Foundry); re-SAVE"
        )
    with open(layout_path) as f:
        _saved_layout = json.load(f)
    path = os.path.join(cfg.workspace_dir, _FINGERPRINT_FILE)
    if os.environ.get("FOUNDRY_VLLM_SKIP_FINGERPRINT") == "1":
        log.warning("[foundry] archive fingerprint not checked (FOUNDRY_VLLM_SKIP_FINGERPRINT=1)")
        return
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry LOAD: {path} not found: the archive was written by an older Foundry; "
            "re-SAVE (FOUNDRY_VLLM_SKIP_FINGERPRINT=1 skips this check)"
        )
    with open(path) as f:
        saved = json.load(f)
    current = archive_fingerprint(vllm_config)
    missing = sorted(set(current) - set(saved))
    if missing:
        raise RuntimeError(
            f"Foundry LOAD: archive {cfg.workspace_dir} was saved by an older Foundry, whose "
            f"fingerprint does not record {', '.join(missing)}: re-SAVE "
            "(FOUNDRY_VLLM_SKIP_FINGERPRINT=1 skips this check)."
        )
    diff = [
        f"{key}: SAVE {saved.get(key)!r}, LOAD {current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if diff:
        raise RuntimeError(
            f"Foundry LOAD: archive {cfg.workspace_dir} was saved under other settings "
            f"({'; '.join(diff)}). Serve with SAVE's settings or re-SAVE "
            "(FOUNDRY_VLLM_SKIP_FINGERPRINT=1 skips this check)."
        )
    log.info("[foundry] archive fingerprint matches (%d entries)", len(current))


def log_alloc_offset(label: str) -> None:
    """Checkpoint of the allocation layout: SAVE records the cursor offset, LOAD
    stops at the first checkpoint where it differs from SAVE's (the buffers
    allocated since the previous checkpoint differ in size or number)."""
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    offset = fops.get_current_alloc_offset()
    log.info("[foundry] alloc_offset[%s] = %d", label, offset)
    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        _layout.setdefault("offsets", {})[label] = offset
        return
    saved = (_saved_layout_value("offsets") or {}).get(label)
    if saved is not None and saved != offset:
        raise RuntimeError(
            f"Foundry LOAD: allocation offset at {label} is {offset}, SAVE had {saved}: the "
            "buffers allocated since the previous checkpoint differ (settings, model or "
            "software that differ from SAVE's). Serve with SAVE's settings or re-SAVE."
        )


def record_region_layout() -> None:
    """SAVE, after the last capture: write the rank's layout record."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.SAVE or cfg.workspace_dir is None:
        return
    with open(os.path.join(cfg.workspace_dir, _LAYOUT_FILE), "w") as f:
        json.dump(_layout, f, indent=1)


def capture_final_alloc_offset() -> int:
    """Record per-rank watermark after all graph captures on SAVE."""
    cfg = get_config()
    offset = fops.get_current_alloc_offset()
    set_final_alloc_offset(offset)
    log.info(
        "[foundry] Captured final_alloc_offset: %d bytes (%.2f GB)", offset, offset / (1024**3)
    )
    if cfg is not None and cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        with open(path, "w") as f:
            json.dump({"final_alloc_offset": offset}, f)
        log.info("[foundry] Saved per-rank final_alloc_offset to %s", path)
    return offset


# ---------------------------------------------------------------------------
# VMM preallocation (LOAD only)
# ---------------------------------------------------------------------------


def preallocate_for_load_mode() -> None:
    """One cuMemCreate + cuMemMap covering the deterministic region."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    # Per-rank file preferred (EP), fall back to shared warmup state.
    final = 0
    if cfg.workspace_dir is not None:
        p = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        if os.path.exists(p):
            with open(p) as f:
                final = json.load(f).get("final_alloc_offset", 0)
    if final <= 0 and cfg.workspace_root is not None:
        final = load_warmup_state(cfg.workspace_root).final_alloc_offset
    if final <= 0:
        return

    current = fops.get_current_alloc_offset()
    remaining = final - current
    if remaining <= 0:
        log.info("[foundry] LOAD: no preallocation needed")
        return

    log.info("[foundry] LOAD: preallocating %d bytes (%.2f GB)", remaining, remaining / (1024**3))
    t = time.perf_counter()
    if fops.preallocate_region(remaining):
        log.info("[foundry TIMING] preallocate_region: %.3f s", time.perf_counter() - t)
    else:
        log.warning("[foundry] preallocate_region failed for %d bytes", remaining)


# ---------------------------------------------------------------------------
# Graph-capture workspace pinning (SAVE + LOAD)
# ---------------------------------------------------------------------------


def _preallocate_attention_workspaces(metadata_builders: list | None) -> None:
    try:
        from vllm.v1.attention.backends import flashinfer as fi

        FlashInferMetadataBuilder = fi.FlashInferMetadataBuilder
    except ImportError:
        return
    # Renamed upstream: _get_trtllm_gen_workspace_buffer (≤ foundry branch)
    # → _get_trtllm_workspace_buffer (main, 2026-07).
    get_ws = getattr(fi, "_get_trtllm_workspace_buffer", None) or getattr(
        fi, "_get_trtllm_gen_workspace_buffer", None
    )
    if get_ws is not None:
        get_ws()
    if metadata_builders:
        for b in metadata_builders:
            if isinstance(b, FlashInferMetadataBuilder):
                b._get_workspace_buffer()


def _eager_inductor_lazy_init(device: torch.device) -> None:
    """Pre-fire ``torch._inductor.fx_passes.joint_graph.lazy_init`` outside
    any CUDA-graph capture window.

    The first in-capture compile inside ``capture_model`` triggers
    ``joint_graph.lazy_init`` which calls ``_sfdp_init`` — that builds SDPA
    pattern templates by copying CPU constants onto the device, which is
    illegal inside a capture window:

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        graph capture unless the CPU tensor is pinned.

    Pre-warming retires the lazy init before capture starts.

    Why only ``joint_graph`` and not the other ``lazy_init`` functions in
    ``torch._inductor.fx_passes.*``: ``post_grad.lazy_init`` and friends
    are also ``@init_once_fakemode`` (``@functools.cache``-keyed on
    ``input_device``), but inductor's own ``post_grad_passes`` calls them
    with **no argument** (``input_device=None``). If we pre-warm with the
    indexed ``cuda:N`` device the cache keys don't match, both calls run,
    and they re-register the same patterns:

        torch._inductor.exc.InductorError: Duplicate pattern: amax_default

    Only ``joint_graph.lazy_init`` is dangerous to leave for capture-time
    (its body actually allocates on the device); the others just register
    patterns and are safe to leave to inductor.
    """
    try:
        from torch._inductor.fx_passes.joint_graph import lazy_init
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init not importable: %s", e)
        return
    try:
        lazy_init(device)
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init(%s) skipped: %s", device, e)


def prepare_graph_capture(
    metadata_builders: list | None = None,
    device: torch.device | None = None,
) -> None:
    """Force cuBLAS + attention workspaces to land at deterministic offsets,
    and pre-warm torch._inductor's pattern-matcher lazy init so the first
    in-capture compile doesn't try to copy CPU tensors onto the device.

    ``device`` is the model_runner's device; passing it lets the inductor
    lazy-init cache key match the in-capture compile path.
    """
    import foundry as foundry_pkg

    if device is not None:
        _eager_inductor_lazy_init(device)
    fops.preallocate_cublas_workspaces()
    _preallocate_attention_workspaces(metadata_builders)
    if foundry_pkg.graph.default_capture_stream is None:
        foundry_pkg.graph.default_capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(foundry_pkg.graph.default_capture_stream):
        fops.preallocate_cublas_workspaces()
        _preallocate_attention_workspaces(metadata_builders)


# ---------------------------------------------------------------------------
# Subprocess env setup (LD_PRELOAD + FOUNDRY_MODE consumed by foundry/csrc/hook.cpp)
# ---------------------------------------------------------------------------


def setup_ld_preload_env() -> None:
    """Mutate os.environ so child processes inherit the foundry hook.

    Idempotent, non-restoring — keeps the mutation for the parent's
    lifetime. Called from patched subprocess-spawn sites.

    Also stamps ``FOUNDRY_SPAWN_T0_NS`` right before the child fork so
    the child can log ``spawn → install_hooks`` wall time (dominated by
    Python + torch + vllm cold import with LD_PRELOAD'd hook).
    """
    current = os.environ.get("LD_PRELOAD", "")
    hook = get_hook_library_path()
    if hook and hook not in current:
        current = f"{hook}:{current}" if current else hook
    nvs = get_nvshmem_host_path()
    if nvs and nvs not in current:
        current = f"{nvs}:{current}" if current else nvs
    if current:
        os.environ["LD_PRELOAD"] = current

    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value

    # Early region reserve: on LOAD the hook claims [base, base+region_size)
    # right after context creation, before module loading can trigger the
    # CUDA driver's lazily-placed VA arena. Size must equal region_size
    # exactly so setup_graph_extension's set_allocation_region call is an
    # idempotent no-op against the early reservation.
    cfg = get_config()
    if cfg is not None and mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_EARLY_RESERVE_BASE"] = hex(cfg.base_addr)
        os.environ["FOUNDRY_EARLY_RESERVE_SIZE"] = str(parse_size(cfg.region_size))

    # Overwrite every spawn site — child reads it on entry.
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())

    log.info("[foundry] LD_PRELOAD set (mode=%s)", mode.value)
