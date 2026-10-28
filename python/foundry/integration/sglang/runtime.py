# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state and VMM setup for the Foundry SGLang integration."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from foundry import ops as cge
from foundry.allocation_region import parse_size
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    attention_dp_active,
    compute_workspace_rank,
    dp_replica,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
    get_verbs_udev_wait_shim_path,
)
from foundry.integration.sglang.nvtx import nvtx_traced

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    sglang_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    memory_pool_config: dict = field(default_factory=dict)
    # Runtime-context overrides the memory-pool resolver made on SAVE
    # ([source, {field: value}] entries, e.g. the Mamba cache size of hybrid
    # models); LOAD skips the resolver and replays them instead.
    context_overrides: list = field(default_factory=list)
    final_alloc_offset: int = 0


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    rank: int = 0
    loaded_graphs: dict = field(default_factory=dict)
    # Set by the first runner capture (prefill when prefill graphs are on,
    # decode otherwise): the pre-capture bootstraps and the layout start run
    # there once, and LOAD's preallocation right after them.
    layout_started: bool = False
    preallocated: bool = False
    # Whether this process restored (LOAD) prefill graphs.
    prefill_graphs_restored: bool = False


_state: CUDAGraphExtensionState | None = None

# Per-rank record of the deterministic allocation layout, written at the end
# of SAVE and consumed by LOAD's preallocation (see record_region_layout).
_LAYOUT_FILE = "region_layout.json"
# Version of the allocation sequence between the layout start and the end of
# the decode capture, recorded in _LAYOUT_FILE on SAVE and required on LOAD.
# 2: both modes run sglang's own decode capture loop and substitute only the
# capture (1, unmarked: a FlashInfer metadata pre-pass before the loop, which
# allocates the per-bs wrappers in another order; LOAD cannot replay it).
CAPTURE_LOOP_VERSION = 2
_layout_start_offset: int | None = None
_final_alloc_offset: int = 0
# Per-rank record of what SAVE ran with, written at SAVE's setup and compared at
# LOAD's (check_archive_fingerprint): the server arguments below, which shape the
# captured graphs or size the buffers they address, the variables in
# _FINGERPRINT_ENV, the software versions and the GPU.
_FINGERPRINT_FILE = "fingerprint.json"
_FINGERPRINT_ARGS = (
    "model_path", "revision", "dtype", "quantization", "kv_cache_dtype", "context_length",
    "tp_size", "pp_size", "dp_size", "attn_dp_size", "attn_cp_size", "ep_size", "moe_dp_size",
    "nnodes", "enable_dp_attention", "enable_dp_lm_head", "moe_dense_tp_size",
    "ep_num_redundant_experts", "attention_backend", "prefill_attention_backend",
    "decode_attention_backend", "moe_a2a_backend", "deepep_mode", "moe_runner_backend",
    "enable_torch_symm_mem", "disable_custom_all_reduce", "enable_two_batch_overlap",
    "enable_single_batch_overlap", "page_size", "max_running_requests", "chunked_prefill_size",
    "max_prefill_tokens", "cuda_graph_config", "cuda_graph_bs_decode", "cuda_graph_max_bs_decode",
    "cuda_graph_backend_prefill", "cuda_graph_bs_prefill", "disable_cuda_graph_padding",
    "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
    "speculative_num_draft_tokens", "speculative_draft_model_path", "enable_lora",
    "max_loras_per_batch",
)  # fmt: skip
_FINGERPRINT_ENV = ("CUDA_MODULE_LOADING", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
# Where weight loading starts: the scratch boundary, or (SAVE) the cursor when
# distributed init allocated past it. Recorded in _LAYOUT_FILE; LOAD moves to it.
_scratch_end_offset: int | None = None


def get_state() -> CUDAGraphExtensionState | None:
    return _state


def _workspace_dir() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_dir


def _workspace_root() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_root


def create_warmup_state(
    memory_pool_config: dict | None = None, context_overrides: list | None = None
) -> WarmupState:
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"

    props = torch.cuda.get_device_properties(0)
    return WarmupState(
        sglang_version=sglang_version,
        timestamp=datetime.now().isoformat(),
        cuda_version=torch.version.cuda or "unknown",
        gpu_name=props.name,
        gpu_total_memory=props.total_memory,
        memory_pool_config=memory_pool_config or {},
        context_overrides=list(context_overrides or []),
    )


def save_warmup_state(state: WarmupState) -> None:
    """Write the rank's warmup state into its own workspace dir, plus the
    workspace-root copy (written once, by whichever rank gets there first).

    The memory pool config is per rank: ranks on the same node can see
    different free memory (the launcher / DP controller context sits on the
    first visible GPU), so a single shared file forced the wrong KV pool size
    onto the other rank on LOAD and its restored graphs read a pool shifted by
    that difference (elastic-EP recover, 2026-09-07)."""
    workspace_root = _workspace_root()
    if workspace_root is None:
        return
    payload = asdict(state)
    rank_dir = _workspace_dir()
    if rank_dir is not None:
        os.makedirs(rank_dir, exist_ok=True)
        rank_path = os.path.join(rank_dir, "warmup_state.json")
        with open(rank_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("[Foundry] Saved SGLang warmup state to %s", rank_path)
    ext_state = get_state()
    path = os.path.join(workspace_root, "warmup_state.json")
    if ext_state is not None and ext_state.rank != 0 and os.path.exists(path):
        return
    os.makedirs(workspace_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[Foundry] Saved SGLang warmup state to %s", path)


def load_warmup_state() -> WarmupState:
    workspace_root = _workspace_root()
    if workspace_root is None:
        raise RuntimeError("Foundry workspace_root is not initialized")
    path = os.path.join(workspace_root, "warmup_state.json")
    rank_dir = _workspace_dir()
    if rank_dir is not None and os.path.exists(os.path.join(rank_dir, "warmup_state.json")):
        path = os.path.join(rank_dir, "warmup_state.json")  # per-rank state wins
    if not os.path.exists(path):
        raise RuntimeError(f"Foundry warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


def _model_identity(model_path) -> str:
    """The digest of config.json for a local model directory (the path may
    differ between hosts), else the path or hub id as given."""
    config = os.path.join(str(model_path), "config.json")
    if os.path.isfile(config):
        with open(config, "rb") as f:
            return "config.json sha256:" + hashlib.sha256(f.read()).hexdigest()
    return str(model_path)


def _driver_version() -> str | None:
    """The kernel driver version (NVML), else the driver API version."""
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


def _full_server_args(server_args):
    """The operator's ServerArgs record. The plugin route's setup receives the
    parallel bag (get_parallel()), which lacks most fields; sglang's runtime
    context holds the whole record."""
    try:
        from sglang.srt.runtime_context import get_server_args

        full = get_server_args()
    except Exception:
        full = None
    return server_args if full is None else full


def archive_fingerprint(server_args) -> dict:
    """What an archive is valid for (see _FINGERPRINT_ARGS), as JSON values."""
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"
    from foundry import __version__ as foundry_version

    server_args = _full_server_args(server_args)
    fp = {name: getattr(server_args, name, None) for name in _FINGERPRINT_ARGS}
    fp["model_path"] = _model_identity(fp["model_path"])
    fp.update({f"env {name}": os.environ.get(name) for name in _FINGERPRINT_ENV})
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    fp.update(
        {
            "sglang version": sglang_version,
            "foundry version": foundry_version,
            "torch version": torch.__version__,
            "cuda version": torch.version.cuda,
            "driver version": _driver_version(),
            "gpu": props.name,
            "gpu compute capability": f"{props.major}.{props.minor}",
            "gpu SMs": props.multi_processor_count,
            "gpu memory": props.total_memory,
        }
    )
    return json.loads(json.dumps(fp, default=_json_value))


def _json_value(value) -> str:
    # An object whose str() is the default repr would differ between runs.
    text = str(value)
    return type(value).__qualname__ if " at 0x" in text else text


def check_archive_fingerprint(server_args) -> None:
    """LOAD, at setup (before the weights load): refuse an archive that SAVE
    wrote under other settings, software or GPU (archive_fingerprint): its
    graphs address buffers whose sizes and offsets those decide, and its
    kernels were selected for SAVE's GPU. An archive written before the
    fingerprint existed is accepted; FOUNDRY_SGLANG_SKIP_FINGERPRINT=1 skips
    the check."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD or cfg.workspace_dir is None:
        return
    path = os.path.join(cfg.workspace_dir, _FINGERPRINT_FILE)
    if not os.path.exists(path) or os.environ.get("FOUNDRY_SGLANG_SKIP_FINGERPRINT") == "1":
        logger.warning("[Foundry] SGLang archive fingerprint not checked (%s)", path)
        return
    with open(path) as f:
        saved = json.load(f)
    current = archive_fingerprint(server_args)
    diff = [
        f"{key}: SAVE {saved.get(key)!r}, LOAD {current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if diff:
        raise RuntimeError(
            f"Foundry LOAD: archive {cfg.workspace_dir} was saved under other settings "
            f"({'; '.join(diff)}). Serve with SAVE's settings or re-SAVE "
            "(FOUNDRY_SGLANG_SKIP_FINGERPRINT=1 skips this check)."
        )
    logger.info("[Foundry] SGLang archive fingerprint matches (%d entries)", len(current))


@nvtx_traced("foundry.setup_graph_ext (binary restore)")
def setup_graph_extension(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> None:
    """Set up the VMM region before SGLang initializes NCCL/process groups."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return

    t0 = time.perf_counter()
    rank = compute_workspace_rank(server_args, tp_rank, pp_rank, dp_rank)
    Path(cfg.workspace_root).mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    logger.info(
        "[Foundry] SGLang rank=%d (tp_rank=%d pp_rank=%d dp_rank=%s replica=%d "
        "attention_dp=%s) workspace_dir=%s",
        rank,
        tp_rank,
        pp_rank,
        dp_rank,
        dp_replica(server_args, dp_rank),
        attention_dp_active(server_args),
        workspace_dir,
    )

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        with open(workspace_dir / _FINGERPRINT_FILE, "w") as f:
            json.dump(archive_fingerprint(server_args), f, indent=1)
    t_bin = 0.0
    if cfg.mode == CUDAGraphExtensionMode.LOAD:
        cge.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            raise RuntimeError(f"Foundry workspace for rank {rank} does not exist: {workspace_dir}")
        t1 = time.perf_counter()
        cge.load_cuda_modules_and_libraries(str(workspace_dir))
        t_bin = time.perf_counter() - t1

    region_size = parse_size(cfg.region_size)
    t1 = time.perf_counter()
    cge.set_allocation_region(cfg.base_addr, region_size)
    t_region = time.perf_counter() - t1
    t1 = time.perf_counter()
    _ = torch._C._cuda_getCurrentBlasHandle()
    t_blas = time.perf_counter() - t1
    _state = CUDAGraphExtensionState(rank=rank)
    logger.info(
        "[Foundry] SGLang graph extension setup completed in %.3f s "
        "(binary restore %.3f s, region reserve %.3f s, cuBLAS handle %.3f s)",
        time.perf_counter() - t0,
        t_bin,
        t_region,
        t_blas,
    )


def _saved_scratch_end(workspace_dir: str) -> int | None:
    """LOAD: SAVE's scratch boundary from the archive (None: an archive written
    before it was recorded)."""
    path = os.path.join(workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        value = json.load(f).get("scratch_end_offset")
    return None if value is None else int(value)


def skip_to_scratch_boundary() -> None:
    """End of distributed init: move the cursor to the scratch boundary, so the
    weights start at the same offset on SAVE and LOAD however much distributed
    init allocated below it (NCCL, symmetric memory and DeepEP buffers depend on
    the host and the library versions).

    SAVE uses ``scratch_space_size``, or the cursor when distributed init
    allocated past it (with a warning: no margin is left), and records the
    boundary in the archive. LOAD moves to SAVE's boundary, and stops when its
    own distributed init allocated past it: every later buffer would sit above
    the offsets the restored graphs address."""
    global _scratch_end_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = cge.get_current_alloc_offset()
    boundary = scratch
    if cfg.mode == CUDAGraphExtensionMode.LOAD and cfg.workspace_dir is not None:
        saved = _saved_scratch_end(cfg.workspace_dir)
        if saved is None and current > scratch:
            # Archive without the record: SAVE kept its cursor; so does LOAD.
            saved = current
        boundary = scratch if saved is None else saved
        if current > boundary:
            raise RuntimeError(
                f"Foundry LOAD: distributed init allocated {current / 2**20:.0f} MB before the "
                f"weights, past SAVE's scratch boundary at {boundary / 2**20:.0f} MB. Set "
                f"scratch_space_size above {current / 2**20:.0f} MB and re-SAVE."
            )
    elif current > scratch:
        logger.warning(
            "[Foundry] SGLang distributed init allocated %.0f MB, more than scratch_space_size "
            "(%.0f MB): the weights start right after it, and a LOAD whose distributed init "
            "allocates more will refuse the archive. Raise scratch_space_size above it.",
            current / 2**20,
            scratch / 2**20,
        )
        boundary = current
    _scratch_end_offset = boundary
    cge.set_current_alloc_offset(boundary)
    logger.info("[Foundry] SGLang skipped allocator to scratch boundary %d", boundary)


@contextlib.contextmanager
def allocation_region_suspended():
    """Run a block whose device allocations must not touch the deterministic
    layout: inside it the hook passes cuMemAlloc through to the driver (plain
    allocations outside the region, cursor untouched). On exit the torch cache
    is emptied before the region is resumed, so no cached segment from outside
    the region can later be handed to an allocation that belongs inside it."""
    cge.stop_allocation_region()
    try:
        yield
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        cge.resume_allocation_region()


def mark_layout_start() -> None:
    """Begin the deterministic layout; the same sequence point on SAVE and LOAD.

    Both modes first return torch's cached-but-free blocks to the driver, so
    the caching allocator is in the same state (empty) when the layout begins
    and every later allocation takes the same cuMemAlloc path on both sides.
    SAVE then records the cursor as the layout start. Everything SAVE allocates
    after this point is either replayed at recorded absolute offsets (graph
    memory) or allocated at the same offset on both sides (the FlashInfer
    metadata pre-pass); the pre-capture bootstraps run before it on both modes,
    so LOAD's cursor equals the recorded start (a difference is a divergence:
    preallocate_for_load_mode moves a LOAD that is behind up to it and stops
    one that is ahead)."""
    global _layout_start_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.mode != CUDAGraphExtensionMode.SAVE:
        return
    _layout_start_offset = cge.get_current_alloc_offset()
    logger.info("[Foundry] SGLang layout start offset=%d", _layout_start_offset)


@nvtx_traced("foundry.save.record_layout")
def record_region_layout() -> int:
    """SAVE: persist the rank's allocation layout for LOAD.

    ``final_alloc_offset`` is the cursor watermark; ``live_ranges`` are the
    (offset, size) ranges still mapped at this point. LOAD backs only the live
    ranges, so memory SAVE allocated and freed inside the span costs LOAD
    nothing."""
    global _final_alloc_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return 0
    _final_alloc_offset = cge.get_current_alloc_offset()
    if cfg.workspace_dir is not None:
        ranges = cge.get_live_region_ranges()
        layout = {
            "scratch_end_offset": _scratch_end_offset,
            "start_offset": _layout_start_offset,
            "final_alloc_offset": _final_alloc_offset,
            "live_ranges": ranges,
            "capture_loop_version": CAPTURE_LOOP_VERSION,
        }
        with open(os.path.join(cfg.workspace_dir, _LAYOUT_FILE), "w") as f:
            json.dump(layout, f)
        logger.info(
            "[Foundry] SGLang region layout: %d live ranges (%.0f MB) in the region, "
            "%.0f MB of them above the layout start; span to the watermark %.0f MB",
            len(ranges),
            sum(r[1] for r in ranges) / 2**20,
            sum(r[1] for r in ranges if r[0] >= (_layout_start_offset or 0)) / 2**20,
            (_final_alloc_offset - (_layout_start_offset or 0)) / 2**20,
        )
    if cfg.workspace_root is not None:
        warmup_state_path = os.path.join(cfg.workspace_root, "warmup_state.json")
        if os.path.exists(warmup_state_path):
            state = load_warmup_state()
            state.final_alloc_offset = _final_alloc_offset
            save_warmup_state(state)
    logger.info("[Foundry] SGLang final_alloc_offset=%d", _final_alloc_offset)
    if cfg.workspace_dir is not None and os.environ.get("FOUNDRY_SGLANG_CHECK_ARCHIVE") == "1":
        # Opt-in (reads every .cugraph): report graph pointers that LOAD will not have.
        from foundry.integration.sglang import archive_check

        archive_check.report(
            archive_check.check_rank_archive(
                cfg.workspace_dir, cfg.base_addr, parse_size(cfg.region_size)
            )
        )
    return _final_alloc_offset


def check_capture_loop_version() -> None:
    """LOAD, at setup (before the weights load): refuse an archive whose
    allocation sequence this code cannot replay (see CAPTURE_LOOP_VERSION)."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD or cfg.workspace_dir is None:
        return
    path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry region layout not found: {path} (re-SAVE with the current code)"
        )
    with open(path) as f:
        version = json.load(f).get("capture_loop_version")
    if version != CAPTURE_LOOP_VERSION:
        raise RuntimeError(
            f"Foundry archive {cfg.workspace_dir} has capture_loop_version={version}, this "
            f"code needs {CAPTURE_LOOP_VERSION}: re-SAVE with the current code"
        )


@nvtx_traced("foundry.memory_restore")
def preallocate_for_load_mode() -> None:
    """LOAD: back the recorded layout ahead of the graph restore.

    Maps, in one step, every range that was live at the end of SAVE between
    LOAD's current cursor and the watermark, then checks the cursor against
    SAVE's recorded layout start (equal by construction; a difference means
    the allocation sequences diverged: a LOAD that is behind is moved up and
    logged, one that is ahead is an error). Allocations inside the span are
    pointer bumps; a hole is backed on demand by the hook and logged, since it
    too means the sequences diverged."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return
    if cfg.workspace_dir is None:
        raise RuntimeError("Foundry workspace_dir is not initialized")
    path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry region layout not found: {path} (re-save the archive with this version)"
        )
    with open(path) as f:
        layout = json.load(f)
    final = int(layout["final_alloc_offset"])
    start = layout.get("start_offset")
    current = cge.get_current_alloc_offset()
    if final > current:
        ranges = [
            (int(off), int(size))
            for off, size in layout.get("live_ranges", [])
            if off + size > current and off < final
        ]
        t0 = time.perf_counter()
        ok = cge.preallocate_ranges(ranges, final)
        logger.info(
            "[Foundry] LOAD memory restore: %d live ranges (%.0f MB) mapped in %.1f ms",
            len(ranges),
            sum(size for _, size in ranges) / (1 << 20),
            1000 * (time.perf_counter() - t0),
        )
        if not ok:
            free, total = torch.cuda.mem_get_info()
            raise RuntimeError(
                f"Foundry LOAD could not back the {(final - current) / 2**20:.0f} MB layout span "
                f"({free / 2**20:.0f} MB of {total / 2**20:.0f} MB free): LOAD keeps the recorded "
                "kernel images resident and maps the span in one step, so it needs more headroom "
                "than SAVE. Lower --mem-fraction-static (for SAVE and LOAD alike) or capture "
                "fewer graphs."
            )
    if start is None:
        return
    if current == start:
        return
    # The two cursors coincide by construction; a difference means the SAVE and
    # LOAD allocation sequences diverged before the layout start. Moving up
    # keeps later allocations at SAVE's offsets; being ahead cannot be repaired:
    # LOAD's own buffers then occupy offsets the restored graphs use.
    if current > start:
        raise RuntimeError(
            f"Foundry LOAD: the cursor before the graph restore is {current}, past SAVE's layout "
            f"start {start} ({(current - start) / 2**20:.0f} MB): LOAD allocated more than SAVE "
            "before the graphs (compare the 'alloc_offset[...]' lines of both logs for the first "
            "difference). Serve with the settings and environment of the SAVE run, or re-SAVE."
        )
    cge.set_current_alloc_offset(start)
    logger.warning(
        "[Foundry] SGLang LOAD cursor %d != SAVE's layout start %d (moved up to it): the "
        "allocation sequences differ before the graph restore",
        current,
        start,
    )


def log_alloc_offset(label: str) -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    offset = cge.get_current_alloc_offset()
    # Free device memory alongside the region offset: LOAD ends with less free
    # memory than SAVE at identical offsets (resident kernel images, reserved
    # ranges SAVE had already freed), and this trail is how that gap is measured.
    try:
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        free_mb = float("nan")
    logger.info(
        "[Foundry] SGLang alloc_offset[%s]=%d (%.2f MB) free=%.0f MB",
        label,
        offset,
        offset / (1024 * 1024),
        free_mb,
    )


def _set_subprocess_preload_env() -> None:
    """LD_PRELOAD (prepended, never overwritten: the launcher's own entries
    stay), FOUNDRY_MODE (read by the hook library) and the spawn timestamp."""
    current = os.environ.get("LD_PRELOAD", "")
    for path in (get_hook_library_path(), get_nvshmem_host_path(), get_verbs_udev_wait_shim_path()):
        if path and path not in current:
            current = f"{path}:{current}" if current else path
    if current:
        os.environ["LD_PRELOAD"] = current
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())


def setup_ld_preload_env(server_args=None) -> None:
    """Plugin route, spawn sites (launcher before the schedulers, DP
    controller before its TP groups): the preload environment for the
    children, set for the rest of the process, and the plugin's environment
    pins re-asserted (``plugin.apply_env_pins``; with ``server_args`` also
    the all-to-all-dependent ``NCCL_CUMEM_ENABLE``) so every child inherits
    them whichever route loaded Foundry."""
    _set_subprocess_preload_env()
    if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
        from foundry.integration.sglang.plugin import apply_env_pins

        apply_env_pins(server_args)


_SUBPROCESS_ENV_KEYS = ("LD_PRELOAD", "FOUNDRY_MODE", "FOUNDRY_SPAWN_T0_NS")


def _start_resource_tracker() -> None:
    """Start the launcher's multiprocessing resource tracker before the
    preload window: under the spawn start method Python starts it lazily in
    the first ``Process.start()``, which would otherwise be inside the window
    and give the tracker the hook too."""
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()


@contextlib.contextmanager
def subprocess_preload_env():
    """Dependency route (``foundry.integration.sglang.api``): the same
    preload environment, set only around ``Process.start()`` (the child
    copies ``os.environ`` at that point) and restored afterwards, so the
    launcher's other processes do not load the hook."""
    _start_resource_tracker()
    saved = {key: os.environ.get(key) for key in _SUBPROCESS_ENV_KEYS}
    try:
        _set_subprocess_preload_env()
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
