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
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from foundry import ops as cge
from foundry.allocation_region import parse_size
from foundry.integration.sglang.config import (
    ARCHIVE_FINGERPRINT_FILE,
    ARCHIVE_LAYOUT_FILE,
    ARCHIVE_SAVE_ID_KEY,
    CAPTURE_LOOP_VERSION,
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

# The archive records and the capture-loop version: config.py.
_LAYOUT_FILE = ARCHIVE_LAYOUT_FILE
_layout_start_offset: int | None = None
_final_alloc_offset: int = 0
# Per-rank record of what SAVE ran with, written at SAVE's setup and compared at
# LOAD's (check_archive_fingerprint): the server arguments below, which shape the
# captured graphs or size the buffers they address, the variables in
# _FINGERPRINT_ENV, the region's base and size, the software versions and the GPU.
_FINGERPRINT_FILE = ARCHIVE_FINGERPRINT_FILE
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
    "max_loras_per_batch", "enable_return_hidden_states", "return_hidden_states_mode",
    "enable_return_aux_hidden_states", "cuda_graph_max_bs_prefill",
    "cuda_graph_max_seq_len_prefill", "cuda_graph_prefill_max_context",
    "enable_deterministic_inference", "rl_on_policy_target",
    "enable_flashinfer_allreduce_fusion", "enforce_disable_flashinfer_allreduce_fusion",
    "flashinfer_allreduce_fusion_backend", "enable_symm_mem", "enable_torch_symm_mem",
    "enable_nccl_nvls", "dcp_size", "dcp_comm_backend", "dcp_replicate_q_proj",
    "enable_fused_qk_norm_rope", "triton_attention_num_kv_splits",
    "triton_attention_split_tile_size", "linear_attn_backend", "linear_attn_decode_backend",
    "linear_attn_prefill_backend", "mamba_backend", "mamba_ssm_dtype", "max_mamba_cache_size",
    "dsa_decode_backend", "dsa_prefill_backend", "dsa_paged_mqa_logits_backend",
    "dsa_topk_backend", "enable_torch_compile", "disable_radix_cache",
    "mamba_radix_cache_strategy", "mamba_track_interval", "disable_overlap_schedule",
    "weight_cache_mode",
)  # fmt: skip
# Variables that select kernels or shape the communicators and buffers the graphs address.
_FINGERPRINT_ENV = (
    "CUDA_MODULE_LOADING", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "SGLANG_NCCL_SO_PATH", "CUBLAS_WORKSPACE_CONFIG", "TORCH_SYMM_MEM_DISABLE_MULTICAST",
    "NCCL_ALGO", "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS",
    "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
    "NCCL_RUNTIME_CONNECT", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", "NVSHMEM_SYMMETRIC_SIZE",
)  # fmt: skip
# Distributions whose device code the archive holds while LOAD's install builds
# the host-side state it reads (communicators, attention plans, DeepEP buffers).
_FINGERPRINT_DISTRIBUTIONS = (
    "flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin", "sglang-kernel",
    "sgl-kernel", "sgl-deep-ep", "deep-ep", "sgl-deep-gemm", "deep-gemm", "triton",
    "nvidia-nccl-cu12", "nvidia-nccl-cu13", "nvidia-nvshmem-cu12", "nvidia-nvshmem-cu13",
)  # fmt: skip
_SAVE_ID_KEY = ARCHIVE_SAVE_ID_KEY
_save_id: str | None = None
# Allocation offsets at every checkpoint from the region's setup on (the
# alloc_offset[...] log lines), distributed init included (the graphs address
# the communicators and symmetric-memory buffers it allocates): SAVE records
# them in _LAYOUT_FILE, LOAD compares every checkpoint it shares with SAVE and
# stops at the first difference.
_TRAIL_FROM = "after_setup_graph_ext"
_offset_trail: dict[str, int] = {}
_trail_started = False
_saved_trail: dict[str, int] | None = None
# Where weight loading starts: the scratch boundary, or (SAVE) the cursor when
# distributed init allocated past it. Recorded in _LAYOUT_FILE; LOAD moves to it.
_scratch_end_offset: int | None = None
# The (offset, size) ranges live below the cursor when distributed init ends:
# SAVE records them, LOAD must have the same (equal offsets can still hide a
# reordered or resized allocation).
_scratch_live_ranges: list | None = None


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


def _model_identity(model_path, revision=None, trust_remote_code=False) -> str:
    """The digest of the model's config: the config class's view of it
    (AutoConfig, as the engine reads it: a checkpoint a trainer re-wrote with
    save_pretrained under another transformers has the same view), else the
    config.json file, from a local directory or the hub cache's snapshot of a
    hub id (one model may be named either way, and its path may differ between
    hosts), else the path or hub id as given; without the keys that name the
    writer either way."""
    try:
        from transformers import AutoConfig

        data = AutoConfig.from_pretrained(
            str(model_path), revision=revision, trust_remote_code=bool(trust_remote_code)
        ).to_dict()
        for key in ("transformers_version", "_name_or_path"):
            data.pop(key, None)
        raw = json.dumps(data, sort_keys=True, default=str).encode()
        return "config sha256:" + hashlib.sha256(raw).hexdigest()
    except Exception:
        pass
    config = os.path.join(str(model_path), "config.json")
    if not os.path.isfile(config):
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(str(model_path), "config.json", revision=revision)
            config = cached if isinstance(cached, str) else None
        except Exception:
            config = None
    if config is None or not os.path.isfile(config):
        return str(model_path)
    with open(config, "rb") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        for key in ("transformers_version", "_name_or_path"):
            data.pop(key, None)
        raw = json.dumps(data, sort_keys=True).encode()
    except ValueError:
        pass
    return "config.json sha256:" + hashlib.sha256(raw).hexdigest()


def _distribution_versions() -> dict:
    from importlib import metadata

    versions = {}
    for name in _FINGERPRINT_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sglang_source_identity() -> str | None:
    """For a git checkout of sglang (an editable install, whose version string
    stays as installed): HEAD and a digest of the uncommitted changes to the
    package's tracked files; None otherwise (the version string identifies a
    wheel)."""
    import subprocess

    try:
        import sglang

        pkg = Path(sglang.__file__).resolve().parent

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(pkg), *args], capture_output=True, text=True, timeout=30
            )

        # The package must be tracked by the repository git finds (a wheel in a
        # venv inside another work tree is not).
        tracked = git("ls-files", "--error-unmatch", "__init__.py")
        if tracked.returncode != 0:
            # Not a work tree at all: a wheel. A work tree git refuses to read
            # (another owner) or cannot answer: not known.
            if "not a git repository" in tracked.stderr or "did not match" in tracked.stderr:
                return None
            return f"unknown ({tracked.stderr.strip()[:80]})"
        head = git("rev-parse", "HEAD").stdout.strip()
        # Tracked files only: files sglang or its JIT caches write into the tree
        # between SAVE and LOAD must not count.
        changes = git("diff", "HEAD", "--", ".").stdout
        return f"{head} changes:{hashlib.sha256(changes.encode()).hexdigest()[:16]}"
    except Exception:
        return None


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
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _device_allocator_settings() -> list:
    """torch's caching-allocator settings that shape device allocations, as
    sorted ``key:value`` items: the pinned_* keys (how pinned host memory is
    obtained) are left out, values are normalized, and either variable
    spelling gives the same list."""
    items, depth, current = [], 0, ""
    for ch in _allocator_settings():
        depth += {"[": 1, "]": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    items.append(current)
    pairs = []
    for item in items:
        key, _, value = item.partition(":")
        key, value = key.strip().lower(), value.strip().replace(" ", "")
        if key and not key.startswith("pinned_"):
            pairs.append(f"{key}:{value.lower() if value.lower() in ('true', 'false') else value}")
    return sorted(pairs)


def _pynccl_library_identity() -> str | None:
    """The NCCL the captured collectives run on: the library sglang's PyNccl
    loads (SGLANG_NCCL_SO_PATH or the installed one), by its reported version
    and, for an explicit path, the file's size."""
    try:
        from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
            NCCLLibrary,
            find_nccl_library,
        )

        path = find_nccl_library()
        identity = f"{NCCLLibrary(path).ncclGetVersion()}"
        explicit = os.environ.get("SGLANG_NCCL_SO_PATH")
        if explicit and os.path.isfile(explicit):
            identity += f" {os.path.getsize(explicit)}B"  # not mtime: pip writes the install time
        return identity
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _full_server_args(server_args):
    """The operator's ServerArgs record, as resolved. The plugin route's setup
    receives the parallel bag (get_parallel()), which lacks most fields;
    sglang's runtime context holds the whole record, whose resolution
    decisions (a backend sglang picked, a pin) its resolving view applies."""
    try:
        from sglang.srt.runtime_context import get_server_args

        full = get_server_args()
    except Exception:
        full = None
    record = server_args if full is None else full
    try:
        from sglang.srt.arg_groups.overrides import resolving_view

        return resolving_view(record)
    except Exception:
        return record


def archive_fingerprint(server_args, ranks: dict | None = None) -> dict:
    """What an archive is valid for (see _FINGERPRINT_ARGS), as JSON values.
    ``ranks``: the writing process's ranks (a rank directory read by another
    rank holds that rank's kernel arguments, e.g. DeepEP's rank)."""
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"
    from foundry import __version__ as foundry_version

    server_args = _full_server_args(server_args)
    fp = {name: getattr(server_args, name, None) for name in _FINGERPRINT_ARGS}
    fp["model_path"] = _model_identity(
        fp["model_path"], fp["revision"], getattr(server_args, "trust_remote_code", False)
    )
    fp.update({f"rank {k}": v for k, v in (ranks or {}).items()})
    fp.update({f"env {name}": os.environ.get(name) for name in _FINGERPRINT_ENV})
    fp.update({f"package {k}": v for k, v in _distribution_versions().items()})
    fp["sglang source"] = _sglang_source_identity()
    fp["device allocator settings"] = _device_allocator_settings()
    try:  # derived from the radix-cache and overlap settings: adds a state copy to every graph
        from sglang.srt.runtime_context import get_exec

        fp["mamba extra buffer"] = bool(get_exec().mamba.enable_mamba_extra_buffer)
    except Exception:
        fp["mamba extra buffer"] = None
    try:
        fp["nccl version"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except Exception:
        fp["nccl version"] = None
    fp["pynccl library"] = _pynccl_library_identity()
    cfg = get_config()
    if cfg is not None:
        # The graphs address the region by absolute address, and the hook places every
        # reservation of 2 GiB or more (the NVSHMEM heap of DeepEP) at base + region_size.
        fp["region base_addr"] = hex(cfg.base_addr)
        fp["region_size"] = parse_size(cfg.region_size)
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


def check_archive_fingerprint(server_args, ranks: dict | None = None) -> None:
    """LOAD, at setup (before the weights load): refuse an archive that SAVE
    wrote under other settings, software or GPU (archive_fingerprint): its
    graphs address buffers whose sizes and offsets those decide, and its
    kernels were selected for SAVE's GPU. FOUNDRY_SGLANG_SKIP_FINGERPRINT=1
    skips the check."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD or cfg.workspace_dir is None:
        return
    path = os.path.join(cfg.workspace_dir, _FINGERPRINT_FILE)
    if os.environ.get("FOUNDRY_SGLANG_SKIP_FINGERPRINT") == "1":
        logger.warning("[Foundry] SGLang archive fingerprint not checked (%s)", path)
        return
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry LOAD: {path} not found: the archive was not written by a SAVE of this "
            "code; re-SAVE (FOUNDRY_SGLANG_SKIP_FINGERPRINT=1 skips this check)"
        )
    with open(path) as f:
        saved = json.load(f)
    save_id = saved.pop(_SAVE_ID_KEY, None)
    layout_path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            layout_id = json.load(f).get(_SAVE_ID_KEY)
        if layout_id != save_id:
            raise RuntimeError(
                f"Foundry LOAD: {path} and {layout_path} come from different SAVE runs "
                f"({save_id} / {layout_id}): another SAVE wrote this rank directory at the same "
                "time, or the archive was assembled from several copies; re-SAVE with one "
                "workspace_root per engine"
            )
    current = archive_fingerprint(server_args, ranks)
    missing = sorted(set(current) - set(saved))
    if missing:
        raise RuntimeError(
            f"Foundry LOAD: archive {cfg.workspace_dir} was saved by an older Foundry, whose "
            f"fingerprint does not record {', '.join(missing)}: re-SAVE "
            "(FOUNDRY_SGLANG_SKIP_FINGERPRINT=1 skips this check)."
        )

    def unknown(value) -> bool:  # a value one side could not determine
        return isinstance(value, str) and value.startswith("unknown (")

    diff = [
        f"{key}: SAVE {saved.get(key)!r}, LOAD {current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
        and not (unknown(saved.get(key)) or unknown(current.get(key)))
    ]
    if diff:
        raise RuntimeError(
            f"Foundry LOAD: archive {cfg.workspace_dir} was saved under other settings "
            f"({'; '.join(diff)}). Serve with SAVE's settings or re-SAVE "
            "(FOUNDRY_SGLANG_SKIP_FINGERPRINT=1 skips this check)."
        )
    not_known = sorted(k for k in current if unknown(saved.get(k)) or unknown(current.get(k)))
    if not_known:
        logger.warning("[Foundry] SGLang archive fingerprint: not compared: %s", not_known)
    logger.info("[Foundry] SGLang archive fingerprint matches (%d entries)", len(current))


def _check_hook_preloaded() -> None:
    """The hook must interpose this process's driver calls: torch's allocations
    reach it only when libcuda_hook.so is preloaded (foundry.ops also links it,
    and that copy intercepts nothing). Without it every buffer gets a
    driver-chosen address and the archive describes none of them."""
    import ctypes

    class _DlInfo(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    process = ctypes.CDLL(None)
    try:
        symbol = ctypes.cast(process.cuMemAlloc_v2, ctypes.c_void_p)
    except AttributeError:
        symbol = None
    where = "no library of the process" if symbol is None else "an unidentified library"
    if symbol is not None:
        for lib in (None, "libdl.so.2"):  # dladdr: in libc since glibc 2.34
            try:
                dladdr = (process if lib is None else ctypes.CDLL(lib)).dladdr
            except (AttributeError, OSError):
                continue
            info = _DlInfo()
            if dladdr(symbol, ctypes.byref(info)) and info.dli_fname:
                where = os.fsdecode(info.dli_fname)
            break
    configured = get_hook_library_path()
    if os.path.basename(where).startswith("libcuda_hook") or (
        configured and os.path.realpath(where) == os.path.realpath(configured)
    ):
        return
    raise RuntimeError(
        f"[Foundry] the CUDA hook is not preloaded in this process: cuMemAlloc_v2 resolves to "
        f"{where} (LD_PRELOAD={os.environ.get('LD_PRELOAD', '')!r}, hook_library_path="
        f"{get_hook_library_path()!r}). Check hook_library_path, and launch through a path "
        "Foundry's spawn sites cover (sglang serve, python -m sglang.launch_server, "
        "sglang.Engine) in a launcher that had the Foundry config before it first loaded "
        "sglang's plugins (once per process)."
    )


def _allocator_settings() -> str:
    for name in ("_accelerator_getAllocatorSettings", "_cuda_getAllocatorSettings"):
        getter = getattr(torch._C, name, None)
        if getter is not None:
            try:
                return str(getter())
            except Exception:
                pass
    return ",".join(
        os.environ.get(k, "") for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    )


def _check_torch_allocator() -> None:
    """Refuse the caching-allocator configurations whose memory the hook does
    not see: expandable segments (torch maps its own VMM memory with
    cuMemCreate / cuMemMap) and the cudaMallocAsync backend (driver pools)."""
    backend = torch.cuda.memory.get_allocator_backend()
    expandable = re.search(r"expandable_segments\s*:\s*true", _allocator_settings(), re.I)
    if backend != "native" or expandable:
        raise RuntimeError(
            f"[Foundry] SGLang save/load needs torch's native caching allocator without "
            f"expandable segments (allocator backend {backend!r}, settings "
            f"{_allocator_settings()!r}): its memory is placed through cuMemAlloc, which "
            "the hook records. Unset expandable_segments / backend:cudaMallocAsync in "
            "PYTORCH_CUDA_ALLOC_CONF."
        )


# Caching-allocator segments that existed outside the region when it was set up
# (allocated before Foundry's setup); _check_tensors_in_region ignores them.
_segments_before_region: set[int] = set()


def check_model_tensors_in_region(model) -> None:
    """Both modes, at the layout start: every CUDA parameter and buffer of the
    model must be in the region, or in a reservation of 2 GiB or more that the
    hook placed at its fixed address after the region (DeepSeek-V4.1 Engram
    peer shards). One elsewhere (weights a loader imported from another
    process, a host table mapped for zero-copy access) has an address that
    differs between SAVE and LOAD, and the restored graphs would read SAVE's."""
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE or model is None:
        return
    base, end = cfg.base_addr, cfg.base_addr + parse_size(cfg.region_size)
    placed = [(base + o, base + o + n) for o, n in cge.get_placed_reservation_ranges()]
    outside, seen = [], set()
    for kind, named in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in named:
            if tensor is None or not tensor.is_cuda or tensor.numel() == 0:
                continue
            ptr = tensor.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            if not (base <= ptr < end or any(lo <= ptr < hi for lo, hi in placed)):
                outside.append(f"{kind} {name} at 0x{ptr:x}")
    if outside:
        raise RuntimeError(
            f"[Foundry] {len(outside)} model tensor(s) are outside the allocation region "
            f"({', '.join(outside[:6])}): their addresses differ between SAVE and LOAD, and "
            "the captured graphs address them. The model's weights must be allocated by this "
            "process on its scheduler thread (no weight cache daemon, no host-resident tables)."
        )


def _check_tensors_in_region(label: str) -> None:
    """Every tensor live at ``label`` must be in the region. A caching-allocator
    segment outside it holds memory with a driver-chosen address that differs
    between SAVE and LOAD: allocated on a thread without the region (threaded
    weight loaders) or while it was stopped. A tensor there that a graph
    addresses (a weight placed in such a segment, e.g. by post-load processing)
    would sit elsewhere on LOAD."""
    cfg = get_config()
    base, end = cfg.base_addr, cfg.base_addr + parse_size(cfg.region_size)
    device = torch.cuda.current_device()
    outside = [
        seg
        for seg in torch.cuda.memory_snapshot()
        if seg.get("device", device) == device
        and seg.get("active_size", 0) > 0
        and not (base <= seg["address"] and seg["address"] + seg["total_size"] <= end)
        and seg["address"] not in _segments_before_region
    ]
    if not outside:
        return
    detail = ", ".join(
        f"0x{seg['address']:x} ({seg['active_size'] / 2**20:.1f} of "
        f"{seg['total_size'] / 2**20:.1f} MB in use)"
        for seg in outside[:8]
    )
    raise RuntimeError(
        f"[Foundry] at {label}, {len(outside)} caching-allocator segment(s) outside the "
        f"allocation region hold live tensors: {detail}. They were allocated on another "
        "thread (a threaded weight loader) or with the region stopped, at addresses that "
        "differ between SAVE and LOAD."
    )


@nvtx_traced("foundry.setup_graph_ext (binary restore)")
def setup_graph_extension(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> None:
    """Set up the VMM region before SGLang initializes NCCL/process groups."""
    global _state, _save_id
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    _check_hook_preloaded()
    _check_torch_allocator()

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

    ranks = {"workspace": rank, "tp": tp_rank, "pp": pp_rank, "dp": dp_rank}
    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        _save_id = uuid.uuid4().hex
        with open(workspace_dir / _FINGERPRINT_FILE, "w") as f:
            json.dump(
                {**archive_fingerprint(server_args, ranks), _SAVE_ID_KEY: _save_id}, f, indent=1
            )
    t_bin = 0.0
    if cfg.mode == CUDAGraphExtensionMode.LOAD:
        if os.environ.get("FOUNDRY_ENABLE_EXEC_SHARE") == "1":
            logger.warning(
                "[Foundry] FOUNDRY_ENABLE_EXEC_SHARE=1: exec share mode is experimental: why "
                "switching members gave tokens that differ from native on a tested model is "
                "not determined, and members whose copy extents or clustered launch dims differ "
                "from their template now keep their own exec (docs/sglang/known-issues.md); "
                "serve with the default mode"
            )
        cge.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            raise RuntimeError(f"Foundry workspace for rank {rank} does not exist: {workspace_dir}")
        # Before the archived modules load: an archive from another GPU or
        # settings is refused with the difference named (its kernels may not
        # load on this GPU at all).
        check_capture_loop_version()
        check_archive_fingerprint(server_args, ranks)
        t1 = time.perf_counter()
        cge.load_cuda_modules_and_libraries(str(workspace_dir))
        t_bin = time.perf_counter() - t1

    region_size = parse_size(cfg.region_size)
    t1 = time.perf_counter()
    cge.set_allocation_region(cfg.base_addr, region_size)
    t_region = time.perf_counter() - t1
    _segments_before_region.update(seg["address"] for seg in torch.cuda.memory_snapshot())
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


def _saved_scratch_end(workspace_dir: str) -> int:
    """LOAD: SAVE's scratch boundary from the archive."""
    path = os.path.join(workspace_dir, _LAYOUT_FILE)
    value = None
    if os.path.exists(path):
        with open(path) as f:
            value = json.load(f).get("scratch_end_offset")
    if value is None:
        raise RuntimeError(
            f"Foundry LOAD: {path} does not record SAVE's scratch boundary: the archive was "
            "not written by a completed SAVE of this code; re-SAVE"
        )
    return int(value)


def skip_to_scratch_boundary() -> None:
    """End of distributed init: move the cursor to the scratch boundary, where
    the weights start on SAVE and LOAD. Distributed init must have allocated
    exactly as on SAVE (the restored graphs address its NCCL communicators and
    symmetric-memory buffers): the offset trail compares its checkpoints, and
    LOAD compares the allocations and carved reservations live below the cursor
    with SAVE's here.

    SAVE uses ``scratch_space_size``, or the cursor when distributed init
    allocated past it (with a warning: no margin is left), and records the
    boundary in the archive. LOAD moves to SAVE's boundary, and stops when its
    own distributed init allocated past it: every later buffer would sit above
    the offsets the restored graphs address."""
    global _scratch_end_offset, _scratch_live_ranges
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = cge.get_current_alloc_offset()
    # Allocations ("a") and the reservations carved from the region ("r": torch
    # symmetric memory, NCCL cuMem buffers), in offset order.
    live = sorted(
        [["a", int(off), int(size)] for off, size in cge.get_live_region_ranges() if off < current]
        + [
            ["r", int(off), int(size)]
            for off, size in cge.get_reserved_region_ranges()
            if off < current
        ],
        key=lambda r: (r[1], r[0]),
    )
    boundary = scratch
    if cfg.mode == CUDAGraphExtensionMode.LOAD and cfg.workspace_dir is not None:
        boundary = _saved_scratch_end(cfg.workspace_dir)
        with open(os.path.join(cfg.workspace_dir, _LAYOUT_FILE)) as f:
            saved_live = json.load(f).get("scratch_live_ranges")
        if saved_live is not None and saved_live != live:
            first = next(
                (i for i, (a, b) in enumerate(zip(saved_live, live)) if a != b),
                min(len(saved_live), len(live)),
            )
            raise RuntimeError(
                f"Foundry LOAD: distributed init left other allocations below the weights than "
                f"on SAVE ({len(live)} live ranges, SAVE {len(saved_live)}; first difference at "
                f"#{first}: LOAD {live[first] if first < len(live) else None}, SAVE "
                f"{saved_live[first] if first < len(saved_live) else None}). The restored "
                "graphs address the communicators and symmetric-memory buffers allocated there "
                "(another NCCL build or setting, host multicast support, cuBLAS workspace). "
                "Serve with the settings and environment of the SAVE run, or re-SAVE."
            )
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
    _scratch_live_ranges = live
    cge.set_current_alloc_offset(boundary)
    logger.info("[Foundry] SGLang skipped allocator to scratch boundary %d", boundary)


_region_closed = False


def close_allocation_region() -> None:
    """End of startup (after the last startup capture or restore): this
    thread's later allocations go to the driver as usual.

    Nothing allocated after this point is addressed by a captured or restored
    graph. The region hands out address range bump-style and never reuses a
    freed range, so allocations made while serving (the receive buffers of
    every weight update, which torch releases at each empty_cache and requests
    again) would walk the cursor to the end of the region, and region memory
    is VMM memory, which a process without Foundry's hook cannot open over
    CUDA IPC."""
    global _region_closed
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE or _region_closed:
        return
    cge.stop_allocation_region()
    _region_closed = True
    logger.info(
        "[Foundry] SGLang allocation region closed after startup at offset %d: later "
        "allocations are plain",
        cge.get_current_alloc_offset(),
    )


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
        if not _region_closed:
            cge.resume_allocation_region()


def mark_layout_start() -> None:
    """Begin the deterministic layout; the same sequence point on SAVE and LOAD.

    Both modes first return torch's cached-but-free blocks to the driver, so
    the caching allocator is in the same state (empty) when the layout begins
    and every later allocation takes the same cuMemAlloc path on both sides.
    SAVE then records the cursor as the layout start. Everything SAVE allocates
    after this point is either replayed at recorded absolute offsets (graph
    memory) or allocated at the same offset on both sides; the pre-capture
    bootstraps run before it on both modes, so LOAD's cursor equals the
    recorded start (any difference stops LOAD in preallocate_for_load_mode)."""
    global _layout_start_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    gc.collect()
    torch.cuda.empty_cache()
    _check_tensors_in_region("the layout start")
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
            "scratch_live_ranges": _scratch_live_ranges,
            # Carved before the layout start, so made at the same offsets on LOAD (for
            # archive_check; the scratch and layout-start checks compare them).
            "reserved_ranges": [
                [int(o), int(n)]
                for o, n in cge.get_reserved_region_ranges()
                if _layout_start_offset is not None and o < _layout_start_offset
            ],
            "offset_trail": dict(_offset_trail),
            "start_offset": _layout_start_offset,
            "final_alloc_offset": _final_alloc_offset,
            "live_ranges": ranges,
            "capture_loop_version": CAPTURE_LOOP_VERSION,
            _SAVE_ID_KEY: _save_id,
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
    the allocation sequences diverged, and LOAD stops). Allocations inside the
    span are pointer bumps; a hole (memory SAVE released before its end, the
    warm-up forward's for one) is backed on demand by the hook, logged, and
    released again when it is freed."""
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
    if start is None or current == start:
        return
    # The two cursors coincide by construction; a difference means the SAVE and
    # LOAD allocation sequences diverged before the layout start, and a buffer
    # allocated since the last compared checkpoint sits elsewhere than where the
    # restored graphs address it.
    raise RuntimeError(
        f"Foundry LOAD: the cursor before the graph restore is {current}, SAVE's layout start "
        f"was {start} ({(current - start) / 2**20:+.0f} MB): the allocation sequences diverged "
        "before the graphs (compare the 'alloc_offset[...]' lines of both logs for the first "
        "difference). Serve with the settings and environment of the SAVE run, or re-SAVE."
    )


def _check_offset_trail(cfg, label: str, offset: int) -> None:
    """LOAD: the offset at ``label`` must equal SAVE's. A difference means the
    allocation sequences diverged (another loader path, other server settings
    or environment); communicators, weights, KV pool and the graphs' buffers
    would no longer sit where the archive addresses them."""
    global _saved_trail
    if _saved_trail is None:
        path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
        with open(path) as f:
            trail = json.load(f).get("offset_trail")
        if trail is None:
            raise RuntimeError(
                f"Foundry LOAD: {path} does not record SAVE's allocation offsets: the archive "
                "was not written by a completed SAVE of this code; re-SAVE"
            )
        _saved_trail = {k: int(v) for k, v in trail.items()}
    saved = _saved_trail.get(label)
    if saved is not None and saved != offset:
        raise RuntimeError(
            f"Foundry LOAD: allocation offset at {label} is {offset}, SAVE had {saved} "
            f"({(offset - saved) / 2**20:+.0f} MB): the allocation sequences diverged since the "
            "previous checkpoint (compare the 'alloc_offset[...]' lines of both logs). Serve "
            "with the settings and environment of the SAVE run, or re-SAVE."
        )


def log_alloc_offset(label: str) -> None:
    global _trail_started
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
    _trail_started = _trail_started or label == _TRAIL_FROM
    if _trail_started:
        if cfg.mode == CUDAGraphExtensionMode.SAVE:
            _offset_trail[label] = offset
        elif cfg.workspace_dir is not None:
            _check_offset_trail(cfg, label, offset)


def _set_subprocess_preload_env() -> None:
    """LD_PRELOAD (Foundry's libraries first: the udev-wait shim, DeepEP's
    NVSHMEM host library, the hook; then the launcher's own entries, without
    the copies of Foundry's), FOUNDRY_MODE (read by the hook library), the
    region for the hook's early reserve and the spawn timestamp. A library
    placed after the hook that looks up the next definition of its own
    symbol (the shim: ``dlsym(RTLD_NEXT, "inotify_add_watch")``) reaches the
    hook's ``dlsym``, which forwards the lookup from the hook's position."""
    ours = [
        p
        for p in (get_verbs_udev_wait_shim_path(), get_nvshmem_host_path(), get_hook_library_path())
        if p
    ]
    launcher = re.split(r"[:\s]+", os.environ.get("LD_PRELOAD", ""))
    entries = list(dict.fromkeys(ours + [p for p in launcher if p and p not in ours]))
    if entries:
        os.environ["LD_PRELOAD"] = ":".join(entries)
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value
        # LOAD reserves the region as the first VA operation of
        # load_cuda_modules_and_libraries: the ~0.5 GB of restored modules can
        # otherwise make the driver place its own VA where the fixed-address
        # reserve then fails. The values match setup_graph_extension's, so its
        # set_allocation_region finds the region already set.
        cfg = get_config()
        if cfg is not None:
            os.environ["FOUNDRY_EARLY_RESERVE_BASE"] = hex(cfg.base_addr)
            os.environ["FOUNDRY_EARLY_RESERVE_SIZE"] = str(parse_size(cfg.region_size))
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


_SUBPROCESS_ENV_KEYS = (
    "LD_PRELOAD",
    "FOUNDRY_MODE",
    "FOUNDRY_EARLY_RESERVE_BASE",
    "FOUNDRY_EARLY_RESERVE_SIZE",
    "FOUNDRY_SPAWN_T0_NS",
)


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
