# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry as an SGLang plugin.

Activation: ``FOUNDRY_GRAPH_EXTENSION_CONFIG=<foundry toml>`` in the launcher's
environment (spawned processes inherit it). SGLang's ``load_plugins()`` runs
the ``foundry`` entry point (``foundry_sglang_plugin:load``) in the launcher,
before the server args are built, and at the top of every scheduler process.
Each run of :func:`activate` registers the resolution hooks below and installs
the runtime patches (``hooks.install``).

Since upstream #35594 the data-parallel controller and the detokenizer call
``load_plugins()`` too, so the plugin activates there as well. Neither needs
it: the controller's spawn environment (LD_PRELOAD, the pins) is the one it
inherits from the launcher, whose ``Engine._launch_scheduler_processes`` sets
it before the controller is spawned, and its patched
``launch_tensor_parallel_group`` sets the same values again. Activation in a
process that never runs the model costs ~0.7 s of imports (H200 host, in
parallel with the schedulers' start) and initializes nothing on the GPU.

Resolution hooks (``register_resolution_hook`` on whitelisted steps, so the
pins land at the same pipeline positions as the former in-tree step):

- ``apply_inkling_prefill_cuda_graph_default`` (wrapped, pins first): the
  step the in-tree ``handle_graph_extension`` ran right before. Decode is
  pinned to ``full``, prefill to ``disabled`` unless ``full`` / ``disabled`` was
  given, profiling and FlashInfer autotune off (SAVE and LOAD must allocate
  identically). Conflicting flags raise instead of being overridden. The
  same step pins the flags that select state Foundry cannot replay
  (:data:`FIELD_PINS`, see "Pins" below).
- ``handle_cuda_graph_config`` (wrapped, validates after): the resolved
  config, which also covers ``--cuda-graph-config`` JSON (it outranks the
  per-phase flags).
- ``handle_other_validations`` (wrapped, validates after): features Foundry
  does not support, once speculative decoding, LoRA and elastic EP are
  resolved; the graph config and the pinned flags are checked again after
  the later cascades, and ``NCCL_CUMEM_ENABLE`` is pinned once the MoE
  all-to-all backend is known.

Pins. State that cannot be made static across processes is not supported;
Foundry uses the cuMem-backed alternatives instead. A graph restored on LOAD
replays only what was recorded inside the capture region: a buffer
registration, IPC handle exchange or pool carve-out made outside it (at
``graph_capture()`` exit, at first use inside the forward, by a second graph
or allocator) does not exist in the LOAD process, and the restored kernels
read addresses nobody set up. Every such feature that a flag or an
environment variable can switch off is pinned here rather than left to the
recipe: each pin prints one ``[Foundry] pin:`` line, and a value the user
set explicitly to the contrary raises with the reason (never a silent
override). Features with no Foundry logic at all are rejected instead
(:func:`reject_unsupported_features`, ``hooks.reject_unsupported_decode_runner``).

Environment pins (:data:`ENV_PINS`) are applied in :func:`activate`, which
runs in the launcher before the server args are built (so the resolution
steps that read them see the pinned values) and before any engine process is
spawned (spawned processes inherit ``os.environ``; values that sglang caches
at import time, such as ``SGLANG_JIT_DEEPGEMM_PRECOMPILE``, are only correct
this way). They are re-asserted at the top of every scheduler process
(``activate`` runs there too) and at each spawn site
(``runtime.setup_ld_preload_env``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_ENV = "FOUNDRY_GRAPH_EXTENSION_CONFIG"
# Declaration source recorded in the resolution stash.
RESOLUTION_SOURCE = "foundry_graph_extension"
# Source of the unreplayable-state pins (FIELD_PINS), kept apart so that each
# declaration can be read back on its own.
PIN_RESOLUTION_SOURCE = "foundry_graph_extension_pins"
# How the messages below name the activation. The dependency route
# (foundry.integration.sglang.api) switches them to SGLang's flag through
# use_flag_labels(); the reasons stay the same on both routes.
_FLAG = f"Foundry graph persistence ({CONFIG_ENV})"
_DISABLE_HINT = f"unset {CONFIG_ENV} to run without graphs"
_EXTENSION = f"the Foundry graph extension ({CONFIG_ENV})"
# Whitelisted resolution steps the plugin wraps (register_resolution_hooks);
# the preflight checks that the installed sglang still offers them.
RESOLUTION_HOOK_STEPS = (
    "apply_inkling_prefill_cuda_graph_default",
    "handle_cuda_graph_config",
    "handle_other_validations",
)
_ACTIVATED = False
# The FOUNDRY_GRAPH_EXTENSION_CONFIG value this process activated with.
_ACTIVATED_CONFIG: str | None = None


def activate(cfg_path: str) -> None:
    """Plugin body; called by ``foundry_sglang_plugin.load`` when the env is set.

    ``load_plugins()`` logs and swallows exceptions from a plugin, which would
    leave SGLang running natively while the user asked for SAVE or LOAD, so a
    failure here exits the process (SystemExit is not an ``Exception``)."""
    global _ACTIVATED, _ACTIVATED_CONFIG
    if _ACTIVATED:
        return
    from foundry.integration.sglang import api

    if api.is_active() or _sglang_has_dependency_route():
        raise SystemExit(
            f"[Foundry] {CONFIG_ENV} is set, but this sglang calls Foundry directly: "
            "use --cuda-graph-persistence {save,load} [--cuda-graph-persistence-config "
            f"{cfg_path}] and unset {CONFIG_ENV}"
        )
    try:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"{cfg_path!r} does not exist")
        apply_env_pins()
        register_resolution_hooks()
        from foundry.integration.sglang.hooks import install

        install(cfg_path)
    except Exception as exc:
        raise SystemExit(
            f"[Foundry] sglang plugin failed to activate ({CONFIG_ENV}={cfg_path}): {exc!r}"
        ) from exc
    _ACTIVATED = True
    _ACTIVATED_CONFIG = cfg_path
    # Logging is not configured yet in every process that loads plugins.
    print(
        f"[Foundry] sglang plugin active: pid={os.getpid()} config={cfg_path}",
        file=sys.stderr,
        flush=True,
    )


def check_config_unchanged() -> None:
    """Launcher spawn sites: sglang loads plugins once per process, so the
    mode, the resolution hooks and the preload environment stay those of the
    config this process activated with. Schedulers activate from the variable
    they inherit; a later engine started after it changed (or was unset)
    would mix the two."""
    current = os.environ.get(CONFIG_ENV)
    if _ACTIVATED_CONFIG is not None and current != _ACTIVATED_CONFIG:
        raise RuntimeError(
            f"[Foundry] {CONFIG_ENV} is {current!r}, but this process activated Foundry with "
            f"{_ACTIVATED_CONFIG!r}: sglang loads plugins once per process, so a later engine "
            "keeps the first config's mode, resolution hooks and preload environment. Start "
            "each engine with another config, or a native engine, in a new process."
        )


def use_flag_labels(flag: str = "--cuda-graph-persistence") -> None:
    """Dependency route: name SGLang's flag instead of the plugin's
    environment variable in every pin and reject message."""
    global _FLAG, _DISABLE_HINT, _EXTENSION
    _FLAG = f"Foundry graph persistence ({flag})"
    _DISABLE_HINT = f"drop {flag} to run without graphs"
    _EXTENSION = f"the Foundry graph extension ({flag})"


def _sglang_has_dependency_route() -> bool:
    """True for an sglang that declares Foundry as a dependency and calls it
    at its own call sites (sglang.srt.utils.foundry_adapter); the plugin's
    patches would then run on top of those calls."""
    import importlib.util

    try:
        return importlib.util.find_spec("sglang.srt.utils.foundry_adapter") is not None
    except (ImportError, ValueError):
        return False


def register_resolution_hooks() -> None:
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    wrappers = (
        _pin_then_previous,
        _previous_then_validate_graph_config,
        _previous_then_reject_unsupported,
    )
    for name, wrapper in zip(RESOLUTION_HOOK_STEPS, wrappers, strict=True):
        register_resolution_hook(name)(wrapper)


# ---------------------------------------------------------------------------
# Resolution-hook wrappers: fn(server_args, previous)
# ---------------------------------------------------------------------------


def _pin_then_previous(server_args: Any, previous) -> None:
    pin_graph_fields(server_args)
    pin_unreplayable_fields(server_args)
    previous(server_args)


def _previous_then_validate_graph_config(server_args: Any, previous) -> None:
    previous(server_args)
    validate_resolved_graph_config(server_args)


def _previous_then_reject_unsupported(server_args: Any, previous) -> None:
    previous(server_args)
    reject_unsupported_features(server_args)
    validate_resolved_graph_config(server_args)
    validate_pinned_fields(server_args)
    apply_env_pins(server_args)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _backend():
    from sglang.srt.model_executor.cuda_graph_config import Backend

    return Backend


def pin_graph_fields(server_args: Any) -> None:
    """Decode ``full``; prefill ``full`` or ``disabled`` (default ``disabled``, not
    the platform default, which on CUDA is ``breakable``; leaving it unset
    would also let a model default such as Inkling's pick it); profiling and
    FlashInfer autotune off."""
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    Backend = _backend()
    cfg = resolving_view(server_args)

    disabled = getattr(cfg, "disable_cuda_graph", False)
    if disabled or getattr(cfg, "disable_decode_cuda_graph", False):
        raise ValueError(
            "--disable-cuda-graph / --disable-decode-cuda-graph cannot be combined with "
            f"{_FLAG}: "
            f"Foundry saves and restores the decode CUDA graphs; {_DISABLE_HINT}"
        )
    decode = getattr(cfg, "cuda_graph_backend_decode", None)
    if decode not in (None, Backend.FULL):
        raise ValueError(
            f"--cuda-graph-backend-decode={decode!r} is not supported with {_FLAG}: "
            "Foundry persists full decode graphs only; use 'full' or leave it unset"
        )
    prefill = getattr(cfg, "cuda_graph_backend_prefill", None)
    if prefill not in (None, Backend.FULL, Backend.DISABLED):
        raise ValueError(
            f"--cuda-graph-backend-prefill={prefill!r} is not supported with {_FLAG}: "
            "Foundry persists full prefill graphs only; use 'full' or 'disabled'"
        )
    if getattr(cfg, "enable_profile_cuda_graph", False):
        raise ValueError(
            f"--enable-profile-cuda-graph is not supported with {_FLAG}: profiling "
            "adds allocations to the capture that LOAD does not reproduce"
        )
    fields = dict(
        cuda_graph_backend_decode=Backend.FULL,
        enable_profile_cuda_graph=False,
        disable_flashinfer_autotune=True,
    )
    if prefill is None:
        fields["cuda_graph_backend_prefill"] = Backend.DISABLED
    declare_resolution(server_args, RESOLUTION_SOURCE, **fields)


def validate_resolved_graph_config(server_args: Any) -> None:
    """The resolved ``cuda_graph_config`` (JSON > per-phase flags > legacy
    flags > defaults, then the compatibility cascades) must still be decode
    ``full`` and prefill ``full`` / ``disabled``."""
    from sglang.srt.arg_groups.overrides import resolving_view

    Backend = _backend()
    graph_cfg = resolving_view(server_args).cuda_graph_config
    if graph_cfg is None:
        return
    decode = graph_cfg.decode.backend
    prefill = graph_cfg.prefill.backend
    if decode != Backend.FULL:
        raise ValueError(
            f"cuda_graph_config[decode].backend={decode!r} is not supported with {_FLAG}: "
            "Foundry persists full decode graphs only (check --cuda-graph-config, "
            "and disaggregation roles that turn decode graphs off)"
        )
    if prefill not in (Backend.FULL, Backend.DISABLED):
        raise ValueError(
            f"cuda_graph_config[prefill].backend={prefill!r} is not supported with {_FLAG}: "
            "Foundry persists full prefill graphs only; use 'full' or 'disabled'"
        )


def _elastic_recapture_path_exists() -> bool:
    try:
        from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
    except ImportError:
        return False
    return hasattr(cuda_graph_setup, "recapture_elastic_cuda_graph")


def reject_unsupported_features(server_args: Any) -> None:
    """Features whose graphs Foundry cannot save or restore, rejected once
    they are resolved. Pipeline parallelism is supported: the workspace rank
    includes the PP rank and the non-last stages' PPProxyTensors outputs are
    persisted (decode: decode_graphs.json; prefill: prefill_graphs.json)."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    unsupported = []
    spec = getattr(cfg, "speculative_algorithm", None)
    if spec and str(spec).upper() != "NONE":
        unsupported.append(
            f"speculative decoding (--speculative-algorithm={spec}): target-verify and "
            "draft graphs are not persisted"
        )
    if getattr(cfg, "enable_lora", False) or getattr(cfg, "lora_paths", None):
        unsupported.append("LoRA (--enable-lora / --lora-paths): not persisted or tested")
    if getattr(cfg, "enable_pdmux", False):
        unsupported.append("PD multiplexing (--enable-pdmux): multi-stream graphs")
    if getattr(cfg, "use_ray", False):
        unsupported.append(
            "Ray scheduler actors (--use-ray): Foundry's preload environment and plugin do not "
            "reach them, so SAVE would write no archive and LOAD would capture natively"
        )
    if getattr(cfg, "weight_cache_mode", "off") not in (None, "off"):
        unsupported.append(
            f"the weight cache (--weight-cache-mode {cfg.weight_cache_mode}): the parameters "
            "are CUDA IPC mappings of the daemon's memory at addresses the driver picks per "
            "process, outside the allocation region"
        )
    host_memory = []
    if getattr(cfg, "enable_hisparse", False):
        host_memory.append("--enable-hisparse (host KV pool)")
    engram = _engram_host_tables(server_args, cfg)
    if engram:
        host_memory.append(engram)
    if getattr(cfg, "ple_offload_embedding", False):
        host_memory.append(
            "--ple-offload-embedding (on by default for BF16 Qwen4-Exp on CUDA; disable it with "
            "--no-ple-offload-embedding)"
        )
    if (getattr(cfg, "cpu_offload_gb", 0) or 0) > 0:
        host_memory.append("--cpu-offload-gb")
    if (getattr(cfg, "offload_group_size", -1) or -1) > 0:
        host_memory.append("--offload-group-size")
    if host_memory:
        unsupported.append(
            f"{', '.join(host_memory)}: captured kernels read pinned host memory through "
            "addresses each process picks anew, which a restored graph would not have"
        )
    if getattr(cfg, "enable_torch_compile", False):
        unsupported.append(
            "--enable-torch-compile: Foundry pins TORCHDYNAMO_DISABLE=1, so the compiled "
            "batch sizes would capture sglang's uncompiled reference ops"
        )
    if (
        str(getattr(cfg, "disaggregation_mode", "") or "").lower() in ("prefill", "decode")
        and str(getattr(cfg, "disaggregation_transfer_backend", "") or "").lower() == "mooncake"
        and (
            _normalize_env(os.environ.get("MC_INTRANODE_NVLINK", "")) == "1"
            or os.environ.get("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "").upper() == "INTRA_NODE_NVLINK"
        )
    ):
        unsupported.append(
            "Mooncake's intra-node NVLink transport for PD disaggregation: it exports the KV "
            "pool, which is Foundry region memory, through CUDA IPC handles a process without "
            "Foundry's hook cannot open; use the RDMA protocol"
        )
    if getattr(cfg, "is_embedding", False):
        unsupported.append(
            "embedding models (--is-embedding): sglang captures no CUDA graphs for them"
        )
    max_ep = getattr(cfg, "max_ep_size", None)
    if (
        getattr(cfg, "elastic_ep_backend", None) is not None
        and max_ep is not None
        and max_ep > cfg.tp_size
        and _elastic_recapture_path_exists()
    ):
        unsupported.append(
            f"elastic-EP CUDA-graph recapture (--elastic-ep-backend with --max-ep-size={max_ep} "
            f"> tp size {cfg.tp_size}): graphs recaptured after a scale event cannot be restored"
        )
    if _normalize_env(os.environ.get("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING", "")) == "1":
        unsupported.append(
            "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING: the KV pool is re-sized from post-capture "
            "free memory, which LOAD cannot reproduce"
        )
    if unsupported:
        raise ValueError(f"not supported with {_EXTENSION}: " + "; ".join(unsupported))


# ---------------------------------------------------------------------------
# Pins: configs that select state Foundry cannot replay (module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldPin:
    """A ServerArgs field the plugin declares to ``value``. ``default`` is
    sglang's default (fa090f7755, unchanged at 2c83dc6656): a raw value other
    than the pin and the default can only come from the user, and raises.
    ``strict`` pins are re-checked after the later resolution steps; a
    non-strict one may be turned off by sglang (the fallback is still
    replayable)."""

    field: str
    value: Any
    default: Any
    flag: str
    reason: str
    strict: bool = True


FIELD_PINS: tuple[FieldPin, ...] = (
    FieldPin(
        "disable_custom_all_reduce",
        True,
        False,
        "--disable-custom-all-reduce",
        "custom all-reduce (v1 and v2) registers the graphs' buffers over CUDA IPC when "
        "graph_capture() exits, outside the recorded region",
    ),
    FieldPin(
        "enable_torch_symm_mem",
        True,
        False,
        "--enable-torch-symm-mem",
        "the in-graph all-reduce uses torch symmetric memory, a cuMem buffer Foundry places "
        "at the same address on SAVE and LOAD",
        strict=False,
    ),
    FieldPin(
        "enable_symm_mem",
        False,
        False,
        "--enable-symm-mem",
        "NCCL symmetric-memory windows are registered at first use inside the forward and "
        "need NCCL's cuMem buffers",
    ),
    FieldPin(
        "enable_nccl_nvls",
        False,
        False,
        "--enable-nccl-nvls",
        "NVLS multicast buffers are registered at first use inside the forward",
    ),
    FieldPin(
        "enable_mscclpp",
        False,
        False,
        "--enable-mscclpp",
        "MSCCL++ registers its buffers at first use inside the forward",
    ),
    FieldPin(
        "enable_two_batch_overlap",
        False,
        False,
        "--enable-two-batch-overlap",
        "not validated with Foundry (children micro-batch graphs and DeepEP capture events); "
        "pinned off until a TBO row is validated",
    ),
    FieldPin(
        "enable_memory_saver",
        False,
        False,
        "--enable-memory-saver",
        "torch_memory_saver owns the pools (and, with SGLANG_MEMORY_SAVER_CUDA_GRAPH, the "
        "graph memory) outside Foundry's region",
    ),
)

# --dsv4-attn-backend: 'auto' (resolves to flashmla) or 'flashmla'; never trtllm.
DSV4_ATTN_FIELD = "dsv4_attn_backend"
DSV4_ATTN_REASON = (
    "the trtllm DeepSeek-V4 attention backend asserts at its first call, which falls "
    "inside the capture, and needs an eager bootstrap Foundry does not run"
)


@dataclass(frozen=True)
class EnvPin:
    name: str
    value: str
    reason: str


ENV_PINS: tuple[EnvPin, ...] = (
    # NCCL registers user buffers of graph-captured collectives (above a size
    # threshold) and the kernels then read peers' remote addresses from an
    # array NCCL fills on the host at capture time. A restored graph replays
    # those kernels without the registration, so the array holds garbage at
    # LOAD (illegal address in the DP-attention all-gather at bs>=4 with NCCL
    # 2.30). Keep every size on the unregistered path.
    EnvPin(
        "NCCL_GRAPH_REGISTER",
        "0",
        "NCCL registers the user buffers of graph-captured collectives on the host at capture "
        "time; a restored graph replays the kernels without the registration",
    ),
    EnvPin(
        "NCCL_LOCAL_REGISTER",
        "0",
        "same as NCCL_GRAPH_REGISTER for buffers registered outside a graph",
    ),
    EnvPin(
        "NCCL_NVLS_ENABLE",
        "0",
        "NVLS multicast buffers are mapped with driver flags Foundry's VMM region does not "
        "carry and are registered outside the recorded region",
    ),
    # NVSHMEM builds one NCCL communicator per team (5 on a 4-rank node, ~0.4 s
    # each) while DeepEP creates its buffer; the low-latency kernels never use
    # them, and the pin changes no allocation, so existing archives stay valid.
    EnvPin(
        "NVSHMEM_DISABLE_NCCL",
        "1",
        "NVSHMEM's per-team NCCL communicators cost ~2.4 s in the DeepEP bootstrap and are "
        "unused by the low-latency kernels",
    ),
    EnvPin(
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
        "0",
        "the DeepGEMM precompile sweep runs on the first rank inside the first capture: it "
        "synchronizes the device and allocates scratch only SAVE sees; kernels still JIT "
        "per shape",
    ),
    EnvPin(
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH",
        "0",
        "the memory-saver graph context owns the graph memory outside Foundry's region",
    ),
    EnvPin(
        "SGLANG_ENABLE_METADATA_GLUE_GRAPH",
        "0",
        "the attention-metadata prep is captured into a second graph Foundry does not save",
    ),
    EnvPin(
        "SGLANG_ENABLE_GRAPH_POOL_PRECARVE",
        "0",
        "only upstream capture_one runs the precarve (measured over its two eager warmups, "
        "minted at the first capture); Foundry's capture_one replaces it on SAVE and LOAD, so "
        "the flag would be a no-op and the graph pool would not match a native run's layout",
    ),
    EnvPin(
        "SGLANG_ENABLE_GRAPH_POOL_BORROW",
        "0",
        "eager allocations would borrow free graph-pool extents whose addresses the restored "
        "graphs reference",
    ),
    # Read by torch.compile when it decorates (sglang applies it at import),
    # so it must be in the environment before the scheduler imports sglang;
    # the scheduler also sets torch._dynamo.config.disable
    # (hooks.before_parallel_init), which dynamo reads per frame.
    EnvPin(
        "TORCHDYNAMO_DISABLE",
        "1",
        "SAVE runs each shape's first forward inside the capture and torch >= 2.14's dynamo "
        "refuses to compile while a stream is capturing; sglang's compiled helpers run eagerly",
    ),
)

# NCCL_CUMEM_ENABLE=0 unless the MoE all-to-all is DeepEP v2, whose NCCL
# windows need cuMem (sglang defaults it to 1 for deepep_v2); pinned during
# resolution, once the backend is known, and at the spawn sites.
NCCL_CUMEM_PIN = EnvPin(
    "NCCL_CUMEM_ENABLE",
    "0",
    "NCCL's cuMem buffers (P2P, NVLS) are mapped with driver flags Foundry's VMM region "
    "does not carry; the plain allocator keeps them at deterministic offsets",
)
_CUMEM_EXEMPT_A2A = ("deepep_v2",)
# Set once the launcher applied the env pins: descendants log only changes.
_ENV_PINNED_MARK = "FOUNDRY_SGLANG_ENV_PINNED"
_CUMEM_EXEMPT_LOGGED = False
_KEPT_LOGGED: set[str] = set()


def _pin_log(message: str) -> None:
    # Logging is not configured yet in the launcher when resolution runs.
    print(f"[Foundry] pin: {message}", file=sys.stderr, flush=True)


def _contrary(pin: FieldPin, raw: Any) -> bool:
    return raw != pin.value and raw != pin.default


def pin_unreplayable_fields(server_args: Any) -> None:
    """Declare :data:`FIELD_PINS` and the DeepSeek-V4 attention backend. Raw
    (user) values are read from ``server_args`` itself: declarations never
    change the raw inputs. Fields the installed sglang does not have are
    skipped."""
    from sglang.srt.arg_groups.overrides import declare_resolution

    errors = []
    fields: dict[str, Any] = {}
    lines = []
    for pin in FIELD_PINS:
        if not hasattr(server_args, pin.field):
            continue
        raw = getattr(server_args, pin.field)
        if _contrary(pin, raw):
            errors.append(f"{pin.flag}={raw!r}: {pin.reason}")
            continue
        fields[pin.field] = pin.value
        if raw != pin.value:
            state = f"overrides sglang default {pin.default!r}"
        elif raw == pin.default:
            state = "sglang default"
        else:
            state = "as given"
        lines.append(f"{pin.field}={pin.value!r} ({state}): {pin.reason}")
    if hasattr(server_args, DSV4_ATTN_FIELD):
        raw = getattr(server_args, DSV4_ATTN_FIELD)
        if raw == "trtllm":
            errors.append(f"--dsv4-attn-backend=trtllm: {DSV4_ATTN_REASON}")
        else:
            value = raw or "auto"
            fields[DSV4_ATTN_FIELD] = value
            lines.append(f"{DSV4_ATTN_FIELD}={value!r} (not trtllm): {DSV4_ATTN_REASON}")
    if errors:
        raise ValueError(f"not supported with {_FLAG}: " + "; ".join(errors))
    for line in lines:
        _pin_log(line)
    if fields:
        declare_resolution(server_args, PIN_RESOLUTION_SOURCE, **fields)


def _last_source(server_args: Any, field: str) -> str:
    for source, declared in reversed(getattr(server_args, "_resolved_overrides", None) or ()):
        if field in declared:
            return source
    return "the raw input"


def validate_pinned_fields(server_args: Any) -> None:
    """After the later resolution steps (model overrides, platform fallbacks):
    a strict pin must still hold. A later sglang step that switched one back
    raises (naming it) rather than being overridden after other fields were
    derived from it."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    errors = []
    for pin in FIELD_PINS:
        if not hasattr(server_args, pin.field):
            continue
        value = getattr(cfg, pin.field)
        if value == pin.value:
            continue
        source = _last_source(server_args, pin.field)
        if pin.strict:
            errors.append(f"{pin.field}={value!r} (set by {source}): {pin.reason}")
        else:
            _pin_log(f"{pin.field}={value!r} (set by {source}); the collective falls back to NCCL")
    if hasattr(server_args, DSV4_ATTN_FIELD) and getattr(cfg, DSV4_ATTN_FIELD) == "trtllm":
        errors.append(
            f"{DSV4_ATTN_FIELD}='trtllm' (set by {_last_source(server_args, DSV4_ATTN_FIELD)}): "
            f"{DSV4_ATTN_REASON}"
        )
    if errors:
        raise ValueError(f"not supported with {_FLAG}: " + "; ".join(errors))


def _engram_host_tables(server_args: Any, cfg: Any) -> str | None:
    """Why DeepSeek-V4.1's Engram tables would be in host memory, or None."""
    if hasattr(server_args, "engram_config"):
        # sglang with --engram-config: _resolve_engram_config (DeepSeek-V4.1 only;
        # it runs before this step) folds the deprecated variables into the
        # declared value, the only one the Engram layers read.
        if _engram_placement(cfg) == "cpu":
            return "Engram tables in host memory (--engram-config placement cpu)"
        return None
    # Older sglang: the Engram layers read the variable; peer shards (from
    # 79701a2549) take precedence with TP > 1 on CUDA. Both as sglang parses them.
    host_table = _sglang_env_flag("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE")
    if host_table is None:
        host_table = (
            _normalize_env(os.environ.get("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "")) == "1"
        )
    if not host_table:
        return None
    if (
        _sglang_env_flag("SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS")
        and (getattr(cfg, "tp_size", 1) or 1) > 1
    ):
        return None
    try:
        from sglang.srt.arg_groups.model_override_base import model_config_of

        if not model_config_of(server_args).use_engram:
            return None
    except Exception:
        pass  # model not known here: keep the refusal
    return "Engram tables in host memory (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)"


def _sglang_env_flag(name: str) -> bool | None:
    """The value sglang reads for the boolean variable ``name`` (its EnvBool:
    true/1/yes/y, an invalid spelling reads as the default), or None when the
    installed sglang does not define it."""
    try:
        from sglang.srt.environ import envs

        field = getattr(envs, name, None)
        return None if field is None else bool(field.get())
    except Exception:
        return None


def _engram_placement(cfg: Any) -> str | None:
    """The resolved --engram-config placement (a config object, a dict or its
    JSON string), or None."""
    value = getattr(cfg, "engram_config", None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        return value.get("placement")
    return getattr(value, "placement", None)


def _normalize_env(value: str) -> str:
    v = value.strip().lower()
    if v in ("0", "false", "no", "n", "off"):
        return "0"
    if v in ("1", "true", "yes", "y", "on"):
        return "1"
    return v


def _a2a_backend(server_args: Any) -> str | None:
    try:
        from sglang.srt.arg_groups.overrides import resolving_view

        value = getattr(resolving_view(server_args), "moe_a2a_backend", None)
    except ImportError:
        value = getattr(server_args, "moe_a2a_backend", None)
    return None if value is None else str(getattr(value, "value", value))


def apply_env_pins(server_args: Any = None) -> None:
    """Set :data:`ENV_PINS` in ``os.environ`` (and :data:`NCCL_CUMEM_PIN` when
    ``server_args`` is given, i.e. once the all-to-all backend is known). An
    unset variable is set; one set to the pinned value is kept; one set to
    anything else raises. Idempotent: every process that loads the plugin and
    every spawn site calls it."""
    global _CUMEM_EXEMPT_LOGGED
    pins = list(ENV_PINS)
    if server_args is not None:
        a2a = _a2a_backend(server_args)
        if a2a not in _CUMEM_EXEMPT_A2A:
            pins.append(NCCL_CUMEM_PIN)
        elif not _CUMEM_EXEMPT_LOGGED:
            _CUMEM_EXEMPT_LOGGED = True
            _pin_log(
                f"{NCCL_CUMEM_PIN.name} left to sglang: --moe-a2a-backend {a2a} needs NCCL's "
                "cuMem windows"
            )
    first = os.environ.get(_ENV_PINNED_MARK) is None
    errors = []
    for pin in pins:
        current = os.environ.get(pin.name)
        if current is None:
            os.environ[pin.name] = pin.value
            _pin_log(f"{pin.name}={pin.value} (was unset): {pin.reason}")
        elif _normalize_env(current) != pin.value:
            errors.append(f"{pin.name}={current!r} (Foundry needs {pin.value}): {pin.reason}")
        elif current != pin.value:
            # Same meaning in another spelling ("off", " 0", "true"): written in
            # the canonical form, which every consumer's parser reads alike.
            os.environ[pin.name] = pin.value
            _pin_log(f"{pin.name}={pin.value} (was {current!r}): {pin.reason}")
        elif (first or pin is NCCL_CUMEM_PIN) and pin.name not in _KEPT_LOGGED:
            _KEPT_LOGGED.add(pin.name)
            _pin_log(f"{pin.name}={pin.value} (already set): {pin.reason}")
    if errors:
        raise ValueError(
            f"environment not supported with {_FLAG}: "
            + "; ".join(errors)
            + " -- unset these variables, Foundry sets them"
        )
    os.environ[_ENV_PINNED_MARK] = "1"
