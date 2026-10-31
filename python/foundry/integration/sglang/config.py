# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry SGLang integration configuration."""

from __future__ import annotations

import enum
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import tomllib


class CUDAGraphExtensionMode(str, enum.Enum):
    NONE = "none"
    SAVE = "save"
    LOAD = "load"


# Per-rank archive records LOAD reads before anything else (runtime.py; the
# preflight reads them too, without torch).
# region_layout.json: the deterministic allocation layout, written at the end
# of SAVE and consumed by LOAD's preallocation (runtime.record_region_layout).
ARCHIVE_LAYOUT_FILE = "region_layout.json"
# fingerprint.json: what SAVE ran with (runtime.archive_fingerprint).
ARCHIVE_FINGERPRINT_FILE = "fingerprint.json"
# Per SAVE run, in both files: LOAD refuses a rank directory whose files come
# from different runs (two engines writing one workspace_root).
ARCHIVE_SAVE_ID_KEY = "save id"
# Version of the allocation sequence between the layout start and the end of
# the decode capture, recorded in region_layout.json on SAVE and required on
# LOAD. 2: both modes run sglang's own decode capture loop and substitute only
# the capture (1, unmarked: a FlashInfer metadata pre-pass before the loop,
# which allocates the per-bs wrappers in another order; LOAD cannot replay it).
# 3: one eager warm-up forward at each capture scope's first shape on both
# modes (hooks._warm_up_first_shape); every record LOAD checks (fingerprint,
# scratch boundary, offset trail) is required.
CAPTURE_LOOP_VERSION = 3


def _toml_bool(data: dict, key: str, default: bool) -> bool:
    """A TOML boolean (``true`` / ``false``); a string such as ``"false"``
    would otherwise read as True."""
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Foundry config: {key} must be true or false, got {value!r}")
    return value


@dataclass
class CUDAGraphExtensionConfig:
    mode: CUDAGraphExtensionMode = CUDAGraphExtensionMode.NONE
    hook_library_path: str | None = None
    nvshmem_host_path: str | None = None
    # Bare hosts whose verbs devices exist but cannot be opened make rdma-core
    # wait 5 s per HCA for udev in DeepEP's first NVSHMEM init (~40 s per
    # engine). Path to the preload shim that removes the wait
    # (tools/host/no_cdev_wait.c, docs/bare-host-verbs-udev-wait.md); None
    # leaves LD_PRELOAD as the launcher exported it.
    verbs_udev_wait_shim_path: str | None = None
    base_addr: int = 0x600000000000
    region_size: str = "256GB"
    workspace_root: str = "foundry_archive"
    workspace_dir: str | None = None
    scratch_space_size: str = "64MB"
    # SAVE: group same-topology graphs under one template with on-demand
    # members. False makes every graph its own template so LOAD rebuilds each
    # from its full node list (slower load, no shared-graph re-parameterization).
    graph_templates: bool = True

    @classmethod
    def from_toml(cls, path: str | Path) -> CUDAGraphExtensionConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_mapping(data, source=str(path))

    @classmethod
    def from_mapping(cls, data: dict, source: str = "<defaults>") -> CUDAGraphExtensionConfig:
        """Build from parsed TOML keys; a missing key keeps the default and the
        library paths are detected from the installed wheels."""
        base_addr_value = data.get("base_addr", cls.base_addr)
        base_addr = int(base_addr_value, 0) if isinstance(base_addr_value, str) else base_addr_value

        hook_library_path = data.get("hook_library_path")
        if hook_library_path is None:
            hook_library_path = cls._detect_hook_so_path()
        elif "/" in hook_library_path and not Path(hook_library_path).is_file():
            # ld.so ignores a missing LD_PRELOAD entry with a warning only, and
            # every buffer would then get a driver-chosen address (a bare name
            # is resolved through the loader's search path).
            raise FileNotFoundError(
                f"hook_library_path={hook_library_path!r} in {source} does not exist; "
                "omit it to use the installed foundry's libcuda_hook.so"
            )

        nvshmem_host_path = data.get("nvshmem_host_path")
        if nvshmem_host_path is None:
            nvshmem_host_path = cls._detect_nvshmem_host_path()

        verbs_udev_wait_shim_path = data.get("verbs_udev_wait_shim_path")
        if verbs_udev_wait_shim_path is not None and not Path(verbs_udev_wait_shim_path).is_file():
            raise FileNotFoundError(
                f"verbs_udev_wait_shim_path={verbs_udev_wait_shim_path!r} in {source} "
                "does not exist; "
                "build it with `make -C tools/host` (docs/bare-host-verbs-udev-wait.md)"
            )

        return cls(
            mode=CUDAGraphExtensionMode(data.get("mode", cls.mode.value)),
            hook_library_path=hook_library_path,
            nvshmem_host_path=nvshmem_host_path,
            verbs_udev_wait_shim_path=verbs_udev_wait_shim_path,
            base_addr=base_addr,
            region_size=data.get("region_size", cls.region_size),
            workspace_root=data.get("workspace_root", cls.workspace_root),
            scratch_space_size=data.get("scratch_space_size", cls.scratch_space_size),
            graph_templates=_toml_bool(data, "graph_templates", cls.graph_templates),
        )

    @staticmethod
    def _detect_hook_so_path() -> str | None:
        spec = importlib.util.find_spec("foundry.ops")
        if spec and spec.origin:
            ops_so_path = Path(spec.origin).resolve()
            hook_so_path = ops_so_path.parent / "libcuda_hook.so"
            if hook_so_path.exists():
                return str(hook_so_path)
        return None

    @staticmethod
    def _detect_nvshmem_host_path() -> str | None:
        """Locate DeepEP's NVSHMEM host lib from the installed wheel.

        ``nvidia-nvshmem-cuXX`` ships ``libnvshmem_host.so.3`` (cu13 ``torch``
        pulls it as a dependency), but its ``site-packages/nvidia/nvshmem/lib``
        dir is not on the loader search path — ``deep_ep`` reaches it only via
        an RPATH. Foundry's hook must interpose NVSHMEM's module-init symbols,
        so it has to ``LD_PRELOAD`` the lib by absolute path. Resolve that path
        from the wheel here, mirroring ``_detect_hook_so_path``; an explicit
        ``nvshmem_host_path`` in the TOML still overrides this. Returns None when
        the wheel isn't installed (non-EP runs simply don't preload it).
        """
        try:
            spec = importlib.util.find_spec("nvidia.nvshmem")
        except (ImportError, ValueError):
            return None
        if spec is None:
            return None
        roots = list(spec.submodule_search_locations or [])
        if spec.origin:
            roots.append(str(Path(spec.origin).parent))
        for root in roots:
            for name in ("libnvshmem_host.so.3", "libnvshmem_host.so"):
                candidate = Path(root) / "lib" / name
                if candidate.exists():
                    return str(candidate)
        return None


_config: CUDAGraphExtensionConfig | None = None


def load_graph_extension_config(path: str | None, mode: str | None = None) -> None:
    """Load the TOML at ``path`` (None: all defaults). ``mode``, when given,
    replaces the TOML's ``mode`` (SGLang's ``--cuda-graph-persistence`` wins)."""
    global _config
    if path is None:
        config = CUDAGraphExtensionConfig.from_mapping({})
    else:
        config = CUDAGraphExtensionConfig.from_toml(path)
    if mode is not None:
        config.mode = CUDAGraphExtensionMode(mode)
    _config = config


def get_config() -> CUDAGraphExtensionConfig | None:
    return _config


def get_graph_extension_mode() -> CUDAGraphExtensionMode:
    if _config is None:
        return CUDAGraphExtensionMode.NONE
    return _config.mode


def get_workspace_root() -> str | None:
    if _config is None:
        return None
    return _config.workspace_root


def get_hook_library_path() -> str | None:
    if _config is None:
        return None
    return _config.hook_library_path


def get_nvshmem_host_path() -> str | None:
    if _config is None:
        return None
    return _config.nvshmem_host_path


def get_verbs_udev_wait_shim_path() -> str | None:
    if _config is None:
        return None
    return _config.verbs_udev_wait_shim_path


def attention_dp_active(server_args) -> bool:
    """Whether attention runs data parallel: ``attn_dp_enabled`` on sglang
    from #41818 on (``--attn-dp-size N``; the deprecated ``--dp-size N
    --enable-dp-attention`` resolves to it and leaves ``enable_dp_attention``
    False), ``enable_dp_attention`` before."""
    if hasattr(server_args, "attn_dp_enabled"):
        return bool(server_args.attn_dp_enabled)
    return bool(getattr(server_args, "enable_dp_attention", False))


def _attn_dp_size(server_args) -> int:
    # Derived on get_parallel() on both layouts; a ServerArgs from before
    # #41818 counts the attention-DP groups in dp_size.
    size = getattr(server_args, "attn_dp_size", None)
    if not size:
        size = getattr(server_args, "dp_size", 1)
    return max(int(size or 1), 1)


def dp_replica(server_args, dp_rank: int | None) -> int:
    """The data-parallel replica this process belongs to.

    Without attention DP, ``dp_rank`` is the replica: the DP controller spawns
    one TP x PP world per replica. With attention DP (either sglang layout)
    the DP controller spawns the ``tp_size`` schedulers of ONE TP world and
    passes each ``dp_rank`` = its attention-DP group,
    ``tp_rank // (attn_tp_size * attn_cp_size)``; sglang rejects
    ``--dp-size > 1`` together with ``--attn-dp-size > 1``. The replica is then
    ``dp_rank // attn_dp_size`` (0), which also numbers replica-major
    ``replica * attn_dp_size + group`` ranks should the combination appear."""
    dp_index = dp_rank or 0
    if attention_dp_active(server_args):
        return dp_index // _attn_dp_size(server_args)
    return dp_index


def compute_workspace_rank(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> int:
    """``rank_<N>`` of this process:
    ``replica * tp_size * pp_size + pp_rank * tp_size + tp_rank`` with the
    replica from :func:`dp_replica`. EP4 with DP attention (tp 4, attn_dp 4,
    dp_rank 0..3 = tp_rank) is rank_0..3, as before #41818."""
    replica = dp_replica(server_args, dp_rank)
    return (
        replica * server_args.tp_size * server_args.pp_size
        + pp_rank * server_args.tp_size
        + tp_rank
    )
