# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime contracts that hold on both sglang routes.

pytest tests/sgl/test_runtime_contracts.py -v
"""

import json

import pytest

pytest.importorskip("sglang")


def test_triton_key_is_cached_process_wide():
    """The triton_key prewarm thread only helps if the main thread's later
    call hits the same lru_cache."""
    pytest.importorskip("triton")
    from foundry.integration.sglang.hooks import _resolve_triton_key

    assert hasattr(_resolve_triton_key(), "cache_info")


def test_foundry_graphs_support_backend_cleanup():
    """Upstream main's FullCudaGraphBackend.cleanup() calls graph.reset() on
    every graph in ``_graphs``; the graphs Foundry places there are
    foundry.ops.CUDAGraph objects (the fork base's cleanup() does not reset)."""
    from foundry import ops

    assert callable(getattr(ops.CUDAGraph, "reset", None))


def test_decode_output_of_a_non_last_pipeline_stage_round_trips():
    """A non-last pipeline stage's decode graph returns PPProxyTensors, not
    the logits: SAVE records the tensor names (decode_graphs.json, as JSON)
    and LOAD rebuilds the same structure around the restored tensors."""
    import torch
    from foundry.integration.sglang import graph_ops
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    hidden, residual = torch.zeros(4, 8), torch.ones(4, 8)
    packed, spec = graph_ops._pack_output(
        PPProxyTensors({"hidden_states": hidden, "residual": residual})
    )
    assert spec == {"pp": ["hidden_states", "residual"]}
    assert packed[0] is hidden and packed[1] is residual
    restored = graph_ops._unpack_output(packed, json.loads(json.dumps(spec)))
    assert isinstance(restored, PPProxyTensors)
    assert restored.tensors["hidden_states"] is hidden and restored.tensors["residual"] is residual
    with pytest.raises(RuntimeError, match="records 2"):
        graph_ops._unpack_output([hidden], spec)

    logits = torch.zeros(4, 16)
    packed, spec = graph_ops._pack_output(LogitsProcessorOutput(next_token_logits=logits))
    assert packed is logits and spec is None
    assert graph_ops._unpack_output(logits).next_token_logits is logits
    assert graph_ops._unpack_output([logits]).next_token_logits is logits


def test_decode_hidden_states_round_trip():
    """Under a return-hidden-states mode the captured decode output also holds
    the hidden states; LOAD must return them, not the logits alone."""
    import torch
    from foundry.integration.sglang import graph_ops
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    logits, hidden, last = torch.zeros(4, 16), torch.ones(4, 8), torch.full((4, 8), 2.0)
    packed, spec = graph_ops._pack_output(
        LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden, last_hidden_states=last
        )
    )
    assert spec == {"lpo": ["next_token_logits", "hidden_states", "last_hidden_states"]}
    restored = graph_ops._unpack_output(packed, json.loads(json.dumps(spec)))
    assert isinstance(restored, LogitsProcessorOutput)
    assert restored.next_token_logits is logits
    assert restored.hidden_states is hidden and restored.last_hidden_states is last
    with pytest.raises(TypeError, match="next_token_top_logprobs_val is a list"):
        graph_ops._pack_output(
            LogitsProcessorOutput(next_token_logits=logits, next_token_top_logprobs_val=[1])
        )


@pytest.fixture
def region(monkeypatch, tmp_path):
    """runtime with a fake allocation cursor and a config for ``mode``."""
    from foundry.integration.sglang import runtime as rt
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig, CUDAGraphExtensionMode

    cursor = [0]
    monkeypatch.setattr(rt.cge, "get_current_alloc_offset", lambda: cursor[0])
    monkeypatch.setattr(rt.cge, "set_current_alloc_offset", lambda v: cursor.__setitem__(0, v))
    monkeypatch.setattr(rt, "_scratch_end_offset", None)
    monkeypatch.setattr(rt, "_scratch_live_ranges", None)
    monkeypatch.setattr(rt, "_offset_trail", {})
    monkeypatch.setattr(rt, "_trail_started", False)
    monkeypatch.setattr(rt, "_saved_trail", None)

    def use(mode, offset, layout=None, **settings):
        cfg = CUDAGraphExtensionConfig(
            mode=CUDAGraphExtensionMode(mode),
            scratch_space_size="1024MB",
            workspace_dir=str(tmp_path),
            **settings,
        )
        monkeypatch.setattr(rt, "get_config", lambda: cfg)
        if layout is not None:
            (tmp_path / rt._LAYOUT_FILE).write_text(json.dumps(layout))
        cursor[0] = offset
        return rt

    use.cursor = cursor
    return use


MB = 1 << 20


def test_scratch_boundary_save_and_load(region):
    """SAVE moves to the scratch size and records it; LOAD moves to the same
    boundary (whether its distributed init matched SAVE's is the offset
    trail's and the live ranges' business)."""
    rt = region("save", 300 * MB)
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 1024 * MB and rt._scratch_end_offset == 1024 * MB
    rt = region("load", 280 * MB, layout={"scratch_end_offset": 1024 * MB})
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 1024 * MB


def test_scratch_overrun_on_save_becomes_the_boundary(region):
    """Distributed init past the scratch size: SAVE keeps the cursor as the
    boundary; a LOAD that allocated less moves up to it, one that allocated
    more stops before loading weights at shifted offsets."""
    rt = region("save", 4932 * MB)
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 4932 * MB and rt._scratch_end_offset == 4932 * MB
    rt = region("load", 4900 * MB, layout={"scratch_end_offset": 4932 * MB})
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 4932 * MB
    rt = region("load", 4940 * MB, layout={"scratch_end_offset": 4932 * MB})
    with pytest.raises(RuntimeError, match="re-SAVE"):
        rt.skip_to_scratch_boundary()


def test_archive_without_the_layout_records_is_refused(region):
    """An archive whose layout lacks the scratch boundary or the offset trail
    (an older Foundry's): LOAD stops instead of guessing them."""
    rt = region("load", 300 * MB, layout={})
    with pytest.raises(RuntimeError, match="scratch boundary.*re-SAVE"):
        rt.skip_to_scratch_boundary()
    rt = region("load", 300 * MB, layout={"scratch_end_offset": 1024 * MB})
    with pytest.raises(RuntimeError, match="allocation offsets.*re-SAVE"):
        rt._check_offset_trail(rt.get_config(), "after_scratch_skip", 1024 * MB)


def test_load_cursor_off_the_layout_start_is_an_error(region, monkeypatch):
    """A LOAD cursor behind or ahead of SAVE's layout start: some buffer sits
    elsewhere than where the restored graphs address it, so the restore stops."""
    layout = {"start_offset": 5000 * MB, "final_alloc_offset": 6000 * MB, "live_ranges": []}
    for cursor in (4990 * MB, 5010 * MB):
        rt = region("load", cursor, layout=layout)
        monkeypatch.setattr(rt.cge, "preallocate_ranges", lambda ranges, final: True)
        with pytest.raises(RuntimeError, match="SAVE's layout start was"):
            rt.preallocate_for_load_mode()
    rt = region("load", 5000 * MB, layout=layout)
    rt.preallocate_for_load_mode()


def _server_args(model_dir, **overrides):
    import types

    args = {"model_path": str(model_dir), "tp_size": 2, "ep_size": 2, "port": 30000}
    args.update(overrides)
    return types.SimpleNamespace(**args)


def test_archive_fingerprint_refuses_other_settings(region, tmp_path, monkeypatch):
    """SAVE records the settings that shape the archive; LOAD under another
    parallel layout, batch limit or region size stops at setup, while a setting
    outside the record (the port) or a moved model directory with the same
    config.json does not. A record without a field this code records (an older
    Foundry's), or no record, stops it too; the override skips the check."""
    from types import SimpleNamespace

    props = SimpleNamespace(
        name="GPU-X", major=9, minor=0, multi_processor_count=132, total_memory=1
    )
    rt = region("save", 0)
    monkeypatch.setattr(rt, "_full_server_args", lambda server_args: server_args)
    monkeypatch.setattr(rt.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(rt.torch.cuda, "get_device_properties", lambda device: props)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 16}')
    saved = rt.archive_fingerprint(_server_args(model))
    (tmp_path / rt._FINGERPRINT_FILE).write_text(json.dumps(saved))
    assert saved["model_path"].startswith("config.json sha256:") and saved["gpu"] == "GPU-X"

    rt = region("load", 0)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "config.json").write_text('{"hidden_size": 16}')
    rt.check_archive_fingerprint(_server_args(moved, port=31000))
    with pytest.raises(RuntimeError, match="tp_size: SAVE 2, LOAD 4"):
        rt.check_archive_fingerprint(_server_args(model, tp_size=4))
    with pytest.raises(RuntimeError, match="max_running_requests"):
        rt.check_archive_fingerprint(_server_args(model, max_running_requests=64))
    with pytest.raises(RuntimeError, match="region_size: SAVE"):
        region("load", 0, region_size="64GB").check_archive_fingerprint(_server_args(model))
    rt = region("load", 0)
    older = {k: v for k, v in saved.items() if k != "enable_return_hidden_states"}
    (tmp_path / rt._FINGERPRINT_FILE).write_text(json.dumps(older))
    with pytest.raises(RuntimeError, match="older Foundry.*enable_return_hidden_states"):
        rt.check_archive_fingerprint(_server_args(model))
    (tmp_path / rt._FINGERPRINT_FILE).write_text(json.dumps(saved))
    monkeypatch.setenv("FOUNDRY_SGLANG_SKIP_FINGERPRINT", "1")
    rt.check_archive_fingerprint(_server_args(model, tp_size=4))
    monkeypatch.delenv("FOUNDRY_SGLANG_SKIP_FINGERPRINT")
    (tmp_path / rt._FINGERPRINT_FILE).unlink()
    with pytest.raises(RuntimeError, match="not found.*re-SAVE"):
        rt.check_archive_fingerprint(_server_args(model))


def test_archive_fingerprint_reads_the_full_server_args(monkeypatch):
    """The plugin route's setup receives sglang's parallel bag, which has no
    model or batch fields; the fingerprint reads the operator's whole record
    from sglang's runtime context."""
    import types

    import sglang.srt.runtime_context as context
    from foundry.integration.sglang import runtime as rt

    full = types.SimpleNamespace(max_running_requests=8)
    monkeypatch.setattr(context, "get_server_args", lambda: full)
    assert rt._full_server_args(types.SimpleNamespace(tp_size=1)).max_running_requests == 8


def test_recapture_after_startup_captures_in_place(monkeypatch):
    """A capture scope of a phase whose startup scope already ran (sglang
    recaptures after update_weights_from_disk(recapture_cuda_graph=True))
    neither restores nor archives: capture_one captures the shape in place."""
    import contextlib

    import sglang.srt.model_executor.runner_utils.pool as pool
    from foundry.integration.sglang import graph_ops, hooks
    from foundry.integration.sglang.config import CUDAGraphExtensionMode

    captured, bootstraps = [], []
    monkeypatch.setattr(hooks, "get_graph_extension_mode", lambda: CUDAGraphExtensionMode.LOAD)
    monkeypatch.setattr(hooks, "_startup_captured", {"decode"})
    monkeypatch.setattr(hooks, "_recapture_bootstrapped", False)
    monkeypatch.setattr(pool, "graph_pool_capture_scope", contextlib.nullcontext)
    monkeypatch.setattr(
        graph_ops, "capture_graph", lambda p, s, fn: captured.append((p, s)) or ("graph", fn())
    )
    monkeypatch.setattr(graph_ops, "bootstrap_lazy_runtimes", lambda: bootstraps.append(1))
    monkeypatch.setattr(
        graph_ops, "restore_next_decode_graph", lambda key: pytest.fail("restored on a recapture")
    )
    for _ in range(2):
        with hooks.capture_scope("decode", None):
            assert hooks.capture_one("key", lambda: "out", pool="p", stream="s") == ("graph", "out")
    assert captured == [("p", "s")] * 2
    # The one-time runtime inits run before the first recaptured shape only.
    assert bootstraps == [1]
    assert hooks._scope is None and not hooks._recapturing


def test_load_reserves_the_region_before_restoring_modules(monkeypatch):
    """Spawned schedulers get the region in the environment, so the hook
    reserves it as the first VA operation of load_cuda_modules_and_libraries
    (the restored modules can otherwise take the base)."""
    import os

    from foundry.integration.sglang import runtime as rt
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig, CUDAGraphExtensionMode

    cfg = CUDAGraphExtensionConfig(
        mode=CUDAGraphExtensionMode.LOAD, base_addr=0x500000000000, region_size="16GB"
    )
    monkeypatch.setattr(rt, "get_config", lambda: cfg)
    monkeypatch.setattr(rt, "get_graph_extension_mode", lambda: cfg.mode)
    for key in rt._SUBPROCESS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with rt.subprocess_preload_env():  # dependency route: around Process.start()
        assert os.environ["FOUNDRY_EARLY_RESERVE_BASE"] == "0x500000000000"
        assert os.environ["FOUNDRY_EARLY_RESERVE_SIZE"] == str(16 << 30)
    assert "FOUNDRY_EARLY_RESERVE_BASE" not in os.environ

    import foundry.integration.sglang.plugin as plugin

    monkeypatch.setattr(plugin, "apply_env_pins", lambda server_args=None: None)
    try:  # plugin route: for the rest of the launcher process
        rt.setup_ld_preload_env()
        assert os.environ["FOUNDRY_EARLY_RESERVE_BASE"] == "0x500000000000"
        assert os.environ["FOUNDRY_EARLY_RESERVE_SIZE"] == str(16 << 30)
        assert os.environ["FOUNDRY_MODE"] == "load"
    finally:
        for key in rt._SUBPROCESS_ENV_KEYS:
            os.environ.pop(key, None)


def test_foundry_preloads_come_before_the_launchers(monkeypatch):
    """The shim, the NVSHMEM host library and the hook go first, also when the
    launcher already preloads one of them (the documented shell export of the
    shim): a library after the hook that looks up the next definition of its
    own symbol would get its own definition back."""
    import os

    from foundry.integration.sglang import runtime as rt

    monkeypatch.setattr(rt, "get_hook_library_path", lambda: "/f/libcuda_hook.so")
    monkeypatch.setattr(rt, "get_nvshmem_host_path", lambda: "/f/libnvshmem_host.so")
    monkeypatch.setattr(rt, "get_verbs_udev_wait_shim_path", lambda: "/f/libno_cdev_wait.so")
    monkeypatch.setenv("LD_PRELOAD", "/u/own.so /f/libno_cdev_wait.so:/u/other.so")
    with rt.subprocess_preload_env():
        assert os.environ["LD_PRELOAD"].split(":") == [
            "/f/libno_cdev_wait.so",
            "/f/libnvshmem_host.so",
            "/f/libcuda_hook.so",
            "/u/own.so",
            "/u/other.so",
        ]
    assert os.environ["LD_PRELOAD"] == "/u/own.so /f/libno_cdev_wait.so:/u/other.so"


def test_graph_templates_must_be_a_toml_boolean(tmp_path):
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig

    path = tmp_path / "foundry.toml"
    path.write_text('mode = "save"\ngraph_templates = false\n')
    assert CUDAGraphExtensionConfig.from_toml(path).graph_templates is False
    path.write_text('mode = "save"\ngraph_templates = "false"\n')
    with pytest.raises(TypeError, match="graph_templates must be true or false"):
        CUDAGraphExtensionConfig.from_toml(path)


def test_plugin_entry_point_exits_when_foundry_cannot_load(monkeypatch):
    """sglang's load_plugins() logs a plugin's exception and serves on: an
    import failure must stop the process instead."""
    import sys

    import foundry_sglang_plugin

    monkeypatch.setenv(foundry_sglang_plugin.CONFIG_ENV, "/tmp/foundry.toml")
    monkeypatch.setitem(sys.modules, "foundry.integration.sglang.plugin", None)
    with pytest.raises(SystemExit, match="sglang plugin failed to load"):
        foundry_sglang_plugin.load()


def test_load_stops_at_the_first_checkpoint_that_differs_from_save(region, monkeypatch):
    """From the region's setup on (distributed init included), SAVE records
    the offset at every alloc_offset checkpoint; LOAD compares each checkpoint
    it shares with SAVE and stops at the first difference. Checkpoints of one
    mode only are not compared."""
    rt = region("save", 0)
    for label, offset in (("after_setup_graph_ext", 32), ("after_init_torch_dist", 300),
                          ("after_scratch_skip", 1024), ("before_init_memory_pool", 5000),
                          ("after_lazy_runtimes", 6000)):  # fmt: skip
        region.cursor[0] = offset * MB
        rt.log_alloc_offset(label)
    assert set(rt._offset_trail) == {
        "after_setup_graph_ext",
        "after_init_torch_dist",
        "after_scratch_skip",
        "before_init_memory_pool",
        "after_lazy_runtimes",
    }
    layout = {"offset_trail": rt._offset_trail}

    rt = region("load", 0, layout=layout)
    monkeypatch.setattr(rt, "_trail_started", False)
    monkeypatch.setattr(rt, "_saved_trail", None)
    for label, offset in (("after_setup_graph_ext", 32), ("after_init_torch_dist", 300),
                          ("after_scratch_skip", 1024), ("before_init_memory_pool", 5000),
                          ("before_preallocate", 7000)):  # fmt: skip
        region.cursor[0] = offset * MB
        rt.log_alloc_offset(label)
    region.cursor[0] = 5016 * MB
    with pytest.raises(RuntimeError, match=r"before_init_memory_pool is .*\+16 MB"):
        rt.log_alloc_offset("before_init_memory_pool")
    # Distributed init allocating less on LOAD is a difference too.
    rt = region("load", 0, layout=layout)
    monkeypatch.setattr(rt, "_trail_started", False)
    monkeypatch.setattr(rt, "_saved_trail", None)
    region.cursor[0] = 32 * MB
    rt.log_alloc_offset("after_setup_graph_ext")
    region.cursor[0] = 280 * MB
    with pytest.raises(RuntimeError, match=r"after_init_torch_dist is .*-20 MB"):
        rt.log_alloc_offset("after_init_torch_dist")


def test_first_shape_of_each_scope_runs_one_eager_forward(monkeypatch):
    """SAVE and LOAD alike run one eager forward, then the attention backend's
    reset, at a capture scope's first shape, before its graph is captured or
    restored (what the model builds lazily on its first forward then exists on
    both modes); later shapes run none."""
    import contextlib

    import sglang.srt.model_executor.runner_utils.pool as pool
    import torch
    from foundry.integration.sglang import graph_ops, hooks
    from foundry.integration.sglang.config import CUDAGraphExtensionMode

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(pool, "graph_pool_capture_scope", contextlib.nullcontext)
    monkeypatch.setattr(graph_ops, "save_graph", lambda *args, **kwargs: None)

    def run(mode, hook):
        calls = []
        monkeypatch.setattr(hooks, "get_graph_extension_mode", lambda: mode)
        monkeypatch.setattr(hooks, "_scope", ("decode", None))
        monkeypatch.setattr(hooks, "_recapturing", False)
        monkeypatch.setattr(hooks, "_warmed_up", set())
        monkeypatch.setattr(hooks.rt, "log_alloc_offset", calls.append)
        monkeypatch.setattr(
            graph_ops, "capture_graph", lambda p, s, fn: calls.append("capture") or ("g", "o")
        )
        monkeypatch.setattr(
            graph_ops,
            "restore_next_decode_graph",
            lambda key: calls.append("restore") or ("g", "o"),
        )

        def forward():
            calls.append("forward")
            return "out"

        kwargs = {}
        if hook == "reset":
            kwargs["post_warmup_hook"] = lambda: calls.append("reset")
        elif hook is None:
            kwargs["post_warmup_hook"] = None
        for key in ("bs4", "bs2"):
            hooks.capture_one(key, forward, pool="p", stream="s", **kwargs)
        return calls

    for mode, step in (
        (CUDAGraphExtensionMode.SAVE, "capture"),
        (CUDAGraphExtensionMode.LOAD, "restore"),
    ):
        warm = ["before_decode_warmup", "forward"]
        after = ["after_decode_warmup", step, step]
        assert run(mode, "reset") == [*warm, "reset", *after], mode
        # sglang passes None for a backend without the reset: the warm-up still runs.
        assert run(mode, None) == [*warm, *after], mode
        # A dependency-route adapter that passes no hook (API < 2.1) runs without it.
        assert run(mode, "absent") == [step, step], mode


def test_live_tensors_outside_the_region_are_refused(region, monkeypatch):
    """A caching-allocator segment outside the region that holds live tensors
    at the layout start (allocated on another thread, e.g. by a threaded weight
    loader) stops SAVE and LOAD; segments that existed before the region was
    set up, or hold nothing, do not."""
    rt = region("save", 0)
    base = rt.get_config().base_addr
    segments = [
        {"device": 0, "address": base + (2 << 20), "total_size": 2 << 20, "active_size": 1 << 20},
        {"device": 0, "address": 0x7F0000000000, "total_size": 2 << 20, "active_size": 0},
        {"device": 0, "address": 0x7E0000000000, "total_size": 2 << 20, "active_size": 1024},
    ]
    monkeypatch.setattr(rt.torch.cuda, "memory_snapshot", lambda: segments)
    monkeypatch.setattr(rt.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(rt, "_segments_before_region", set())
    with pytest.raises(RuntimeError, match=r"1 caching-allocator segment.*0x7e0000000000"):
        rt._check_tensors_in_region("the layout start")
    monkeypatch.setattr(rt, "_segments_before_region", {0x7E0000000000})
    rt._check_tensors_in_region("the layout start")


def test_model_tensors_outside_the_region_are_refused(region, monkeypatch):
    """A CUDA parameter or buffer outside the region (weights imported from
    another process, a host table mapped for zero-copy access) stops SAVE and
    LOAD at the layout start; CPU and empty tensors do not count."""
    import types

    rt = region("load", 0)
    base = rt.get_config().base_addr

    def tensor(ptr, cuda=True, numel=4):
        return types.SimpleNamespace(is_cuda=cuda, numel=lambda: numel, data_ptr=lambda: ptr)

    def model(params, buffers=()):
        return types.SimpleNamespace(
            named_parameters=lambda: iter(params), named_buffers=lambda: iter(buffers)
        )

    inside = [("w", tensor(base + 4096)), ("tied", tensor(base + 4096))]
    ignored = [("cpu", tensor(0x1000, cuda=False)), ("empty", tensor(0, numel=0))]
    rt.check_model_tensors_in_region(model(inside + ignored))
    with pytest.raises(RuntimeError, match=r"1 model tensor.*buffer table at 0x7f0000000000"):
        rt.check_model_tensors_in_region(model(inside, [("table", tensor(0x7F0000000000))]))
    # A reservation of 2 GiB or more the hook placed at its fixed address after
    # the region (Engram peer shards) has the same address on SAVE and LOAD.
    after = rt.parse_size(rt.get_config().region_size)
    monkeypatch.setattr(rt.cge, "get_placed_reservation_ranges", lambda: [(after, 4 << 30)])
    rt.check_model_tensors_in_region(model([("shard", tensor(base + after + 8192))]))
    with pytest.raises(RuntimeError, match=r"1 model tensor.*parameter past"):
        rt.check_model_tensors_in_region(model([("past", tensor(base + after + (4 << 30)))]))


def test_allocator_configurations_the_hook_does_not_see_are_refused(monkeypatch):
    from foundry.integration.sglang import runtime as rt

    monkeypatch.setattr(rt.torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(rt, "_allocator_settings", lambda: "expandable_segments:True")
    with pytest.raises(RuntimeError, match="expandable segments"):
        rt._check_torch_allocator()
    monkeypatch.setattr(rt, "_allocator_settings", lambda: "pinned_use_cuda_host_register:True")
    rt._check_torch_allocator()
    monkeypatch.setattr(rt.torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync")
    with pytest.raises(RuntimeError, match="cudaMallocAsync"):
        rt._check_torch_allocator()


def test_allocator_settings_record_the_device_keys_only(monkeypatch):
    """The fingerprint's allocator entry leaves out the pinned-host keys and
    normalizes the rest, lists included."""
    from foundry.integration.sglang import runtime as rt

    monkeypatch.setattr(
        rt,
        "_allocator_settings",
        lambda: "pinned_use_cuda_host_register:True,max_split_size_mb:512,"
        " roundup_power2_divisions:[256:1,512:2],expandable_segments:False",
    )
    assert rt._device_allocator_settings() == [
        "expandable_segments:false",
        "max_split_size_mb:512",
        "roundup_power2_divisions:[256:1,512:2]",
    ]
    monkeypatch.setattr(rt, "_allocator_settings", lambda: "")
    assert rt._device_allocator_settings() == []


def test_model_identity_names_the_config_not_the_path(tmp_path, monkeypatch):
    """One model named by two paths, by a hub id and its cached snapshot, or
    re-written by another transformers (same config-class view), is one
    identity; another config is another."""
    import types

    import huggingface_hub
    import transformers
    from foundry.integration.sglang import runtime as rt

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "config.json").write_text('{"hidden_size": 16, "transformers_version": "4.1"}')
    (b / "config.json").write_text('{"transformers_version": "4.9",\n "hidden_size": 16}')

    # Config class view: the same parsed config is one identity whatever the file holds.
    views = {
        str(a): {"hidden_size": 16, "dtype": "bfloat16"},
        str(b): {"hidden_size": 16, "dtype": "bfloat16"},
    }

    def from_pretrained(path, **kwargs):
        if path not in views:
            raise OSError(path)
        return types.SimpleNamespace(to_dict=lambda: {**views[path], "_name_or_path": path})

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    assert rt._model_identity(a).startswith("config sha256:")
    assert rt._model_identity(a) == rt._model_identity(b)
    views[str(b)] = {"hidden_size": 32, "dtype": "bfloat16"}
    assert rt._model_identity(a) != rt._model_identity(b)

    # Without the config class: the file without the writer keys, hub ids through the cache.
    views.clear()
    (b / "config.json").write_text('{"transformers_version": "4.9",\n "hidden_size": 16}')
    assert rt._model_identity(a) == rt._model_identity(b)
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, revision=None: str(a / "config.json"),
    )
    assert rt._model_identity("org/model") == rt._model_identity(a)
    (b / "config.json").write_text('{"hidden_size": 32}')
    assert rt._model_identity(a) != rt._model_identity(b)


def test_files_of_two_save_runs_are_refused(region, tmp_path):
    """fingerprint.json and region_layout.json written by different SAVE runs
    (two engines writing one workspace_root) stop LOAD."""
    rt = region("load", 0, layout={"save id": "run-b"})
    (tmp_path / rt._FINGERPRINT_FILE).write_text(json.dumps({"save id": "run-a"}))
    with pytest.raises(RuntimeError, match="different SAVE runs"):
        rt.check_archive_fingerprint(None)


def test_distributed_init_layout_must_match_save(region, monkeypatch):
    """The allocations and carved reservations distributed init leaves below
    the weights (communicators, symmetric-memory buffers the graphs address)
    must be SAVE's: one resized range stops LOAD before the weights load."""
    rt = region("save", 300 * MB)
    monkeypatch.setattr(rt.cge, "get_live_region_ranges", lambda: [(0, 2 * MB), (2 * MB, 64 * MB)])
    monkeypatch.setattr(rt.cge, "get_reserved_region_ranges", lambda: [(66 * MB, 128 * MB)])
    rt.skip_to_scratch_boundary()
    layout = {"scratch_end_offset": 1024 * MB, "scratch_live_ranges": rt._scratch_live_ranges}
    assert [r[0] for r in layout["scratch_live_ranges"]] == ["a", "a", "r"]
    rt = region("load", 300 * MB, layout=layout)
    monkeypatch.setattr(rt.cge, "get_reserved_region_ranges", lambda: [(66 * MB, 64 * MB)])
    with pytest.raises(RuntimeError, match="first difference at #2"):
        rt.skip_to_scratch_boundary()
    rt = region("load", 300 * MB, layout=layout)
    monkeypatch.setattr(rt.cge, "get_reserved_region_ranges", lambda: [(66 * MB, 128 * MB)])
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 1024 * MB


def test_weights_load_on_the_scheduler_thread(monkeypatch):
    """From the end of distributed init to the layout start, on SAVE and LOAD,
    every module's should_async_load (sglang's switch to a weight-loading
    thread pool) returns False, the modules imported meanwhile included; the
    original is restored afterwards."""
    import sys
    import types

    import sglang.srt.model_loader.utils as loader_utils
    import torch
    from foundry.integration.sglang import hooks

    orig = loader_utils.should_async_load
    cpu_weight = torch.empty(1)
    assert orig(cpu_weight)
    importer = types.ModuleType("foundry_test_importer")
    importer.should_async_load = orig
    monkeypatch.setitem(sys.modules, importer.__name__, importer)
    late = types.ModuleType("foundry_test_late_importer")
    monkeypatch.setattr(hooks, "_async_load_original", None)
    try:
        hooks._synchronous_weight_loading(True)
        late.should_async_load = loader_utils.should_async_load  # imported during the load
        sys.modules[late.__name__] = late
        assert not loader_utils.should_async_load(cpu_weight)
        assert not importer.should_async_load(cpu_weight)
    finally:
        hooks._synchronous_weight_loading(False)
        sys.modules.pop(late.__name__, None)
    assert loader_utils.should_async_load is orig
    assert importer.should_async_load is orig and late.should_async_load is orig
