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


@pytest.fixture
def region(monkeypatch, tmp_path):
    """runtime with a fake allocation cursor and a config for ``mode``."""
    from foundry.integration.sglang import runtime as rt
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig, CUDAGraphExtensionMode

    cursor = [0]
    monkeypatch.setattr(rt.cge, "get_current_alloc_offset", lambda: cursor[0])
    monkeypatch.setattr(rt.cge, "set_current_alloc_offset", lambda v: cursor.__setitem__(0, v))
    monkeypatch.setattr(rt, "_scratch_end_offset", None)

    def use(mode, offset, layout=None):
        cfg = CUDAGraphExtensionConfig(
            mode=CUDAGraphExtensionMode(mode),
            scratch_space_size="1024MB",
            workspace_dir=str(tmp_path),
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
    """SAVE moves to the scratch size and records it; LOAD, whose distributed
    init allocated a different amount below it, moves to the same boundary."""
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


def test_scratch_boundary_of_an_archive_without_the_record(region):
    """An archive written before the boundary was recorded: LOAD behaves as
    before (scratch size, or its own cursor when past it)."""
    rt = region("load", 300 * MB, layout={})
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 1024 * MB
    rt = region("load", 2000 * MB, layout={})
    rt.skip_to_scratch_boundary()
    assert region.cursor[0] == 2000 * MB


def test_load_cursor_ahead_of_the_layout_start_is_an_error(region, monkeypatch):
    """LOAD behind SAVE's layout start is moved up to it; ahead of it, its own
    buffers occupy offsets the restored graphs use, so the restore stops."""
    layout = {"start_offset": 5000 * MB, "final_alloc_offset": 6000 * MB, "live_ranges": []}
    rt = region("load", 4990 * MB, layout=layout)
    monkeypatch.setattr(rt.cge, "preallocate_ranges", lambda ranges, final: True)
    rt.preallocate_for_load_mode()
    assert region.cursor[0] == 5000 * MB
    rt = region("load", 5010 * MB, layout=layout)
    with pytest.raises(RuntimeError, match="past SAVE's layout start"):
        rt.preallocate_for_load_mode()


def _server_args(model_dir, **overrides):
    import types

    args = {"model_path": str(model_dir), "tp_size": 2, "ep_size": 2, "port": 30000}
    args.update(overrides)
    return types.SimpleNamespace(**args)


def test_archive_fingerprint_refuses_other_settings(region, tmp_path, monkeypatch):
    """SAVE records the settings that shape the archive; LOAD under another
    parallel layout or batch limit stops at setup, while a setting outside the
    record (the port) or a moved model directory with the same config.json
    does not. An archive without the record, or the override, skips the check."""
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
    monkeypatch.setenv("FOUNDRY_SGLANG_SKIP_FINGERPRINT", "1")
    rt.check_archive_fingerprint(_server_args(model, tp_size=4))
    monkeypatch.delenv("FOUNDRY_SGLANG_SKIP_FINGERPRINT")
    (tmp_path / rt._FINGERPRINT_FILE).unlink()
    rt.check_archive_fingerprint(_server_args(model, tp_size=4))


def test_archive_fingerprint_reads_the_full_server_args(monkeypatch):
    """The plugin route's setup receives sglang's parallel bag, which has no
    model or batch fields; the fingerprint reads the operator's whole record
    from sglang's runtime context."""
    import types

    import sglang.srt.runtime_context as context
    from foundry.integration.sglang import runtime as rt

    full = types.SimpleNamespace(max_running_requests=8)
    monkeypatch.setattr(context, "get_server_args", lambda: full)
    assert rt._full_server_args(types.SimpleNamespace(tp_size=1)) is full
