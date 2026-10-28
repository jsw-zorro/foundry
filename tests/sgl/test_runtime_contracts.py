# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime contracts that hold on both sglang routes.

pytest tests/sgl/test_runtime_contracts.py -v
"""

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
    import json

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
