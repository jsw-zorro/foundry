# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""A damaged binary archive is not read (1 GPU, no vLLM/SGLang).

    pytest tests/core/test_archive_damage.py -q -s

SAVE writes each graph twice, as JSON and as a .cugraph binary, and LOAD reads
the binary. A binary whose counts or offsets reach outside the file (a copy cut
short, a write that failed part-way) is refused with a warning, and that graph
is read from its JSON: every restored graph must still replay like SAVE. Two
graphs of one topology group are saved; the template's binary is cut short and
the member's header claims more nodes than the file holds. The exec-pool
prewarm reads the same binaries and must skip both.
"""

import os
import struct
import sys

import pytest
import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

BASE_ADDR = 0x4F0000000000
REGION_SIZE_STR = "1GB"
HIDDEN = 1024
# (file name, rows); save_graph_manifest expects graph_<index>_<suffix>.json
GRAPHS = [("graph_0_rows64.json", 64), ("graph_1_rows128.json", 128)]
_NUM_NODES_OFFSET = 16  # FileHeader: magic[8], version, flags, num_nodes


def _body(x, y, n):
    return [x[:n] * 2.0 + y[:n], torch.relu(y[:n] - 1.0)]


def _prefix():
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.randn(128, HIDDEN, generator=g).cuda()
    y = torch.randn(128, HIDDEN, generator=g).cuda()
    with torch.cuda.stream(stream):
        _body(x, y, 64)  # warm-up: lazy module loads
    torch.cuda.synchronize()
    return x, y


def _run_save(archive):
    import foundry as fdry

    x, y = _prefix()
    graphs, outs = [], []
    for name, n in GRAPHS:
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = _body(x, y, n)
        graph.replay()
        torch.cuda.synchronize()
        graph.save(os.path.join(archive, name), output_tensors=out)
        graphs.append(graph)
        outs.append([t.cpu().clone() for t in out])
    fdry.save_graph_manifest(archive)
    fdry.save_graph_manifest(archive)  # a second call on the same archive is harmless
    save_reference(archive, outs, [x, y])
    fdry.stop_allocation_region()


def _run_load(archive):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    x, y = _prefix()
    ref_outs = load_reference(archive, [x, y])
    paths = [os.path.join(archive, name) for name, _ in GRAPHS]
    with pytest.raises(RuntimeError, match="num_threads must be positive"):
        fdry.CUDAGraph.start_graph_builds(paths, num_threads=0)
    fdry.CUDAGraph.start_exec_pool_prewarm(paths)
    pending = fdry.CUDAGraph.start_graph_builds(paths)
    for i, path in enumerate(paths):
        graph, out = fdry.CUDAGraph.finish_one_graph_load(pending, i)
        graph.replay()
        torch.cuda.synchronize()
        for k, (got, want) in enumerate(zip(out, ref_outs[i], strict=True)):
            assert torch.equal(got.cpu(), want), f"{os.path.basename(path)} output {k} != SAVE"
    fdry.stop_allocation_region()


def _cugraph(archive, name):
    return archive / name.replace(".json", ".cugraph")


def test_damaged_binaries_fall_back_to_the_json(tmp_path):
    run_phase(__file__, "save", tmp_path)
    archive = tmp_path / "archive"
    template, member = (_cugraph(archive, name) for name, _ in GRAPHS)
    data = template.read_bytes()
    template.write_bytes(data[: len(data) // 2])
    data = bytearray(member.read_bytes())
    struct.pack_into("<I", data, _NUM_NODES_OFFSET, 1_000_000)
    member.write_bytes(bytes(data))
    out = run_phase(__file__, "load", tmp_path)
    for path in (template, member):
        assert f"{path}: " in out and "LOAD uses the JSON" in out, (
            f"no fallback warning for {path.name}"
        )


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
