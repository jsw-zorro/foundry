# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Graphs with the same nodes and different edges get separate templates (1 GPU, no vLLM/SGLang).

    pytest tests/core/test_template_edges.py -q -s

LOAD rebuilds each member of a topology group on its template's graph, so the
member runs with the template's dependency edges (a member's own edges are not
stored). Kernels that use programmatic dependent launch for some batch sizes only
give graphs with the same nodes and a programmatic edge in one size, a full one in
another; a member that inherited a programmatic edge it did not have started
before its producer was done (sporadic logprob differences on LOAD).

SAVE captures the same two kernels three times: serially (the second kernel
depends on the first) at 64 and at 128 rows, and forked at 64 rows (the second
kernel on a side stream, no edge between the two). The two serial graphs must
share a template, the forked one must get its own, and every restored graph must
replay like SAVE.
"""

import json
import os
import sys

import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

BASE_ADDR = 0x4E0000000000
REGION_SIZE_STR = "1GB"
HIDDEN = 1024
# (file name, rows, forked); save_graph_manifest expects graph_<index>_<suffix>.json
GRAPHS = [("graph_0_serial64.json", 64, False), ("graph_1_forked64.json", 64, True),
          ("graph_2_serial128.json", 128, False)]  # fmt: skip


def _body(x, y, n, side):
    if side is None:
        return [x[:n] * 2.0, y[:n] + 1.0]
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    a = x[:n] * 2.0
    with torch.cuda.stream(side):
        b = y[:n] + 1.0
    cur.wait_stream(side)
    return [a, b]


def _prefix():
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator."""
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.randn(128, HIDDEN, generator=g).cuda()
    y = torch.randn(128, HIDDEN, generator=g).cuda()
    side = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _body(x, y, 64, None)  # warm-up: lazy module loads
    torch.cuda.synchronize()
    return x, y, side


def _run_save(archive):
    import foundry as fdry

    x, y, side = _prefix()
    graphs, outs = [], []
    for name, n, forked in GRAPHS:
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = _body(x, y, n, side if forked else None)
        graph.replay()
        torch.cuda.synchronize()
        graph.save(os.path.join(archive, name), output_tensors=out)
        graphs.append(graph)
        outs.append([t.cpu().clone() for t in out])
    fdry.save_graph_manifest(archive)
    save_reference(archive, outs, [x, y])
    fdry.stop_allocation_region()


def _run_load(archive):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    x, y, _ = _prefix()
    ref_outs = load_reference(archive, [x, y])
    paths = [os.path.join(archive, name) for name, _, _ in GRAPHS]
    pending = fdry.CUDAGraph.start_graph_builds(paths)
    for i, path in enumerate(paths):
        graph, out = fdry.CUDAGraph.finish_one_graph_load(pending, i)
        graph.replay()
        torch.cuda.synchronize()
        for k, (got, want) in enumerate(zip(out, ref_outs[i], strict=True)):
            assert torch.equal(got.cpu(), want), f"{os.path.basename(path)} output {k} != SAVE"
    fdry.stop_allocation_region()


def test_edges_split_topology_groups(tmp_path):
    run_phase(__file__, "save", tmp_path)
    archive = tmp_path / "archive"
    keys = {name: json.loads((archive / name).read_text())["topology_key"] for name, _, _ in GRAPHS}
    serial64, forked64, serial128 = (keys[name] for name, _, _ in GRAPHS)
    assert serial64 == serial128, "same kernels and edges at two sizes: one topology key"
    assert serial64.rsplit("|E", 1)[0] == forked64.rsplit("|E", 1)[0], "same node sequence"
    assert serial64 != forked64, "different edges: different topology keys"
    groups = json.loads((archive / "graph_manifest.json").read_text())["topology_groups"]
    members = sorted(sorted(g["members"]) for g in groups)
    assert members == [
        ["graph_0_serial64.json", "graph_2_serial128.json"],
        ["graph_1_forked64.json"],
    ]
    run_phase(__file__, "load", tmp_path)


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
