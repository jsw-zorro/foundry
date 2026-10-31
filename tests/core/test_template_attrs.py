# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Members of a topology group launch with their own kernel-node attributes
(1 GPU, no vLLM/SGLang).

    pytest tests/core/test_template_attrs.py -q -s

LOAD rebuilds each member of a topology group on its template's graph. SAVE
records a kernel-node attribute only when it is not zero, and the topology key
does not include it, so graphs that differ only in an attribute share a
template. A member that records none of an attribute the template (or the
member rewritten before it) set must still launch without it.

The probe kernel writes whether it was launched cooperatively
(cooperative_groups::this_grid().is_valid()). SAVE captures it six times, each
graph writing to its own int: cooperatively in graphs 0 and 4, plainly in the
others. All six share graph 0 as their template; every restored graph must
report its own launch kind.
"""

import os
import sys

import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

BASE_ADDR = 0x4E8000000000
REGION_SIZE_STR = "1GB"
COOPERATIVE = [True, False, False, False, True, False]
# save_graph_manifest expects graph_<index>_<suffix>.json
NAMES = [f"graph_{i}_{'coop' if c else 'plain'}.json" for i, c in enumerate(COOPERATIVE)]

_SOURCE = r"""
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

__global__ void probe(int* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = cg::this_grid().is_valid() ? 1 : 0;
}

void launch(torch::Tensor out, bool cooperative) {
  int* p = out.data_ptr<int>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (cooperative) {
    void* args[] = {&p};
    TORCH_CHECK(cudaLaunchCooperativeKernel((void*)probe, dim3(1), dim3(32), args, 0, stream) ==
                cudaSuccess);
  } else {
    probe<<<1, 32, 0, stream>>>(p);
  }
}
"""


def _extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        "foundry_test_template_attrs",
        cpp_sources="void launch(torch::Tensor out, bool cooperative);",
        cuda_sources=_SOURCE,
        functions=["launch"],
        verbose=False,
    )


def _prefix(ext):
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    outs = [torch.full((1,), -1, dtype=torch.int32, device="cuda") for _ in COOPERATIVE]
    with torch.cuda.stream(stream):
        ext.launch(outs[0], True)  # warm-up: lazy module load
        ext.launch(outs[0], False)
    torch.cuda.synchronize()
    return outs


def _run_save(archive):
    import foundry as fdry

    ext = _extension()
    outs = _prefix(ext)
    graphs = []
    for name, out, coop in zip(NAMES, outs, COOPERATIVE, strict=True):
        out.fill_(-1)
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            ext.launch(out, coop)
        graph.replay()
        torch.cuda.synchronize()
        assert out.item() == int(coop), f"{name}: the probe reports {out.item()} on SAVE"
        graph.save(os.path.join(archive, name))
        graphs.append(graph)
    fdry.save_graph_manifest(archive)
    save_reference(archive, torch.tensor([int(c) for c in COOPERATIVE]), outs)
    fdry.stop_allocation_region()


def _run_load(archive):
    import foundry as fdry

    ext = _extension()
    fdry.load_cuda_modules_and_libraries("hook_archive")
    outs = _prefix(ext)
    want = load_reference(archive, outs).tolist()
    paths = [os.path.join(archive, name) for name in NAMES]
    pending = fdry.CUDAGraph.start_graph_builds(paths)
    got = []
    for i, out in enumerate(outs):
        graph, _ = fdry.CUDAGraph.finish_one_graph_load(pending, i)
        out.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        got.append(out.item())
    assert got == want, f"cooperative launch per graph: LOAD {got}, SAVE {want}"
    fdry.stop_allocation_region()


def test_members_launch_with_their_own_attributes(tmp_path):
    import json

    run_phase(__file__, "save", tmp_path)
    groups = json.loads((tmp_path / "archive" / "graph_manifest.json").read_text())
    members = [sorted(g["members"]) for g in groups["topology_groups"]]
    assert members == [sorted(NAMES)], f"the six graphs must share one template: {members}"
    run_phase(__file__, "load", tmp_path)


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
