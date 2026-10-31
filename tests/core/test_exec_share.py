# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Eager vs exec-share LOAD (1 GPU, no vLLM/SGLang).

Usage: pytest tests/core/test_exec_share.py -q -s
With cudagraph-patch preloaded (the patched-route case; the real-route case then skips):
    eval "$(cudagraph-patch env)" && pytest tests/core/test_exec_share.py -q -s

SAVE captures one elementwise chain (>128 kernel nodes) at several batch sizes:
same kernels and DAG, different grids, so one template plus members. Two LOAD
processes, eager (default) and share (FOUNDRY_ENABLE_EXEC_SHARE=1), replay the
sizes in alternation, which in share mode switches the template's exec on
every replay, and must match the SAVE outputs bit for bit. Share mode must log
its mode line, instantiate no member exec and start no exec-pool prewarm.
Per-replay times of a switch vs a repeat are printed, not asserted. The route
line must say cudagraph-patch when cgp_exec_update resolves in this process
(LD_PRELOAD, inherited by the LOAD processes), the real update otherwise.

A second workload (needs Triton) launches one Triton kernel with two BLOCK
specializations: same function name and DAG, different binary. Share mode
must split the saved group into one template per binary.

A third workload copies the first n rows of the input into a persistent
buffer: one device-to-device memcpy node whose size differs per member. Every
replay must write exactly n rows (the buffer is filled with -1 before each
replay, so a copy of the template's size shows in the rows past n). It saves
the largest size first, as sglang does, so every member is smaller than its
template; so do the fourth and fifth.

A fourth workload (needs nvcc and sm90+) launches a kernel in 2-CTA clusters
with 2n blocks, each marking its own entry of a persistent buffer: the grid of
a clustered kernel node differs per member, and every replay must mark exactly
2n entries. Share mode gives each graph of these two workloads its own template
(members whose copy extents or clustered launch dims differ are not switched).
A fifth one launches a producer and a consumer joined by a programmatic (PDL)
edge, whose members are switched.
"""

import ctypes
import json
import os
import re
import shutil
import statistics
import sys
import time

import pytest
import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Weak cgp_* symbols bind in the LOAD processes only when the patch is preloaded,
# and they inherit this process's LD_PRELOAD.
_HAS_CGP = hasattr(ctypes.CDLL(None), "cgp_exec_update")
ROUTE_LINE = {
    True: "[foundry] exec share: patched route (cgp)",
    False: "[foundry] exec share: real cuGraphExecUpdate",
}

BASE_ADDR = 0x4F0000000000
REGION_SIZE_STR = "1GB"
HIDDEN = 1024
SIZES = [64, 128, 256, 512, 1024]  # rows; one graph each
DEPTH = 48  # 3 kernels per step: 144 kernel nodes per graph
# Every consecutive pair differs: each replay is a member switch in share mode.
ORDER = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4, 3, 2, 1, 0, 4, 0]
TIMING_ITERS = 50
SHARE_ENV = "FOUNDRY_ENABLE_EXEC_SHARE"
WORKLOAD_ENV = "FOUNDRY_TEST_EXEC_SHARE_WORKLOAD"
TRITON_DEPTH = 8
PHASE2_RE = re.compile(
    r"Phase 2: (\d+) templates \+ (\d+) on-demand = (\d+) graphs built in [\d.]+ ms "
    r"\(members: (\d+) execs instantiated"
)
REFINE_RE = re.compile(r"\[foundry\] exec share: (\d+) saved groups -> (\d+) templates")
if os.environ.get(WORKLOAD_ENV) in ("memcpy", "cluster", "pdl"):
    # Largest first, as sglang captures: every member is smaller than its template.
    SIZES = SIZES[::-1]


def _body(x, n):
    y = x[:n]
    for _ in range(DEPTH):
        y = torch.sin(y)
        y = y * 0.5
        y = y + 0.25
    return y


if _HAS_TRITON:

    @triton.jit
    def _affine_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 0.5 + 0.25, mask=mask)


def _triton_block(n):
    """Rows 64 and 128 use one specialization, 256..1024 the other."""
    return 512 if n <= 128 else 1024


def _body_triton(x, n):
    y = x[:n]
    block = _triton_block(n)
    for _ in range(TRITON_DEPTH):
        out = torch.empty_like(y)
        _affine_kernel[(triton.cdiv(y.numel(), block),)](y, out, y.numel(), BLOCK=block)
        y = out
    return y


_dst: torch.Tensor | None = None  # memcpy / cluster workloads: the persistent destination

_CLUSTER_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void foundry_test_mark(float* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = 1.0f;
}

void launch_clustered(torch::Tensor out, int64_t blocks) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(blocks));
  cfg.blockDim = dim3(32);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, foundry_test_mark, out.data_ptr<float>()) == cudaSuccess);
}
"""
_cluster_ext = None

# Programmatic dependent launch: the producer marks 2n entries and triggers its dependent early;
# the consumer (launched with programmatic stream serialization) waits for the producer's
# memory with cudaGridDependencySynchronize and doubles every entry it finds marked.
_PDL_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void foundry_test_produce(float* out) {
  cudaTriggerProgrammaticLaunchCompletion();
  // Some work after the trigger, so a dependent released too early reads stale entries.
  float v = 1.0f;
  for (int k = 0; k < 2000; ++k) v = v * 0.999999f + 0.000001f;
  if (threadIdx.x == 0) out[blockIdx.x] = v > 0.0f ? 1.0f : 0.0f;
}

__global__ void foundry_test_consume(float* out) {
  cudaGridDependencySynchronize();
  if (threadIdx.x == 0 && out[blockIdx.x] == 1.0f) out[blockIdx.x] = 2.0f;
}

void launch_pdl(torch::Tensor out, int64_t blocks) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  foundry_test_produce<<<static_cast<unsigned>(blocks), 32, 0, stream>>>(out.data_ptr<float>());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(blocks));
  cfg.blockDim = dim3(32);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, foundry_test_consume, out.data_ptr<float>()) == cudaSuccess);
}
"""
_pdl_ext = None


def _extension(archive, name, source, function):
    from torch.utils.cpp_extension import load_inline

    build = os.path.join(os.path.dirname(archive), "ext")
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name,
        cpp_sources=f"void {function}(torch::Tensor out, int64_t blocks);",
        cuda_sources=source,
        functions=[function],
        build_directory=build,
        verbose=False,
    )


def _body_memcpy(x, n):
    _dst[:n].copy_(x[:n])
    return _dst


def _body_cluster(x, n):
    _cluster_ext.launch_clustered(_dst, 2 * n)
    return _dst


def _body_pdl(x, n):
    _pdl_ext.launch_pdl(_dst, 2 * n)
    return _dst


def _before_replay():
    if _dst is not None:
        _dst.fill_(-1.0)


def _workload_body():
    workload = os.environ.get(WORKLOAD_ENV)
    bodies = {
        "triton": _body_triton,
        "memcpy": _body_memcpy,
        "cluster": _body_cluster,
        "pdl": _body_pdl,
    }
    return bodies.get(workload, _body)


def _prefix():
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator."""
    global _dst
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.randn(max(SIZES), HIDDEN, generator=g).cuda()
    if os.environ.get(WORKLOAD_ENV) == "memcpy":
        _dst = torch.full_like(x, -1.0)
    elif os.environ.get(WORKLOAD_ENV) in ("cluster", "pdl"):
        _dst = torch.full((2 * max(SIZES),), -1.0, device="cuda")
    body = _workload_body()
    with torch.cuda.stream(stream):
        for n in SIZES if body is _body_triton else SIZES[:1]:
            body(x, n)  # warm-up: lazy module loads, Triton compiles
    torch.cuda.synchronize()
    return x


def _graph_paths(archive):
    # save_graph_manifest expects graph_<index>_<suffix>.json
    return [os.path.join(archive, f"graph_{i}_bs{n}.json") for i, n in enumerate(SIZES)]


def _load_workload_extension(archive):
    global _cluster_ext, _pdl_ext
    if os.environ.get(WORKLOAD_ENV) == "cluster":
        _cluster_ext = _extension(
            archive, "foundry_test_exec_share_cluster", _CLUSTER_SOURCE, "launch_clustered"
        )
    elif os.environ.get(WORKLOAD_ENV) == "pdl":
        _pdl_ext = _extension(archive, "foundry_test_exec_share_pdl", _PDL_SOURCE, "launch_pdl")


def _run_save(archive):
    import foundry as fdry

    _load_workload_extension(archive)
    x = _prefix()
    body = _workload_body()
    graphs, outs = [], []
    for path, n in zip(_graph_paths(archive), SIZES):
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = body(x, n)
        _before_replay()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.clone(), body(x, n))
        graph.save(path, output_tensors=out)
        graphs.append(graph)
        outs.append(out.cpu().clone())
    fdry.save_graph_manifest(archive)
    save_reference(archive, outs, [x])
    fdry.stop_allocation_region()


def _replay_ms(graph, iters):
    """Per-replay GPU (event) and CPU (replay call) medians in us."""
    gpu, cpu = [], []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t = time.perf_counter()
        graph.replay()
        cpu.append((time.perf_counter() - t) * 1e6)
        end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end) * 1e3)
    return statistics.median(gpu), statistics.median(cpu)


def _run_load(archive):
    import foundry as fdry

    mode = "share" if os.environ.get(SHARE_ENV) == "1" else "eager"
    _load_workload_extension(archive)
    fdry.load_cuda_modules_and_libraries("hook_archive")
    x = _prefix()
    ref_outs = load_reference(archive, [x])

    paths = _graph_paths(archive)
    fdry.CUDAGraph.start_exec_pool_prewarm(paths)  # as SGLang LOAD does; no-op in share mode
    pending = fdry.CUDAGraph.start_graph_builds(paths)
    loaded = [fdry.CUDAGraph.finish_one_graph_load(pending, i) for i in range(len(paths))]
    graphs = [g for g, _ in loaded]
    outs = [o if isinstance(o, torch.Tensor) else o[0] for _, o in loaded]

    replays = []
    for i in ORDER:
        _before_replay()
        graphs[i].replay()
        torch.cuda.synchronize()
        got = outs[i].cpu().clone()
        assert torch.equal(got, ref_outs[i]), f"{mode}: graph {i} (rows {SIZES[i]}) != SAVE"
        replays.append(got)
    torch.save(replays, os.path.join(archive, f"load_{mode}.pt"))

    # Timing (printed only): a switch alternates two members, a repeat replays one.
    a, b = len(SIZES) - 2, len(SIZES) - 1
    sw = [_replay_ms(graphs[i], 1) for _ in range(TIMING_ITERS) for i in (a, b)]
    sw_gpu = statistics.median(t[0] for t in sw)
    sw_cpu = statistics.median(t[1] for t in sw)
    graphs[a].replay()
    rep_gpu, rep_cpu = _replay_ms(graphs[a], 2 * TIMING_ITERS)
    print(
        f"[exec-share-test] {mode}: switch replay gpu {sw_gpu:.1f} us, cpu {sw_cpu:.1f} us; "
        f"repeat replay gpu {rep_gpu:.1f} us, cpu {rep_cpu:.1f} us"
    )

    stats = {}
    for g in graphs:
        s = json.loads(g.exec_share_stats())
        stats.setdefault(str(s.get("exec")), s)  # one entry per shared exec
    print("EXEC_SHARE_STATS " + json.dumps({"mode": mode, "execs": list(stats.values())}))
    fdry.stop_allocation_region()


def _spawn(cmd_mode, workdir, share=False, workload="elementwise"):
    """SAVE/LOAD child; returns its combined stdout/stderr."""
    env = {WORKLOAD_ENV: workload}
    if share:
        env[SHARE_ENV] = "1"
    return run_phase(__file__, cmd_mode, workdir, env=env)


def _phase2(log):
    m = PHASE2_RE.search(log)
    assert m, "no Phase 2 summary line"
    templates, members, total, member_execs = map(int, m.groups())
    return templates, members, total, member_execs


def _stats(log):
    line = next(ln for ln in log.splitlines() if ln.startswith("EXEC_SHARE_STATS "))
    return json.loads(line.split(" ", 1)[1])


@pytest.fixture(scope="module", params=["elementwise", "triton", "memcpy", "cluster", "pdl"])
def runs(request, tmp_path_factory):
    workload = request.param
    if workload == "triton" and not _HAS_TRITON:
        pytest.skip("triton not installed")
    if workload in ("cluster", "pdl") and (
        shutil.which("nvcc") is None or torch.cuda.get_device_capability()[0] < 9
    ):
        pytest.skip("CUDA toolkit and sm90+ required")
    work = tmp_path_factory.mktemp(f"exec_share_{workload}")
    _spawn("save", work, workload=workload)
    assert (work / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
    return (
        work,
        _spawn("load", work, workload=workload),
        _spawn("load", work, share=True, workload=workload),
    )


def test_eager_instantiates_every_member(runs):
    _, eager, _ = runs
    templates, members, total, member_execs = _phase2(eager)
    assert total == len(SIZES)
    assert members > 0, "SAVE produced no template members; nothing to share"
    assert member_execs == members
    assert "[foundry] exec share:" not in eager
    assert all(not s["enabled"] for s in _stats(eager)["execs"])


# Workloads whose members differ from the template in copy extents or clustered launch dims:
# share mode gives each graph its own template.
_SPLIT_WORKLOADS = ("exec_share_memcpy", "exec_share_cluster")


def test_share_instantiates_templates_only(runs):
    work, _, share = runs
    templates, members, total, member_execs = _phase2(share)
    assert total == len(SIZES)
    if work.name.startswith(_SPLIT_WORKLOADS):
        assert (templates, members) == (len(SIZES), 0), (templates, members)
        assert REFINE_RE.search(share).groups() == ("1", str(len(SIZES)))
        return
    assert members > 0, "share mode refined every group to one graph; no switch tested"
    assert member_execs == 0
    assert "[foundry] exec share:" in share
    assert not [ln for ln in share.splitlines() if ln.startswith("[foundry] exec pool prewarm")]
    execs = _stats(share)["execs"]
    assert all(s["enabled"] and s["valid"] for s in execs)
    assert sum(s["updates"] for s in execs) > 0, "no member switch happened"


def test_share_splits_kernel_binaries(runs):
    work, eager, share = runs
    if not work.name.startswith("exec_share_triton"):
        pytest.skip("Triton workload only")
    assert _phase2(eager)[0] == 1, "eager keeps the saved group"
    m = REFINE_RE.search(share)
    assert m, "no share-mode refinement line"
    saved, templates = map(int, m.groups())
    # Same names and DAG, two binaries of _affine_kernel: one template per binary.
    assert (saved, templates) == (1, 2)
    assert _phase2(share)[:2] == (2, len(SIZES) - 2)


@pytest.mark.skipif(_HAS_CGP, reason="cudagraph-patch preloaded")
def test_share_route_real(runs):
    _, _, share = runs
    assert ROUTE_LINE[False] in share
    assert all(s["route"] == "cuGraphExecUpdate" for s in _stats(share)["execs"])


@pytest.mark.skipif(
    not (_HAS_CGP and os.environ.get("CUDAGRAPH_PATCH") == "1"),
    reason=(
        "cudagraph-patch not preloaded (cgp_exec_update unresolved)"
        if not _HAS_CGP
        else "CUDAGRAPH_PATCH=1 not set: the preloaded patch passes every update through"
    ),
)
def test_share_route_patched(runs):
    """The patch must be active, not merely resolvable: with LD_PRELOAD alone it logs
    'patch inactive' and passes every update to the driver."""
    _, _, share = runs
    assert ROUTE_LINE[True] in share
    assert "patch inactive" not in share
    assert "installed execupdate_prefetch profile" in share
    assert all(s["route"] == "cgp" for s in _stats(share)["execs"])


def test_outputs_identical_between_modes(runs):
    work, _, _ = runs
    eager = torch.load(work / "archive" / "load_eager.pt")
    share = torch.load(work / "archive" / "load_share.pt")
    assert len(eager) == len(share) == len(ORDER)
    for k, (a, b) in enumerate(zip(eager, share)):
        assert torch.equal(a, b), f"replay {k} (graph {ORDER[k]}): eager != share"


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
