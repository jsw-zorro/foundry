# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""A graph kernel from a module loaded as PTX text is restored (1 GPU, no vLLM/SGLang).

    pytest tests/core/test_ptx_module.py -q -s

The driver reads in-memory PTX as a C string, and SAVE archives the text
without its terminating NUL. SAVE loads a one-kernel PTX module with
cuModuleLoadData (through the preloaded hook, which records it), captures one
launch of it and saves the graph; LOAD restores the module from the archive and
the graph must replay like SAVE.
"""

import ctypes
import os
import sys

import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

BASE_ADDR = 0x4EC000000000
REGION_SIZE_STR = "1GB"
N = 32

# out[tid] += 1.0 for one block of N threads.
PTX = r"""
.version 7.0
.target sm_80
.address_size 64

.visible .entry foundry_ptx_add_one(
	.param .u64 foundry_ptx_add_one_param_0
)
{
	.reg .f32 	%f<3>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<5>;

	ld.param.u64 	%rd1, [foundry_ptx_add_one_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	add.f32 	%f2, %f1, 0f3F800000;
	st.global.f32 	[%rd4], %f2;
	ret;
}
"""


def _driver():
    """(cuModuleLoadData of the preloaded hook, which records the image; the driver itself)."""
    hooked, real = ctypes.CDLL(None), ctypes.CDLL("libcuda.so.1")
    hooked.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    real.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    real.cuLaunchKernel.argtypes = (
        [ctypes.c_void_p]
        + [ctypes.c_uint] * 7
        + [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
    )
    return hooked, real


def _prefix():
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    out = torch.arange(N, dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    return out


def _run_save(archive):
    import foundry as fdry

    out = _prefix()
    hooked, real = _driver()
    module, func = ctypes.c_void_p(), ctypes.c_void_p()
    assert hooked.cuModuleLoadData(ctypes.byref(module), PTX.encode()) == 0, "cuModuleLoadData"
    assert real.cuModuleGetFunction(ctypes.byref(func), module, b"foundry_ptx_add_one") == 0
    ptr = ctypes.c_void_p(out.data_ptr())
    args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(ptr), ctypes.c_void_p))

    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        stream = torch.cuda.current_stream().cuda_stream
        rc = real.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, stream, args, None)
        assert rc == 0, f"cuLaunchKernel {rc}"
    graph.replay()
    torch.cuda.synchronize()
    want = torch.arange(N, dtype=torch.float32) + 1
    assert torch.equal(out.cpu(), want), "SAVE replay"
    graph.save(os.path.join(archive, "graph.json"))
    save_reference(archive, out, [out])
    fdry.stop_allocation_region()


def _run_load(archive):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    out = _prefix()
    want = load_reference(archive, [out])
    pending = fdry.CUDAGraph.start_graph_builds([os.path.join(archive, "graph.json")])
    graph, _ = fdry.CUDAGraph.finish_one_graph_load(pending, 0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), want), f"LOAD replay {out.cpu()} != SAVE {want}"
    fdry.stop_allocation_region()


def test_ptx_module_kernel_is_restored(tmp_path):
    run_phase(__file__, "save", tmp_path)
    run_phase(__file__, "load", tmp_path)


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
