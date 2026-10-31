# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The VMM allocation region (1 GPU): placement, preallocation, stop / resume,
live ranges and the allocation cursor. Each case runs in a child process with
libcuda_hook.so preloaded.

    pytest tests/core/test_allocation_region.py -q -s

The allocation timing benchmarks are in tools/bench/bench_preallocation.py.
"""

import json
import os
import sys

import pytest
import torch
from _helpers import run_phase

BASE_ADDR = 0x500000000000
REGION_SIZE = 1 << 30
MB = 1 << 20


def _in_region(t, base=BASE_ADDR, size=REGION_SIZE):
    return base <= t.data_ptr() and t.data_ptr() + t.numel() * t.element_size() <= base + size


def _assert_in_region(tensors, base=BASE_ADDR, size=REGION_SIZE):
    for i, t in enumerate(tensors):
        assert _in_region(t, base, size), (
            f"tensor {i} at 0x{t.data_ptr():x} is outside [0x{base:x}, 0x{base + size:x})"
        )


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _alloc_three():
    # 4 MB, 16 MB, 1 MB: different caching-allocator size classes.
    return [
        torch.randn(1000, 1000, device="cuda"),
        torch.randn(2000, 2000, device="cuda"),
        torch.randn(500, 500, device="cuda"),
    ]


def _run_placement(archive, api, prealloc):
    """Allocations land inside the region, through either API, with or without
    a preallocated span."""
    import foundry as fdry

    torch.cuda.init()
    prealloc_size = "128MB" if prealloc == "prealloc" else None

    def check():
        tensors = _alloc_three()
        _assert_in_region(tensors)
        assert tensors[0].data_ptr() < tensors[1].data_ptr(), "region addresses must increase"
        del tensors
        torch.cuda.empty_cache()

    if api == "ctx":
        with fdry.allocation_region(BASE_ADDR, "1GB", prealloc_size=prealloc_size):
            check()
    else:
        fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
        try:
            if prealloc_size is not None:
                assert fdry.preallocate_region(fdry.parse_size(prealloc_size)), "preallocation"
            check()
            if prealloc_size is not None:
                fdry.free_preallocated_region()
        finally:
            fdry.stop_allocation_region()
    print("[TEST] placement OK")


def _run_size_formats(archive):
    """Region bases as int or hex string, sizes as strings or bytes."""
    import foundry as fdry

    torch.cuda.init()
    # Well below the mmap area near 0x7f...: a base there can collide with a
    # mapping the process already has (an 8 GB PROT_NONE reservation at
    # 0x7effb0000000 on one host), and the hook then disables the region.
    cases = [
        (0x510000000000, "1GB"),
        (0x511000000000, "512MB"),
        (0x512000000000, "100MB"),
        ("0x513000000000", 50 * MB),
    ]
    for base, size in cases:
        base_int = int(base, 16) if isinstance(base, str) else base
        with fdry.allocation_region(base, size):
            t = torch.randn(100, 100, device="cuda")
            _assert_in_region([t], base_int, fdry.parse_size(size))
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[TEST] size formats OK")


def _run_fallback(archive):
    """An allocation past the end of the preallocated span is mapped on demand
    (slow path), in the region and contiguous with the span."""
    import foundry as fdry

    torch.cuda.init()
    span = 32 * MB
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    try:
        start = fdry.get_current_alloc_offset()
        assert fdry.preallocate_region(span), "preallocation"
        a = torch.empty(span, dtype=torch.uint8, device="cuda")  # fills the span (fast path)
        b = torch.empty(span, dtype=torch.uint8, device="cuda")  # past it (slow path)
        _assert_in_region([a, b])
        assert a.data_ptr() == BASE_ADDR + start, hex(a.data_ptr())
        assert b.data_ptr() == a.data_ptr() + span, "slow path must continue at the span end"
        a.fill_(1)
        b.fill_(2)
        torch.cuda.synchronize()
        assert int(a.sum().item()) == span and int(b.sum().item()) == 2 * span
        del a, b
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()
    finally:
        fdry.stop_allocation_region()
    print("[TEST] fallback OK")


def _run_stop_resume(archive):
    """stop_allocation_region routes allocations (here a private MemPool) out
    of the region; resume_allocation_region continues at the region cursor."""
    import foundry as fdry

    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    t1 = torch.full((2 * MB,), 1.0, device="cuda")
    fdry.stop_allocation_region()
    pool = torch.cuda.memory.MemPool()
    with torch.cuda.use_mem_pool(pool):
        t2 = torch.full((2 * MB,), 2.0, device="cuda")
    fdry.resume_allocation_region()
    t3 = torch.full((2 * MB,), 3.0, device="cuda")
    _assert_in_region([t1, t3])
    assert not _in_region(t2), f"t2 at 0x{t2.data_ptr():x} was allocated while stopped"
    assert t3.data_ptr() > t1.data_ptr(), "resume must continue above the earlier allocations"
    fdry.stop_allocation_region()
    print("[TEST] stop/resume OK")


def _offsets_prefix():
    import foundry as fdry

    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    return fdry.get_current_alloc_offset()


def _covered(ranges, off, size):
    return any(o <= off and off + size <= o + n for o, n in ranges)


def _run_record(archive):
    """SAVE side of the sglang LOAD memory restore: A, B, C allocated, A freed;
    the live ranges cover B and C only."""
    import foundry as fdry

    start = _offsets_prefix()
    a = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    b = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    a_off = a.data_ptr() - BASE_ADDR
    del a
    torch.cuda.empty_cache()
    ranges = [list(r) for r in fdry.get_live_region_ranges()]
    final = fdry.get_current_alloc_offset()
    b_off, c_off = b.data_ptr() - BASE_ADDR, c.data_ptr() - BASE_ADDR
    assert not any(o <= a_off < o + n for o, n in ranges), f"freed A still live: {ranges}"
    assert _covered(ranges, b_off, b.numel()) and _covered(ranges, c_off, c.numel()), ranges
    assert final >= c_off + c.numel(), (final, c_off)
    with open(os.path.join(archive, "layout.json"), "w") as f:
        json.dump({"start": start, "ranges": ranges, "final": final, "b": b_off, "c": c_off}, f)
    fdry.stop_allocation_region()
    print("[TEST] record OK")


def _run_replay(archive):
    """LOAD side: map only the recorded live ranges, skip A's hole with
    set_current_alloc_offset, and B, C land at their SAVE addresses."""
    import foundry as fdry

    with open(os.path.join(archive, "layout.json")) as f:
        rec = json.load(f)
    start = _offsets_prefix()
    assert start == rec["start"], (start, rec["start"])
    ranges = [tuple(r) for r in rec["ranges"]]
    assert fdry.preallocate_ranges(ranges, rec["final"]), "preallocate_ranges"
    assert fdry.get_current_alloc_offset() == start, "preallocate_ranges must not move the cursor"
    fdry.set_current_alloc_offset(rec["b"])
    assert fdry.get_current_alloc_offset() == rec["b"]
    fdry.set_current_alloc_offset(start)  # backwards: refused, cursor unchanged
    assert fdry.get_current_alloc_offset() == rec["b"]
    b = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    assert b.data_ptr() - BASE_ADDR == rec["b"], hex(b.data_ptr())
    assert c.data_ptr() - BASE_ADDR == rec["c"], hex(c.data_ptr())
    assert fdry.get_current_alloc_offset() == rec["final"]
    assert [tuple(r) for r in fdry.get_live_region_ranges()] == ranges
    b.fill_(1)
    c.fill_(2)
    torch.cuda.synchronize()
    assert int(b.sum().item()) == b.numel() and int(c.sum().item()) == 2 * c.numel()
    del b, c
    torch.cuda.empty_cache()
    fdry.free_preallocated_region()
    fdry.stop_allocation_region()
    print("[TEST] replay OK")


def _run_exhaustion(archive):
    """An allocation that does not fit in the rest of the region fails with
    out-of-memory and leaves the cursor in place: an enabled region never hands
    out an address of the driver's choosing (a graph captured with it would
    point elsewhere on LOAD)."""
    import foundry as fdry

    torch.cuda.init()
    size = 64 * MB
    fdry.set_allocation_region(BASE_ADDR, size)
    try:
        a = torch.empty(48 * MB, dtype=torch.uint8, device="cuda")
        _assert_in_region([a], BASE_ADDR, size)
        offset = fdry.get_current_alloc_offset()
        with pytest.raises(torch.cuda.OutOfMemoryError):
            torch.empty(48 * MB, dtype=torch.uint8, device="cuda")
        assert fdry.get_current_alloc_offset() == offset, "a refused allocation moved the cursor"
        # 48 + 12 MB still fits (above 10 MB torch's segment is the request rounded to 2 MB).
        b = torch.empty(12 * MB, dtype=torch.uint8, device="cuda")
        _assert_in_region([b], BASE_ADDR, size)
    finally:
        fdry.stop_allocation_region()
    print("[TEST] exhaustion OK")


def _run_reserve_failure(archive):
    """A region whose range is already reserved (here by this process, on
    another thread: the region state is per thread) raises, through both APIs,
    instead of leaving the allocations outside the region."""
    import threading

    import foundry as fdry

    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    errors = []

    def other_thread():
        torch.cuda.set_device(0)
        try:
            fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            with fdry.allocation_region(BASE_ADDR, REGION_SIZE):
                pass
        except RuntimeError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    fdry.stop_allocation_region()
    assert len(errors) == 2 and all("could not be reserved" in e for e in errors), errors
    print("[TEST] reserve failure OK")


def _run_free_live_span(archive):
    """free_preallocated_region keeps the span mapped while an allocation in
    it is live: unmapping it would leave the tensor on unmapped memory."""
    import foundry as fdry

    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    try:
        assert fdry.preallocate_region(64 * MB), "preallocation"
        t = torch.full((4 * MB,), 3, dtype=torch.uint8, device="cuda")
        assert not fdry.free_preallocated_region(), "released a span with a live tensor in it"
        assert int(t.sum().item()) == 3 * 4 * MB, "the live tensor lost its memory"
        del t
        torch.cuda.empty_cache()
        assert fdry.free_preallocated_region(), "the empty span was not released"
    finally:
        fdry.stop_allocation_region()
    print("[TEST] free live span OK")


def _run_span_and_later_region(archive):
    """A preallocated span that stays mapped (a live tensor in it) serves no
    allocation of a region set later at a lower base: releasing the span
    afterwards must leave that region's tensors mapped."""
    import foundry as fdry

    torch.cuda.init()
    high = BASE_ADDR + 4 * REGION_SIZE
    fdry.set_allocation_region(high, REGION_SIZE)
    assert fdry.preallocate_region(64 * MB), "preallocation"
    kept = torch.full((4 * MB,), 1, dtype=torch.uint8, device="cuda")
    assert not fdry.free_preallocated_region(), "released a span with a live tensor in it"
    fdry.stop_allocation_region()
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    try:
        # Larger than what is left of the segment holding `kept`: a new allocation.
        t = torch.full((64 * MB,), 2, dtype=torch.uint8, device="cuda")
        _assert_in_region([t])
        del kept
        torch.cuda.empty_cache()
        assert fdry.free_preallocated_region(), "the empty span was not released"
        assert int(t.sum().item()) == 2 * 64 * MB, "a tensor of the later region lost its memory"
    finally:
        fdry.stop_allocation_region()
    print("[TEST] span and later region OK")


def _ctx_create_and_alloc(count):
    """cuCtxCreate on this thread, then ``count`` 1 GiB cuMemAlloc_v2 calls
    through the hook; returns their addresses (freed, context destroyed)."""
    import ctypes

    hooked, real = ctypes.CDLL(None), ctypes.CDLL("libcuda.so.1")
    u64, ptr = ctypes.c_uint64, ctypes.POINTER
    real.cuInit.argtypes = [ctypes.c_uint]
    real.cuDeviceGet.argtypes = [ptr(ctypes.c_int), ctypes.c_int]
    real.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    hooked.cuCtxCreate_v2.argtypes = [ptr(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    hooked.cuMemAlloc_v2.argtypes = [ptr(u64), ctypes.c_size_t]
    hooked.cuMemFree_v2.argtypes = [u64]
    assert real.cuInit(0) == 0
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    assert real.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert hooked.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
    ptrs = []
    try:
        for _ in range(count):
            p = u64()
            rc = hooked.cuMemAlloc_v2(ctypes.byref(p), 1 << 30)
            assert rc == 0, f"cuMemAlloc_v2 of 1 GiB returned {rc}"
            ptrs.append(p.value)
    finally:
        for p in ptrs:
            hooked.cuMemFree_v2(p)
        real.cuCtxDestroy_v2(ctx)
    return ptrs


def _in_default_region(ptrs):
    base, size = 0x500000000000, 2 << 30
    return [base <= p < base + size for p in ptrs]


def _run_ctx_default_region(archive):
    """With FOUNDRY_CTX_DEFAULT_REGION=1 the first cuCtxCreate gives its thread
    a default 2 GiB region at 0x500000000000, which no SAVE or LOAD
    configures: an allocation past its end goes to the driver, as before
    regions were checked, and does not fail."""
    ptrs = _ctx_create_and_alloc(3)  # 3 x 1 GiB: the third does not fit
    assert _in_default_region(ptrs) == [True, True, False], [hex(p) for p in ptrs]
    print("[TEST] cuCtxCreate default region OK")


def _run_ctx_keeps_region(archive):
    """With the opt-in, a cuCtxCreate on a thread that already has a region
    keeps that region (strict: what does not fit fails)."""
    import ctypes

    import foundry as fdry

    torch.cuda.init()
    torch.cuda.set_device(0)
    base = BASE_ADDR + 8 * REGION_SIZE  # away from the default region
    fdry.set_allocation_region(base, REGION_SIZE)
    hooked, real = ctypes.CDLL(None), ctypes.CDLL("libcuda.so.1")
    u64, ptr = ctypes.c_uint64, ctypes.POINTER
    real.cuDeviceGet.argtypes = [ptr(ctypes.c_int), ctypes.c_int]
    real.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    hooked.cuCtxCreate_v2.argtypes = [ptr(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    hooked.cuMemAlloc_v2.argtypes = [ptr(u64), ctypes.c_size_t]
    hooked.cuMemFree_v2.argtypes = [u64]
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    assert real.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert hooked.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
    try:
        p = u64()
        assert hooked.cuMemAlloc_v2(ctypes.byref(p), 512 * MB) == 0
        try:
            assert base <= p.value < base + REGION_SIZE, hex(p.value)
            q = u64()
            rc = hooked.cuMemAlloc_v2(ctypes.byref(q), REGION_SIZE)
            assert rc == 2, f"1 GiB past the kept region returned {rc}, expected out-of-memory"
        finally:
            hooked.cuMemFree_v2(p.value)
    finally:
        real.cuCtxDestroy_v2(ctx)
        fdry.stop_allocation_region()
    print("[TEST] cuCtxCreate keeps the thread's region OK")


def _run_ctx_no_default_region(archive):
    """Without the opt-in a thread that creates its own context keeps the
    driver's allocator (processes that only inherit the preload)."""
    ptrs = _ctx_create_and_alloc(1)
    assert _in_default_region(ptrs) == [False], [hex(p) for p in ptrs]
    print("[TEST] cuCtxCreate without a default region OK")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prealloc", ["no_prealloc", "prealloc"])
@pytest.mark.parametrize("api", ["ctx", "set_stop"])
def test_allocations_land_in_region(workdir, api, prealloc):
    run_phase(__file__, "placement", workdir, api, prealloc)


def test_region_size_formats(workdir):
    run_phase(__file__, "size-formats", workdir)


def test_fallback_past_the_preallocated_span(workdir):
    run_phase(__file__, "fallback", workdir)


def test_stop_and_resume(workdir):
    run_phase(__file__, "stop-resume", workdir)


def test_live_ranges_and_cursor_restore_the_layout(workdir):
    run_phase(__file__, "record", workdir)
    run_phase(__file__, "replay", workdir)


def test_allocation_past_the_region_end_fails(workdir):
    run_phase(__file__, "exhaustion", workdir)


def test_occupied_region_base_raises(workdir):
    run_phase(__file__, "reserve-failure", workdir)


def test_live_preallocation_is_not_released(workdir):
    run_phase(__file__, "free-live-span", workdir)


def test_kept_span_serves_no_later_region(workdir):
    out = run_phase(__file__, "span-later-region", workdir)
    assert "on demand at" not in out, "an allocation of the later region was served from the span"


def test_context_default_region_falls_back_to_the_driver(workdir):
    run_phase(__file__, "ctx-default-region", workdir, env={"FOUNDRY_CTX_DEFAULT_REGION": "1"})


def test_context_gets_no_default_region_without_the_opt_in(workdir):
    run_phase(__file__, "ctx-no-default-region", workdir)


def test_context_keeps_an_existing_region(workdir):
    run_phase(__file__, "ctx-keeps-region", workdir, env={"FOUNDRY_CTX_DEFAULT_REGION": "1"})


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {
        "--placement": _run_placement,
        "--size-formats": _run_size_formats,
        "--fallback": _run_fallback,
        "--stop-resume": _run_stop_resume,
        "--record": _run_record,
        "--replay": _run_replay,
        "--exhaustion": _run_exhaustion,
        "--reserve-failure": _run_reserve_failure,
        "--free-live-span": _run_free_live_span,
        "--span-later-region": _run_span_and_later_region,
        "--ctx-default-region": _run_ctx_default_region,
        "--ctx-no-default-region": _run_ctx_no_default_region,
        "--ctx-keeps-region": _run_ctx_keeps_region,
    }[mode](archive, *rest)
