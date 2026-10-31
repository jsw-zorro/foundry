# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The hook's dlsym (no GPU work).

    pytest tests/core/test_hook_dlsym.py -q -s

libcuda_hook.so defines dlsym: it returns its own functions for the driver
entry points it intercepts and passes every other lookup to the real dlsym,
which must resolve RTLD_NEXT and RTLD_DEFAULT from the original caller as
without the hook. Checked with two preloaded wrappers of one libc function
after the hook (the first must get the second's definition, the second
libc's), an RTLD_LOCAL tree (RTLD_DEFAULT must see its own dependency,
RTLD_NEXT gets libc's definition), a name only the calling wrapper defines
(NULL with dlerror set), and the vDSO-backed time functions. On x86-64 dlsym
is an assembly entry point that jumps into the real dlsym; elsewhere the hook
forwards from C++ and only the one-wrapper case holds.
"""

import os
import platform
import shutil
import subprocess
import sys

import pytest
from _helpers import run_phase

# Each wrapper records the definition its RTLD_NEXT lookup returned
# (<prefix>next); getppid returns -12345 when that is the wrapper's own, -1
# when the lookup found none, -999 when the wrappers call each other.
WRAPPER_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct timeval;
static __thread int depth;
void* WRAPPER_next;

pid_t getppid(void) {
  if (++depth > 50) { depth = 0; return -999; }
  pid_t (*next)(void) = (pid_t (*)(void))dlsym(RTLD_NEXT, "getppid");
  WRAPPER_next = (void*)next;
  pid_t r = next == NULL ? -1 : (next == getppid ? -12345 : next());
  depth--;
  return r;
}

#ifdef WITH_TIME
time_t time(time_t* t) {
  time_t (*next)(time_t*) = (time_t (*)(time_t*))dlsym(RTLD_NEXT, "time");
  return next && next != time ? next(t) : (time_t)-1;
}

int gettimeofday(struct timeval* tv, void* tz) {
  int (*next)(struct timeval*, void*) =
      (int (*)(struct timeval*, void*))dlsym(RTLD_NEXT, "gettimeofday");
  return next && next != gettimeofday ? next(tv, tz) : -1;
}

int foundry_only_in_wrapper_a(void) { return 0; }

int WRAPPER_missing(void) {
  dlerror();
  void* p = dlsym(RTLD_NEXT, "foundry_only_in_wrapper_a");
  const char* e = dlerror();
  return p == NULL && e != NULL && strstr(e, "foundry_only_in_wrapper_a") != NULL;
}
#endif
"""

DEP_C = "int foundry_dep_fn(void) { return 77; }\n"

# Loaded RTLD_LOCAL with DEP as its DT_NEEDED: RTLD_DEFAULT from here must see
# DEP (the scope of an RTLD_LOCAL tree, as for mooncake -> libibverbs).
TREE_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
int foundry_dep_fn(void);
void* tree_next;
int tree_default_lookup(void) {
  int (*f)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "foundry_dep_fn");
  return f ? f() : -1;
}
int tree_uses_dep(void) { return foundry_dep_fn(); }
pid_t tree_next_getppid(void) {
  pid_t (*f)(void) = (pid_t (*)(void))dlsym(RTLD_NEXT, "getppid");
  tree_next = (void*)f;
  return f ? f() : -1;
}
"""

ASM_DLSYM = platform.machine() == "x86_64"


def _owner(process, address):
    import ctypes

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    info = DlInfo()
    assert address, "no definition"
    assert process.dladdr(ctypes.c_void_p(address), ctypes.byref(info)) != 0
    return os.path.basename(os.fsdecode(info.dli_fname))


def _run_lookups(archive, expected_ppid, tree):
    import ctypes
    import time as pytime

    process = ctypes.CDLL(None)

    def pointer(lib, name):
        return ctypes.c_void_p.in_dll(lib, name).value

    got = process.getppid()
    assert got == int(expected_ppid), f"getppid through the two wrappers returned {got}"
    a_next, b_next = pointer(process, "wrapper_a_next"), pointer(process, "wrapper_b_next")
    assert b_next, "wrapper B was not reached"
    assert _owner(process, a_next) == "libwrap_b.so", _owner(process, a_next)
    assert _owner(process, b_next).startswith("libc."), _owner(process, b_next)
    assert process.wrapper_a_missing() == 1, (
        "a name only the caller defines: not NULL, or no dlerror"
    )
    process.time.restype = ctypes.c_long
    process.time.argtypes = [ctypes.c_void_p]
    assert abs(process.time(None) - pytime.time()) < 60, "time() through the wrapper"
    local = ctypes.CDLL(tree, mode=os.RTLD_LOCAL)
    assert local.tree_default_lookup() == 77, "RTLD_DEFAULT from an RTLD_LOCAL tree"
    assert local.tree_next_getppid() == int(expected_ppid), "RTLD_NEXT from an RTLD_LOCAL tree"
    tree_owner = _owner(process, pointer(local, "tree_next"))
    assert tree_owner.startswith("libc."), f"RTLD_NEXT from an RTLD_LOCAL tree: {tree_owner}"

    # Intercepted names come from the hook, through the process and through libcuda's handle.
    alloc = ctypes.cast(process.cuMemAlloc_v2, ctypes.c_void_p).value
    assert _owner(process, alloc).startswith("libcuda_hook"), _owner(process, alloc)
    driver = ctypes.CDLL("libcuda.so.1")
    proc_address = ctypes.cast(driver.cuGetProcAddress, ctypes.c_void_p).value
    assert _owner(process, proc_address).startswith("libcuda_hook"), _owner(process, proc_address)
    print("[TEST] dlsym lookups after the hook OK")


def _run_one_wrapper(archive, expected_ppid, tree):
    import ctypes

    got = ctypes.CDLL(None).getppid()
    assert got == int(expected_ppid), f"getppid through the wrapper returned {got}"
    local = ctypes.CDLL(tree, mode=os.RTLD_LOCAL)
    assert local.tree_default_lookup() == 77, "RTLD_DEFAULT from an RTLD_LOCAL tree"
    print("[TEST] one wrapper after the hook OK")


def _build(cc, workdir, name, source, *flags):
    src, out = workdir / f"{name}.c", workdir / f"lib{name}.so"
    src.write_text(source)
    subprocess.run([cc, "-O2", "-fPIC", "-shared", "-o", str(out), str(src), *flags], check=True)
    return out


@pytest.fixture
def libraries(workdir):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    wrap_a = WRAPPER_C.replace("WRAPPER_", "wrapper_a_")
    wrap_b = WRAPPER_C.replace("WRAPPER_", "wrapper_b_")
    _build(cc, workdir, "foundry_dep", DEP_C)
    rpath = [f"-L{workdir}", "-lfoundry_dep", f"-Wl,-rpath,{workdir}", "-ldl"]
    return {
        "a": _build(cc, workdir, "wrap_a", wrap_a, "-DWITH_TIME", "-ldl"),
        "b": _build(cc, workdir, "wrap_b", wrap_b, "-ldl"),
        "tree": _build(cc, workdir, "tree", TREE_C, *rpath),
    }


@pytest.mark.xfail(
    not ASM_DLSYM,
    strict=True,
    raises=AssertionError,
    reason="off x86-64 the hook forwards dlsym from C++ (known-issues.md, Preloads row)",
)
def test_lookups_after_the_hook_resolve_as_without_it(workdir, libraries):
    # run_phase preloads the hook first, then the LD_PRELOAD given here.
    preload = f"{libraries['a']}:{libraries['b']}"
    args = (os.getpid(), libraries["tree"])
    run_phase(__file__, "lookups", workdir, *args, env={"LD_PRELOAD": preload})


def test_one_wrapper_after_the_hook_reaches_libc(workdir, libraries):
    args = (os.getpid(), libraries["tree"])
    run_phase(__file__, "one-wrapper", workdir, *args, env={"LD_PRELOAD": str(libraries["b"])})


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--lookups": _run_lookups, "--one-wrapper": _run_one_wrapper}[mode](archive, *rest)
