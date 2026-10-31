# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The preflight CLI (python -P -m foundry.integration.sglang.preflight) on the
recipe TOML: SAVE and LOAD checks, the FAILED line and the exit codes
(0 ok, 1 a failed check, 2 bad arguments). No GPU.

    pytest tests/sgl/test_preflight.py -v

recipe/sglang/foundry.toml has no ``mode`` key (the mode comes from
--cuda-graph-persistence), so it passes only on the dependency route; on the
plugin route the CLI must refuse it.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import CONFIG_ENV, plugin_route_only, sglang_has_adapter  # noqa: E402
from foundry.integration.sglang import preflight  # noqa: E402

RECIPE_TOML = Path(__file__).resolve().parents[2] / "recipe" / "sglang" / "foundry.toml"
ARCHIVE = "foundry_archive"  # workspace_root of the recipe TOML

dependency_route_only = pytest.mark.skipif(
    not sglang_has_adapter(), reason="dependency route only: this sglang has no foundry_adapter"
)


def _preflight(cwd, *args):
    """Run the CLI from ``cwd`` (no sglang checkout to shadow the package)."""
    env = {k: v for k, v in os.environ.items() if k not in (CONFIG_ENV, "SGLANG_PLUGINS")}
    proc = subprocess.run(
        [sys.executable, "-P", "-m", "foundry.integration.sglang.preflight", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(proc.stdout + proc.stderr)
    return proc


def _recipe(tmp_path, *mode):
    return _preflight(tmp_path, "--toml", str(RECIPE_TOML), "--cwd", str(tmp_path), *mode)


@dependency_route_only
def test_recipe_toml_save_passes(tmp_path):
    proc = _recipe(tmp_path, "--save")
    assert proc.returncode == 0, proc.stderr
    assert f"{preflight.PREFIX} ok: TOML {RECIPE_TOML} mode=save" in proc.stdout
    assert "FAILED" not in proc.stderr


@dependency_route_only
def test_recipe_toml_load_without_archive_fails(tmp_path):
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 1
    assert f"{preflight.PREFIX} FAILED: LOAD archive {tmp_path / ARCHIVE}" in proc.stderr
    assert "run --save first" in proc.stderr


@dependency_route_only
def test_recipe_toml_load_with_empty_archive_fails(tmp_path):
    (tmp_path / ARCHIVE).mkdir()
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 1
    assert "has no rank_* dirs" in proc.stderr


def _rank_records(rank_dir, version=None, layout_id="run", fingerprint_id="run"):
    import json

    from foundry.integration.sglang import config

    rank_dir.mkdir(parents=True)
    layout = {
        "capture_loop_version": config.CAPTURE_LOOP_VERSION if version is None else version,
        config.ARCHIVE_SAVE_ID_KEY: layout_id,
    }
    (rank_dir / config.ARCHIVE_LAYOUT_FILE).write_text(json.dumps(layout))
    (rank_dir / config.ARCHIVE_FINGERPRINT_FILE).write_text(
        json.dumps({config.ARCHIVE_SAVE_ID_KEY: fingerprint_id})
    )


@dependency_route_only
def test_recipe_toml_load_with_rank_dirs_passes(tmp_path):
    for rank in (0, 1):
        _rank_records(tmp_path / ARCHIVE / f"rank_{rank}")
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 0, proc.stderr
    assert "mode=load" in proc.stdout and "(2 rank dirs)" in proc.stdout


@dependency_route_only
@pytest.mark.parametrize(
    "records, message",
    [
        (None, "region_layout.json not found"),
        ({"version": 2}, "capture_loop_version=2"),
        ({"layout_id": "b"}, "different SAVE runs"),
    ],
)
def test_recipe_toml_load_with_an_incomplete_rank_dir_fails(tmp_path, records, message):
    """The records LOAD reads first, checked as LOAD checks them."""
    if records is None:
        (tmp_path / ARCHIVE / "rank_0").mkdir(parents=True)
    else:
        _rank_records(tmp_path / ARCHIVE / "rank_0", **records)
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 1
    assert message in proc.stderr and "re-SAVE" in proc.stderr


def _load_toml(tmp_path, hook=None):
    """A LOAD TOML with its mode in the file, for check_toml on either route."""
    if hook is None:
        hook = tmp_path / "libcuda_hook.so"
        hook.write_bytes(b"")
    toml = tmp_path / "load.toml"
    toml.write_text(f'mode = "load"\nhook_library_path = "{hook}"\nworkspace_root = "{ARCHIVE}"\n')
    return str(toml)


def test_load_check_passes_complete_rank_dirs(tmp_path):
    for rank in (0, 1):
        _rank_records(tmp_path / ARCHIVE / f"rank_{rank}")
    assert "(2 rank dirs)" in preflight.check_toml(_load_toml(tmp_path), "load", tmp_path)


@pytest.mark.parametrize(
    "records, message",
    [
        (None, "region_layout.json not found"),
        ({"version": 2}, "capture_loop_version=2"),
        ({"layout_id": "b"}, "different SAVE runs"),
    ],
)
def test_load_check_reads_every_rank_dir(tmp_path, records, message):
    _rank_records(tmp_path / ARCHIVE / "rank_0")
    if records is None:
        (tmp_path / ARCHIVE / "rank_1").mkdir(parents=True)
    else:
        _rank_records(tmp_path / ARCHIVE / "rank_1", **records)
    pattern = rf"rank_1: .*{re.escape(message)}.*re-SAVE"
    with pytest.raises(preflight.PreflightError, match=pattern):
        preflight.check_toml(_load_toml(tmp_path), "load", tmp_path)


def _system_shared_object():
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1] if len(line.split()) >= 6 else ""
        if os.path.basename(path).startswith(("libm.so", "libm-")):
            return path
    pytest.skip("no libm mapped in this process")


def test_bare_hook_name_is_resolved_by_the_loader(tmp_path, monkeypatch):
    """A bare hook_library_path passes when ld.so finds it for this interpreter
    (here through LD_LIBRARY_PATH) and fails when it does not."""
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(_system_shared_object(), lib / "libfoundry_test_hook.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(lib))
    assert preflight._loader_resolves("libfoundry_test_hook.so") is True
    assert preflight._loader_resolves("libfoundry_missing_hook.so") is False
    _rank_records(tmp_path / ARCHIVE / "rank_0")
    preflight.check_toml(_load_toml(tmp_path, "libfoundry_test_hook.so"), "load", tmp_path)
    with pytest.raises(preflight.PreflightError, match="not found by ld.so"):
        preflight.check_toml(_load_toml(tmp_path, "libfoundry_missing_hook.so"), "load", tmp_path)
    # A relative element is relative to the directory the engine is served from.
    monkeypatch.setenv("LD_LIBRARY_PATH", "lib")
    assert preflight._loader_resolves("libfoundry_test_hook.so", tmp_path) is True
    assert preflight._loader_resolves("libfoundry_test_hook.so", lib) is False
    preflight.check_toml(_load_toml(tmp_path, "libfoundry_test_hook.so"), "load", tmp_path)


@dependency_route_only
def test_recipe_toml_needs_a_mode_flag(tmp_path):
    proc = _recipe(tmp_path)
    assert proc.returncode == 1
    assert "pass --save or --load" in proc.stderr


@plugin_route_only
def test_recipe_toml_is_refused_on_the_plugin_route(tmp_path):
    proc = _recipe(tmp_path, "--save")
    assert proc.returncode == 1
    assert "has mode = 'none', expected 'save'" in proc.stderr


def test_missing_toml_fails(tmp_path):
    proc = _preflight(tmp_path, "--toml", str(tmp_path / "missing.toml"), "--save")
    assert proc.returncode == 1
    assert (
        f"{preflight.PREFIX} FAILED: Foundry TOML {tmp_path / 'missing.toml'} does not exist"
        in (proc.stderr)
    )


@pytest.mark.parametrize(
    "args", [["--save"], ["--toml", str(RECIPE_TOML), "--save", "--load"]], ids=["no_toml", "both"]
)
def test_bad_arguments_exit_2(tmp_path, args):
    assert _preflight(tmp_path, *args).returncode == 2


@pytest.mark.parametrize("value", ["", "foundry", "other, foundry"])
def test_plugin_allowlist_accepts(value):
    preflight.check_plugin_allowlist({"SGLANG_PLUGINS": value} if value else {})


def test_plugin_allowlist_without_foundry_fails():
    with pytest.raises(preflight.PreflightError, match="does not include 'foundry'"):
        preflight.check_plugin_allowlist({"SGLANG_PLUGINS": "other"})
