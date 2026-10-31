# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Archive files without a GPU: the graph manifest and the archive scan.

pytest tests/unit/test_archive_files.py -q
"""

import json
import struct

import pytest
from foundry import save_graph_manifest

BASE = 0x600000000000
SIZE = 1 << 30


def _graph(path, key, deps=True):
    data = {"topology_key": key, "nodes": []}
    if deps:
        data["dependencies"] = [{"from": 0, "to": 1}]
    path.write_text(json.dumps(data))


def test_manifest_can_be_written_twice(tmp_path):
    """Graphs are ordered by index (graph_2 before graph_10), the first of a
    group is its template; the second call finds graph_manifest.json next to
    the graphs, which is not a graph, and writes the same manifest."""
    _graph(tmp_path / "graph_10_a.json", "K")
    _graph(tmp_path / "graph_2_b.json", "K")
    _graph(tmp_path / "graph_1_c.json", "L")
    save_graph_manifest(str(tmp_path))
    first = json.loads((tmp_path / "graph_manifest.json").read_text())
    save_graph_manifest(str(tmp_path))
    assert json.loads((tmp_path / "graph_manifest.json").read_text()) == first
    groups = {g["template"]: g["members"] for g in first["topology_groups"]}
    assert groups == {
        "graph_1_c.json": ["graph_1_c.json"],
        "graph_2_b.json": ["graph_2_b.json", "graph_10_a.json"],
    }
    # Members lose their dependency lists (LOAD rebuilds them on the template).
    assert "dependencies" not in json.loads((tmp_path / "graph_10_a.json").read_text())
    assert "dependencies" in json.loads((tmp_path / "graph_2_b.json").read_text())
    assert not list(tmp_path.glob("*.tmp"))


def _cugraph(path, *, extra_args: bytes | None, params: bytes = b"", events=b"{}"):
    """A one-kernel-node .cugraph (BinaryGraphFormat.h, v3). ``extra_args``: the
    node is an 'extra' launch with this argument buffer and two 8-byte
    parameters in it; otherwise ``params`` is its one parameter."""
    name = b"kernel_k\0"
    entry = bytearray(168)
    struct.pack_into("<IB", entry, 0, 0, 0)  # node 0, NODE_KERNEL
    body = 8
    struct.pack_into("<II", entry, body + 36, 0, len(name) - 1)  # function name
    if extra_args is not None:
        index = struct.pack("<IIII", 0, 8, 8, 8)
        struct.pack_into("<IIII", entry, body + 120, 0, 2, 0, len(extra_args))
        param_data, arg_data = b"", extra_args
    else:
        index = struct.pack("<II", 0, len(params))
        struct.pack_into("<IIII", entry, body + 120, 0, 1, 0xFFFFFFFF, 0)
        param_data, arg_data = params, b""
    sections = [name, bytes(entry), b"", index, param_data, arg_data, b"", b"", events, b"", b"K"]
    offset = 64 + 24 * len(sections)
    directory, blob = b"", b""
    for i, data in enumerate(sections):
        directory += struct.pack("<IIQQ", i, 0, offset, len(data))
        pad = (-len(data)) % 8
        blob += data + b"\0" * pad
        offset += len(data) + pad
    header = struct.pack("<8sIIIIIII28x", b"CUGRAPH\0", 3, 4, 1, 0, 0, len(sections), 1)
    path.write_bytes(header + directory + blob)


def test_archive_scan_reads_extra_launch_parameters(tmp_path):
    """An 'extra' launch (NCCL's kernels) keeps its parameters in the node's
    argument buffer; the scan must read them there."""
    from foundry.integration.sglang import archive_check

    outside, inside = 0x1234, BASE + 0x4000
    _cugraph(tmp_path / "graph_0_x.cugraph", extra_args=struct.pack("<QQ", outside, inside))
    findings = archive_check.check_rank_archive(str(tmp_path), BASE, SIZE)
    assert findings == {"kernel_k": {"nodes": 1, "targets": [0x4000]}}


@pytest.mark.parametrize("recorded", [False, True])
def test_archive_scan_reads_per_parameter_launches(tmp_path, recorded):
    from foundry.integration.sglang import archive_check

    target = BASE + 0x8000
    events = json.dumps({"events": [{"type": "alloc", "ptr": target, "size": 64}]}).encode()
    _cugraph(
        tmp_path / "graph_0_x.cugraph",
        extra_args=None,
        params=struct.pack("<Q", target),
        events=events if recorded else b"{}",
    )
    findings = archive_check.check_rank_archive(str(tmp_path), BASE, SIZE)
    assert findings == ({} if recorded else {"kernel_k": {"nodes": 1, "targets": [0x8000]}})
