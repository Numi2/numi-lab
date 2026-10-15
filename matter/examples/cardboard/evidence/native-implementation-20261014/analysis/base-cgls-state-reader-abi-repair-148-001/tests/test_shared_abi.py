#!/usr/bin/env python3
"""Compile the actual frozen shared.h ABI and compare parser record widths."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE144 = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-state-sidecar-144/matter/include/numi/matter/shared.h")
SOURCE144_SHA256 = "cd173c076905f660aa8aecc900e6f6aba8d69406996f3cccf2e0656c753ff871"
REPO_ROOT = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-state-sidecar-144")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reader_widths() -> dict[str, int]:
    reader_path = PACKAGE / "tools/read_state_capture.py"
    spec = importlib.util.spec_from_file_location("state_capture_reader", reader_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {key: value for key, value in module.RECORD_WIDTHS.items() if key.startswith("record:")}


def verify_actual_abi() -> dict[str, object]:
    assert SOURCE144.is_file(), SOURCE144
    actual_hash = sha256(SOURCE144)
    assert actual_hash == SOURCE144_SHA256, f"source144 shared.h hash mismatch: {actual_hash}"
    compiler = os.environ.get("CXX", "clang++")
    probe_source = PACKAGE / "tests/shared_abi_probe.cpp"
    with tempfile.TemporaryDirectory(prefix="cgls-abi-148-") as tmp:
        binary = Path(tmp) / "shared_abi_probe"
        subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(REPO_ROOT / "matter/include"), str(probe_source), "-o", str(binary)],
            check=True,
        )
        output = subprocess.run([str(binary)], check=True, text=True, capture_output=True).stdout
    observed: dict[str, int] = {}
    for line in output.splitlines():
        name, value = line.split()
        observed[name] = int(value)
    parser_widths = load_reader_widths()
    abi_widths = {name: observed[name] for name in parser_widths}
    assert parser_widths == abi_widths, f"reader ABI widths {parser_widths} != compiled widths {abi_widths}"

    expected_offsets = {
        "NMFEMNodeStateGPU.positionAndMass": 0,
        "NMFEMNodeStateGPU.velocityAndInverseMass": 16,
        "NMFEMNodeStateGPU.restAndFixed": 32,
        "NMFEMNodeStateGPU.deltaVelocity": 48,
        "NMFEMNodeStateGPU.referenceDisplacementAndMode": 64,
        "NMTetrahedronGPU.nodes": 0,
        "NMTetrahedronGPU.inverseRestRow0": 16,
        "NMTetrahedronGPU.inverseRestRow1": 32,
        "NMTetrahedronGPU.inverseRestRow2": 48,
        "NMTetrahedronGPU.identity": 64,
        "NMTetrahedronGPU.materialFrameRotation": 80,
    }
    actual_offsets = {name: observed[name] for name in expected_offsets}
    assert actual_offsets == expected_offsets, f"compiled member offsets mismatch: {actual_offsets}"
    return {
        "shared_header_sha256": actual_hash,
        "record_widths": parser_widths,
        "member_offsets": actual_offsets,
    }


class SharedHeaderABITests(unittest.TestCase):
    def test_reader_widths_and_offsets_match_compiled_source144_header(self) -> None:
        result = verify_actual_abi()
        self.assertEqual(result["record_widths"], {
            "record:NMFEMNodeStateGPU": 80,
            "record:NMTetrahedronGPU": 96,
        })
        self.assertEqual(len(result["member_offsets"]), 11)


if __name__ == "__main__":
    unittest.main()


