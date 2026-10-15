#!/usr/bin/env python3
"""Recheck the sealed CPU shipping-arena audit against source106."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    audit = json.loads((HERE / "audit.json").read_text())
    preflight = json.loads((HERE / "preflight.json").read_text())
    frozen = json.loads((SOURCE / "frozen-source.json").read_text())
    assert audit["schema"] == "numi.cardboard.explicit_flute_shipping_arena_audit.v1"
    assert audit["preflight_sha256"] == sha(HERE / "preflight.json")
    assert audit["source_manifest_sha256"] == sha(SOURCE / "frozen-source.json")
    assert audit["source_file_count"] == len(frozen["source_sha256"])
    for rel, expected in audit["selected_source_sha256"].items():
        assert frozen["source_sha256"][rel] == expected
        assert sha(SOURCE / rel) == expected
    assert preflight["whole_process_preallocation_admission"]["complete"] is False
    assert preflight["whole_process_preallocation_admission"]["actual_metal_peak_measured"] is False
    assert preflight["whole_process_preallocation_admission"]["actual_host_peak_measured"] is False
    scenarios = preflight["memory_model"]["modeled_scenarios"]
    compact = scenarios["capacity_131072_compact_flag_on_stride15_best_case"]
    assert compact["combined_allocation_plus_tangent_cache_within_16_gib"] is True
    assert compact["within_shipping_admission_model"] is False
    assert preflight["current_admission"]["nodes_rejected"] is True
    assert preflight["current_admission"]["tetrahedra_rejected"] is True
    assert preflight["current_admission"]["dense_pairs_rejected"] is True
    assert audit["independent_formula_checks"]["compact_surface_bytes"] == 218607808
    assert audit["independent_formula_checks"]["legacy_surface_bytes"] == 1009743872
    assert audit["independent_formula_checks"]["bounded_deformable_slot_bytes"] == 81788928
    assert audit["independent_formula_checks"]["optional_sorted_incidence_workspace_bytes"] == 12582912
    assert audit["independent_formula_checks"]["persistent_cohesive_bytes"] == 484109760
    with tempfile.TemporaryDirectory(prefix="cardboard-sizeof-audit-") as temp:
        exe = Path(temp) / "sizeof"
        subprocess.run([
            "clang++", "-std=c++20", "-I", str(SOURCE / "include"),
            "-I", str(SOURCE / "matter/include"), str(HERE / "sizeof.cpp"),
            "-o", str(exe),
        ], check=True)
        output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
        assert output == (HERE / "sizeof.txt").read_text()
    print("PASS: source106 bindings, preflight verdict, cap blockers, byte formulas, C++ ABI sizes")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
