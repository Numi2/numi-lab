#!/usr/bin/env python3
"""Recheck retained temporal and adjacent-mesh evidence; this runs no simulation."""
import hashlib
import json
from pathlib import Path

from large_spatial_reference import compare, read_reference

BASE = Path("docs/evidence/midpoint-spatial-cost-dynamic-20261108")
LATE = Path("docs/evidence/midpoint-first-failure-20261109")
OUT = LATE / "historical-mesh-ladder-audit-001.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(n):
    if n <= 32:
        return BASE / f"rk4-n{n}-h050.json", BASE / f"rk4-n{n}-h025.json"
    if n == 64:
        return LATE / "rk4-n64-h050.json", LATE / "rk4-n64-parallel-h025.json"
    return LATE / "rk4-n128-retry1-h050.json", LATE / "rk4-n128-retry2-h025.json"


def main():
    result = {"schema": "numi.historical-mesh-ladder-audit.v1",
              "scope": "Retained independent CPU references, not a new simulation or native rollout.",
              "limits": ["The historical n64 references do not bind both original executions to a binary hash; a provenance-bound repeat is running.",
                         "Output checks do not independently establish input material or contact-law identity; consult the retained source and execution records.",
                         "A temporally qualified mesh and adjacent-mesh agreement do not by themselves establish continuum or experimental validation.",
                         "Peak values are integration-step sampled peaks; binned force is a different metric."],
              "validator_sha256": sha256(Path("tests/contact/large_spatial_reference.py")),
              "temporal": [], "spatial": []}
    prev = None
    for n in (2, 4, 8, 16, 32, 64, 128):
        coarse_path, fine_path = paths(n)
        coarse = read_reference(coarse_path, n, 5e-7)
        fine = read_reference(fine_path, n, 2.5e-7)
        temporal = compare(coarse, fine)
        result["temporal"].append({"divisions": n, "nodes": fine["nodes"],
            "coarse": {"path": str(coarse_path), "sha256": sha256(coarse_path), "h_s": coarse["h"]},
            "fine": {"path": str(fine_path), "sha256": sha256(fine_path), "h_s": fine["h"]},
            "area_m2": fine["verified_area_m2"], "mass_kg": fine["mass_kg"],
            "fine_sampled_peak_force_N": fine["peak_force_N"], "metrics": temporal})
        if prev is not None:
            prior_n, prior, prior_temporal = prev
            result["spatial"].append({"coarse_divisions": prior_n, "fine_divisions": n,
                "both_temporal_checks_pass": prior_temporal["passes_refinement"] and temporal["passes_refinement"],
                "metrics": compare(prior, fine)})
        prev = n, fine, temporal
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUT)
    for item in result["temporal"]:
        m = item["metrics"]
        print("temporal", item["divisions"], m["passes_refinement"], "binned", m["binned_force"], "peak", m["sampled_peak_force"])
    for item in result["spatial"]:
        m = item["metrics"]
        print("spatial", item["coarse_divisions"], "->", item["fine_divisions"], m["passes_refinement"], "binned", m["binned_force"], "peak", m["sampled_peak_force"])


if __name__ == "__main__":
    main()
