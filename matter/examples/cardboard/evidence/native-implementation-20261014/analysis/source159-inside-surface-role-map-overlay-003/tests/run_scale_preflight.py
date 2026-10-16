#!/usr/bin/env python3
"""Compile and run the nominal-count topology-only stress preflight."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

BASE = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = BASE / "analysis/source159-inside-surface-role-map-overlay-003"
SOURCE157 = BASE / "snapshots/source-right-grip-telemetry-157"
SOURCE = PACKAGE / "tests/topology_scale_preflight.cpp"
OUT = PACKAGE / "validation/topology-scale-preflight.json"


def main() -> None:
    command = [
        "clang++", "-std=c++20", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-I" + str(SOURCE157 / "matter/tools"),
        "-I" + str(SOURCE157 / "matter/include"),
        "-I" + str(SOURCE157 / "include"),
        "-I" + str(PACKAGE / "candidate/matter/tools"), str(SOURCE),
        "-o", "/tmp/source159-topology-scale-preflight",
    ]
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    completed = subprocess.run(["/tmp/source159-topology-scale-preflight"],
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    expected = {
        "schema": "numi.cardboard.inside_topology_scale_preflight.v1",
        "node_count": 528444,
        "planar_boundary_vertices": 34862,
        "planar_triangles": 492576,
        "tetrahedra": 1477728,
        "surface_faces": 1054876,
        "hash_capacity": 16777216,
        "hash_bytes": 268435456,
        "surface_mappings_verified": 1054876,
        "all_exterior_faces_seen": True,
        "geometry_or_material_evaluated": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise SystemExit(f"scale preflight mismatch {key}: {result.get(key)!r} != {value!r}")
    if result.get("workspace_estimate_bytes", 1 << 63) > 1 << 30:
        raise SystemExit("scale preflight workspace estimate exceeded 1 GiB")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"scale preflight: PASS; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
