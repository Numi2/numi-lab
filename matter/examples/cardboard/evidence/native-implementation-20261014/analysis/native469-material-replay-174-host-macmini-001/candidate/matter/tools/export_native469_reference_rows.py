#!/usr/bin/env python3
"""Export a bounded, hash-bound subset of native469 FP64/MP reference arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


SCHEMA = "numi.cardboard.native469-selected-reference-v1"
MATERIALS = {
    0: "hajali2009_liner_hill_bilinear_fit",
    1: "hajali2009_medium_hill_bilinear_fit",
    2: "starch2007_finite_glue_elastic",
}
REPRESENTATIVE_TETS = {0: 0, 1: 576, 2: 864}
STATE_KEYS = ("center", "s1", "s2", "s3", "s4")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def f64_list(values: np.ndarray) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError("reference contains a nonfinite value")
    return [float(x) for x in flat]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--mp-manifest", type=Path, required=True)
    ap.add_argument("--capture-sha256", required=True)
    ap.add_argument("--report-sha256", required=True)
    ap.add_argument("--arrays-sha256", required=True)
    ap.add_argument("--mp-manifest-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    expected = (args.capture_sha256, args.report_sha256, args.arrays_sha256, args.mp_manifest_sha256)
    actual = tuple(sha256(path) for path in (args.capture, args.report, args.arrays, args.mp_manifest))
    if actual != expected or any(
        len(x) != 64 or any(c not in "0123456789abcdefABCDEF" for c in x)
        for x in expected
    ):
        raise ValueError("one or more source469/MP input hashes differ from explicit bindings")
    report = json.loads(args.report.read_text())
    package_manifest = json.loads(args.mp_manifest.read_text())
    included = package_manifest["included_files_sha256"]
    if included.get("run-002/report.json") != actual[1] or included.get("run-002/mixed_reference_arrays.npz") != actual[2]:
        raise ValueError("MP package manifest does not bind the supplied report/array outputs")
    bindings = report["bindings"]
    if bindings["native_files"]["final-newton"] != actual[0]:
        raise ValueError("MP report is not bound to the supplied native469 capture")
    source_snapshot = bindings["source_snapshot"]
    if source_snapshot != "/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157":
        raise ValueError("MP reference source snapshot differs")

    mp = {}
    for row in report["mp_results"]:
        key = (int(row["tet"]), int(row["state_index"]))
        if row["fallback_status"] != "mp_gates_pass" or key in mp:
            raise ValueError("MP report contains an unqualified or duplicate root")
        mp[key] = int(row["material_index"])

    tet_material: dict[int, int] = {}
    with args.capture.open() as stream:
        header = json.loads(next(stream))
        if header["environment_count"] != 2 or header["tetrahedron_count_per_environment"] != 1920:
            raise ValueError("native469 capture dimensions changed")
        for line in stream:
            row = json.loads(line)
            if row.get("record") != "tetrahedron" or row.get("environment") != 0:
                continue
            tet = int(row["tet_id"])
            material = int(row["material_index"])
            if tet in tet_material or not 0 <= tet < 1920 or material not in MATERIALS:
                raise ValueError("duplicate/out-of-range native469 env0 tetrahedron")
            if row["material_name"] != MATERIALS[material]:
                raise ValueError("native469 material name/index mismatch")
            tet_material[tet] = material
    if len(tet_material) != 1920:
        raise ValueError("native469 env0 does not contain all 1920 tetrahedra")
    for material, tet in REPRESENTATIVE_TETS.items():
        if tet_material.get(tet) != material:
            raise ValueError("representative material/tet binding changed")
    for (tet, _state), material in mp.items():
        if tet_material.get(tet) != material:
            raise ValueError("MP root material differs from exact native469 tet")

    selected_tets = set(REPRESENTATIVE_TETS.values()) | {tet for tet, _ in mp}
    with np.load(args.arrays, allow_pickle=False) as arrays:
        expected_shapes = {
            "center_stress": (1920, 3, 3), "s1_stress": (1920, 3, 3),
            "s2_stress": (1920, 3, 3), "s3_stress": (1920, 3, 3),
            "s4_stress": (1920, 3, 3), "center_tangent": (1920, 9, 9),
            "s1_tangent": (1920, 9, 9), "s2_tangent": (1920, 9, 9),
            "s3_tangent": (1920, 9, 9), "s4_tangent": (1920, 9, 9),
            "paper_material_index": (1728,), "paper_parameters_f32": (1728, 28),
        }
        if any(k not in arrays or arrays[k].shape != shape for k, shape in expected_shapes.items()):
            raise ValueError("source469 independent array names/shapes changed")
        rows = [{"record": "header", "schema": SCHEMA, "capture_sha256": actual[0],
                 "report_sha256": actual[1], "arrays_sha256": actual[2], "mp_package_manifest_sha256": actual[3],
                 "source_snapshot": source_snapshot, "environment": 0, "rows_per_tet": 5,
                 "selected_tets": sorted(selected_tets),
                 "interpretation": "FP64 independent reference plus MP-only fallback where present; no native accuracy qualification"}]
        for tet in sorted(selected_tets):
            material = tet_material[tet]
            for state_idx, state_name in enumerate(STATE_KEYS):
                prefix = state_name + "_"
                stress = np.asarray(arrays[prefix + "stress"][tet], dtype=np.float64).reshape(9)
                tangent = np.asarray(arrays[prefix + "tangent"][tet], dtype=np.float64).reshape(81)
                row = {"record": "row", "tet_id": tet, "material_index": material,
                       "state_index": state_idx, "state_label": state_name,
                       "fp64_stress": f64_list(stress), "fp64_tangent": f64_list(tangent),
                       "mp_state": None, "mp_stress": None, "mp_tangent": None}
                key = (tet, state_idx)
                if key in mp:
                    state_key = f"mp_state_{state_idx}_{tet}"
                    stress_key = f"mp_stress_{state_idx}_{tet}"
                    tangent_key = f"mp_tangent_{state_idx}_{tet}"
                    if any(k not in arrays for k in (state_key, stress_key, tangent_key)):
                        raise ValueError("qualified MP row is missing exact root/stress/tangent arrays")
                    row["mp_state"] = f64_list(arrays[state_key])
                    row["mp_stress"] = f64_list(arrays[stress_key])
                    row["mp_tangent"] = f64_list(arrays[tangent_key])
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()
    write_exclusive(args.output, payload)
    print(json.dumps({"status": "pass", "output_sha256": sha256(args.output),
                      "output_bytes": len(payload), "rows": len(rows) - 1,
                      "selected_tets": len(selected_tets), "mp_roots": len(mp),
                      "input_sha256": dict(zip(("capture", "report", "arrays", "mp_manifest"), actual))},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # retain concise, explicit failure for source-bound callers
        print(f"FAIL native469 reference export: {exc}", file=sys.stderr)
        raise SystemExit(1)
