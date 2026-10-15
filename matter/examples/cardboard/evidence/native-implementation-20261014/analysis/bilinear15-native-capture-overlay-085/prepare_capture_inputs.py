#!/usr/bin/env python3
"""Prepare a small, hash-bound Metal point test from the retained N0/N63 captures.

This only selects/copies source-bound rows. It does not compute or modify native
outputs. The source capture, FP64 equations, and reference files stay read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path("/Users/home/cardboard-implementation-20261014")
N0 = BASE / "solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture"
N63 = BASE / "solver/sourcefit15-fg32-control1-newton63-capture-266/operator-capture"
MP = BASE / "analysis/local-projection-scaled-correction-082/mpmath_transition_reference.json"
N63_REPORT = BASE / "analysis/bilinear15-fp64-replay-267/late-output-002/report.json"
N63_NPZ = BASE / "analysis/bilinear15-fp64-replay-267/late-output-002/rows.npz"
EQ_DIR = BASE / "analysis/bilinear15-fp64-replay-267"
MATERIAL = BASE / "snapshots/source-scaled-projection-and-coupled-capture-084/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter"
MATERIAL_SHA = "76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9"
CAPTURE_SHA = {
    "N0": "14246b2ade8e301d68b33d97f14eb6468e7f8f7dbc4a36249a83e33a25db2541",
    "N63": "13f264ff1bb9688f8ac48f3c940020ff1f1f45d7fbb1713cd2144ba5320b91c2",
}
TET_IDS = [750, 751, 758, 608, 763, 1722, 1723, 1704]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f32words(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def capture_rows(path: Path, expected_sha: str):
    actual_sha = sha(path)
    if actual_sha != expected_sha:
        raise ValueError(f"capture hash changed: {path}: {actual_sha}")
    rows = []
    with path.open() as f:
        header = json.loads(next(f))
        if header.get("record") != "header" or header.get("schema") != "numi.cardboard.fem-newton-capture.v1":
            raise ValueError(f"unsupported capture header in {path}")
        if header.get("root") != 1:
            raise ValueError(f"unexpected capture root for {path}")
        for line in f:
            rec = json.loads(line)
            if rec.get("record") == "tetrahedron" and rec.get("environment") == 0 and rec.get("material_index") == 1:
                rows.append(rec)
    by_id = {int(r["tet_id"]): r for r in rows}
    if len(by_id) != len(rows) or not set(TET_IDS).issubset(by_id):
        raise ValueError(f"selected medium tet IDs are missing or duplicated in {path}")
    return actual_sha, header, by_id


def matrix_flat(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (3, 3):
        raise ValueError(f"expected 3x3 captured matrix, got {a.shape}")
    return a.reshape(9).astype(np.float64).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")

    sys.path.insert(0, str(EQ_DIR))
    import bilinear15_reference as ref

    cap_binding = {}
    selected = {}
    for label, path in (("N0", N0), ("N63", N63)):
        digest, header, rows = capture_rows(path, CAPTURE_SHA[label])
        cap_binding[label] = {"path": str(path), "sha256": digest,
                              "root": header["root"],
                              "newton_iteration": header["newton_iteration"],
                              "environment_count": header["environment_count"],
                              "tetrahedron_count_per_environment": header["tetrahedron_count_per_environment"]}
        selected[label] = (header, rows)

    mpdoc = json.loads(MP.read_text())
    if mpdoc.get("input_capture_sha256") != CAPTURE_SHA["N0"] or mpdoc.get("dps") != [50, 80]:
        raise ValueError("N0 arbitrary-precision reference is not bound to the expected capture/dps")
    mp_rows = {int(r["tet_id"]): r for r in mpdoc["rows"]
               if r.get("environment") == 0 and r.get("material_index") == 1}
    if not set(TET_IDS).issubset(mp_rows):
        raise ValueError("N0 mpmath reference omitted selected rows")

    n63_report = json.loads(N63_REPORT.read_text())
    if n63_report.get("capture_sha256") != CAPTURE_SHA["N63"]:
        raise ValueError("N63 FP64 report is not bound to the expected capture")
    n63 = np.load(N63_NPZ)
    all_records = []
    with N63.open() as f:
        next(f)
        for line in f:
            rec = json.loads(line)
            if rec.get("record") == "tetrahedron" and rec.get("material_index") in (0, 1):
                all_records.append(rec)
    n63_index = {(int(r["environment"]), int(r["material_index"]), int(r["tet_id"])): i
                 for i, r in enumerate(all_records)}
    if n63["fp64_refined"].shape[0] != len(all_records):
        raise ValueError("N63 strict-reference row ordering/length mismatch")

    material_sha = sha(MATERIAL)
    if material_sha != MATERIAL_SHA:
        raise ValueError(f"bilinear medium material asset hash changed: {material_sha}")
    equation_sha = sha(EQ_DIR / "bilinear15_reference.py")

    cases = []
    for label in ("N0", "N63"):
        header, rows = selected[label]
        material = next((m for m in header["materials"] if int(m["index"]) == 1), None)
        if material is None or material["name"] != "hajali2009_medium_hill_bilinear_fit":
            raise ValueError(f"{label} capture material index 1 identity changed")
        offset = int(material["parameter_offset"])
        count_expected = 28
        params_env = header["parameters_by_environment"][0]
        params = f32words(params_env[offset:offset + count_expected])
        if len(params) != count_expected:
            raise ValueError(f"{label} capture parameter count mismatch")
        for tet_id in TET_IDS:
            rec = rows[tet_id]
            c = rec["center"]
            old = f32words(c["accepted_state"])
            fhigh = np.asarray(c["F_local"], dtype=np.float32).reshape(9)
            flow = np.asarray(c["F_local_low"], dtype=np.float32).reshape(9)
            fdouble = fhigh.astype(np.float64) + flow.astype(np.float64)
            direction = np.asarray(c["F_direction_local_per_s"], dtype=np.float32).reshape(9)
            fdot = np.asarray(c["Fdot_local_per_s"], dtype=np.float32).reshape(9)
            rate_direction = np.asarray(c["rate_direction_local_per_s2"], dtype=np.float32).reshape(9)
            if label == "N0":
                mp = mp_rows[tet_id]
                reference_state = np.asarray(mp["solutions"]["80"]["root"], dtype=np.float64)
                reference_tangent = np.asarray(mp["solutions"]["80"]["tangent"], dtype=np.float64).reshape(9)
                reference_source = "mpmath-dps80 selected transition reference"
                branch_details = mp.get("captured_candidate_fb", {})
            else:
                index = n63_index[(0, 1, tet_id)]
                reference_state = np.asarray(n63["states_refined"][index], dtype=np.float64)
                reference_tangent = np.asarray(n63["fp64_refined"][index], dtype=np.float64).reshape(9)
                reference_source = "independent FP64 safeguarded projection/tangent replay"
                branch_details = {}
            reference_stress = np.asarray(ref.stress(params, fdouble.reshape(3, 3), reference_state), dtype=np.float64).reshape(9)
            state_delta = float(reference_state[12] - old[12])
            branch = "multiplier_increased" if state_delta > 1.0e-8 else "no_resolved_multiplier_increase"
            cases.append({
                "id": f"{label}-medium-tet{tet_id}", "capture_label": label,
                "environment": 0, "material_index": 1, "tet_id": tet_id,
                "branch_indicator": branch,
                "multiplier_increment_from_reference": state_delta,
                "branch_threshold": 1.0e-8,
                "branch_details_from_reference": branch_details,
                "reference_source": reference_source,
                "parameters": params.tolist(),
                "accepted_state": old.tolist(),
                "F_high": fhigh.astype(np.float64).tolist(),
                "F_low": flow.astype(np.float64).tolist(),
                "deformation_direction": direction.astype(np.float64).tolist(),
                "Fdot": fdot.astype(np.float64).tolist(),
                "rate_direction": rate_direction.astype(np.float64).tolist(),
                "dt_seconds": float(np.float32(c["dt_seconds_bulk_modulus_kind_frame_active"][0])),
                "temperature_K": float(np.float32(c["dt_seconds_bulk_modulus_kind_frame_active"][1])),
                "captured_default_state": (f32words(c["projected_state"]) +
                                           f32words(c["projected_state_low"])).tolist(),
                "captured_default_stress_Pa": matrix_flat(c["P_local_Pa"]),
                "captured_default_tangent_Pa_per_s": matrix_flat(c["tangent_local_Pa_per_s"]),
                "captured_projection_report": c["projection"],
                "reference_state": reference_state.tolist(),
                "reference_stress_Pa": reference_stress.tolist(),
                "reference_tangent_Pa_per_s": reference_tangent.tolist(),
            })

    doc = {
        "schema": "numi.cardboard.bilinear15-native-captured-point-inputs.v1",
        "classification": "Bounded native production evaluator replay inputs; no full-cycle or material qualification claim.",
        "source_material": {"path": str(MATERIAL), "sha256": material_sha,
                            "name": "hajali2009_medium_hill_bilinear_fit", "state_count": 15,
                            "parameter_count": 28},
        "captures": cap_binding,
        "independent_equations": {"path": str(EQ_DIR / "bilinear15_reference.py"),
                                  "sha256": equation_sha,
                                  "N0_mpmath_reference": {"path": str(MP), "sha256": sha(MP), "dps": [50, 80]},
                                  "N63_FP64_reference_report": {"path": str(N63_REPORT), "sha256": sha(N63_REPORT),
                                                                 "arrays": str(N63_NPZ), "arrays_sha256": sha(N63_NPZ)}},
        "selected_tet_ids_environment0_medium": TET_IDS,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "case_count": len(cases), "capture_hashes": CAPTURE_SHA,
                      "material_sha256": material_sha}, indent=2))


if __name__ == "__main__":
    main()
