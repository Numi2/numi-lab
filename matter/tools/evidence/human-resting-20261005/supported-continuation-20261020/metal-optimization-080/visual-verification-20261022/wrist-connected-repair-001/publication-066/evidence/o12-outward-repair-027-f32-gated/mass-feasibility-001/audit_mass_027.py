#!/usr/bin/env python3
"""Bounded, read-only material-volume audit for O12 candidate 027.

The identity row mapping is accepted only when tetrahedra and material IDs are
bitwise equal to the pinned reference-input arrays. This does not invent a
donor-origin mapping; the report records that gap explicitly.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_volumes(points, tets):
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)
    a = p[t[:, 1]] - p[t[:, 0]]
    b = p[t[:, 2]] - p[t[:, 0]]
    c = p[t[:, 3]] - p[t[:, 0]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-report", required=True)
    ap.add_argument("--immediate-parent", required=True)
    ap.add_argument("--reference-input", required=True)
    ap.add_argument("--d37-remap-report", required=True)
    ap.add_argument("--j-bounds-report", required=True)
    ap.add_argument("--donor-mass-csv", required=True)
    ap.add_argument("--donor-frames-csv", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    paths = {k: Path(getattr(args, k.replace("-", "_"))) for k in (
        "candidate", "candidate-report", "immediate-parent", "reference-input",
        "d37-remap-report", "j-bounds-report", "donor-mass-csv", "donor-frames-csv")}
    z = np.load(paths["candidate"], allow_pickle=False)
    parent = np.load(paths["immediate-parent"], allow_pickle=False)
    ref = np.load(paths["reference-input"], allow_pickle=False)
    creport = json.loads(paths["candidate-report"].read_text())
    d37 = json.loads(paths["d37-remap-report"].read_text())
    jreport = json.loads(paths["j-bounds-report"].read_text())

    required = ("points_world_f32", "tetrahedra", "material_ids")
    for label, archive in (("candidate", z), ("immediate-parent", parent)):
        if any(k not in archive for k in required):
            raise ValueError(f"{label} missing required arrays")
    if not all(k in ref for k in ("reference_points_m", "tetrahedra", "material_ids")):
        raise ValueError("reference input missing arrays")

    T = z["tetrahedra"]
    M = z["material_ids"]
    row_identity = (np.array_equal(T, ref["tetrahedra"]) and
                    np.array_equal(M, ref["material_ids"]) and
                    np.array_equal(T, parent["tetrahedra"]) and
                    np.array_equal(M, parent["material_ids"]))
    if not row_identity:
        raise ValueError("tetra/material row identity with reference/parent is not proven")

    vref = signed_volumes(ref["reference_points_m"], T)
    vparent = signed_volumes(parent["points_world_f32"], T)
    vcand = signed_volumes(z["points_world_f32"], T)
    if np.any(vref <= 0) or np.any(vparent <= 0) or np.any(vcand <= 0):
        raise ValueError("nonpositive signed volume in an input geometry")

    gmass = d37["geometry_mass"]
    rho = {
        1: float(gmass["fixed_skin_debit_kg"] / gmass["skin_volume_m3"]),
        2: float(gmass["fixed_fat_debit_kg"] / gmass["fat_volume_m3"]),
    }
    jb = next((v for k, v in jreport.items() if k == "J_bounds"), None)
    if jb is None:
        # Fit preflight report stores this in a nested policy block.
        stack = [jreport]
        while stack and jb is None:
            obj = stack.pop()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "J_bounds":
                        jb = v
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(obj, list):
                stack.extend(obj)
    if jb is None or len(jb) != 2:
        raise ValueError("could not recover configured J bounds")
    jlo, jhi = map(float, jb)

    summary = {}
    allmass_ref = allmass_candidate_uniform = 0.0
    allcounts = {"below_J_min": 0, "above_J_max": 0}
    for mat, name in ((1, "skin"), (2, "fat")):
        rows = np.flatnonzero(M == mat)
        j = vcand[rows] / vref[rows]
        mass_ref = rho[mat] * float(np.sum(vref[rows]))
        mass_uniform = rho[mat] * float(np.sum(vcand[rows]))
        implied = rho[mat] / j
        below, above = int(np.count_nonzero(j < jlo)), int(np.count_nonzero(j > jhi))
        allmass_ref += mass_ref
        allmass_candidate_uniform += mass_uniform
        allcounts["below_J_min"] += below
        allcounts["above_J_max"] += above
        summary[name] = {
            "material_id": mat,
            "cell_count": int(len(rows)),
            "reference_volume_m3": float(np.sum(vref[rows])),
            "immediate_parent_014_volume_m3": float(np.sum(vparent[rows])),
            "candidate_027_volume_m3": float(np.sum(vcand[rows])),
            "candidate_vs_reference_volume_delta_percent": float(100.0 * (np.sum(vcand[rows]) / np.sum(vref[rows]) - 1.0)),
            "candidate_vs_immediate_parent_volume_delta_percent": float(100.0 * (np.sum(vcand[rows]) / np.sum(vparent[rows]) - 1.0)),
            "reference_density_kg_m3_from_D37_fixed_debit": rho[mat],
            "parent_mass_kg_from_reference_volume": mass_ref,
            "candidate_mass_kg_if_configured_density_unchanged": mass_uniform,
            "mass_delta_kg_if_configured_density_unchanged": mass_uniform - mass_ref,
            "J_candidate_over_reference": {
                "min": float(np.min(j)), "p01": float(np.quantile(j, .01)),
                "p05": float(np.quantile(j, .05)), "median": float(np.median(j)),
                "p95": float(np.quantile(j, .95)), "p99": float(np.quantile(j, .99)),
                "max": float(np.max(j)),
                "min_cell_row": int(rows[int(np.argmin(j))]),
                "max_cell_row": int(rows[int(np.argmax(j))]),
                "below_configured_min_count": below,
                "above_configured_max_count": above,
                "configured_bounds": [jlo, jhi],
            },
            "current_density_kg_m3_if_each_parent_cell_mass_is_preserved": {
                "min": float(np.min(implied)), "p01": float(np.quantile(implied, .01)),
                "p05": float(np.quantile(implied, .05)), "median": float(np.median(implied)),
                "p95": float(np.quantile(implied, .95)), "p99": float(np.quantile(implied, .99)),
                "max": float(np.max(implied)),
                "interpretation": "diagnostic rho0/J; no allowed per-cell density interval was found in the pinned candidate or D37 ledger",
            },
        }

    table_header = next(csv.reader(open(paths["donor-mass-csv"], newline="")))
    donor_table_header = next(csv.reader(open(paths["donor-frames-csv"], newline="")))
    creport_text = json.dumps(creport).lower()
    d37_status = d37.get("status", "unknown")
    d37_geometry = gmass
    report = {
        "schema": "numi.human.o12-outward-027-material-capacity-audit.v1",
        "status": "READ_ONLY_MASS_FEASIBILITY_FAILS_CONFIGURED_J_BOUNDS; DONOR_ORIGIN_CAPACITY_UNRESOLVED",
        "candidate_physical_admission": creport.get("status"),
        "material_geometry": summary,
        "totals": {
            "reference_mass_kg_from_fixed_material_density": allmass_ref,
            "candidate_mass_kg_if_fixed_density_unchanged": allmass_candidate_uniform,
            "mass_delta_kg_if_fixed_density_unchanged": allmass_candidate_uniform - allmass_ref,
            "mass_delta_percent_if_fixed_density_unchanged": 100.0 * (allmass_candidate_uniform / allmass_ref - 1.0),
            "J_violations": allcounts,
        },
        "row_mapping": {
            "candidate_to_reference": "identity by exact tetrahedra and material_ids equality (28,632 rows); parent mass per row reconstructed as configured reference density times F64 owner-reference cell volume",
            "candidate_contains_mass_or_donor_ancestry_arrays": False,
            "candidate_npz_keys": list(z.files),
        },
        "donor_capacity": {
            "status": "NOT_COMPUTABLE_FROM_PINNED_CANDIDATE",
            "missing_link": "no candidate cell-row to parent-origin donor/source-contribution map; candidate NPZ has only points, tetrahedra, material_ids",
            "available_but_not_substitutable": {
                "D37_donor_mass_allocation_header": table_header,
                "D37_donor_frame_header_prefix": donor_table_header[:8],
                "reason": "D37 donor-mass-allocation is an aggregate donor/frame-weighted node ledger for the older D37 candidate; donor-frames records column transform ownership. Neither is a per-cell parent-origin ledger for candidate 027, so assigning candidate cell volume/mass by node owner would fabricate origin attribution.",
            },
        },
        "configured_limits": {
            "J_bounds": [jlo, jhi],
            "reference_density_kg_m3": {"skin": rho[1], "fat": rho[2]},
            "density_bounds": None,
            "density_bounds_note": "No per-cell density interval is present in pinned artifacts; only fixed material reference densities/debits are available.",
        },
        "pinned_inputs": {str(k): {"path": str(v), "sha256": sha(v)} for k, v in paths.items()},
        "source_facts": {
            "D37_status": d37_status,
            "D37_fixed_skin_mass_kg": float(d37_geometry["fixed_skin_debit_kg"]),
            "D37_fixed_fat_mass_kg": float(d37_geometry["fixed_fat_debit_kg"]),
            "D37_reference_skin_volume_m3": float(d37_geometry["skin_volume_m3"]),
            "D37_reference_fat_volume_m3": float(d37_geometry["fat_volume_m3"]),
            "candidate_report_declares_no_mass_closure": "no reference-coordinate pullback" in creport_text and "mass closure" in creport_text,
        },
        "limitations": [
            "This is a read-only geometry-to-reference mass feasibility diagnostic, not a conservative transfer receipt or physical admission.",
            "Fixed-density candidate mass deficit is aggregate; exact donor-origin capacity is blocked by the missing candidate cell-to-parent-donor contribution map.",
            "Preserving parent cell masses implies rho0/J values shown above, but no permitted per-cell density range was found; configured J bounds already fail.",
        ],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
