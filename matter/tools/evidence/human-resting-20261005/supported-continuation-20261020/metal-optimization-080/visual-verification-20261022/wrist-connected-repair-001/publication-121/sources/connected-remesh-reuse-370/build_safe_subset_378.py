#!/usr/bin/env python3
"""Build a fixed-coordinate 358 flip subset excluding all 130 self-witness rows.

This is a construction proposal only. It starts from frozen 368 and retains
only originally selected 370 edges whose incident face rows do not touch a
reported 130 improper self-pair. It does not run geometry or bone audits.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE = ROOT / "barycentric-connected-clearance-170/same-parent-remesh-368/run-001/candidate.npz"
BASE159 = ROOT / "barycentric-connected-clearance-170/same-parent-remesh-368/run-001/audit-159/report.json"
SELECTOR = ROOT / "barycentric-connected-clearance-170/bone-aware-remesh-370/run-001/selector-report.json"
SELECTED_CANDIDATE = ROOT / "barycentric-connected-clearance-170/bone-aware-remesh-370/run-001/candidate.npz"
SELF_SUMMARY = ROOT / "barycentric-connected-clearance-170/bone-aware-remesh-370/run-001/audit-130/summary.json"
SELF_PAIRS = ROOT / "barycentric-connected-clearance-170/bone-aware-remesh-370/run-001/audit-130/surface-self-pairs.npy"
FLIP_PATH = ROOT / "D/connected-remesh-reuse-358/same_parent_flips_358.py"
OUT = ROOT / "barycentric-connected-clearance-170/bone-aware-remesh-370/run-002-self-safe-subset"

PINS = {
    "base_candidate": "9eb42003871a9a95bed5c56906f97bdc3c3f996486ddade3a10fd9bac0437a88",
    "base_159_report": "d5674389a55adeae71c20f9f3c81ad05c04ca8fbf2c4167351973fdefaf0c97d",
    "selector_report": "d003e99f2621fa95dcb8f8fca9b0b1a58cb2223757072b1dfd83b14a3d0c9ab8",
    "selector_candidate": "b6ddb223677eed04d98df36d48968525415d7c780eef3f2222fbfc9315cc3d96",
    "self_summary": "3988939f093a34bc3e47d3d77309c94b4025bba568ac83552ac47b440c044daf",
    "flip_358": "a08e1f7e6392854f72ffab09af6f2b1f2afe885b1a84f586afe71160ff3ccbee",
}
EXPECTED_SELF_PAIR_SHA256 = "a0c13070db484b1ef033372b60cd4c8a84320a19a5f4f7bba2e95abadaed2b78"

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def require_sha(name: str, path: Path, expected: str) -> str:
    got = sha_file(path)
    if got != expected:
        raise RuntimeError(f"{name} SHA mismatch: {got} != {expected}")
    return got

def load_flip():
    require_sha("358 operator", FLIP_PATH, PINS["flip_358"])
    spec = importlib.util.spec_from_file_location("same_parent_flips_358_subset378", FLIP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned 358 operator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def baseline_target_rows(report: dict, face_count: int) -> list[int]:
    if (report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
            or report.get("complete") is not True
            or report.get("bank_count") != 185
            or report.get("expected_bank_count") != 185
            or len(report.get("banks", [])) != 185
            or report.get("surface_face_count") != face_count):
        raise RuntimeError("baseline 159 report is not the complete 185-bank audit of this topology")
    rows = set()
    bank_keys = set()
    for bank in report["banks"]:
        key = (bank.get("body"), bank.get("stable_bone_id"))
        if key in bank_keys or type(key[0]) is not int or type(key[1]) is not int:
            raise RuntimeError("baseline 159 has duplicate or malformed typed bank")
        bank_keys.add(key)
        if bank.get("skin_faces") != face_count:
            raise RuntimeError("baseline 159 skin face domain mismatch")
        result = bank.get("result")
        if result is None:
            if bank.get("skin_aabb_active") != 0 or bank.get("bank_aabb_active") != 0:
                raise RuntimeError("null bank result lacks explicit AABB exclusion")
            continue
        pairs = result.get("triangle_pairs")
        if result.get("count") != len(pairs):
            raise RuntimeError("baseline 159 exact pair count mismatch")
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2
                    or type(pair[0]) is not int or type(pair[1]) is not int
                    or not 0 <= pair[0] < face_count):
                raise RuntimeError("baseline 159 pair identity malformed or out of range")
            rows.add(pair[0])
    if len(bank_keys) != 185:
        raise RuntimeError("baseline typed bank inventory incomplete")
    return sorted(rows)

def op_signature(op: dict):
    return (tuple(sorted(map(int, op["edge"]))),
            tuple(map(int, op["face_rows"])),
            int(op["parent_row"]))

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    require_sha("base candidate", BASE, PINS["base_candidate"])
    require_sha("base 159 report", BASE159, PINS["base_159_report"])
    require_sha("370 selector report", SELECTOR, PINS["selector_report"])
    require_sha("370 self-audit summary", SELF_SUMMARY, PINS["self_summary"])
    require_sha("370 self-pair array", SELF_PAIRS, EXPECTED_SELF_PAIR_SHA256)
    selected_hash = require_sha("370 selected candidate", SELECTED_CANDIDATE, PINS["selector_candidate"])
    base = json.loads(BASE159.read_text())
    selector = json.loads(SELECTOR.read_text())
    self_summary = json.loads(SELF_SUMMARY.read_text())
    if selector.get("candidate_sha256") != PINS["base_candidate"]:
        raise RuntimeError("370 selector did not use the pinned 368 base")
    if selector.get("output_candidate_sha256") != selected_hash or self_summary.get("candidate_sha256") != selected_hash:
        raise RuntimeError("self audit and selector output candidate hashes differ")
    if self_summary.get("status") != "STOPPED_AT_SURFACE_GATE":
        raise RuntimeError("expected 370 self audit failure is not present; do not use this exclusion run")
    with np.load(BASE, allow_pickle=False) as z:
        source = {k: np.asarray(z[k]).copy() for k in z.files}
    faces = np.asarray(source["source_O_faces"])
    target_rows = baseline_target_rows(base, len(faces))
    self_pairs = np.load(SELF_PAIRS, allow_pickle=False)
    if self_pairs.ndim != 2 or self_pairs.shape[1] != 2 or not np.issubdtype(self_pairs.dtype, np.integer):
        raise RuntimeError("130 self-pair witnesses must be integer Nx2")
    if len(self_pairs) != int(self_summary.get("exact_PLC_surface_pairs", -1)):
        raise RuntimeError("130 self-pair witness count differs from summary")
    if np.any(self_pairs < 0) or np.any(self_pairs >= len(faces)):
        raise RuntimeError("130 self-pair row outside 368 face domain")
    witness_rows = set(map(int, self_pairs.ravel()))
    proposed = selector.get("operations")
    if not isinstance(proposed, list):
        raise RuntimeError("370 selector operation list missing")
    chosen = [op for op in proposed if op.get("selected") is True]
    if len(chosen) != selector.get("strict_pair_reducing_flip_count"):
        raise RuntimeError("370 selected operation count mismatch")
    implicated, safe = [], []
    for op in chosen:
        if witness_rows.intersection(map(int, op["face_rows"])):
            implicated.append(op)
        else:
            safe.append(op)
    # 370 emitted a nonoverlapping batch; require that invariant before subset.
    used_faces = [int(row) for op in chosen for row in op["face_rows"]]
    if len(set(used_faces)) != len(used_faces):
        raise RuntimeError("370 selected flips are not face-disjoint")
    expected = {op_signature(op) for op in safe}
    allowed_edges = [list(map(int, op["edge"])) for op in safe]
    flip = load_flip()
    result, replay = flip.propose_flip_batch(
        source, target_rows, max_flips=len(safe), min_angle_gain_deg=1.0e-8,
        min_normal_dot=0.0, allowed_edges=allowed_edges,
        excluded_face_rows=sorted(witness_rows),
    )
    generated = replay.get("operations", [])
    got = {op_signature(op) for op in generated}
    if got != expected or len(generated) != len(safe):
        raise RuntimeError(f"358 regenerated a different safe subset: expected {len(expected)}, got {len(got)}")
    if not np.array_equal(result["points_world_m"], source["points_world_m"]):
        raise RuntimeError("subset flip moved points")
    if not np.array_equal(result["source_O_faces"], result["area_reference_037_source_O_faces"]):
        raise RuntimeError("subset flip broke matched-037 face synchronization")
    for key, value in source.items():
        if ("anchor" in key or key.startswith(("source_prolongation_", "prior_input_parent__source_prolongation_",
                                                "prior_input_006__source_prolongation_", "prior_run028__source_prolongation_"))
                or key == "area_reference_037_points_world_m_f64"):
            if key not in result or not np.array_equal(result[key], value):
                raise RuntimeError(f"point/source-anchor/reference invariant changed: {key}")
    changed_rows = set(map(int, np.flatnonzero(np.any(result["source_O_faces"] != faces, axis=1))))
    expected_changed = {int(row) for op in safe for row in op["face_rows"]}
    if changed_rows != expected_changed:
        raise RuntimeError("output changed face rows outside the safe 370 subset")
    if changed_rows & witness_rows:
        raise RuntimeError("output still changes a row implicated by a 130 self witness")
    # Same-parent boundary rings and exact boundary edges must remain unchanged.
    edges_before, edges_after = flip._edge_map(faces), flip._edge_map(result["source_O_faces"])
    boundary_before = {edge for edge, inc in edges_before.items() if len(inc) == 1}
    boundary_after = {edge for edge, inc in edges_after.items() if len(inc) == 1}
    if boundary_before != boundary_after:
        raise RuntimeError("subset changed the exact surface boundary edge set")
    if not np.array_equal(result["refinement_face_parent_face_row"], source["refinement_face_parent_face_row"]):
        raise RuntimeError("immediate source-parent face IDs changed")
    # The 358 owner recomputes barycentric ancestry and unions CSR source ancestry
    # on changed pairs. Check exact union preservation from original parent faces.
    parent = np.asarray(source["refinement_face_parent_face_row"])
    for op in generated:
        rows = list(map(int, op["face_rows"]))
        if parent[rows[0]] != parent[rows[1]] or any(parent[r] != op["parent_row"] for r in rows):
            raise RuntimeError("flip escaped its exact immediate parent")
    OUT.mkdir(parents=True)
    candidate_out = OUT / "candidate.npz"
    with candidate_out.open("xb") as stream:
        np.savez_compressed(stream, **result)
    def csr_rows(data, ptr_name, val_name, row):
        ptr, vals = np.asarray(data[ptr_name]), np.asarray(data[val_name])
        return set(map(int, vals[ptr[row]:ptr[row+1]]))
    ancestry_checks = []
    for op in generated:
        rows = list(map(int, op["face_rows"]))
        source_union = csr_rows(source, "face_source_ancestry_indptr", "face_source_ancestry_face_rows", rows[0]) | csr_rows(source, "face_source_ancestry_indptr", "face_source_ancestry_face_rows", rows[1])
        sub_union = csr_rows(source, "face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows", rows[0]) | csr_rows(source, "face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows", rows[1])
        if any(csr_rows(result, "face_source_ancestry_indptr", "face_source_ancestry_face_rows", r) != source_union for r in rows):
            raise RuntimeError("changed face source ancestry is not the exact old-pair union")
        if any(csr_rows(result, "face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows", r) != sub_union for r in rows):
            raise RuntimeError("changed face fine-subface ancestry is not the exact old-pair union")
        ancestry_checks.append({"face_rows": rows, "parent_row": int(op["parent_row"]),
                                "source_ancestry_rows": sorted(source_union),
                                "fine_subface_ancestry_rows": sorted(sub_union)})
    out_sha = sha_file(candidate_out)
    report = {
        "status": "SELF_WITNESS_FILTERED_358_SUBSET_PROPOSAL",
        "geometry_admitted": False, "native_admission": False, "accepted_physical_steps": 0,
        "method": "Rebuild only 370-selected edges not incident to any 130 self-witness face row; no replacement/fill flips.",
        "base_candidate": str(BASE), "base_candidate_sha256": PINS["base_candidate"],
        "base_159_report": str(BASE159), "base_159_report_sha256": PINS["base_159_report"],
        "selector_370_report": str(SELECTOR), "selector_370_report_sha256": PINS["selector_report"],
        "selector_370_candidate_sha256": selected_hash,
        "self_audit_130_summary_sha256": PINS["self_summary"],
        "self_pair_witness_sha256": EXPECTED_SELF_PAIR_SHA256,
        "self_pair_count": len(self_pairs), "unique_self_witness_face_rows": len(witness_rows),
        "original_370_selected_flip_count": len(chosen),
        "excluded_implicated_flip_count": len(implicated),
        "retained_flip_count": len(safe),
        "excluded_operations": [{"edge": op["edge"], "face_rows": op["face_rows"],
                                 "implicated_self_witness_rows": sorted(witness_rows.intersection(op["face_rows"]))}
                                for op in implicated],
        "retained_edges": [op["edge"] for op in safe],
        "retained_operation_signatures": [{"edge": list(sig[0]), "face_rows": list(sig[1]), "parent_row": sig[2]}
                                          for sig in sorted(expected)],
        "target_face_rows_from_complete_159_baseline": len(target_rows),
        "rebuilt_operation_count": len(generated),
        "rebuilt_358_report": replay,
        "changed_face_rows": sorted(changed_rows),
        "coordinates_bitwise_unchanged": True,
        "anchor_source_and_output_arrays_bitwise_unchanged": True,
        "area_reference_037_points_bitwise_unchanged": True,
        "matched_037_face_connectivity_synchronized": True,
        "exact_boundary_edge_set_unchanged": True,
        "source_O_face_parent_ids_unchanged": True,
        "changed_face_ancestry_union_checks": ancestry_checks,
        "output_candidate": str(candidate_out), "output_candidate_sha256": out_sha,
        "output_points_sha256": array_sha(result["points_world_m"]),
        "output_faces_sha256": array_sha(result["source_O_faces"]),
        "full_exact_self_and_185_bank_audits_required": True,
    }
    (OUT / "subset-report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": report["status"], "excluded": len(implicated),
                      "retained": len(safe), "output_candidate_sha256": out_sha}, sort_keys=True))

if __name__ == "__main__":
    main()
