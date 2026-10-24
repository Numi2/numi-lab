#!/usr/bin/env python3
"""Compose disjoint regional NHSKIN topology repairs as a diagnostic artifact.

The command validates source lineage and exact untouched records. It does not
admit mechanics, tissue mass, pose behavior, or geometric overlap. The outer
D36 O descriptor is pinned and recorded separately; it is never coerced into
an NHSKIN patch.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def require_sha(path, expected, label):
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def native_admission_decision(candidate_sha, component_receipt_path=None,
                              mechanical_receipt_path=None):
    """Fail closed unless one receipt certifies this exact composite candidate."""
    reasons = []
    if not component_receipt_path or not Path(component_receipt_path).is_file():
        reasons.append("missing composite all-component geometry/pose gate receipt")
    else:
        c = load_json(component_receipt_path)
        if c.get("candidate_sha256") != candidate_sha:
            reasons.append("composite gate receipt candidate hash mismatch")
        required = ("exact_self_pass", "exact_bone_pass", "exact_tissue_pass",
                    "all_required_poses_pass", "source_outside_exact",
                    "native_admission")
        for key in required:
            if c.get(key) is not True:
                reasons.append(f"composite gate {key} is not true")
    if not mechanical_receipt_path or not Path(mechanical_receipt_path).is_file():
        reasons.append("missing candidate-matched mechanical map receipt")
    else:
        m = load_json(mechanical_receipt_path)
        if m.get("candidate_sha256") != candidate_sha:
            reasons.append("mechanical receipt candidate hash mismatch")
        for key in ("positive_cells_pass", "material_mass_closure_pass",
                    "all_required_poses_pass"):
            if m.get(key) is not True:
                reasons.append(f"mechanical map {key} is not true")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="frozen compositor module")
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-sha256", required=True)
    ap.add_argument("--arm", required=True, help="edge-refined NHSKIN candidate")
    ap.add_argument("--arm-sha256", required=True)
    ap.add_argument("--arm-ancestry", required=True)
    ap.add_argument("--arm-ancestry-sha256", required=True)
    ap.add_argument("--arm-report", required=True)
    ap.add_argument("--arm-report-sha256", required=True)
    ap.add_argument("--hand", required=True, help="duplicate-source NHSKIN candidate")
    ap.add_argument("--hand-sha256", required=True)
    ap.add_argument("--hand-patch", required=True)
    ap.add_argument("--hand-patch-sha256", required=True)
    ap.add_argument("--hand-report", required=True)
    ap.add_argument("--hand-report-sha256", required=True)
    ap.add_argument("--outer271", required=True, help="separate physical O descriptor NPZ")
    ap.add_argument("--outer271-sha256", required=True)
    ap.add_argument("--outer-audit", required=True)
    ap.add_argument("--outer-audit-sha256", required=True)
    ap.add_argument("--outer-tissue-audit", required=True)
    ap.add_argument("--outer-tissue-audit-sha256", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--component-gate", default=None)
    ap.add_argument("--mechanical-gate", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    module_path = Path(args.module).resolve()
    spec = importlib.util.spec_from_file_location("frozen_skin_compositor_256", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    pins = {
        "compositor_module": {"path": str(module_path), "sha256": sha256(module_path)},
        "base": {"path": str(Path(args.base).resolve()), "sha256": require_sha(args.base, args.base_sha256, "base")},
        "arm_candidate": {"path": str(Path(args.arm).resolve()), "sha256": require_sha(args.arm, args.arm_sha256, "arm candidate")},
        "arm_ancestry": {"path": str(Path(args.arm_ancestry).resolve()), "sha256": require_sha(args.arm_ancestry, args.arm_ancestry_sha256, "arm ancestry")},
        "arm_report": {"path": str(Path(args.arm_report).resolve()), "sha256": require_sha(args.arm_report, args.arm_report_sha256, "arm report")},
        "hand_candidate": {"path": str(Path(args.hand).resolve()), "sha256": require_sha(args.hand, args.hand_sha256, "hand candidate")},
        "hand_patch": {"path": str(Path(args.hand_patch).resolve()), "sha256": require_sha(args.hand_patch, args.hand_patch_sha256, "hand patch")},
        "hand_report": {"path": str(Path(args.hand_report).resolve()), "sha256": require_sha(args.hand_report, args.hand_report_sha256, "hand report")},
        "outer271_descriptor": {"path": str(Path(args.outer271).resolve()), "sha256": require_sha(args.outer271, args.outer271_sha256, "outer271 descriptor")},
        "outer271_audit": {"path": str(Path(args.outer_audit).resolve()), "sha256": require_sha(args.outer_audit, args.outer_audit_sha256, "outer271 audit")},
        "outer271_tissue_audit": {"path": str(Path(args.outer_tissue_audit).resolve()), "sha256": require_sha(args.outer_tissue_audit, args.outer_tissue_audit_sha256, "outer271 tissue audit")},
    }
    base = module.decode(Path(args.base).read_bytes())
    arm = module.decode(Path(args.arm).read_bytes())
    hand = module.decode(Path(args.hand).read_bytes())
    arm_anc = load_npz(args.arm_ancestry)
    hp = load_npz(args.hand_patch)
    if base["nv"] != 61950 or len(base["inds"]) != 108987:
        raise ValueError("unexpected source128 baseline domain")
    if arm["nv"] != 62056 or len(arm["inds"]) != 109199:
        raise ValueError("unexpected arm candidate domain")
    if hand["nv"] != 62353 or len(hand["inds"]) != 108987:
        raise ValueError("unexpected hand duplicate-source domain")
    if arm_anc["face_parent_source_rows"].shape != (len(arm["inds"]),):
        raise ValueError("arm face parent table does not cover arm faces")

    edge_repair = (arm,
                   arm_anc["face_parent_source_rows"],
                   arm_anc["new_vertex_edge_source_ids"],
                   arm_anc["new_vertex_edge_weights"])
    duplicate_repair = dict(
        payload=hand,
        face_parent_source_rows=hp["face_parent_source_rows"],
        protected_source_face_ids=hp["protected_source_face_ids"],
        replaced_source_face_ids=hp["replaced_source_face_ids"],
        duplicated_output_vertex_ids=hp["duplicated_output_vertex_ids"],
        duplicated_source_vertex_ids=hp["duplicated_source_vertex_ids"],
        ancestry_offsets=hp["ancestry_offsets"],
        source_vertex_ids=hp["source_vertex_ids"],
        source_weights=hp["source_weights"])
    raw, report = module.merge_source_topology_repairs(
        base, edge_repairs=[edge_repair], duplicate_repairs=[duplicate_repair])
    result = module.decode(raw)

    arm_parent = set(map(int, arm_anc["changed_parent_face_rows"]))
    hand_touched = set(map(int, report["regions"][-1]["touched_source_faces"]))
    hand_parent = set(map(int, hp["replaced_source_face_ids"]))
    if arm_parent & hand_touched:
        raise ValueError("arm and hand parent-face domains overlap")
    face_parent = np.asarray(report["face_parent_source_rows"], dtype=np.int64)
    if face_parent.shape != (len(result["inds"]),):
        raise ValueError("composed output face ancestry is incomplete")
    if not np.array_equal(base["binds"].tobytes(), result["binds"].tobytes()):
        raise ValueError("composed output changed source body bindings")

    changed = set(map(int, report["changed_original_rows"]))
    untouched = np.asarray(sorted(set(range(base["nv"])) - changed), dtype=np.int64)
    base_vertex_bytes = np.frombuffer(base["raw"][base["vertoff"]:base["vertoff"]+base["nv"]*56], dtype=np.uint8).reshape(base["nv"],56)
    result_vertex_bytes = np.frombuffer(result["raw"][result["vertoff"]:result["vertoff"]+base["nv"]*56], dtype=np.uint8).reshape(base["nv"],56)
    untouched_vertex_records_exact = bool(np.array_equal(base_vertex_bytes[untouched], result_vertex_bytes[untouched]))
    if not untouched_vertex_records_exact:
        raise ValueError("one or more declared untouched source vertex records changed")
    if base["weights"].tobytes() != result["weights"][:base["nv"]].tobytes():
        raise ValueError("source dense weights changed during geometry composition")
    untouched_parent_exact = 0
    changed_parent_set = arm_parent | hand_touched
    for outrow, parent in enumerate(face_parent):
        parent = int(parent)
        if parent not in changed_parent_set:
            if not np.array_equal(result["inds"][outrow], base["inds"][parent]):
                raise ValueError(f"untouched source face changed at parent {parent}")
            untouched_parent_exact += 1

    # The output CSR is an explicit lineage map, not a volume-overlap proof.
    edge_src = np.asarray(arm_anc["new_vertex_edge_source_ids"], dtype=np.int64)
    edge_w = np.asarray(arm_anc["new_vertex_edge_weights"], dtype=np.float64)
    dup_src = np.asarray(hp["duplicated_source_vertex_ids"], dtype=np.int64)
    lengths = np.r_[np.ones(base["nv"],dtype=np.int64),
                    np.full(len(edge_src),2,dtype=np.int64),
                    np.ones(len(dup_src),dtype=np.int64)]
    offsets = np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    ids = np.concatenate((np.arange(base["nv"],dtype=np.int64),edge_src.ravel(),dup_src))
    weights = np.concatenate((np.ones(base["nv"],dtype=np.float64),edge_w.ravel(),np.ones(len(dup_src),dtype=np.float64)))
    if len(lengths) != result["nv"] or int(offsets[-1]) != len(ids):
        raise ValueError("composed vertex ancestry does not cover output rows")
    np.savez_compressed(out/"lineage.npz", face_parent_source_rows=face_parent,
                        vertex_ancestry_offsets=offsets, vertex_ancestry_source_rows=ids,
                        vertex_ancestry_weights=weights,
                        original_source_vertex_count=np.asarray([base["nv"]],dtype=np.int64),
                        arm_new_vertex_source_rows=edge_src,
                        arm_new_vertex_source_weights=edge_w,
                        hand_duplicate_output_rows=np.asarray(hp["duplicated_output_vertex_ids"],dtype=np.int64)+len(edge_src),
                        hand_duplicate_source_rows=dup_src)
    payload_path = out/"candidate.nhskin"
    payload_path.write_bytes(raw)
    result_sha = sha256(payload_path)

    outer = load_npz(args.outer271)
    outer_audit = load_json(args.outer_audit)
    outer_tissue = load_json(args.outer_tissue_audit)
    arm_report = load_json(args.arm_report)
    hand_report = load_json(args.hand_report)
    if outer["outer_world_actual007_m"].ndim != 2 or outer["surface_faces"].ndim != 2:
        raise ValueError("outer271 descriptor arrays are malformed")
    if outer_audit.get("pins", {}).get("candidate_sha256") != pins["outer271_descriptor"]["sha256"]:
        raise ValueError("outer271 exact audit is not pinned to this descriptor")
    if outer_tissue.get("candidate_sha256") != pins["outer271_descriptor"]["sha256"]:
        raise ValueError("outer271 tissue audit is not pinned to this descriptor")
    if outer_audit.get("status") != "pass-exact-surface-and-bone":
        raise ValueError("outer271 exact surface/bone audit did not pass")

    admitted, blocked_reasons = native_admission_decision(
        result_sha, args.component_gate, args.mechanical_gate)
    receipt = dict(
        schema="numi.human.diagnostic-regional-skin-compose.v1",
        output_candidate_sha256=result_sha,
        output_vertex_count=result["nv"], output_face_count=len(result["inds"]),
        source_vertex_count=base["nv"], source_face_count=len(base["inds"]),
        arm_added_edge_vertices=len(edge_src), hand_added_duplicate_vertices=len(dup_src),
        arm_source_parent_faces=len(arm_parent), hand_declared_replaced_parent_faces=len(hand_parent),
        hand_touched_parent_faces=len(hand_touched),
        arm_hand_parent_overlap_count=0,
        changed_original_vertex_rows=len(changed),
        untouched_original_vertex_records_exact=untouched_vertex_records_exact,
        untouched_parent_faces_exact=untouched_parent_exact,
        original_dense_weights_exact=True, source_bindings_exact=True,
        face_parent_map_complete=True, vertex_ancestry_rows=len(lengths),
        ancestry="arm vertices use exact source-edge interpolation; hand vertices use declared direct source-row duplication",
        geometric_overlap_certified=False,
        physical_admission=False,
        native_admission=bool(admitted),
        native_admission_blockers=blocked_reasons,
        components={
            "arm": {"candidate_sha256": pins["arm_candidate"]["sha256"], "report_sha256": pins["arm_report"]["sha256"],
                    "status":arm_report.get("status"), "exact_poses":arm_report.get("exact_poses"),
                    "geometry_only":arm_report.get("geometry_only")},
            "hand": {"candidate_sha256": pins["hand_candidate"]["sha256"], "report_sha256": pins["hand_report"]["sha256"],
                     "status":hand_report.get("status"), "admission":hand_report.get("admission"),
                     "exact_skin_intersections":hand_report.get("exact_skin_intersections"),
                     "exact_all_nearby_bones":hand_report.get("exact_all_nearby_bones")},
            "outer271": {"descriptor_sha256":pins["outer271_descriptor"]["sha256"], "descriptor_only":True,
                         "surface_bone_audit_status":outer_audit.get("status"),
                         "bone_intersection_pairs":sum(int(v.get("exact_intersection_pairs",0)) for v in outer_audit.get("registered_anatomy_bone_intersections",{}).values()),
                         "tissue_intersection_pairs":outer_tissue.get("exact_nonbone_intersection_pairs"),
                         "physical_admission":outer_tissue.get("physical_admission")}},
        input_pins=pins,
        entrypoint_sha256=sha256(Path(__file__).resolve()),
        invocation_argv=sys.argv,
        compositor_report=report,
        notes=["Diagnostic source composition only; it does not repair component geometry.",
               "The O-only outer271 descriptor remains separate and is not coerced into the NHSKIN surface.",
               "No matching mechanical map with positive cells, material mass closure, and all-pose pass was supplied."])
    with open(out/"compose-report.json","w",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
        f.write("\n")
    print(json.dumps({"output":str(out),"candidate_sha256":result_sha,
                      "vertices":result["nv"],"faces":len(result["inds"]),
                      "changed_original_rows":len(changed),"native_admission":admitted,
                      "blockers":blocked_reasons},indent=2))

if __name__ == "__main__":
    main()
