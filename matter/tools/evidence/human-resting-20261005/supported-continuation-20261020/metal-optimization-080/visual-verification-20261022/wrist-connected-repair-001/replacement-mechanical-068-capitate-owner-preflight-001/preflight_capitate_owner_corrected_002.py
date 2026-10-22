from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = E / "replacement-mechanical-068-capitate-owner-preflight-001"
CURRENT = E / "finger-profile-068-whole-hand-minimax"
OLD = E / "replacement-mechanical-061-normal-cone-001"
LAYER = E / "replacement-mechanical-066-joint-3d-layer-fit/joint-3d-layers.npz"
sys.path.insert(0, "/Users/n/numi-human-anatomy-finish-1378/src")
sys.path.insert(0, str(E / "replacement-mechanical-055"))
from numilab_human.skin_source_payload_preflight import decode_payload
from native_prism_checks_vectorized import signed_prisms

MANIFEST = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
OLD_REPORT = OUT / "capitate-attachment-owner-preflight.json"
OUTPUT = OUT / "capitate-attachment-owner-preflight-002.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    geometry_path = CURRENT / "replacement-geometry.npz"
    payload_path = CURRENT / "replacement-bilateral-hands038.nhskin"
    source_geometry_path = E / "replacement-wrists-058/replacement-geometry.npz"
    source_payload_path = E / "replacement-wrists-058/replacement-bilateral-hands038.nhskin"
    source_map_path = OLD / "coarse-to-source.npz"
    coarse_path = OLD / "coarse-surface.npz"
    sites_path = OLD / "attachment-audit.stdout.json"
    determinant_owner_path = E / "replacement-mechanical-055/native_prism_checks_vectorized.py"

    geometry = np.load(geometry_path, allow_pickle=False)
    source_geometry = np.load(source_geometry_path, allow_pickle=False)
    source_map = np.load(source_map_path, allow_pickle=False)
    coarse = np.load(coarse_path, allow_pickle=False)
    assert np.array_equal(geometry["faces"], source_geometry["faces"])
    assert np.array_equal(geometry["weights"], source_geometry["weights"])

    payload = decode_payload(payload_path.read_bytes())
    source_payload = decode_payload(source_payload_path.read_bytes())
    bind_ids = np.asarray(payload["bindings_u"][:, 0], dtype=np.int64)
    assert np.array_equal(bind_ids, np.asarray(source_payload["bindings_u"][:, 0], dtype=np.int64))
    manifest = json.loads(MANIFEST.read_text())
    body_names = {int(x["core_body_index"]): str(x["myosim_body"]) for x in manifest["source"]["anchors"]}
    names = [body_names.get(int(body), f"core_body_{int(body)}") for body in bind_ids]

    text = sites_path.read_text()
    sites = json.loads(text.split("\n", 1)[1])
    sites = [s for s in sites if str(s.get("donor_name", "")).startswith("capitate_")]
    incidence = np.bincount(np.asarray(coarse["faces"], dtype=np.int64).ravel(), minlength=len(coarse["world"]))
    shift = np.asarray([-0.05, 0.20, 2.00], dtype=np.float64)
    stale_rows = []
    for site in sites:
        row = int(site["reduced_vertex"])
        face_id = int(source_map["source_face"][row])
        bary = np.asarray(source_map["source_barycentric"][row], dtype=np.float64)
        tri = np.asarray(geometry["faces"][face_id], dtype=np.int64)
        weights = np.einsum("i,ij->j", bary, geometry["weights"][tri])
        order = np.argsort(weights)[::-1]
        point = np.einsum("i,ij->j", bary, geometry["world"][tri])
        point_coarse = point + shift
        source_residual = float(np.linalg.norm(np.asarray(coarse["world"][row]) - shift - point) * 1000)
        best = int(order[0])
        stale_rows.append({
            "old_reduced_vertex": row,
            "old_source_vertex_label_only": int(site["source_vertex"]),
            "old_inferred_owner": str(site["donor_name"]),
            "old_donor_surface_distance_mm": float(site["donor_surface_distance_mm"]),
            "old_site_provenance": str(site["site_provenance"]),
            "current_coarse_face_incidence_count": int(incidence[row]),
            "active_current_coarse_node": bool(incidence[row] > 0),
            "current_registered_source_face": face_id,
            "current_registered_source_face_vertices": [int(v) for v in tri],
            "current_source_barycentric": [float(v) for v in bary],
            "current_source_face_body_weight_ranking": [
                {
                    "myosim_body": names[int(i)],
                    "core_body_index": int(bind_ids[int(i)]),
                    "weight": float(weights[int(i)]),
                }
                for i in order[:5]
            ],
            "current_registered_surface_point_world_m": [float(v) for v in point],
            "current_registered_surface_point_in_coarse_frame_m": [float(v) for v in point_coarse],
            "current_coarse_to_registered_surface_residual_mm": source_residual,
            "classification": "stale_identity_evidence_only",
            "attachment_owner_selected": None,
            "attachment_payload_emitted": False,
        })

    layer = np.load(LAYER, allow_pickle=False)
    faces = np.asarray(coarse["faces"], dtype=np.int64)
    oriented_checks = {}
    for label, outer, inner, thickness in (
        ("skin", layer["outer_world"], layer["inner_world"], layer["skin_thickness_m"]),
        ("fat", layer["inner_world"], layer["deep_world"], layer["fat_thickness_m"]),
    ):
        bad, values = signed_prisms(outer, inner, thickness, faces)
        oriented_checks[label] = {
            "owner_predicate": "native_prism_checks_vectorized.signed_prisms",
            "owner_predicate_source_sha256": sha(determinant_owner_path),
            "triangle_count": int(len(faces)),
            "wrong_signed_or_degenerate_prism_count": int(bad.sum()),
            "wrong_face_rows": [int(i) for i in np.flatnonzero(bad)],
            "all_inputs_finite": bool(np.isfinite(values).all()),
            "minimum_abs_tet_volume_m3": float(np.min(np.abs(values[:, :3]))),
            "float32_magnitude_gate_run": False,
            "float32_gate_note": "The owner rule computes Float32 magnitude qualification only after signed-prism admission; this layer fails the signed gate.",
        }

    report = {
        "schema": "numi.human.inferred-hand-fat-attachment-owner-preflight.v2",
        "status": "stale_identity_evidence_only_layer_gate_failed",
        "supersedes": {
            "report": str(OLD_REPORT),
            "sha256": sha(OLD_REPORT),
            "reason": "Preflight 001 incorrectly counted negative determinant components without the native sorted-global prism face-parity rule, and treated an orphan coarse row as an owner candidate.",
            "retained_unchanged": True,
        },
        "scope": "Reaudit of the two old capitate-labelled inferred attachment rows against current 068 skin ancestry, active coarse topology, and exact native layer prism admission.",
        "current_mapping_identity": {
            "registered_068_geometry_sha256": sha(geometry_path),
            "registered_068_payload_sha256": sha(payload_path),
            "registered_058_geometry_sha256": sha(source_geometry_path),
            "registered_058_payload_sha256": sha(source_payload_path),
            "current_faces_equal_058": True,
            "current_weights_equal_058": True,
            "current_binding_order_equal_058": True,
            "source_map_061_sha256": sha(source_map_path),
            "coarse_surface_061_sha256": sha(coarse_path),
            "source_face_count": int(len(geometry["faces"])),
            "coarse_allocated_node_count": int(len(coarse["world"])),
            "coarse_referenced_node_count": int(len(np.unique(faces))),
            "coarse_orphan_node_count": int(len(coarse["world"]) - len(np.unique(faces))),
            "bone_binding_manifest_sha256": sha(MANIFEST),
        },
        "old_capitate_sites": stale_rows,
        "attachment_decision": {
            "selected_sites": [],
            "no_orphan_rows_are_admitted": True,
            "no_owner_retarget_from_source_weight_only": True,
            "reason": "Both legacy rows have zero active coarse-face incidences. Row 297 maps to 94.77% 2proxph_r on a stale ancestry face, not to capitate; row 373 has no body above the 0.85 owner-weight screen. Neither is a current capitate anchor.",
            "next_owner_requirements": [
                "Select a new point on the current active boundary, not by legacy vertex identity.",
                "Require a valid current skin/fat quality and spacing receipt using the owner signed-prism gate.",
                "Keep inferred-source provenance explicit; no measured capitate enthesis evidence is available.",
            ],
        },
        "layer_admission": {
            "diagnostic_layer_sha256": sha(LAYER),
            "oriented_native_signed_prism_checks": oriented_checks,
            "status": "FAIL_SIGNED_PRISM_GATE",
            "admitted_to_attachment_payload": False,
        },
        "limitations": [
            "The 068 skin-weight interpolation is ancestry diagnostic evidence only; it does not locate an enthesis or validate a deep-fat anchor.",
            "The exact native owner predicate rejects the current 066 layer before Float32 determinant magnitude checks.",
            "No new attachment point, reduction header, payload, or physics owner is produced.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(OUTPUT),
        "orphan_rows": [{"row": x["old_reduced_vertex"], "incidences": x["current_coarse_face_incidence_count"], "source_owner": x["current_source_face_body_weight_ranking"][0]} for x in stale_rows],
        "native_oriented_wrong_prisms": {k: v["wrong_signed_or_degenerate_prism_count"] for k,v in oriented_checks.items()},
        "status": report["status"],
    }, indent=2))

if __name__ == "__main__":
    main()
