#!/usr/bin/env python3
"""Read-only source/native joint and toe articulation coverage audit."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
from datetime import datetime, timezone

BASE = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020")
AUDIT_ROOT = BASE / "metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
RIGID = pathlib.Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
SOURCE_DIR = pathlib.Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4")
SOURCE_MANIFEST = SOURCE_DIR / "source-manifest.json"
M4 = SOURCE_DIR / "m4-pro.json"
BONE_MANIFEST = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
ELBOW = AUDIT_ROOT / "arm-actuation-preparation-001/arm-flexor-assessment-006-final.json"
LIGAMENT = BASE / "ligament-audit-current-f15c28/runtime-mapping-check.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).with_name("joint-coverage-audit.json"))
    args = ap.parse_args()

    for p in [RIGID, SOURCE_MANIFEST, M4, BONE_MANIFEST, ELBOW, LIGAMENT]:
        require(p.is_file(), f"missing required input: {p}")
    sm = json.loads(SOURCE_MANIFEST.read_text())
    m4 = json.loads(M4.read_text())
    bm = json.loads(BONE_MANIFEST.read_text())
    BONE_PAYLOAD = BONE_MANIFEST.parent / bm["payload"]["file"]
    require(BONE_PAYLOAD.is_file(), f"missing bone payload: {BONE_PAYLOAD}")
    elbow = json.loads(ELBOW.read_text())
    lig = json.loads(LIGAMENT.read_text())
    rigid_bytes = RIGID.read_bytes()
    require(sha(BONE_PAYLOAD) == bm["payload"]["sha256"], "bone payload hash differs from its retained manifest")
    require(rigid_bytes[:8] == b"NHRIGID2", "unexpected native rigid payload magic")
    version, variant, source_body_count, engine_body_count, joint_count, nq, nv = struct.unpack_from("<7I", rigid_bytes, 8)
    body_order = sm["core_tree"]["body_order"]
    require(version == 1, f"unexpected NHRIGID ABI version {version}")
    require(source_body_count == sm["model"]["source_body_count"], "native/source source-body count mismatch")
    require(engine_body_count == sm["core_tree"]["engine_body_count"] == len(body_order), "native/source engine-body count mismatch")
    require(nq == sm["model"]["source_nq"] and nv == sm["model"]["source_nv"], "native/source q/v count mismatch")
    embedded_archive_sha = rigid_bytes[48:80].hex()
    require(embedded_archive_sha == sm["source"]["archive_sha256"], "native rigid embedded source archive hash differs from selected source manifest")
    body_offset = 224
    joint_offset = body_offset + engine_body_count * 160
    dof_offset = joint_offset + joint_count * 144
    default_q_offset = dof_offset + nv * 64
    default_v_offset = default_q_offset + nq * 4
    source_map_offset = default_v_offset + nv * 4
    source_pose_offset = source_map_offset + source_body_count * 4
    expected_bytes = source_pose_offset + source_body_count * 28
    require(len(rigid_bytes) == expected_bytes, f"NHRIGID stream size mismatch: {len(rigid_bytes)} != {expected_bytes}")
    native_body_prefix = [struct.unpack_from("<4I", rigid_bytes, body_offset + i * 160) for i in range(engine_body_count)]
    native_joint_prefix = [struct.unpack_from("<8I", rigid_bytes, joint_offset + i * 144) for i in range(joint_count)]
    native_dof_prefix = [struct.unpack_from("<8I4f", rigid_bytes, dof_offset + i * 64) for i in range(nv)]

    joint_rows = sm["core_tree"]["source_joint_map"]
    by_name = {r["source_name"]: r for r in joint_rows}
    by_q = {int(r["core_q_index"]): r for r in joint_rows}
    residual_by_v = {int(r["dof"]): r for r in m4["residuals"]}
    require(len(residual_by_v) == nv, f"source oracle has {len(residual_by_v)} residual rows, expected nv={nv}")

    def joint(name, expected_child):
        r = by_name[name]
        q = int(r["core_q_index"])
        v = int(r["core_v_index"])
        require(q == v + 1, f"target hinge q/v convention changed for {name}: q={q}, v={v}")
        o = residual_by_v[v]
        child = o.get("child_body_name")
        child_idx = int(o.get("child_body", -1))
        require(child == expected_child, f"{name} child mismatch: {child} != {expected_child}")
        require(0 <= child_idx < len(body_order) and body_order[child_idx] == child, f"{name} child index/name mismatch: {child_idx}/{child}")
        require(int(o.get("joint", -1)) == int(r["core_joint_index"]), f"{name} joint index mismatch between manifest and source oracle")
        lo, hi = [float(x) for x in r["source_range"]]
        require(abs(lo - float(o["lower_limit"])) < 2e-5 and abs(hi - float(o["upper_limit"])) < 2e-5, f"{name} limits disagree between source manifest and oracle")
        native_dof = native_dof_prefix[v]
        require(int(native_dof[1]) == int(r["core_joint_index"]) and int(native_dof[2]) == q and int(native_dof[3]) == v, f"{name} q/v/joint owner differs in actual NHRIGID DoF table")
        require(abs(float(native_dof[8]) - lo) < 2e-5 and abs(float(native_dof[9]) - hi) < 2e-5, f"{name} limits differ in actual NHRIGID")
        native_joint = native_joint_prefix[int(r["core_joint_index"])]
        parent_idx = int(native_joint[0])
        require(int(native_joint[1]) == child_idx, f"{name} native joint child index mismatch")
        require(int(native_body_prefix[child_idx][1]) == parent_idx, f"{name} body parent index disagrees with its native joint descriptor")
        parent_name = body_order[parent_idx]
        muscles = o.get("muscles", [])
        pos = sorted((m for m in muscles if float(m["generalized_force"]) > 1e-3), key=lambda x: -float(x["generalized_force"]))[:3]
        neg = sorted((m for m in muscles if float(m["generalized_force"]) < -1e-3), key=lambda x: float(x["generalized_force"]))[:3]
        def compact(m):
            return {"id": int(m["index"]), "name": m["name"], "generalized_force_Nm": float(m["generalized_force"])}
        return {
            "source_name": name, "q_index": q, "v_index": v,
            "core_joint_index": int(r["core_joint_index"]),
            "parent_body": parent_name, "parent_body_index": parent_idx,
            "child_body": child, "child_body_index": child_idx,
            "limits_rad": [lo, hi], "limit_status": r.get("core_limit_status"),
            "source_oracle_top_positive_muscles": [compact(m) for m in pos],
            "source_oracle_top_negative_muscles": [compact(m) for m in neg],
        }

    joints = []
    expected = {
        "shoulder_elv_r": "__myosim_serial_humphant1_r_0", "shoulder1_r2_r": "humphant1_r", "shoulder_rot_r": "humerus_r",
        "elbow_flexion_r": "ulna_r",
        "shoulder_elv_l": "__myosim_serial_humphant1_l_0", "shoulder1_r2_l": "humphant1_l", "shoulder_rot_l": "humerus_l",
        "elbow_flexion_l": "ulna_l",
        "knee_angle_r": "__myosim_serial_tibia_r_2", "ankle_angle_r": "talus_r",
        "subtalar_angle_r": "calcn_r", "mtp_angle_r": "toes_r",
        "knee_angle_l": "__myosim_serial_tibia_l_2", "ankle_angle_l": "talus_l",
        "subtalar_angle_l": "calcn_l", "mtp_angle_l": "toes_l",
    }
    for n, child in expected.items():
        joints.append(joint(n, child))
    by_joint_name = {j["source_name"]: j for j in joints}
    for side in ("r", "l"):
        ankle = by_joint_name[f"ankle_angle_{side}"]
        subtalar = by_joint_name[f"subtalar_angle_{side}"]
        mtp = by_joint_name[f"mtp_angle_{side}"]
        require(ankle["child_body"] == subtalar["parent_body"], f"{side} ankle→subtalar parent/child chain broken")
        require(subtalar["child_body"] == mtp["parent_body"], f"{side} subtalar→MTP parent/child chain broken")
        require(mtp["child_body"] == f"toes_{side}", f"{side} MTP is not the single aggregate toes body")

    finger_dofs = []
    for side, qbase in [("r", 50), ("l", 88)]:
        for digit, q in enumerate([qbase, qbase+4, qbase+8, qbase+12], start=2):
            row = by_q[q]
            name = row["source_name"]
            expected_names = {f"md{digit}_flexion_{side}", f"md{digit}_flexion" if side == "r" and digit == 3 else f"md{digit}_flexion_{side}"}
            require(name in expected_names, f"unexpected hand q{q}: {name}")
            child_name = residual_by_v[int(row["core_v_index"])]["child_body_name"]
            cleaned = (child_name or "").replace("__myosim_serial_", "")
            cleaned = re.sub(r"_\d+$", "", cleaned)
            match = re.search(r"(?:distph|midph)([2-5])|([2-5])proxph", cleaned)
            require(match is not None and str(digit) in match.groups() and cleaned.endswith("_" + side), f"md{digit} q{q} is not attached to its same-digit hand phalanx: {child_name}")
            require(not cleaned.startswith("toes_"), f"hand flexion q{q} misclassified as foot")
            finger_dofs.append({"source_name": name, "q_index": q, "v_index": int(row["core_v_index"]), "parent_body": body_order[int(native_joint_prefix[int(row["core_joint_index"])][0])], "child_body": child_name})

    # The model's foot digit segments are grouped by a single rigid-body transform per side.
    toe_compounds = bm["toe_rigid_compounds"]
    require(len(toe_compounds) == 10, f"expected ten side/digit compounds, got {len(toe_compounds)}")
    anchors = {a["member_id"]: a for a in bm["source"]["anchors"]}
    toe_groups = []
    toe_member_ids = set()
    for c in toe_compounds:
        side = "r" if c["myosim_body"] == "toes_r" else "l"
        require(c["myosim_body"] in ("toes_r", "toes_l"), f"unexpected toe body {c['myosim_body']}")
        require(int(c["independent_articulation_count"]) == 0, "toe compound declares independent articulation")
        body_index = int(c["core_body_index"])
        require(body_order[body_index] == c["myosim_body"], f"toe body index/name mismatch: {body_index}/{c['myosim_body']}")
        members = []
        for mid in c["source_member_ids"]:
            require(mid not in toe_member_ids, f"duplicate toe source member {mid}")
            toe_member_ids.add(mid)
            a = anchors[mid]
            require(a["myosim_body"] == c["myosim_body"] and int(a["core_body_index"]) == int(c["core_body_index"]), f"toe member {mid} not on shared body")
            members.append({"member_id": mid, "source_record_index": int(a["source_record_index"]), "vertices": int(a["vertex_count"]), "triangles": int(a["triangle_count"]), "member_sha256": a["member_sha256"]})
        toe_groups.append({
            "side": side, "digit": int(c["digit"]), "myosim_body": c["myosim_body"],
            "core_body_index": int(c["core_body_index"]),
            "binding": c["binding"], "independent_articulation_count": 0,
            "source_member_ids": members,
        })
    source_body_names = [x["name"] for x in sm["core_tree"]["source_body_records"]]
    toe_like_source_bodies = [n for n in source_body_names if re.search(r"(toe|phal|hallux)", n, re.I)]
    require(set(toe_like_source_bodies) == {"toes_r", "toes_l"}, f"source body inventory contains unexpected toe/phalange bodies: {toe_like_source_bodies}")

    foot_joint_names = sorted(n for n in by_name if n.startswith(("ankle_angle_", "subtalar_angle_", "mtp_angle_")))
    require(foot_joint_names == sorted(["ankle_angle_r","subtalar_angle_r","mtp_angle_r","ankle_angle_l","subtalar_angle_l","mtp_angle_l"]), f"unexpected foot joint set: {foot_joint_names}")
    foot_children = {"talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"}
    foot_chain_rows = [r for r in joint_rows if residual_by_v[int(r["core_v_index"])].get("child_body_name") in foot_children]
    require(sorted(r["source_name"] for r in foot_chain_rows) == foot_joint_names, f"unexpected articulation attached to foot/toe child bodies: {[r['source_name'] for r in foot_chain_rows]}")

    input_hashes = {str(p): sha(p) for p in [RIGID, SOURCE_MANIFEST, M4, BONE_MANIFEST, BONE_PAYLOAD, ELBOW, LIGAMENT]}
    rigid_payload_expected = sm["payloads"]["rigid"]["sha256"]
    elbow_summary = {
        "receipt": str(ELBOW), "receipt_sha256": input_hashes[str(ELBOW)],
        "status": elbow.get("derived_status"),
        "scope": elbow.get("scope"),
        "simulated_seconds": elbow["native"].get("simulated_seconds"),
        "step_count": elbow["native"].get("step_count"),
        "replay": elbow["native"].get("replay"),
        "muscle_ids": elbow["native"].get("muscle_ids"),
        "right_q_increment_rad": elbow["elbow_response"]["right"].get("q_increment_rad"),
        "left_q_increment_rad": elbow["elbow_response"]["left"].get("q_increment_rad"),
        "right_v_increment_rad_s": elbow["elbow_response"]["right"].get("v_increment_rad_s"),
        "left_v_increment_rad_s": elbow["elbow_response"]["left"].get("v_increment_rad_s"),
        "right_selected_torque_increment_Nm": elbow["elbow_response"]["right"].get("native_selected_muscle_generalized_torque_increment_nm"),
        "left_selected_torque_increment_Nm": elbow["elbow_response"]["left"].get("native_selected_muscle_generalized_torque_increment_nm"),
    }
    ligament_summary = {
        "receipt": str(LIGAMENT), "receipt_sha256": input_hashes[str(LIGAMENT)],
        "schema": lig.get("schema"), "rows": lig.get("rows"),
        "identity_mapping": lig.get("identity_mapping"),
        "coverage_by_group": lig.get("coverage", {}).get("by_group"),
        "all_185_meshes_checked_each_row": lig.get("coverage", {}).get("all_185_meshes_checked_for_each_row"),
        "total_segment_triangle_hits": lig.get("coverage", {}).get("total_segment_triangle_hits"),
        "explicit_representation_limits": lig.get("explicit_representation_limits"),
    }
    report = {
        "schema": "numi.human-joint-body-chain-coverage.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "read-only cross-check of selected native NHRIGID header, source-manifest joint q/v/body indices, m4-pro child-body and source-limit rows, and retained toe bone member anchors",
        "native_scene": {
            "path": str(RIGID), "sha256": input_hashes[str(RIGID)], "magic": rigid_bytes[:8].decode(),
            "abi_version": version, "header_variant": variant,
            "source_body_count": source_body_count, "engine_body_count": engine_body_count,
            "joint_count": joint_count, "nq": nq, "nv": nv,
            "embedded_source_archive_sha256": embedded_archive_sha,
            "source_manifest_rigid_payload_sha256": rigid_payload_expected,
            "native_file_matches_manifest_rigid_payload_sha256": input_hashes[str(RIGID)] == rigid_payload_expected,
            "source_archive_identity_matches": True,
        },
        "source_inputs": {
            "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": input_hashes[str(SOURCE_MANIFEST)],
            "m4_pro": str(M4), "m4_pro_sha256": input_hashes[str(M4)],
            "bone_manifest": str(BONE_MANIFEST), "bone_manifest_sha256": input_hashes[str(BONE_MANIFEST)],
            "bone_payload": str(BONE_PAYLOAD), "bone_payload_sha256": input_hashes[str(BONE_PAYLOAD)],
            "upstream_archive": sm["source"]["archive"], "upstream_archive_sha256": sm["source"]["archive_sha256"],
            "upstream_revision": sm["source"]["revision"], "upstream_license": sm["source"]["license"],
        },
        "joint_coverage": joints,
        "hand_vs_foot_name_chain_assertion": {
            "hand_md2_to_md5_flexion": finger_dofs,
            "source_body_inventory_toe_or_phalange_names": toe_like_source_bodies,
            "assertion": "md2-md5 flexion q rows resolve to hand phalanx child bodies; foot q rows resolve only to ankle/subtalar calcn bodies and one MTP child toes_r/l.",
        },
        "toe_bone_mesh_bindings": {
            "side_digit_compound_count": len(toe_groups),
            "digit_count_per_side": {"r": 5, "l": 5},
            "current_independent_toe_or_IP_dof_count": 0,
            "source_member_geometry_is_available_for_future_split": True,
            "bone_payload": {"path": str(BONE_PAYLOAD), "sha256": input_hashes[str(BONE_PAYLOAD)], "manifest_expected_sha256": bm["payload"]["sha256"], "member_count": len(toe_member_ids)},
            "groups": toe_groups,
            "interpretation": "Separate anatomical source member meshes/IDs exist for toe segments, but every member is currently bound to one shared toes_r or toes_l rigid-body transform; the present model cannot flex individual toes or their IP joints.",
        },
        "retained_native_selected_actuation": {
            "elbow_flexor": elbow_summary,
            "shoulder_knee_ankle_toe_selected_activation_receipt_included": False,
            "scope_boundary": "This audit establishes structural DOFs and source-oracle muscle contributors. It does not qualify shoulder/knee/ankle/toe active response, contact, or skin/fat mechanics."
        },
        "retained_ligament_mapping": ligament_summary,
        "skin_fat_and_detailed_to_fem": {
            "joint_specific_qualified_mapping_established_by_this_audit": False,
            "status": "not_established",
            "scope_note": "The elbow activation receipt is a core selected-tendon response, not joint-region skin/fat or deformable contact qualification. Existing hand-only embedding evidence does not establish shoulder, elbow, knee, ankle, or toe layer coverage."
        },
        "input_sha256": input_hashes,
        "assertions": {
            "nhrigid_header_counts_match_source_manifest": True,
            "embedded_archive_matches_source_manifest": True,
            "selected_q_v_limits_match_source_oracle": True,
            "child_body_indices_match_source_body_order": True,
            "native_body_parent_matches_joint_descriptor": True,
            "foot_ankle_subtalar_mtp_parent_child_chain_asserted": True,
            "hand_md_flexion_cannot_be_mislabeled_as_toe_motion": True,
            "toe_mesh_members_share_aggregate_body_with_no_independent_articulation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS_SOURCE_BODY_CHAIN_ASSERTIONS",
        "output": str(args.output),
        "output_sha256": sha(args.output),
        "joint_rows": len(joints),
        "toe_compounds": len(toe_groups),
        "toe_independent_dofs": 0,
        "native_nhrigid_sha256": input_hashes[str(RIGID)],
        "source_manifest_rigid_sha256_matches_native": input_hashes[str(RIGID)] == rigid_payload_expected,
        "elbow_receipt_status": elbow.get("derived_status"),
    }, indent=2))

if __name__ == "__main__":
    main()
