"""Config-matched construction-only zero displacement bounds for area blockers.

The three row IDs were observed in the frozen 486 area-SCP diagnostics. This
helper pins the 461 reference topology/provenance and derives the active
candidate/audit hashes from the 490-validated matched config. It does not
declare the vertices anchors, forces, or accepted geometry constraints.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069")
REFERENCE_CANDIDATE_SHA256 = "742315227c1d8225aac9307af828251a73a98a9c40b4f59fe791aeef5da1ba0f"
EXPECTED_FACE_ARRAY_SHA256 = "ca497b20d90d0390e034bed0bacb348cf70f2ff05da23a52d978826cd8a800c8"
EXPECTED_BLOCKER_ANCESTRY_SHA256 = "3aba5eb6af7200ac5b4253596cb8e8130f0a4842f148aa0f4289c4d9dc0f81d0"
BLOCKER_FACE_ROWS = (32434, 11844, 40017)
EXPECTED_FACE_VERTICES = {
    32434: (7587, 7575, 7588),
    11844: (19587, 15333, 18958),
    40017: (7280, 10112, 13430),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _ancestry_digest(z, rows=BLOCKER_FACE_ROWS) -> str:
    h = hashlib.sha256()
    row_fields = (
        "source_O_faces", "area_reference_037_source_O_faces", "face_kind",
        "parent_coarse_face_row", "parent_fine_detail_face_row",
        "parent_source_face_marker", "parent_fine_subface_row",
        "refinement_face_parent_face_row", "refinement_face_parent_barycentric",
    )
    csr_fields = (
        ("face_source_ancestry", "face_source_ancestry_indptr", "face_source_ancestry_face_rows"),
        ("refinement_face_parent_ancestry", "refinement_face_parent_ancestry_indptr",
         "refinement_face_parent_ancestry_face_rows"),
        ("face_fine_subface_ancestry", "face_fine_subface_ancestry_indptr",
         "face_fine_subface_ancestry_rows"),
    )
    for row in rows:
        h.update(int(row).to_bytes(8, "little", signed=True))
        for key in row_fields:
            a = np.asarray(z[key])[row]
            h.update(key.encode() + b"\0")
            h.update(str(a.dtype).encode() + b"\0")
            h.update(np.asarray(a.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(a).view(np.uint8))
        for name, ip_key, values_key in csr_fields:
            ip = np.asarray(z[ip_key]); values = np.asarray(z[values_key])
            segment = values[int(ip[row]):int(ip[row + 1])]
            h.update(name.encode() + b"\0")
            h.update(segment.dtype.str.encode() + b"\0")
            h.update(np.asarray(segment.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(segment).view(np.uint8))
    return h.hexdigest()


def _npz_noncoordinate_identity_digest(z):
    h = hashlib.sha256()
    for key in sorted(k for k in z.files if k != "points_world_m"):
        a = np.ascontiguousarray(np.asarray(z[key]))
        h.update(key.encode() + b"\0")
        h.update(a.dtype.str.encode() + b"\0")
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.view(np.uint8))
    return h.hexdigest()


def _matched_record(matched_inputs, role, actual_path):
    if not isinstance(matched_inputs, dict) or role not in matched_inputs:
        raise ValueError(f"490 validated matched input is missing {role}")
    record = matched_inputs[role]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"490 validated {role} record is malformed")
    path = Path(record["path"]).resolve()
    if path != Path(actual_path).resolve():
        raise ValueError(f"{role} path differs from the 490 matched config")
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{role} digest from 490 matched config is malformed")
    if file_sha256(path) != digest:
        raise ValueError(f"{role} digest from 490 matched config no longer matches")
    return {"path": str(path), "sha256": digest}


def validate_blocker_candidate(candidate_path, audit159_path, *, matched_inputs,
                               matched_config_path, matched_config_sha256):
    """Validate the current 490 candidate against frozen non-coordinate identity."""
    candidate_path = Path(candidate_path).resolve()
    audit159_path = Path(audit159_path).resolve()
    candidate_record = _matched_record(matched_inputs, "candidate", candidate_path)
    audit_record = _matched_record(matched_inputs, "audit159", audit159_path)
    config_path = Path(matched_config_path).resolve()
    if not config_path.is_file() or file_sha256(config_path) != matched_config_sha256:
        raise ValueError("490 matched config path/hash is invalid or changed")
    config = json.loads(config_path.read_text())
    if config.get("schema") != "numi.human.matched_inputs_component_fit_475.v1":
        raise ValueError("505 requires the existing 490 matched-input schema")
    config_inputs = config.get("inputs")
    if not isinstance(config_inputs, dict):
        raise ValueError("490 matched config inputs are malformed")
    for role, record in (("candidate", candidate_record), ("audit159", audit_record)):
        cfg_record = config_inputs.get(role)
        if not isinstance(cfg_record, dict) or set(cfg_record) != {"path", "sha256"}:
            raise ValueError(f"490 config {role} record is malformed")
        if (Path(cfg_record["path"]).resolve() != Path(record["path"]).resolve()
                or cfg_record["sha256"] != record["sha256"]):
            raise ValueError(f"{role} expected hash was not derived from the 490 matched config")
    candidate_sha = file_sha256(candidate_path)
    audit_sha = file_sha256(audit159_path)
    if candidate_sha != candidate_record["sha256"] or audit_sha != audit_record["sha256"]:
        raise ValueError("candidate or audit does not match the 490-validated input hashes")
    reference_path = BASE / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz"
    reference_sha = file_sha256(reference_path)
    if reference_sha != REFERENCE_CANDIDATE_SHA256:
        raise ValueError("frozen 461 geometry reference hash changed")
    audit = json.loads(audit159_path.read_text())
    if audit.get("candidate_sha256") != candidate_sha:
        raise ValueError("159 audit does not bind to the 490-matched candidate")
    if audit.get("complete") is not True:
        raise ValueError("505 requires a complete 159 audit for the exact candidate")
    if int(audit.get("bank_count", -1)) != 185 or int(audit.get("expected_bank_count", -1)) != 185:
        raise ValueError("505 requires complete 185-bank accounting")
    crossing_faces = set()
    banks = audit.get("banks")
    if not isinstance(banks, list) or len(banks) != 185:
        raise ValueError("505 audit bank inventory is malformed")
    for bank in banks:
        result = bank.get("result")
        if result is None:
            continue
        pairs = result.get("triangle_pairs")
        if not isinstance(pairs, list):
            raise ValueError("505 audit pair rows are malformed")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("505 audit pair identity is malformed")
            crossing_faces.add(int(pair[0]))
    if crossing_faces.intersection(BLOCKER_FACE_ROWS):
        raise ValueError("an area blocker face is a current 159 crossing face")

    with np.load(candidate_path, allow_pickle=False) as current, np.load(reference_path, allow_pickle=False) as reference:
        if set(current.files) != set(reference.files):
            raise ValueError("current candidate NPZ keys differ from the frozen geometry reference")
        for key in sorted(k for k in reference.files if k != "points_world_m"):
            a = np.asarray(current[key]); b = np.asarray(reference[key])
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError(f"frozen non-coordinate geometry/provenance identity changed: {key}")
        points = np.asarray(current["points_world_m"])
        reference_points = np.asarray(reference["points_world_m"])
        if points.dtype != reference_points.dtype or points.shape != reference_points.shape:
            raise ValueError("current point array dtype/shape differs from frozen topology")
        if points.shape != (20889, 3) or not np.isfinite(points).all():
            raise ValueError("current candidate points must be finite 20,889 by 3 coordinates")
        faces = np.asarray(current["source_O_faces"])
        area_faces = np.asarray(current["area_reference_037_source_O_faces"])
        if faces.shape != (41733, 3) or array_sha256(faces) != EXPECTED_FACE_ARRAY_SHA256:
            raise ValueError("candidate topology differs from the frozen blocker topology")
        if faces.dtype != area_faces.dtype or faces.shape != area_faces.shape or faces.tobytes() != area_faces.tobytes():
            raise ValueError("fixed-037 face ancestry is not byte-identical to current topology")
        details = []
        for row in BLOCKER_FACE_ROWS:
            actual = tuple(map(int, faces[row]))
            if actual != EXPECTED_FACE_VERTICES[row]:
                raise ValueError(f"blocker face row {row} connectivity changed")
            details.append({"face_row": row, "vertices": list(actual)})
        ancestry_sha = _ancestry_digest(current)
        if ancestry_sha != EXPECTED_BLOCKER_ANCESTRY_SHA256:
            raise ValueError("blocker source ancestry witness changed")
        identity_sha = _npz_noncoordinate_identity_digest(current)
        noncoordinate_count = len(current.files) - 1
    return {
        "schema": "numi.human.area_blocker_zero_bounds_505.v1",
        "classification": "construction-only LP variable bounds; not runtime anchors or physical forces",
        "candidate_path": str(candidate_path), "candidate_sha256": candidate_sha,
        "audit159_path": str(audit159_path), "audit159_sha256": audit_sha,
        "matched_config_path": str(config_path), "matched_config_sha256": matched_config_sha256,
        "matched_candidate_path": candidate_record["path"],
        "matched_candidate_sha256": candidate_record["sha256"],
        "matched_audit159_path": audit_record["path"],
        "matched_audit159_sha256": audit_record["sha256"],
        "reference_candidate_path": str(reference_path),
        "reference_candidate_sha256": REFERENCE_CANDIDATE_SHA256,
        "audit159_complete": True, "audit159_bank_count": 185,
        "candidate_face_array_sha256": EXPECTED_FACE_ARRAY_SHA256,
        "noncoordinate_identity_sha256": identity_sha,
        "noncoordinate_identity_keys": int(noncoordinate_count),
        "blocker_face_rows": details,
        "blocker_source_ancestry_sha256": ancestry_sha,
        "blocker_rows_are_current_159_crossings": False,
        "native_admission": False, "accepted_physical_steps": 0,
    }

def apply_blocker_zero_bounds(lower, upper, support, faces, *, validation, anchors=()):
    """Return copied bounds with only the pinned blocker vertices set to zero."""
    if (not isinstance(validation, dict)
            or validation.get("schema") != "numi.human.area_blocker_zero_bounds_505.v1"
            or validation.get("candidate_sha256") != validation.get("matched_candidate_sha256")
            or validation.get("audit159_sha256") != validation.get("matched_audit159_sha256")
            or not validation.get("matched_config_sha256")
            or validation.get("candidate_face_array_sha256") != EXPECTED_FACE_ARRAY_SHA256
            or validation.get("blocker_source_ancestry_sha256") != EXPECTED_BLOCKER_ANCESTRY_SHA256
            or validation.get("blocker_rows_are_current_159_crossings") is not False):
        raise ValueError("validated pinned blocker receipt required")
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    if lo.ndim != 1 or hi.shape != lo.shape or len(lo) != 3 * len(support):
        raise ValueError("bound vectors must use node-major xyz support layout")
    if faces.shape != (41733, 3) or array_sha256(faces) != EXPECTED_FACE_ARRAY_SHA256:
        raise ValueError("runtime face array differs from the blocker-pinned topology")
    if len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= 20889):
        raise ValueError("support rows must be unique and in the frozen candidate node range")
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()) or np.any(lo > hi):
        raise ValueError("input bounds must be finite and ordered")
    if np.any(lo > 0.0) or np.any(hi < 0.0):
        raise ValueError("original bounds do not admit zero displacement")
    row_to_local = {int(row): i for i, row in enumerate(support)}
    blocker_vertices = sorted({int(v) for row in BLOCKER_FACE_ROWS for v in faces[row]})
    if any(v not in row_to_local for v in blocker_vertices):
        raise ValueError("a blocker vertex is outside LP support; refusing an incomplete restriction")
    out_lo = lo.copy(); out_hi = hi.copy()
    changed = []
    for node in blocker_vertices:
        local = row_to_local[node]
        for axis in range(3):
            k = 3 * local + axis
            if out_lo[k] != 0.0 or out_hi[k] != 0.0:
                changed.append(k)
            out_lo[k] = 0.0; out_hi[k] = 0.0
    unchanged = np.ones(len(lo), dtype=bool)
    unchanged[np.asarray(changed, dtype=np.int64)] = False
    if not (np.array_equal(out_lo[unchanged], lo[unchanged]) and np.array_equal(out_hi[unchanged], hi[unchanged])):
        raise RuntimeError("505 changed bounds outside documented blocker vertices")
    anchor_set = set(map(int, np.asarray(anchors, dtype=np.int64)))
    receipt = dict(validation)
    receipt.update({
        "support_count": int(len(support)),
        "restricted_vertex_rows": blocker_vertices,
        "restricted_scalar_bound_indices": changed,
        "restricted_vertices_already_anchors": sorted(anchor_set.intersection(blocker_vertices)),
        "input_lower_sha256": array_sha256(lo), "input_upper_sha256": array_sha256(hi),
        "output_lower_sha256": array_sha256(out_lo), "output_upper_sha256": array_sha256(out_hi),
        "nonblocker_bounds_bitwise_unchanged": True,
        "runtime_anchor_or_force_semantics": False,
        "native_admission": False, "accepted_physical_steps": 0,
    })
    return out_lo, out_hi, receipt


def install_owner_bounds(owner, *, validation):
    """Patch only the 408 source-safe LP bounds hook and retain its result receipt."""
    original = owner.source_safe_trust_bounds
    with np.load(validation["candidate_path"], allow_pickle=False) as z:
        pinned_points_mm = np.asarray(z["points_world_m"], dtype=np.float64) * 1000.0
        pinned_faces = np.asarray(z["source_O_faces"], dtype=np.int64)
    state = {"calls": 0, "receipts": []}
    def replacement(points_mm, source_mm, support, anchors, trust_mm, source_cap_mm):
        actual_points_mm = np.asarray(points_mm, dtype=np.float64)
        if actual_points_mm.shape != pinned_points_mm.shape or not np.array_equal(actual_points_mm, pinned_points_mm):
            raise ValueError("runtime LP baseline points do not match the blocker-pinned candidate")
        lower, upper, report = original(points_mm, source_mm, support, anchors, trust_mm, source_cap_mm)
        state["calls"] += 1
        lower, upper, receipt = apply_blocker_zero_bounds(
            lower, upper, support, pinned_faces,
            validation=validation, anchors=anchors,
        )
        report = dict(report)
        report["area_blocker_zero_bounds_505"] = receipt
        state["receipts"].append(receipt)
        return lower, upper, report
    owner.source_safe_trust_bounds = replacement
    return state
