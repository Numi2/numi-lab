"""Conservative composition of independently audited coordinate-field components.

No audit status or solver/LP certificate is transferred to the composed field.
A fresh full construction check is required for every emitted mixed candidate.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np


class ValidationError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def connected_changed_components(faces, changed_rows):
    faces = np.asarray(faces)
    changed = np.asarray(changed_rows, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValidationError("faces must be an integer Fx3 array")
    if changed.ndim != 1 or len(np.unique(changed)) != len(changed):
        raise ValidationError("changed rows must be a unique 1D list")
    if np.any(changed < 0) or (faces.size and int(changed.max(initial=-1)) >= int(faces.max()) + 1):
        raise ValidationError("changed node row is outside face domain")
    changed_set = set(map(int, changed))
    if len(changed_set) != len(changed):
        raise ValidationError("duplicate changed node")
    parent = {n: n for n in changed_set}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    touched_nodes = []
    for fi, face in enumerate(faces):
        nodes = sorted(changed_set.intersection(map(int, face)))
        if len(nodes) > 1:
            for n in nodes[1:]:
                union(nodes[0], n)
        touched_nodes.append(nodes)

    groups = defaultdict(list)
    for n in sorted(changed_set):
        groups[find(n)].append(n)
    components = [np.asarray(v, dtype=np.int64) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]
    node_to_component = {}
    for ci, nodes in enumerate(components):
        for n in nodes:
            node_to_component[int(n)] = ci

    face_rows = [[] for _ in components]
    face_owner = {}
    for fi, nodes in enumerate(touched_nodes):
        if not nodes:
            continue
        owners = {node_to_component[n] for n in nodes}
        if len(owners) != 1:
            raise ValidationError("a changed face is split across components")
        ci = next(iter(owners))
        face_rows[ci].append(fi)
        face_owner[fi] = ci
    return components, [np.asarray(v, dtype=np.int64) for v in face_rows], face_owner


def validate_159_report(report, candidate_sha256, expected_banks=185):
    if not isinstance(report, dict):
        raise ValidationError("159 report must be an object")
    if report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT":
        raise ValidationError("159 report is not complete")
    if report.get("complete") is not True:
        raise ValidationError("159 report complete flag is false")
    if report.get("candidate_sha256") != candidate_sha256:
        raise ValidationError("159 report candidate hash mismatch")
    if report.get("bank_count") != expected_banks or report.get("expected_bank_count") != expected_banks:
        raise ValidationError("159 bank coverage mismatch")
    banks = {}
    for b in report.get("banks", []):
        key = (int(b["body"]), int(b["stable_bone_id"]))
        if key in banks:
            raise ValidationError("duplicate typed bank")
        result = b.get("result")
        if result is None:
            if int(b.get("bank_aabb_active", -1)) != 0 and int(b.get("skin_aabb_active", -1)) != 0:
                raise ValidationError("null bank result despite active geometry")
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            count = result.get("count")
            raw_pairs = result.get("triangle_pairs")
            if not isinstance(count, int) or count < 0 or raw_pairs is None:
                raise ValidationError("missing exact triangle-pair list")
            pairs = np.asarray(raw_pairs, dtype=np.int64)
            if pairs.size != count * 2:
                raise ValidationError("159 count/list mismatch")
            pairs = pairs.reshape((-1, 2))
            if np.any(pairs < 0):
                raise ValidationError("negative triangle row in 159 pair list")
        per_face = defaultdict(int)
        for face, _ in pairs:
            per_face[int(face)] += 1
        banks[key] = dict(total=int(len(pairs)), face_counts=dict(per_face))
    if len(banks) != expected_banks:
        raise ValidationError("159 typed bank list length mismatch")
    return banks


def select_safe_components(faces, changed_rows, baseline_banks, target_banks):
    components, component_faces, face_owner = connected_changed_components(faces, changed_rows)
    bank_keys = set(baseline_banks)
    if bank_keys != set(target_banks):
        raise ValidationError("baseline and target typed-bank sets differ")
    if not bank_keys:
        raise ValidationError("empty typed-bank reports")
    details = []
    for ci, (nodes, rows) in enumerate(zip(components, component_faces)):
        fset = set(map(int, rows))
        bank_changes = {}
        no_bank_increase = True
        before_total = after_total = 0
        for bank in sorted(bank_keys):
            b0 = baseline_banks[bank]["face_counts"]
            b1 = target_banks[bank]["face_counts"]
            n0 = sum(v for f, v in b0.items() if f in fset)
            n1 = sum(v for f, v in b1.items() if f in fset)
            before_total += n0
            after_total += n1
            if n1 > n0:
                no_bank_increase = False
            if n1 != n0:
                bank_changes[f"{bank[0]}/{bank[1]}"] = {"baseline": n0, "target": n1, "delta": n1 - n0}
        strict_drop = after_total < before_total
        selected = bool(no_bank_increase and strict_drop)
        details.append({
            "component_id": ci,
            "first_node": int(nodes[0]) if len(nodes) else None,
            "node_rows": nodes.tolist(),
            "face_rows": rows.tolist(),
            "baseline_affected_pairs": int(before_total),
            "target_affected_pairs": int(after_total),
            "affected_pair_delta": int(after_total - before_total),
            "bank_changes": bank_changes,
            "no_bankwise_increase": bool(no_bank_increase),
            "strict_total_decrease": bool(strict_drop),
            "selected": selected,
        })
    selected_ids = [d["component_id"] for d in details if d["selected"]]
    predicted = {f"{b[0]}/{b[1]}": int(baseline_banks[b]["total"]) for b in sorted(bank_keys)}
    for d in details:
        if not d["selected"]:
            continue
        for key, change in d["bank_changes"].items():
            predicted[key] += int(change["delta"])
    base_total = sum(x["total"] for x in baseline_banks.values())
    predicted_total = sum(predicted.values())
    if any(v < 0 for v in predicted.values()):
        raise ValidationError("predicted bank count underflow")
    return {
        "components": details,
        "selected_component_ids": selected_ids,
        "face_owner": face_owner,
        "baseline_bank_counts": {f"{b[0]}/{b[1]}": int(baseline_banks[b]["total"]) for b in sorted(bank_keys)},
        "predicted_bank_counts": predicted,
        "baseline_total": int(base_total),
        "predicted_total": int(predicted_total),
        "predicted_total_delta": int(predicted_total - base_total),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_complete_evidence(candidate_path, full_report_path, summary_path, surface_path, arithmetic_path):
    candidate_path = Path(candidate_path)
    candidate_sha = sha256_file(candidate_path)
    full = load_json(full_report_path)
    summary = load_json(summary_path)
    surface = load_json(surface_path)
    arithmetic = load_json(arithmetic_path)

    if full.get("status") != "COMPLETE_EXACT_CONSTRUCTION_CHECKS":
        raise ValidationError("545 full report not complete")
    if full.get("candidate_sha256") != candidate_sha or Path(full.get("candidate_path", "")).resolve() != candidate_path.resolve():
        raise ValidationError("545 full report candidate binding mismatch")
    if full.get("complete_banks") != 185 or full.get("contained_pairs") != 0 or full.get("self_pairs") != 0:
        raise ValidationError("545 complete/inside/self gates not clear")
    if full.get("arithmetic_pass") is not True or full.get("accepted_physical_steps") != 0 or full.get("native_admission") is not False:
        raise ValidationError("545 arithmetic/admission fields invalid")
    if full.get("anchor_rows") != 237:
        raise ValidationError("545 anchor count mismatch")
    area = full.get("full_f32_area_gate") or {}
    if area.get("pass") is not True or area.get("bad_current_oriented_face_count") != 0 or area.get("bad_fixed_037_reference_face_count") != 0:
        raise ValidationError("545 area/reference gate failed")
    if summary.get("candidate_sha256") != candidate_sha or summary.get("complete_surface_audit") is not True:
        raise ValidationError("130 summary candidate/completeness mismatch")
    if summary.get("inside_nodes") != 0 or summary.get("inside_node_bank_pairs") != 0 or summary.get("exact_PLC_surface_pairs") != 0:
        raise ValidationError("130 inside/self counts are nonzero")
    if surface.get("candidate_sha256") != candidate_sha:
        raise ValidationError("159 candidate mismatch")
    banks = validate_159_report(surface, candidate_sha)
    if sum(v["total"] for v in banks.values()) != int(full.get("surface_pairs", -1)):
        raise ValidationError("159 per-bank pairs do not sum to 545 report")
    if arithmetic.get("status") != "ARITHMETIC_PASS" or arithmetic.get("arithmetic_pass") is not True or arithmetic.get("candidate_sha256") != candidate_sha:
        raise ValidationError("separate arithmetic report invalid")
    if arithmetic.get("accepted_physical_steps") != 0 or arithmetic.get("native_admission") is not False:
        raise ValidationError("arithmetic report admission fields invalid")
    area2 = arithmetic.get("full_f32_area_gate") or {}
    if area2.get("pass") is not True or area2.get("bad_current_oriented_face_count") != 0 or area2.get("bad_fixed_037_reference_face_count") != 0:
        raise ValidationError("arithmetic area/reference gate failed")

    # Verify retained source pins in the completed full-check report.
    pins = full.get("source_pins")
    if not isinstance(pins, dict) or not pins:
        raise ValidationError("545 report has no source pins")
    for path, expected in pins.items():
        p = Path(path)
        if not p.is_file() or sha256_file(p) != expected:
            raise ValidationError(f"545 source pin changed or missing: {path}")
    self_pairs_path = Path(summary_path).with_name("surface-self-pairs.npy")
    self_pairs = np.load(self_pairs_path, allow_pickle=False)
    if self_pairs.size != 0:
        raise ValidationError("130 self-pair array is nonempty")
    return {
        "candidate_sha256": candidate_sha,
        "full_report_sha256": sha256_file(full_report_path),
        "summary_sha256": sha256_file(summary_path),
        "surface_sha256": sha256_file(surface_path),
        "arithmetic_sha256": sha256_file(arithmetic_path),
        "full_report": full,
        "banks": banks,
    }
