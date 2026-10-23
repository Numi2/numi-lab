#!/usr/bin/env python3
"""Select/validate inferred attachments from final compacted source-owned arrays.

Evidence-only: never edits anatomy, weights, native headers, or payloads. It
reuses the native signed-prism and Float32 gates and blocks selection on failure.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E = ROOT / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
DEFAULTS = {
    "coarse": E / "replacement-mechanical-069-ordered-compaction-001/coarse-surface.npz",
    "layers": E / "replacement-mechanical-069-ordered-compaction-001/candidate-layers.npz",
    "source_map": E / "replacement-mechanical-069-ordered-compaction-001/coarse-to-source.npz",
    "row_map": E / "replacement-mechanical-069-ordered-compaction-001/compaction-row-map.npz",
    "source_geometry": E / "finger-profile-068-whole-hand-minimax/replacement-geometry.npz",
    "skin_payload": E / "finger-profile-068-whole-hand-minimax/replacement-bilateral-hands038.nhskin",
    "left_patch": E / "finger-profile-068-whole-hand-minimax/left-patch.npz",
    "right_patch": E / "finger-profile-068-whole-hand-minimax/right-patch.npz",
    "old_sites": E / "replacement-mechanical-061-normal-cone-001/attachment-audit.stdout.json",
    "source_header": ROOT / "apps/NumiHumanWholeBodySkinReduction.hpp",
    "native_source": ROOT / "apps/numilab_human_myosim_visual_probe.mm",
    "native_test": ROOT / "tests/human_wholebody_skin_layers.cpp",
    "cmake": ROOT / "CMakeLists.txt",
    "bone_manifest": Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json"),
    "prism_helper": E / "replacement-mechanical-055/native_prism_checks_vectorized.py",
}
MIN_WEIGHT = 0.85
MIN_ALTITUDE_M = 5.0e-5
MIN_SPACING_M = 0.025
MAX_PER_DONOR = 3
# These are source-field-qualified owner groups with new candidate rows. This
# does not restore their old source point or synthesize/reweight an owner field.
REPLACEMENT_OWNERS = ((74, "right"), (122, "left"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_with_prefix(path: Path) -> Any:
    return json.loads(path.read_text().split("\n", 1)[1])


def constant_from_header(path: Path, name: str) -> float:
    match = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+)\s*;", path.read_text())
    if match is None:
        raise ValueError(f"cannot find native mass owner {name} in {path}")
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"native mass owner {name} is not positive finite")
    return value


def signed_six_volume(points: np.ndarray) -> float:
    return float(np.dot(points[1] - points[0], np.cross(points[2] - points[0], points[3] - points[0])))


def native_layer_gates(outer, inner, deep, faces, skin_thickness, fat_thickness, tets, tet_layer, helper):
    """Apply native signed-prism, emitted-Float32 and tetra-connectivity checks."""
    sys.path.insert(0, str(helper.parent))
    import native_prism_checks_vectorized as npc

    bad_skin, skin_prisms = npc.signed_prisms(outer, inner, skin_thickness, faces)
    bad_fat, fat_prisms = npc.signed_prisms(inner, deep, fat_thickness, faces)
    skin_det, skin_vol = npc.float32_tet_determinants(outer, inner, faces)
    fat_det, fat_vol = npc.float32_tet_determinants(inner, deep, faces)
    n = len(outer)
    expected = []
    for layer in (0, 1):
        outer_offset = layer * n
        inner_offset = (layer + 1) * n
        rows = []
        for face in np.asarray(faces, dtype=np.int64):
            a, b, c = sorted(map(int, face))
            rows.extend(((outer_offset+a, outer_offset+b, outer_offset+c, inner_offset+c),
                         (outer_offset+a, outer_offset+b, inner_offset+b, inner_offset+c),
                         (outer_offset+a, inner_offset+a, inner_offset+b, inner_offset+c)))
        expected.append(collections.Counter(tuple(sorted(row)) for row in rows))
    actual_ok = (set(map(int, np.unique(np.asarray(tet_layer)))) == {0, 1})
    counts = []
    volumes = []
    for layer in (0, 1):
        mask = np.asarray(tet_layer) == layer
        got = collections.Counter(tuple(sorted(map(int, row))) for row in np.asarray(tets)[mask])
        actual_ok &= got == expected[layer]
        counts.append(int(np.count_nonzero(mask)))
        sv = skin_vol if layer == 0 else fat_vol
        volumes.append(float(np.sum(np.abs(np.asarray(sv, dtype=np.float64)))))
    actual_ok &= len(tet_layer) == counts[0] + counts[1]
    return {
        "signed_prism_skin_failures": int(np.count_nonzero(bad_skin)),
        "signed_prism_fat_failures": int(np.count_nonzero(bad_fat)),
        "minimum_skin_prism_abs_tet_volume_m3": float(np.min(np.abs(skin_prisms[:, :3]))),
        "minimum_fat_prism_abs_tet_volume_m3": float(np.min(np.abs(fat_prisms[:, :3]))),
        "skin_volume_from_native_tet_template_m3": volumes[0],
        "fat_volume_from_native_tet_template_m3": volumes[1],
        "skin_float32_min_abs_normalized_det": float(np.min(np.abs(skin_det))),
        "fat_float32_min_abs_normalized_det": float(np.min(np.abs(fat_det))),
        "skin_tet_count": counts[0],
        "fat_tet_count": counts[1],
        "expected_skin_tet_count": int(sum(expected[0].values())),
        "expected_fat_tet_count": int(sum(expected[1].values())),
        "native_tet_connectivity_exact": bool(actual_ok),
        "signed_prisms_pass": bool(not np.any(bad_skin) and not np.any(bad_fat)),
        "float32_reference_conditioning_pass": bool(
            np.all(np.isfinite(skin_det)) and np.all(np.isfinite(fat_det))
            and np.all(np.abs(skin_det) >= 1.0e-5) and np.all(np.abs(fat_det) >= 1.0e-5)),
    }


def calculate_native_lumped_mass_and_altitude(outer, inner, deep, tets, tet_layer, skin_debit_kg, fat_debit_kg):
    """Reproduce the existing per-tet rho*V/4 mass and deep-node altitude rules."""
    n = len(outer)
    xyz = np.concatenate((outer, inner, deep), axis=0).astype(np.float64)
    tet = np.asarray(tets, dtype=np.int64)
    layer = np.asarray(tet_layer, dtype=np.int8)
    if tet.ndim != 2 or tet.shape[1] != 4 or not np.isin(layer, [0, 1]).all():
        raise ValueError("native layer tets must be four-node skin/fat elements")
    if tet.min() < 0 or tet.max() >= len(xyz):
        raise ValueError("native layer tet node index is outside the three-layer object")
    volumes = np.asarray([abs(signed_six_volume(xyz[ids])) / 6.0 for ids in tet], dtype=np.float64)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("native layer tets contain nonpositive or nonfinite volume")
    layer_volume = [float(np.sum(volumes[layer == k])) for k in (0, 1)]
    if any(not math.isfinite(v) or v <= 0 for v in layer_volume):
        raise ValueError("skin/fat native tet layer has nonpositive total volume")
    rho = (skin_debit_kg / layer_volume[0], fat_debit_kg / layer_volume[1])
    node_mass = np.zeros(3*n, dtype=np.float64)
    for i, ids in enumerate(tet):
        node_mass[ids] += rho[int(layer[i])] * volumes[i] / 4.0
    altitude = np.full(n, np.inf, dtype=np.float64)
    for i, ids in enumerate(tet):
        if layer[i] != 1:
            continue
        points = xyz[ids]
        six = 6.0 * volumes[i]
        for opposite, node in enumerate(ids):
            if node < 2*n:
                continue
            others = [points[j] for j in range(4) if j != opposite]
            area2 = float(np.linalg.norm(np.cross(others[1]-others[0], others[2]-others[0])))
            row = int(node - 2*n)
            if not math.isfinite(area2) or area2 <= 0:
                altitude[row] = -math.inf
            else:
                altitude[row] = min(altitude[row], six / area2)
    return node_mass, altitude, {
        "skin_layer_volume_m3": layer_volume[0],
        "fat_layer_volume_m3": layer_volume[1],
        "skin_density_kg_m3_from_existing_source_debit": rho[0],
        "fat_density_kg_m3_from_existing_source_debit": rho[1],
        "skin_mass_debit_kg": skin_debit_kg,
        "fat_mass_debit_kg": fat_debit_kg,
    }


def validate_site_contract(sites, *, active, source_vertex_by_row, body_by_source_vertex,
                           weight_by_source_vertex_body, incident_altitude_m_by_row,
                           deep_node_mass_kg_by_row, world_by_row, budget,
                           min_weight=MIN_WEIGHT, min_altitude_m=MIN_ALTITUDE_M,
                           min_spacing_m=MIN_SPACING_M, max_per_donor=MAX_PER_DONOR):
    """Native frozen-site invariants; deliberately no exact total-count/body-set gate."""
    errors = []
    seen_rows, seen_sources = set(), set()
    per_body = collections.defaultdict(list)
    for ordinal, site in enumerate(sites):
        row, source_vertex, body = (int(site[k]) for k in ("reduced_vertex", "source_vertex", "donor_body"))
        prefix = f"site[{ordinal}] row={row} source={source_vertex} body={body}"
        if row < 0 or row >= len(active) or not bool(active[row]):
            errors.append(f"{prefix}:inactive_or_out_of_range")
            continue
        if row in seen_rows:
            errors.append(f"{prefix}:duplicate_reduced_vertex")
        seen_rows.add(row)
        if source_vertex != int(source_vertex_by_row[row]):
            errors.append(f"{prefix}:source_vertex_identity_mismatch")
        if not 0 <= source_vertex < len(body_by_source_vertex):
            errors.append(f"{prefix}:source_vertex_out_of_range")
            continue
        if source_vertex in seen_sources:
            errors.append(f"{prefix}:duplicate_source_vertex")
        seen_sources.add(source_vertex)
        weight = float(weight_by_source_vertex_body[source_vertex, body]) if 0 <= body < weight_by_source_vertex_body.shape[1] else -math.inf
        if not math.isfinite(weight) or weight < min_weight:
            errors.append(f"{prefix}:source_weight_below_minimum:{weight}")
        if int(body_by_source_vertex[source_vertex]) != body:
            errors.append(f"{prefix}:donor_body_not_dominant_source_owner")
        altitude = float(incident_altitude_m_by_row[row])
        if not math.isfinite(altitude) or altitude < min_altitude_m:
            errors.append(f"{prefix}:incident_altitude_below_minimum:{altitude}")
        mass = float(deep_node_mass_kg_by_row[row])
        if not math.isfinite(mass) or mass <= 0:
            errors.append(f"{prefix}:deep_node_mass_not_positive:{mass}")
        point = np.asarray(world_by_row[row], dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            errors.append(f"{prefix}:world_position_invalid")
        for other_row, other_point in per_body[body]:
            distance = float(np.linalg.norm(point - other_point))
            if not math.isfinite(distance) or distance < min_spacing_m:
                errors.append(f"{prefix}:same_donor_spacing_below_minimum:{other_row}:{distance}")
        per_body[body].append((row, point))
        if len(per_body[body]) > max_per_donor:
            errors.append(f"{prefix}:donor_site_count_exceeds_{max_per_donor}")
    if len(sites) > budget:
        errors.append(f"site_count_exceeds_native_anchor_budget:{len(sites)}>{budget}")
    return errors


def select_replacement_candidates(owner_body, side, *, active, source_vertex_by_row,
                                  source_face_by_row, source_faces, source_weights,
                                  binding_body_ids, side_patch_slots, source_residual_m,
                                  incident_altitude_m_by_row, deep_node_mass_kg_by_row,
                                  world_by_row, used_rows, used_sources, per_body_points):
    slots = np.flatnonzero(binding_body_ids == owner_body)
    if len(slots) != 1:
        return [], {"binding_slot_count": int(len(slots)), "eligible_before_layer_gates": 0}
    slot = int(slots[0])
    rows = np.flatnonzero(active)
    source_rows = source_vertex_by_row[rows]
    parent_triangles = source_faces[source_face_by_row[rows]]
    source_valid = (source_rows >= 0) & (source_rows < len(source_weights))
    safe_source_rows = np.clip(source_rows, 0, len(source_weights)-1)
    side_ok = np.asarray([all(int(v) in side_patch_slots for v in tri) for tri in parent_triangles])
    identity_ok = source_valid & np.asarray([int(source_rows[k]) in set(map(int, tri)) for k, tri in enumerate(parent_triangles)])
    weights = source_weights[safe_source_rows, slot]
    owner_ok = source_valid & (np.argmax(source_weights[safe_source_rows], axis=1) == slot)
    eligible = source_valid & side_ok & identity_ok & (weights >= MIN_WEIGHT) & owner_ok
    candidates = rows[eligible]
    stats = {
        "binding_slot_count": 1,
        "active_rows": int(len(rows)),
        "same_side_source_faces": int(np.count_nonzero(side_ok)),
        "exact_representative_source_identity_rows": int(np.count_nonzero(identity_ok)),
        "rows_with_weight_and_dominant_owner": int(np.count_nonzero((weights >= MIN_WEIGHT) & owner_ok)),
        "eligible_before_layer_gates": int(len(candidates)),
    }
    ranked = sorted((int(row) for row in candidates),
                    key=lambda row: (-float(source_weights[source_vertex_by_row[row], slot]),
                                     float(source_residual_m[row]), row))
    found = []
    for row in ranked:
        source = int(source_vertex_by_row[row])
        if row in used_rows or source in used_sources:
            continue
        if float(incident_altitude_m_by_row[row]) < MIN_ALTITUDE_M:
            continue
        if float(deep_node_mass_kg_by_row[row]) <= 0:
            continue
        candidate_world = np.asarray(world_by_row[row], dtype=np.float64)
        if any(np.linalg.norm(candidate_world - p) < MIN_SPACING_M for p in per_body_points.get(owner_body, [])):
            continue
        found.append({
            "reduced_vertex": row,
            "source_vertex": source,
            "donor_body": int(owner_body),
            "side": side,
            "source_weight": float(source_weights[source, slot]),
            "source_residual_m": float(source_residual_m[row]),
            "incident_altitude_m": float(incident_altitude_m_by_row[row]),
            "deep_node_mass_kg": float(deep_node_mass_kg_by_row[row]),
            "provenance": "inferred current registered skin donor site; not a measured enthesis",
        })
    return found, stats


def load_and_run(args):
    paths = {key: Path(getattr(args, key)) for key in DEFAULTS}
    coarse_z = np.load(paths["coarse"], allow_pickle=False)
    layer_z = np.load(paths["layers"], allow_pickle=False)
    source_map_z = np.load(paths["source_map"], allow_pickle=False)
    row_map_z = np.load(paths["row_map"], allow_pickle=False)
    geometry_z = np.load(paths["source_geometry"], allow_pickle=False)
    payload_bytes = paths["skin_payload"].read_bytes()
    sys.path.insert(0, "/Users/n/numi-human-anatomy-finish-1378/src")
    from numilab_human.skin_source_payload_preflight import decode_payload
    payload = decode_payload(payload_bytes)

    coarse_world = np.asarray(coarse_z["world"], dtype=np.float64)
    faces = np.asarray(coarse_z["faces"], dtype=np.int64)
    source_vertex = np.asarray(coarse_z["source_vertex"], dtype=np.int64)
    source_faces = np.asarray(geometry_z["faces"], dtype=np.int64)
    source_weights = np.asarray(payload["full_weights"], dtype=np.float64)
    binding_body_ids = np.asarray(payload["bindings_u"][:, 0], dtype=np.int64)
    if source_weights.shape != np.asarray(geometry_z["weights"]).shape or not np.array_equal(
            source_weights.astype(np.float32), np.asarray(geometry_z["weights"], dtype=np.float32)):
        raise ValueError("source weight payload differs from registered geometry ancestry")
    if not np.array_equal(source_faces, np.asarray(payload["indices"], dtype=np.int64).reshape(-1, 3)):
        raise ValueError("source face ordering differs from exact NHSKIN payload")
    if len(set(map(int, binding_body_ids))) != len(binding_body_ids):
        raise ValueError("skin payload has duplicate body binding IDs")

    source_face = np.asarray(source_map_z["source_face"], dtype=np.int64)
    source_bary = np.asarray(source_map_z["source_barycentric"], dtype=np.float64)
    source_residual = np.asarray(source_map_z["source_residual_m"], dtype=np.float64)
    old_to_new = np.asarray(row_map_z["old_to_new"], dtype=np.int64)
    new_to_old = np.asarray(row_map_z["new_to_old"], dtype=np.int64)
    n = len(coarse_world)
    if not (len(source_vertex) == len(source_face) == len(source_bary) == len(source_residual) == len(new_to_old) == n):
        raise ValueError("compacted source/layer arrays have mismatched vertex counts")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= n:
        raise ValueError("compacted active surface faces are out of range")
    active = np.zeros(n, dtype=bool)
    active[np.unique(faces)] = True
    if not bool(active.all()):
        raise ValueError("final compacted surface has unreferenced rows")
    if not np.array_equal(new_to_old, np.flatnonzero(old_to_new >= 0)):
        raise ValueError("old/new compaction maps are not exact inverses")
    if np.any(source_vertex >= len(source_weights)):
        raise ValueError("compacted source vertex index is out of range")
    if np.any(source_face < 0) or np.any(source_face >= len(source_faces)):
        raise ValueError("compacted source face index is out of range")
    if source_bary.shape != (n, 3) or not np.isfinite(source_bary).all() or np.any(source_bary < -1e-8) or not np.allclose(source_bary.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("compacted source barycentric mapping is invalid")
    if not np.isfinite(source_residual).all() or np.any(source_residual < 0):
        raise ValueError("compacted source residuals must be finite and nonnegative")
    parent_triangles = source_faces[source_face]
    identity_in_parent = np.asarray([source_vertex[i] >= 0 and int(source_vertex[i]) in set(map(int, parent_triangles[i])) for i in range(n)])

    outer = np.asarray(layer_z["outer_world"], dtype=np.float64)
    inner = np.asarray(layer_z["inner_world"], dtype=np.float64)
    deep = np.asarray(layer_z["deep_world"], dtype=np.float64)
    skin_thickness = np.asarray(layer_z["skin_thickness_m"], dtype=np.float64)
    fat_thickness = np.asarray(layer_z["fat_thickness_m"], dtype=np.float64)
    tets = np.asarray(layer_z["tetrahedra"], dtype=np.int64)
    tet_layer = np.asarray(layer_z["tetra_layer"], dtype=np.int8)
    if not (outer.shape == inner.shape == deep.shape == (n, 3)
            and len(skin_thickness) == len(fat_thickness) == n
            and len(tets) == len(tet_layer)):
        raise ValueError("final layer/tet arrays do not match compacted row count")
    if not np.array_equal(outer, coarse_world):
        raise ValueError("layer outer boundary is not the final compacted skin surface")
    if not np.isfinite(outer).all() or not np.isfinite(inner).all() or not np.isfinite(deep).all():
        raise ValueError("compacted layer coordinates must be finite")
    layer_gates = native_layer_gates(outer, inner, deep, faces, skin_thickness, fat_thickness,
                                     tets, tet_layer, paths["prism_helper"])
    skin_debit = constant_from_header(paths["source_header"], "kWholeBodySkinMassDebitKg")
    fat_debit = constant_from_header(paths["source_header"], "kWholeBodyFatMassDebitKg")
    node_mass, altitude, mass_audit = calculate_native_lumped_mass_and_altitude(
        outer, inner, deep, tets, tet_layer, skin_debit, fat_debit)

    dominant_slots = np.argmax(source_weights, axis=1)
    body_by_source = binding_body_ids[dominant_slots]
    weight_matrix = np.zeros((len(source_weights), int(np.max(binding_body_ids)) + 1), dtype=np.float64)
    weight_matrix[:, binding_body_ids] = source_weights

    old_sites = json_with_prefix(paths["old_sites"])
    retained, retired = [], []
    for old in old_sites:
        old_row = int(old["reduced_vertex"])
        new_row = int(old_to_new[old_row]) if 0 <= old_row < len(old_to_new) else -1
        item = {"reduced_vertex": new_row, "source_vertex": int(old["source_vertex"]),
                "donor_body": int(old["donor_body"]), "old_reduced_vertex": old_row,
                "donor_name": str(old["donor_name"]), "old_provenance": str(old["site_provenance"])}
        if new_row < 0 or not active[new_row]:
            retired.append({**item, "reason": "orphaned_by_exact_compaction"})
            continue
        if int(source_vertex[new_row]) != item["source_vertex"]:
            raise ValueError(f"retained source identity changed at old row {old_row}->{new_row}")
        if not identity_in_parent[new_row]:
            raise ValueError(f"retained representative source vertex is not in its source parent face at row {new_row}")
        if int(body_by_source[item["source_vertex"]]) != item["donor_body"]:
            raise ValueError(f"retained donor body differs from exact NHSKIN dominant owner at old row {old_row}")
        retained.append(item)

    used_rows = {int(x["reduced_vertex"]) for x in retained}
    used_sources = {int(x["source_vertex"]) for x in retained}
    per_body_points = collections.defaultdict(list)
    for site in retained:
        per_body_points[int(site["donor_body"])].append(outer[int(site["reduced_vertex"])])
    candidates, search = [], {}
    for owner, side in REPLACEMENT_OWNERS:
        patch_path = paths["right_patch"] if side == "right" else paths["left_patch"]
        patch_z = np.load(patch_path, allow_pickle=False)
        patch_slots = set(map(int, np.asarray(patch_z["source_slots"], dtype=np.int64)))
        found, stats = select_replacement_candidates(
            owner, side, active=active, source_vertex_by_row=source_vertex,
            source_face_by_row=source_face, source_faces=source_faces,
            source_weights=source_weights, binding_body_ids=binding_body_ids,
            side_patch_slots=patch_slots, source_residual_m=source_residual,
            incident_altitude_m_by_row=altitude, deep_node_mass_kg_by_row=node_mass[2*n:],
            world_by_row=outer, used_rows=used_rows, used_sources=used_sources,
            per_body_points=per_body_points)
        search[str(owner)] = stats
        if found:
            candidates.append(found[0])
            used_rows.add(int(found[0]["reduced_vertex"]))
            used_sources.add(int(found[0]["source_vertex"]))
            per_body_points[int(owner)].append(outer[int(found[0]["reduced_vertex"])])
    provisional = retained + candidates

    # aclNodeCount is nonnegative, so floor(3N/32) is a conservative lower
    # bound on the native budget when exact ACL node count is not provided.
    budget = (3*n + (0 if args.acl_node_count is None else int(args.acl_node_count))) // 32
    errors = validate_site_contract(
        provisional, active=active, source_vertex_by_row=source_vertex,
        body_by_source_vertex=body_by_source, weight_by_source_vertex_body=weight_matrix,
        incident_altitude_m_by_row=altitude, deep_node_mass_kg_by_row=node_mass[2*n:],
        world_by_row=outer, budget=budget)
    layer_pass = (layer_gates["signed_prisms_pass"]
                  and layer_gates["float32_reference_conditioning_pass"]
                  and layer_gates["native_tet_connectivity_exact"])
    status = ("blocked_invalid_or_unqualified_layer_geometry" if not layer_pass
              else "blocked_attachment_contract" if errors
              else "candidate_set_passes_local_native_attachment_contract")
    accepted = provisional if status == "candidate_set_passes_local_native_attachment_contract" else []
    return {
        "schema": "numi.human.compacted-attachment-selection-validation.v1",
        "status": status,
        "scope": "candidate selection/local native attachment-contract validation only; no FEM payload/header/runtime compilation",
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()
                   if path.is_file()},
        "source_binding_count": int(len(binding_body_ids)),
        "compacted_rows": int(n),
        "active_rows": int(np.count_nonzero(active)),
        "old_site_count": len(old_sites),
        "retained_active_old_site_count": len(retained),
        "retired_orphan_old_site_count": len(retired),
        "retired_old_sites": retired,
        "provisional_replacement_candidates": candidates,
        "replacement_search": search,
        "planned_site_count": len(provisional),
        "accepted_sites": accepted,
        "unique_donor_count": len({int(x["donor_body"]) for x in provisional}),
        "native_site_validation_errors": errors,
        "native_thresholds": {
            "minimum_source_weight": MIN_WEIGHT, "minimum_incident_altitude_m": MIN_ALTITUDE_M,
            "minimum_per_donor_spacing_m": MIN_SPACING_M, "maximum_sites_per_donor": MAX_PER_DONOR,
            "runtime_budget_formula": "(aclNodeCount + shell.femNodes.size())/32",
            "shell_node_count": 3*n, "conservative_budget_lower_bound": (3*n)//32,
            "acl_node_count_supplied": args.acl_node_count, "budget_checked": budget,
        },
        "native_layer_gates": layer_gates,
        "native_mass_owner_reproduction": mass_audit,
        "mass_scope": "native rho*V/4 lumping reproduced from final layer tets and retained skin/fat debit constants; diagnostic only if a layer gate fails",
        "ownership": {
            "donor_body": "exact dominant NHSKIN source-weight binding; no field edits",
            "replacement_site_provenance": "inferred current registered skin donor site, not a measured enthesis",
            "replacement_ranking": "highest exact representative source-vertex owner weight, then lowest current source-map residual, then row index; no stale anchor-world ranking",
        },
    }


def contract_receipt(paths):
    native, header, test, cmake = (paths[k].read_text() for k in ("native_source", "source_header", "native_test", "cmake"))
    match = re.search(r"kWholeBodySkinDeepAttachmentSites\s*=\s*\{\{(.*?)\}\};", header, re.S)
    if match is None:
        raise ValueError("cannot parse frozen skin attachment array")
    entries = re.findall(r"\{\s*(\d+)u\s*,\s*(\d+)u\s*,\s*(\d+)u\s*\}", match.group(1))
    donor_ids = sorted({int(row[2]) for row in entries})
    start = native.find("const std::size_t anchorBudget=")
    end = native.find("std::vector<std::pair<float,std::uint32_t>> confidence", start)
    excerpt = native[start:end]
    return {
        "schema": "numi.human.attachment-contract-receipt.v1",
        "source_hashes_sha256": {
            "native_runtime_cpp": sha256(paths["native_source"]),
            "frozen_site_header": sha256(paths["source_header"]),
            "legacy_test_cpp": sha256(paths["native_test"]),
            "cmake": sha256(paths["cmake"]),
        },
        "native_contract": {
            "budget_expression": "(aclNodeCount + shell.femNodes.size())/32u",
            "runtime_count_rule": "appended selected rows equal current frozen array length; no literal-20 minimum",
            "per_site_rules": ["exact source/reduced/donor identity", "source weight >=0.85",
                               "incident altitude >=50um", "at most 3 per donor",
                               "same-donor spacing >=25mm", "matching positive-mass donor-local node",
                               "unique active node index"],
            "required_donor_bodies": None,
            "source_donor_owner": "native registration chooses dominantSkinDonorBody(*skin, sourceVertex); prepared donorMassNodes and anchor rows must carry that same body index",
        "force_owner": "existing FEM external static-anchor rows enter the current anchor buffer",
            "runtime_source_excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        },
        "legacy_test": {
            "assertions": "attachmentSources.size()==20u and attachmentByBody.size()==7u",
            "current_header_site_count": len(entries),
            "current_header_unique_donor_count": len(donor_ids),
            "current_header_donor_ids": donor_ids,
            "wired_in_current_cmake": "human_wholebody_skin_layers" in cmake,
            "status": "unwired stale test count assumptions; not the native runtime contract",
        },
    }


def self_test():
    active = np.ones(5, dtype=bool)
    source_by_row = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    body_by_source = np.zeros(15, dtype=np.int64)
    body_by_source[10:13] = [1, 2, 3]
    weight = np.zeros((15, 6), dtype=np.float64)
    for body, source in enumerate(source_by_row[:3], 1):
        weight[int(source), body] = 0.9
    altitude = np.full(5, 1e-4)
    mass = np.full(5, 1e-6)
    world = np.asarray([[0,0,0],[.03,0,0],[0,.03,0],[0,0,.03],[.03,.03,0]], dtype=np.float64)
    sites = [{"reduced_vertex": i, "source_vertex": int(source_by_row[i]), "donor_body": i+1} for i in range(3)]
    kw = dict(active=active, source_vertex_by_row=source_by_row, body_by_source_vertex=body_by_source,
              weight_by_source_vertex_body=weight, incident_altitude_m_by_row=altitude,
              deep_node_mass_kg_by_row=mass, world_by_row=world, budget=3)
    assert validate_site_contract(sites, **kw) == []
    cases = []
    bad = [dict(x) for x in sites]; bad[0]["source_vertex"] = 11; cases.append((bad, kw))
    bad = [dict(x) for x in sites]; bad[0]["donor_body"] = 2; cases.append((bad, kw))
    bad = [dict(x) for x in sites]; bad[0]["source_vertex"] = 14; cases.append((bad, kw))
    w = weight.copy(); w[10,1] = .84
    cases.append((sites, {**kw, "weight_by_source_vertex_body": w}))
    a = altitude.copy(); a[0] = 4.9e-5
    cases.append((sites, {**kw, "incident_altitude_m_by_row": a}))
    m = mass.copy(); m[0] = 0
    cases.append((sites, {**kw, "deep_node_mass_kg_by_row": m}))
    cases.append((sites, {**kw, "budget": 2}))
    duplicate_sources = [dict(sites[0]), dict(sites[1])]
    duplicate_sources[1]["source_vertex"] = 10
    duplicate_sources[1]["donor_body"] = 1
    source_map = source_by_row.copy(); source_map[1] = 10
    cases.append((duplicate_sources, {**kw, "source_vertex_by_row": source_map}))
    duplicate_rows = [dict(sites[0]), dict(sites[1])]
    duplicate_rows[1]["reduced_vertex"] = 0
    duplicate_rows[1]["source_vertex"] = 10
    cases.append((duplicate_rows, kw))
    inactive = active.copy(); inactive[1] = False
    cases.append((sites, {**kw, "active": inactive}))
    for bad_sites, bad_kw in cases:
        assert validate_site_contract(bad_sites, **bad_kw), "invalid site unexpectedly passed"
    same = [dict(sites[0]), dict(sites[1])]
    same[1]["donor_body"] = 1
    body = body_by_source.copy(); body[11] = 1
    w = weight.copy(); w[11,1] = .9
    p = world.copy(); p[1] = p[0]
    same_kw = {**kw, "body_by_source_vertex": body, "weight_by_source_vertex_body": w, "world_by_row": p}
    assert validate_site_contract(same, **same_kw), "same-donor spacing violation passed"
    same.append(dict(sites[2])); same[2]["donor_body"] = 1
    body[12] = 1; w[12,1] = .9
    fourth = {"reduced_vertex": 3, "source_vertex": 13, "donor_body": 1}
    same.append(fourth)
    body[13] = 1; w[13,1] = .9
    p = np.asarray([[0,0,0],[.03,0,0],[.06,0,0],[.09,0,0],[.12,0,0]], dtype=np.float64)
    same_kw = {**same_kw, "world_by_row": p}
    assert validate_site_contract(same, **same_kw), "four same-donor sites passed the donor cap"
    print("attachment_validation_self_test=PASS cases=valid,identity,donor,duplicate_source,duplicate_row,inactive,weight,altitude,mass,budget,spacing,donor_cap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, key in (("coarse-surface","coarse"), ("layers","layers"), ("source-map","source_map"),
                      ("row-map","row_map"), ("source-geometry","source_geometry"), ("skin-payload","skin_payload"),
                      ("left-patch","left_patch"), ("right-patch","right_patch"), ("old-sites","old_sites"),
                      ("source-header","source_header"), ("native-source","native_source"),
                      ("native-test","native_test"), ("cmake","cmake"), ("bone-manifest","bone_manifest"),
                      ("prism-helper","prism_helper")):
        parser.add_argument("--"+flag, dest=key, default=str(DEFAULTS[key]))
    parser.add_argument("--acl-node-count", type=int, default=None,
                        help="exact native ACL FEM node count; omitted uses conservative ACL=0 budget lower bound")
    parser.add_argument("--output", default=str(E / "replacement-mechanical-071-attachment-selection-001/selection-validation.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.acl_node_count is not None and args.acl_node_count < 0:
        parser.error("--acl-node-count must be nonnegative")
    if args.self_test:
        self_test()
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = load_and_run(args)
    paths = {key: Path(getattr(args, key)) for key in DEFAULTS}
    receipt = contract_receipt(paths)
    report["tool"] = {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))}
    report["invocation"] = {"argv": sys.argv, "python": sys.executable, "cwd": str(Path.cwd())}
    report["contract_receipt"] = receipt
    receipt_path = output.with_name("native-attachment-contract-receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {k: report[k] for k in ("status","retained_active_old_site_count","retired_orphan_old_site_count",
                                        "provisional_replacement_candidates","planned_site_count","accepted_sites")}
    summary["layer_gates"] = report["native_layer_gates"]
    summary["validation_errors"] = report["native_site_validation_errors"]
    summary["result"] = str(output)
    summary["contract_receipt"] = str(receipt_path)
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "candidate_set_passes_local_native_attachment_contract" else 2


if __name__ == "__main__":
    raise SystemExit(main())
