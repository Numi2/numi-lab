#!/usr/bin/env python3
"""Filter same-parent surface flips by exact 159 typed skin/bone pair counts.

This is a construction proposal only. It preserves the existing 349 flip
operator's same-parent, matched-037 area/orientation, ancestry, and boundary
checks. It does not move points or claim geometry/native admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
FLIP_MODULE_PATH = ROOT / "D/connected-remesh-reuse-358/same_parent_flips_358.py"
SURFACE_AUDIT_159_PATH = ROOT / "audit_skin_bone_surfaces_159.py"
BANK_HELPER_PATH = ROOT / "publication-093/canonical/matter/tools/human_skin_fat_bone_audit.py"
EXPECTED_HASHES = {
    "flip_operator_358": "a08e1f7e6392854f72ffab09af6f2b1f2afe885b1a84f586afe71160ff3ccbee",
    "surface_audit_159": "de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1",
    "bank_reader_and_summary": "37c040a442f1fdf312745c1cc1fb6c543383f1cd0782db46c81ca1eadf0d8324",
    "exact_predicate_315": "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
}
EXPECTED_BANK_COUNT = 185
BANK_DIR_DEFAULT = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/"
    "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
)
EVIDENCE_ROOT_DEFAULT = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001"
)


class SelectorError(ValueError):
    pass


@dataclass(frozen=True)
class TypedBank:
    body: int
    stable_bone_id: int
    triangles_local_m: np.ndarray
    stable_triangle_ids: np.ndarray
    world_to_body_m: np.ndarray
    position_world_m: np.ndarray
    com_offset_m: np.ndarray

    @property
    def key(self) -> tuple[int, int]:
        return (self.body, self.stable_bone_id)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SelectorError(f"cannot load required pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pinned_runtime():
    """Load the frozen flip operator and the exact 159 audit/read owners."""
    paths = {
        "flip_operator_358": FLIP_MODULE_PATH,
        "surface_audit_159": SURFACE_AUDIT_159_PATH,
        "bank_reader_and_summary": BANK_HELPER_PATH,
    }
    got = {key: sha256_file(path) for key, path in paths.items()}
    for key, expected in EXPECTED_HASHES.items():
        if key == "exact_predicate_315":
            continue
        if got[key] != expected:
            raise SelectorError(f"{key} source hash mismatch: {got[key]} != {expected}")
    flip = _load_module("same_parent_flips_358_for_370", FLIP_MODULE_PATH)
    surface_owner = _load_module("audit_skin_bone_surfaces_159_for_370", SURFACE_AUDIT_159_PATH)
    bank_helper = _load_module("human_skin_fat_bone_audit_for_370", BANK_HELPER_PATH)
    exact = surface_owner.exact_module()
    exact_path = Path(surface_owner.EXACT)
    exact_hash = sha256_file(exact_path)
    if exact_hash != EXPECTED_HASHES["exact_predicate_315"]:
        raise SelectorError(f"exact predicate hash mismatch: {exact_hash}")
    got["exact_predicate_315"] = exact_hash
    return flip, surface_owner, bank_helper, exact, got


def typed_banks_from_159_inputs(inputs: tuple) -> tuple[list[TypedBank], list[tuple[int, int]]]:
    """Build all 185 typed banks from the pinned read_bank_inputs tuple."""
    if not isinstance(inputs, tuple) or len(inputs) != 10:
        raise SelectorError("159 read_bank_inputs must return its exact 10-field tuple")
    triangles, ids, _gids, _bodies, matrices, offsets, positions, keys, _pins, _exact = inputs
    tris = np.asarray(triangles, dtype=np.float64)
    id_array = np.asarray(ids)
    key_array = np.asarray(keys)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or not np.isfinite(tris).all():
        raise SelectorError("159 bank triangles must be finite Nx3x3")
    if id_array.shape != (len(tris), 4) or not np.issubdtype(id_array.dtype, np.integer):
        raise SelectorError("159 triangle identity rows must be integer Nx4")
    if key_array.shape != (EXPECTED_BANK_COUNT, 2) or not np.issubdtype(key_array.dtype, np.integer):
        raise SelectorError("159 bank key inventory must be integer 185x2")
    bank_keys = [(int(row[0]), int(row[1])) for row in key_array]
    if len(set(bank_keys)) != EXPECTED_BANK_COUNT:
        raise SelectorError("159 bank key inventory contains duplicates")
    if set(map(tuple, np.unique(id_array[:, :2], axis=0).tolist())) != set(bank_keys):
        raise SelectorError("triangle identities differ from 159 typed bank inventory")
    result: list[TypedBank] = []
    for body, bone in bank_keys:
        mask = (id_array[:, 0] == body) & (id_array[:, 1] == bone)
        stable_ids = np.asarray(id_array[mask, 3], dtype=np.int64)
        if not len(stable_ids) or len(set(map(int, stable_ids))) != len(stable_ids):
            raise SelectorError(f"typed bank {(body, bone)} has empty/duplicate stable triangle IDs")
        if body not in matrices or body not in offsets or body not in positions:
            raise SelectorError(f"typed bank {(body, bone)} has no pinned body transform")
        matrix = np.asarray(matrices[body], dtype=np.float64)
        position = np.asarray(positions[body], dtype=np.float64)
        offset = np.asarray(offsets[body], dtype=np.float64)
        if matrix.shape != (3, 3) or position.shape != (3,) or offset.shape != (3,):
            raise SelectorError(f"typed bank {(body, bone)} transform shapes are invalid")
        if not np.isfinite(matrix).all() or not np.isfinite(position).all() or not np.isfinite(offset).all():
            raise SelectorError(f"typed bank {(body, bone)} transform is non-finite")
        result.append(TypedBank(
            body=body,
            stable_bone_id=bone,
            triangles_local_m=tris[mask].copy(),
            stable_triangle_ids=stable_ids.copy(),
            world_to_body_m=matrix.copy(),
            position_world_m=position.copy(),
            com_offset_m=offset.copy(),
        ))
    return result, bank_keys


def _pair_rows_and_counts(
    baseline_report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    face_count: int,
    banks: Sequence[TypedBank],
    expected_bank_keys: Sequence[tuple[int, int]],
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], set[int], dict[tuple[int, int], int]]:
    """Validate the complete 159 report and return typed pair identities."""
    if baseline_report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT":
        raise SelectorError("baseline 159 report is not a complete typed-bank audit")
    if baseline_report.get("complete") is not True:
        raise SelectorError("baseline 159 report is incomplete")
    if baseline_report.get("geometry_or_native_admission") is not False:
        raise SelectorError("baseline 159 report admission marker is malformed")
    if baseline_report.get("candidate_sha256") != candidate_sha256:
        raise SelectorError("baseline 159 report does not audit the exact candidate bytes")
    if (baseline_report.get("bank_count") != EXPECTED_BANK_COUNT or
            baseline_report.get("expected_bank_count") != EXPECTED_BANK_COUNT):
        raise SelectorError("baseline 159 report has an incomplete bank count")
    report_banks = baseline_report.get("banks")
    if not isinstance(report_banks, list) or len(report_banks) != EXPECTED_BANK_COUNT:
        raise SelectorError("baseline 159 report must contain all 185 banks")
    bank_by_key = {bank.key: bank for bank in banks}
    expected = set(expected_bank_keys)
    if len(expected) != EXPECTED_BANK_COUNT or set(bank_by_key) != expected:
        raise SelectorError("loaded typed-bank inventory is incomplete or duplicated")
    if set(baseline_report.get("expected_typed_bank_keys", expected)) != expected:
        raise SelectorError("baseline report typed-bank receipt differs from loaded banks")
    pairs_by_bank: dict[tuple[int, int], set[tuple[int, int]]] = {}
    bank_counts: dict[tuple[int, int], int] = {}
    target_face_rows: set[int] = set()
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(report_banks):
        if not isinstance(row, dict) or type(row.get("body")) is not int or type(row.get("stable_bone_id")) is not int:
            raise SelectorError(f"159 bank row {index} lacks exact typed identity")
        key = (row["body"], row["stable_bone_id"])
        if key in seen or key not in expected:
            raise SelectorError(f"159 report has duplicate or unknown typed bank {key}")
        seen.add(key)
        bank = bank_by_key[key]
        if row.get("skin_faces") != face_count or row.get("bank_faces") != len(bank.triangles_local_m):
            raise SelectorError(f"159 report face domains differ for typed bank {key}")
        result = row.get("result")
        pair_set: set[tuple[int, int]] = set()
        if result is None:
            if (row.get("skin_aabb_active") != 0 or row.get("bank_aabb_active") != 0 or
                    type(row.get("skin_faces")) is not int or type(row.get("bank_faces")) is not int or
                    row["skin_faces"] <= 0 or row["bank_faces"] <= 0):
                raise SelectorError(f"159 null result lacks explicit complete AABB exclusion for {key}")
        else:
            if not isinstance(result, dict):
                raise SelectorError(f"159 result for {key} is malformed")
            pairs = result.get("triangle_pairs")
            count = result.get("count")
            if type(count) is not int or count < 0 or not isinstance(pairs, list) or len(pairs) != count:
                raise SelectorError(f"159 exact pair count/array mismatch for {key}")
            stable_ids = set(map(int, bank.stable_triangle_ids))
            for pair in pairs:
                if (not isinstance(pair, list) or len(pair) != 2 or
                        any(type(value) is not int for value in pair)):
                    raise SelectorError(f"159 exact pair identity is malformed for {key}")
                face_row, triangle_id = pair
                if not 0 <= face_row < face_count or triangle_id not in stable_ids:
                    raise SelectorError(f"159 exact pair identity is outside {key}'s domain")
                identity = (face_row, triangle_id)
                if identity in pair_set:
                    raise SelectorError(f"159 report duplicates exact pair {identity} for {key}")
                pair_set.add(identity)
                target_face_rows.add(face_row)
        pairs_by_bank[key] = pair_set
        bank_counts[key] = len(pair_set)
    if seen != expected:
        raise SelectorError("159 report omits one or more typed banks")
    return pairs_by_bank, target_face_rows, bank_counts


class Exact159PatchCounter:
    """Exact 159 triangle-pair counter for only the two faces of one flip."""

    def __init__(self, points_world_m: np.ndarray, banks: Sequence[TypedBank],
                 exact_predicate: Any, surface_records: Any):
        raw = np.asarray(points_world_m)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise SelectorError("candidate world points must be finite Nx3")
        self.points_f32 = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(self.points_f32).all():
            raise SelectorError("candidate Float32 world points are non-finite")
        self.banks = tuple(banks)
        self.exact = exact_predicate
        self.records = surface_records
        self.local_points_by_body: dict[int, np.ndarray] = {}

    def _local_points(self, bank: TypedBank) -> np.ndarray:
        cached = self.local_points_by_body.get(bank.body)
        if cached is None:
            cached = ((self.points_f32.astype(np.float64) - bank.position_world_m)
                      @ bank.world_to_body_m.T - bank.com_offset_m)
            if not np.isfinite(cached).all():
                raise SelectorError(f"transformed points are non-finite for body {bank.body}")
            self.local_points_by_body[bank.body] = cached
        return cached

    def count_faces(self, face_rows: Sequence[int], face_triangles: np.ndarray) -> dict[tuple[int, int], dict]:
        rows = np.asarray(face_rows)
        faces = np.asarray(face_triangles)
        if rows.ndim != 1 or rows.size == 0 or not np.issubdtype(rows.dtype, np.integer):
            raise SelectorError("patch face rows must be a nonempty integer vector")
        if faces.shape != (len(rows), 3) or not np.issubdtype(faces.dtype, np.integer):
            raise SelectorError("patch faces must align as integer Kx3")
        if len(set(map(int, rows))) != len(rows) or np.any(rows < 0):
            raise SelectorError("patch face rows are duplicated or negative")
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(self.points_f32)):
            raise SelectorError("patch vertex row outside candidate")
        result_by_bank: dict[tuple[int, int], dict] = {}
        for bank in self.banks:
            local = self._local_points(bank)
            skin_triangles = local[faces]
            bank_triangles = bank.triangles_local_m
            bank_lo = bank_triangles.min(axis=1)
            bank_hi = bank_triangles.max(axis=1)
            skin_lo = skin_triangles.min(axis=1)
            skin_hi = skin_triangles.max(axis=1)
            whole_bank_lo = bank_lo.min(axis=0)
            whole_bank_hi = bank_hi.max(axis=0)
            active_skin = np.flatnonzero(
                np.all(skin_hi >= whole_bank_lo, axis=1) &
                np.all(skin_lo <= whole_bank_hi, axis=1)
            )
            if not len(active_skin):
                result_by_bank[bank.key] = {"count": 0, "triangle_pairs": [],
                                            "aabb_candidate_pairs": 0,
                                            "allowed_shared_vertex_or_edge_pairs": 0}
                continue
            active_lo = skin_lo[active_skin].min(axis=0)
            active_hi = skin_hi[active_skin].max(axis=0)
            active_bank = np.flatnonzero(
                np.all(bank_hi >= active_lo, axis=1) &
                np.all(bank_lo <= active_hi, axis=1)
            )
            if not len(active_bank):
                result_by_bank[bank.key] = {"count": 0, "triangle_pairs": [],
                                            "aabb_candidate_pairs": 0,
                                            "allowed_shared_vertex_or_edge_pairs": 0}
                continue
            skin_records = self.records(
                skin_triangles[active_skin],
                rows[active_skin].astype(np.int64, copy=False),
                self.exact,
            )
            bank_records = self.records(
                bank_triangles[active_bank],
                bank.stable_triangle_ids[active_bank],
                self.exact,
            )
            audited = self.exact._audit_pair(skin_records, bank_records, same_surface=False)
            pairs = audited.get("triangle_pairs")
            if not isinstance(pairs, list) or audited.get("count") != len(pairs):
                raise SelectorError("159 exact predicate returned malformed pair results")
            if audited.get("allowed_shared_vertex_or_edge_pairs", 0) != 0:
                raise SelectorError("159 cross-surface audit unexpectedly authorized a shared simplex")
            valid_rows, valid_triangle_ids = set(map(int, rows)), set(map(int, bank.stable_triangle_ids))
            for pair in pairs:
                if (not isinstance(pair, list) or len(pair) != 2 or
                        type(pair[0]) is not int or type(pair[1]) is not int or
                        pair[0] not in valid_rows or pair[1] not in valid_triangle_ids):
                    raise SelectorError("159 exact predicate returned an untyped/out-of-domain pair")
            result_by_bank[bank.key] = {
                "count": len(pairs),
                "triangle_pairs": sorted(map(tuple, pairs)),
                "aabb_candidate_pairs": int(audited.get("aabb_candidate_pairs", 0)),
                "allowed_shared_vertex_or_edge_pairs": 0,
            }
        if set(result_by_bank) != {bank.key for bank in self.banks}:
            raise SelectorError("patch count did not cover every loaded typed bank")
        return result_by_bank


def _operation_key(operation: Mapping[str, Any]) -> tuple:
    try:
        edge = tuple(sorted(map(int, operation["edge"])))
        faces = tuple(map(int, operation["face_rows"]))
        replacement = tuple(tuple(map(int, tri)) for tri in operation["new_faces"])
        parent = int(operation["parent_row"])
    except Exception as exc:
        raise SelectorError("flip proposal operation has malformed identity") from exc
    if len(edge) != 2 or len(faces) != 2 or len(replacement) != 2 or any(len(t) != 3 for t in replacement):
        raise SelectorError("flip proposal operation has invalid dimensions")
    return edge, faces, replacement, parent


def _strict_decrease(old: Mapping[tuple[int, int], int],
                     new: Mapping[tuple[int, int], int]) -> tuple[bool, str, int]:
    if set(old) != set(new):
        raise SelectorError("old/new pair counts do not cover the same typed banks")
    deltas = {key: int(old[key]) - int(new[key]) for key in old}
    if any(delta < 0 for delta in deltas.values()):
        return False, "typed_bank_pair_count_increased", sum(deltas.values())
    total = sum(old.values()) - sum(new.values())
    if total <= 0:
        return False, "no_strict_total_pair_decrease", total
    return True, "strict_exact_pair_decrease_all_banks_nonincreasing", total


def filter_proposed_batch_370(
    candidate: Mapping[str, np.ndarray],
    proposal_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    baseline_report_sha256: str,
    banks: Sequence[TypedBank],
    expected_bank_keys: Sequence[tuple[int, int]],
    exact_predicate: Any,
    surface_records: Any,
    flip_module: Any,
    max_flips: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Retain only pre-existing 349-valid flips that strictly reduce 159 pairs."""
    if len(banks) != EXPECTED_BANK_COUNT:
        raise SelectorError("all 185 exact typed banks are required")
    data = {key: np.asarray(value) for key, value in candidate.items()}
    faces = np.asarray(data.get("source_O_faces"))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise SelectorError("candidate source_O_faces must be Fx3")
    baseline_pairs, target_rows, baseline_bank_counts = _pair_rows_and_counts(
        baseline_report,
        candidate_sha256=candidate_sha256,
        face_count=len(faces),
        banks=banks,
        expected_bank_keys=expected_bank_keys,
    )
    if proposal_report.get("all_185_typed_bank_audits_required") not in (True, None):
        raise SelectorError("proposal report bank audit marker is invalid")
    operations = proposal_report.get("operations")
    if not isinstance(operations, list):
        raise SelectorError("flip proposal must include an operations list")
    if proposal_report.get("accepted_pair_count", len(operations)) != len(operations):
        raise SelectorError("flip proposal accepted count does not match operation list")
    if not target_rows:
        out = {key: np.array(value, copy=True) for key, value in data.items()}
        return out, {
            "status": "NO_TYPED_SKIN_BONE_CROSSING_TARGETS",
            "geometry_admitted": False, "native_admission": False,
            "accepted_physical_steps": 0, "all_185_banks_complete": True,
            "candidate_sha256": candidate_sha256,
            "baseline_159_report_sha256": baseline_report_sha256,
            "baseline_total_pairs": 0,
            "proposed_flip_count": len(operations), "accepted_flip_count": 0,
            "operations": [],
        }
    if proposal_report.get("seed_face_rows") is not None:
        seed_rows = proposal_report["seed_face_rows"]
        if (not isinstance(seed_rows, list) or
                any(type(row) is not int or row < 0 or row >= len(faces) for row in seed_rows)):
            raise SelectorError("proposal seed face rows are malformed")
        if not set(seed_rows).issubset(target_rows):
            raise SelectorError("proposal uses seed rows outside exact 159 crossing faces")
    else:
        seed_rows = sorted(target_rows)
    # Do not trust proposal payload alone: regenerate its exact edge-whitelisted
    # batch through the frozen same-parent operator before counting it.
    operation_keys = [_operation_key(op) for op in operations]
    if len(set(operation_keys)) != len(operation_keys):
        raise SelectorError("proposal has duplicate flip operation identities")
    for operation, (_edge, pair_faces, _new_faces, _parent) in zip(operations, operation_keys):
        seed_faces = operation.get("seed_face_rows")
        if seed_faces is None:
            seed_faces = sorted(set(pair_faces) & target_rows)
        if not isinstance(seed_faces, list) or not seed_faces or any(
                type(face) is not int or face not in target_rows for face in seed_faces):
            raise SelectorError("proposal operation is not rooted in an exact 159 crossing face")
        if not set(seed_faces).issubset(pair_faces):
            raise SelectorError("proposal seed-face receipt is outside its operation pair")
    edges = [list(key[0]) for key in operation_keys]
    angle_gain = float(proposal_report.get("minimum_angle_gain_deg", 1.0e-8))
    normal_dot = float(proposal_report.get("minimum_normal_dot", 0.0))
    regenerated, regen_report = flip_module.propose_flip_batch(
        data, seed_rows, max_flips=len(operations) if max_flips is None else int(max_flips),
        min_angle_gain_deg=angle_gain, min_normal_dot=normal_dot,
        allowed_edges=edges,
    )
    generated = regen_report.get("operations", [])
    generated_keys = {_operation_key(op) for op in generated}
    if generated_keys != set(operation_keys):
        raise SelectorError("proposal operations do not reproduce through pinned 358 same-parent gates")
    # Input proposal can be accompanied by an output candidate; if so it must
    # equal the deterministic regeneration byte-for-byte on every face array.
    for key in ("source_O_faces", "area_reference_037_source_O_faces",
                "refinement_face_parent_barycentric"):
        if key in proposal_report:
            raise SelectorError("proposal report must not inline unauthenticated face arrays")
    counter = Exact159PatchCounter(
        data["points_world_m"], banks, exact_predicate, surface_records
    )
    baseline_face_counts: dict[tuple[int, int], dict[int, int]] = {}
    for bank_key, pairs in baseline_pairs.items():
        by_face: dict[int, int] = {}
        for face_row, _stable_tri_id in pairs:
            by_face[face_row] = by_face.get(face_row, 0) + 1
        baseline_face_counts[bank_key] = by_face
    scored: list[dict[str, Any]] = []
    accepted_edges: list[list[int]] = []
    accepted_keys: set[tuple] = set()
    rejection_counts: dict[str, int] = {}
    for operation in operations:
        edge, pair_faces, new_faces, parent = _operation_key(operation)
        old_by_bank = {
            key: sum(baseline_face_counts[key].get(row, 0) for row in pair_faces)
            for key in expected_bank_keys
        }
        new_results = counter.count_faces(pair_faces, np.asarray(new_faces, dtype=np.int64))
        new_by_bank = {key: result["count"] for key, result in new_results.items()}
        keep, reason, reduction = _strict_decrease(old_by_bank, new_by_bank)
        if keep:
            accepted_edges.append(list(edge))
            accepted_keys.add(_operation_key(operation))
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        per_bank = []
        for key in sorted(old_by_bank):
            old_count, new_count = old_by_bank[key], new_by_bank[key]
            if old_count or new_count:
                old_ids = sorted(tri_id for face, tri_id in baseline_pairs[key] if face in pair_faces)
                new_ids = sorted(pair[1] for pair in new_results[key]["triangle_pairs"])
                per_bank.append({
                    "body": key[0], "stable_bone_id": key[1],
                    "old_pair_count": old_count, "new_pair_count": new_count,
                    "removed_pair_count": max(0, old_count - new_count),
                    "introduced_pair_count": max(0, new_count - old_count),
                    "old_triangle_ids": old_ids, "new_triangle_ids": new_ids,
                })
        scored.append({
            "edge": list(edge), "face_rows": list(pair_faces), "parent_row": parent,
            "old_total_pairs": sum(old_by_bank.values()),
            "new_total_pairs": sum(new_by_bank.values()),
            "strict_pair_reduction": reduction,
            "selected": keep, "selection_reason": reason,
            "old_new_by_typed_bank": per_bank,
            "minimum_angle_gain_deg": float(operation.get("angle_gain_deg", 0.0)),
        })
    selected_order = sorted(
        (row for row in scored if row["selected"]),
        key=lambda row: (-row["strict_pair_reduction"],
                         -row["minimum_angle_gain_deg"],
                         tuple(row["face_rows"]), tuple(row["edge"])),
    )
    if accepted_edges:
        final, final_proposal = flip_module.propose_flip_batch(
            data, seed_rows,
            max_flips=len(accepted_edges),
            min_angle_gain_deg=angle_gain,
            min_normal_dot=normal_dot,
            allowed_edges=[row["edge"] for row in selected_order],
        )
        final_ops = final_proposal.get("operations", [])
        if {_operation_key(op) for op in final_ops} != accepted_keys:
            raise SelectorError("filtered exact-pair subset changed during 349 batch reconstruction")
        if not np.array_equal(final["points_world_m"], data["points_world_m"]):
            raise SelectorError("349 flip reconstruction changed point coordinates")
        if not np.array_equal(final["source_O_faces"], final["area_reference_037_source_O_faces"]):
            raise SelectorError("349 output broke matched-037 face synchronization")
    else:
        final = {key: np.array(value, copy=True) for key, value in data.items()}
        final_proposal = {
            "status": "NO_EXACT_PAIR_REDUCING_FLIPS",
            "accepted_pair_count": 0, "operations": [], "changed_face_rows": [],
        }
    accepted_score_rows = [row for row in scored if row["selected"]]
    old_by_bank_total = {key: 0 for key in expected_bank_keys}
    new_by_bank_total = {key: 0 for key in expected_bank_keys}
    for row in accepted_score_rows:
        for bank_row in row["old_new_by_typed_bank"]:
            key = (bank_row["body"], bank_row["stable_bone_id"])
            old_by_bank_total[key] += bank_row["old_pair_count"]
            new_by_bank_total[key] += bank_row["new_pair_count"]
    total_keep, total_reason, total_reduction = _strict_decrease(old_by_bank_total, new_by_bank_total) if accepted_score_rows else (False, "no_flips_selected", 0)
    if accepted_score_rows and not total_keep:
        raise SelectorError("selected operations fail combined 185-bank strict decrease")
    report = {
        "status": "BONE_PAIR_REDUCING_SAME_PARENT_FLIP_PROPOSAL" if accepted_score_rows else "NO_EXACT_PAIR_REDUCING_FLIPS",
        "geometry_admitted": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "complete_159_baseline_required": True,
        "all_185_banks_complete": True,
        "exact_159_replacement_face_predicate_used": True,
        "exact_pair_reduction_is_local_screen_only": True,
        "full_159_and_exact_self_audits_still_required": True,
        "source_037_area_and_current_reference_face_fold_gates": "re-evaluated by pinned same-parent 358 operator for every retained flip",
        "coordinates_bitwise_unchanged": True,
        "candidate_sha256": candidate_sha256,
        "baseline_159_report_sha256": baseline_report_sha256,
        "flip_operator_sha256": EXPECTED_HASHES["flip_operator_358"],
        "surface_audit_159_sha256": EXPECTED_HASHES["surface_audit_159"],
        "exact_predicate_sha256": EXPECTED_HASHES["exact_predicate_315"],
        "baseline_total_pairs": sum(baseline_bank_counts.values()),
        "target_crossing_face_count": len(target_rows),
        "proposed_flip_count": len(operations),
        "strict_pair_reducing_flip_count": len(accepted_score_rows),
        "strict_pair_reduction_across_selected_operations": total_reduction,
        "combined_pair_reduction_status": total_reason,
        "selected_edges_ranked_by": ["strict_exact_pair_reduction_desc", "349_minimum_angle_gain_desc", "face_rows", "edge"],
        "rejection_reason_counts": rejection_counts,
        "seed_rows_source": "complete 159 result.triangle_pairs first-column union",
        "operation_domain": "only the original 349/358 proposed batch; no replacement/fill operations",
        "final_flip_operator_report": final_proposal,
        "operations": scored,
        "selection_order": [row["edge"] for row in selected_order],
        "selected_per_typed_bank_totals": [
            {"body": key[0], "stable_bone_id": key[1],
             "old_pair_count": old_by_bank_total[key],
             "new_pair_count": new_by_bank_total[key]}
            for key in sorted(old_by_bank_total) if old_by_bank_total[key] or new_by_bank_total[key]
        ],
    }
    return final, report


def _save_candidate(path: Path, candidate: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **candidate)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-159-report", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-baseline-report-sha256", required=True)
    parser.add_argument("--bank-dir", type=Path, default=BANK_DIR_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-flips", type=int, default=136)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    candidate_hash = sha256_file(args.candidate)
    baseline_hash = sha256_file(args.baseline_159_report)
    if candidate_hash != args.expected_candidate_sha256:
        raise SelectorError("candidate SHA pin mismatch")
    if baseline_hash != args.expected_baseline_report_sha256:
        raise SelectorError("baseline 159 report SHA pin mismatch")
    baseline = json.loads(args.baseline_159_report.read_text(encoding="utf-8"))
    flip, surface_owner, bank_helper, exact, runtime_hashes = load_pinned_runtime()
    raw_inputs = bank_helper.read_bank_inputs(args.bank_dir, args.evidence_root)
    banks, keys = typed_banks_from_159_inputs(raw_inputs)
    bank_helper.summarize_surface_crossing_report(
        baseline, expected_banks=EXPECTED_BANK_COUNT, expected_bank_keys=keys
    )
    with np.load(args.candidate, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]).copy() for key in archive.files}
    # _pair_rows_and_counts checks all typed pair IDs against these exact banks.
    _pairs, target_rows, _counts = _pair_rows_and_counts(
        baseline, candidate_sha256=candidate_hash,
        face_count=len(source.get("source_O_faces", [])), banks=banks,
        expected_bank_keys=keys,
    )
    if not target_rows:
        proposal = {"operations": [], "accepted_pair_count": 0, "seed_face_rows": []}
    else:
        _proposed, proposal = flip.propose_flip_batch(
            source, sorted(target_rows), max_flips=args.max_flips,
            min_angle_gain_deg=1.0e-8, min_normal_dot=0.0,
        )
    result, report = filter_proposed_batch_370(
        source, proposal, baseline,
        candidate_sha256=candidate_hash,
        baseline_report_sha256=baseline_hash,
        banks=banks, expected_bank_keys=keys,
        exact_predicate=exact, surface_records=surface_owner.records,
        flip_module=flip, max_flips=args.max_flips,
    )
    args.output_dir.mkdir(parents=True)
    candidate_out = args.output_dir / "candidate.npz"
    report["runtime_source_hashes"] = runtime_hashes
    report["baseline_report_source"] = str(args.baseline_159_report)
    report["base_candidate_source"] = str(args.candidate)
    report["bank_directory_source"] = str(args.bank_dir)
    report["bank_reader_input_pins"] = raw_inputs[8]
    report["output_candidate"] = str(candidate_out)
    report["output_candidate_sha256"] = _save_candidate(candidate_out, result)
    report["output_array_hashes"] = {
        "points_world_m": sha256_array(result["points_world_m"]),
        "source_O_faces": sha256_array(result["source_O_faces"]),
        "area_reference_037_source_O_faces": sha256_array(result["area_reference_037_source_O_faces"]),
        "refinement_face_parent_face_row": sha256_array(result["refinement_face_parent_face_row"]),
        "refinement_face_parent_barycentric": sha256_array(result["refinement_face_parent_barycentric"]),
    }
    args.output_dir.joinpath("selector-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "proposed": report["proposed_flip_count"],
        "selected": report["strict_pair_reducing_flip_count"],
        "pair_reduction": report["strict_pair_reduction_across_selected_operations"],
        "candidate_sha256": report["output_candidate_sha256"],
        "geometry_admitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
