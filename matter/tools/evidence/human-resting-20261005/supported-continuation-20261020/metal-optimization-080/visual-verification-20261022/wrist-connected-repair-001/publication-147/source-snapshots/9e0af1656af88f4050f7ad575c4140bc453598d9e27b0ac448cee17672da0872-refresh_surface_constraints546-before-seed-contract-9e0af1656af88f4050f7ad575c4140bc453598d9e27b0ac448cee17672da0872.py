#!/usr/bin/env python3
"""Refresh 546 self seeds and fixed-bone guards across face-topology edits.

This is construction-only. It wraps the frozen 541 refresh owner, augments the
prior 314 guard payload with prior exact-159 crossings that are now clear, and
maps historical 026 self-seed face pairs only through unique exact unordered
triangle identities. Run under the geometry slot when called outside prep546.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
REFRESH541 = D / "focused-clearance-541/constraints/refresh_guards_541.py"
REFRESH541_SHA = "293e4da49a10f07420d1da7c512823dcb182dca246e411c3775d18866b27a241"
REFRESH541_SHIM = D / "surface-clearance-546/refresh_guards_multi_parent_546.py"
REFRESH541_SHIM_SHA = "c01a6221f8684829cc4dca39cc5596c9387ccebdacdc6b7b5b6a79dda0fe875e"
OWNER026 = ROOT / "publication-101/evidence/204/source/fit/self_separation_026.py"
OWNER313 = ROOT / "publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
OWNER314 = ROOT / "publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
OWNER401 = D / "retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
PIN = {
    "026": "c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3",
    "313": "dc9d4f953ecede00567c59cbbb6b6b63e0a3556b97972689de499cbce29d4c2e",
    "314": "3422f30a61db0724d234c487d70fc3d74237a802261c2f9a72af9e15843a1c78",
    "401": "2f1eda3f0d46d0f5497a1a04075b5dc2c3b37afe906d0389ff636fca8abc5869",
}


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path | str, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def face_identity_index(faces: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    index: dict[tuple[int, int, int], list[int]] = {}
    for row, face in enumerate(np.asarray(faces)):
        index.setdefault(tuple(sorted(map(int, face))), []).append(int(row))
    return index


def map_face_identity(source_faces: np.ndarray, destination_index: dict, source_row: int) -> tuple[int, int, str]:
    faces = np.asarray(source_faces)
    if source_row < 0 or source_row >= len(faces):
        raise ValueError(f"source face row out of range: {source_row}")
    hits = destination_index.get(tuple(sorted(map(int, faces[source_row]))), [])
    if len(hits) == 1:
        return int(hits[0]), 1, "unique_exact_face"
    if len(hits) == 0:
        return -1, 0, "no_exact_face_match"
    return -1, len(hits), "ambiguous_exact_face_match"


def map_seed_pairs_partial(source_faces: np.ndarray, destination_faces: np.ndarray,
                           pairs: np.ndarray) -> dict:
    """Map whole self-pair rows only when both source and destination faces are unique."""
    raw = np.asarray(pairs)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu":
        raise ValueError("self-seed pair payload must be integer (K,2)")
    source_ix = face_identity_index(source_faces)
    dest_ix = face_identity_index(destination_faces)
    source_status = np.zeros(len(raw), dtype=np.uint8)
    match_counts = np.zeros((len(raw), 2), dtype=np.int32)
    mapped_indices = []
    mapped_by_source = []
    for k, pair in enumerate(raw):
        i, j = map(int, pair)
        if i < 0 or j < 0 or i >= len(source_faces) or j >= len(source_faces) or i == j:
            raise ValueError(f"invalid source self-seed pair at row {k}: {(i, j)}")
        faces = np.asarray(source_faces)
        src_counts = (len(source_ix[tuple(sorted(map(int, faces[i])))]),
                      len(source_ix[tuple(sorted(map(int, faces[j])))]))
        a, ac, ar = map_face_identity(source_faces, dest_ix, i)
        b, bc, br = map_face_identity(source_faces, dest_ix, j)
        match_counts[k] = (ac, bc)
        if src_counts[0] != 1 or src_counts[1] != 1 or ar == "ambiguous_exact_face_match" or br == "ambiguous_exact_face_match":
            source_status[k] = 2
        elif ar != "unique_exact_face" or br != "unique_exact_face":
            source_status[k] = 1
        elif a == b:
            source_status[k] = 3
        else:
            mapped_indices.append(k)
            mapped_by_source.append(sorted((a, b)))
    mapped_by_source_arr = np.asarray(mapped_by_source, dtype=np.int64).reshape(-1, 2)
    mapped = np.unique(mapped_by_source_arr, axis=0) if len(mapped_by_source_arr) else np.empty((0, 2), np.int64)
    dropped_indices = np.flatnonzero(source_status != 0).astype(np.int64)
    payload = {"source_pairs": raw.astype(np.int64, copy=False),
        "source_status_code": source_status, "source_match_counts": match_counts,
        "mapped_source_pair_indices": np.asarray(mapped_indices, np.int64),
        "mapped_pairs_by_source_row": mapped_by_source_arr,
        "unique_mapped_pairs": mapped,
        "dropped_source_pair_indices": dropped_indices,
        "dropped_source_pairs": raw[dropped_indices].astype(np.int64, copy=False),
        "dropped_status_code": source_status[dropped_indices],
        "dropped_match_counts": match_counts[dropped_indices]}
    return {"mapped_pairs": mapped, "source_status_code": source_status,
        "source_match_counts": match_counts, "payload": payload,
        "status_names": {0: "mapped_unique", 1: "changed_or_missing_face", 2: "ambiguous_source_or_destination_face", 3: "pair_faces_collapsed"}}


def remap_prior_surface_pairs(source_faces: np.ndarray, destination_faces: np.ndarray,
                              prior_pairs: set[tuple[int, int, int, int]],
                              current_pairs: set[tuple[int, int, int, int]]) -> dict:
    """Classify prior 159 pairs using unique source and destination face identity."""
    source_ix = face_identity_index(source_faces)
    destination_ix = face_identity_index(destination_faces)
    rows = sorted(prior_pairs)
    status = np.zeros((len(rows),), dtype=np.uint8)
    match_counts = np.zeros((len(rows),), dtype=np.int32)
    mapped_source_indices = []
    mapped_current_pairs = []
    newly_cleared_source = []
    newly_cleared_current = []
    for i, (body, bone, source_face, tri) in enumerate(rows):
        key = tuple(sorted(map(int, source_faces[source_face])))
        if len(source_ix.get(key, [])) != 1:
            status[i] = 2
            continue
        dest, nmatch, reason = map_face_identity(source_faces, destination_ix, source_face)
        match_counts[i] = nmatch
        if reason == "no_exact_face_match":
            status[i] = 1
            continue
        if reason == "ambiguous_exact_face_match":
            status[i] = 2
            continue
        identity = (body, bone, dest, tri)
        mapped_source_indices.append(i)
        mapped_current_pairs.append(identity)
        if identity in current_pairs:
            status[i] = 3
        else:
            status[i] = 0
            newly_cleared_source.append((body, bone, source_face, tri))
            newly_cleared_current.append(identity)
    mapped_current_arr = np.asarray(mapped_current_pairs, dtype=np.int64).reshape(-1, 4)
    dropped_indices = np.flatnonzero((status == 1) | (status == 2)).astype(np.int64)
    all_rows = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    payload = {"source_pairs": all_rows, "source_status_code": status,
        "source_match_counts": match_counts,
        "mapped_source_pair_indices": np.asarray(mapped_source_indices, np.int64),
        "mapped_current_pairs": mapped_current_arr,
        "newly_cleared_source_pairs": np.asarray(newly_cleared_source, np.int64).reshape(-1, 4),
        "newly_cleared_current_pairs": np.asarray(newly_cleared_current, np.int64).reshape(-1, 4),
        "dropped_source_pair_indices": dropped_indices,
        "dropped_source_pairs": all_rows[dropped_indices],
        "dropped_status_code": status[dropped_indices],
        "dropped_match_counts": match_counts[dropped_indices]}
    return {"source_pairs": all_rows, "source_status_code": status,
        "newly_cleared_source_pairs": tuple(newly_cleared_source),
        "newly_cleared_current_pairs": tuple(newly_cleared_current),
        "payload": payload,
        "status_names": {0: "newly_cleared", 1: "no_exact_face_match", 2: "ambiguous_source_or_destination_identity", 3: "still_crossing"}}


def write_mapping_npz(path: Path, mapping: dict) -> None:
    np.savez_compressed(path, **mapping["payload"])


def load_audit_pairs(exporter, report_path: Path, candidate_path: Path,
                     points: np.ndarray, faces: np.ndarray, bankkeys: np.ndarray) -> tuple[dict, set]:
    report = read_json(report_path)
    digest = sha(candidate_path)
    pair_set = exporter.validate_audit_report(
        report, candidate_sha256=digest, point_count=len(points), face_count=len(faces),
        expected_bank_keys=tuple((int(row[0]), int(row[1])) for row in np.asarray(bankkeys)),
        expected_script_sha256=exporter.sha256_file(exporter.AUDIT_159),
        expected_predicate_sha256=exporter.sha256_file(exporter.EXACT_159),
    )
    return report, pair_set


def create_augmented_prior_guard(*, out_dir: Path, prior_candidate: Path,
                                 prior_surface_report: Path, prior_guard_npz: Path,
                                 prior_guard_report: Path, current_candidate: Path,
                                 current_surface_report: Path, exporter, validator,
                                 bank_helper, bank_data: tuple, bank_pins: dict,
                                 prior_pairs: set, current_pairs: set) -> tuple[Path, Path, dict]:
    prior = load_npz(prior_candidate)
    current = load_npz(current_candidate)
    prior_sha, current_sha = sha(prior_candidate), sha(current_candidate)
    if sha(prior_guard_npz) != read_json(prior_guard_report).get("output_npz_sha256"):
        raise ValueError("prior fixed guard NPZ/report hash mismatch")
    if read_json(prior_guard_report).get("base_candidate_sha256") != prior_sha:
        raise ValueError("prior fixed guard is not bound to prior candidate")
    prior_guard_arrays = load_npz(prior_guard_npz)
    prior_guard_report_obj = read_json(prior_guard_report)
    existing_records, _ = validator.validate_fixed_boundary_payload(
        prior_guard_arrays, prior_guard_report_obj, candidate_sha256=prior_sha,
        candidate_points=prior["points_world_m"], candidate_faces=prior["source_O_faces"],
        npz_path=prior_guard_npz,
    )
    bank_triangles, bank_ids, _gids, _bodies, matrices, offsets, positions, bankkeys, actual_bank_pins, _ = bank_data
    if actual_bank_pins != bank_pins:
        raise ValueError("pinned bank input changed while building augmented guard")
    if prior_guard_report_obj.get("bank_reader_sha256") != sha(exporter.BANK_HELPER):
        raise ValueError("prior 314 bank reader differs from current pinned reader")
    if prior_guard_report_obj.get("bank_source_pins") != bank_pins:
        raise ValueError("prior 314 bank source pins differ from current pinned bank inputs")
    surface_mapping = remap_prior_surface_pairs(
        prior["source_O_faces"], current["source_O_faces"], prior_pairs, current_pairs
    )
    old_guard_source = {
        (int(r["bank_body"]), int(r["stable_bone_id"]), int(r["skin_face_row"]), int(r["bank_triangle_stable_id"]))
        for r in existing_records
    }
    augmented_pairs = old_guard_source | set(surface_mapping["newly_cleared_source_pairs"])
    triangles_by_identity = exporter.typed_triangle_lookup(bank_triangles, bank_ids)
    transforms = {int(body): (matrices[int(body)], offsets[int(body)], positions[int(body)]) for body in matrices}
    arrays = exporter.make_barrier_arrays(augmented_pairs, triangles_by_identity, transforms)
    out_dir.mkdir(parents=True, exist_ok=False)
    npz = out_dir / "augmented-prior-guard.npz"
    report_path = out_dir / "augmented-prior-guard-report.json"
    np.savez_compressed(npz, **arrays)
    typed = [{"skin_face_row": int(face), "bank_body": int(body), "stable_bone_id": int(bone),
              "bank_triangle_stable_id": int(tri)}
             for body, bone, face, tri in sorted(augmented_pairs)]
    report = {
        "schema": exporter.OUTPUT_SCHEMA,
        "status": "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
        "filtered_builder_path": str(Path(__file__).resolve()),
        "filtered_builder_sha256": sha(Path(__file__).resolve()),
        "base_candidate_path": str(prior_candidate), "base_candidate_sha256": prior_sha,
        "candidate_points_world_m_array_sha256": validator.array_sha256(prior["points_world_m"]),
        "skin_faces_array_sha256": validator.array_sha256(prior["source_O_faces"]),
        "point_count": len(prior["points_world_m"]), "face_count": len(prior["source_O_faces"]),
        "bank_count": 185, "expected_bank_count": 185,
        "fixed_triangles_units": "mm", "fixed_triangles_frame": "world",
        "geometry_or_native_admission": False,
        "bank_reader_path": str(exporter.BANK_HELPER), "bank_reader_sha256": sha(exporter.BANK_HELPER),
        "bank_source_pins": bank_pins,
        "output_npz": npz.name, "output_npz_sha256": sha(npz),
        "introduced_pair_count": len(typed), "typed_pairs": typed,
        "inputs": [{"role": "base", "candidate_path": str(prior_candidate),
                    "candidate_sha256": prior_sha, "audit159_path": str(prior_surface_report),
                    "audit159_sha256": sha(prior_surface_report)}],
        "augmentation_source": {
            "prior_guard_report_path": str(prior_guard_report), "prior_guard_report_sha256": sha(prior_guard_report),
            "prior_guard_npz_path": str(prior_guard_npz), "prior_guard_npz_sha256": sha(prior_guard_npz),
            "prior_surface_report_path": str(prior_surface_report), "prior_surface_report_sha256": sha(prior_surface_report),
            "current_candidate_path": str(current_candidate), "current_candidate_sha256": current_sha,
            "current_surface_report_path": str(current_surface_report), "current_surface_report_sha256": sha(current_surface_report),
            "previous_exact_159_pair_count": len(prior_pairs), "current_exact_159_pair_count": len(current_pairs),
            "newly_cleared_mappable_pair_count": len(surface_mapping["newly_cleared_source_pairs"]),
        },
        "topology_remap": {"mapping": "unique exact unordered source_O_faces vertex triples only",
            "prior_159_pair_source_count": len(surface_mapping["source_pairs"]),
            "newly_cleared_count": int(np.count_nonzero(surface_mapping["source_status_code"] == 0)),
            "unmapped_modified_face_count": int(np.count_nonzero(surface_mapping["source_status_code"] == 1)),
            "ambiguous_face_identity_count": int(np.count_nonzero(surface_mapping["source_status_code"] == 2)),
            "still_currently_crossing_count": int(np.count_nonzero(surface_mapping["source_status_code"] == 3)),
            "mapping_payload_path": str(out_dir / "prior-surface-pair-remap.npz")},
        "prior_guard_count": len(existing_records), "augmented_guard_count": len(typed),
        "limitations": ["Construction constraints only; final exact 159 and 130 checks remain mandatory.",
            "A prior surface pair is added only if both face identity and typed bank identity map exactly and it is absent from the current exact-159 pair set."]
    }
    # Mapping payload is saved beside this report before its hash is recorded.
    write_mapping_npz(out_dir / "prior-surface-pair-remap.npz", surface_mapping)
    report["topology_remap"]["mapping_payload_sha256"] = sha(out_dir / "prior-surface-pair-remap.npz")
    write_json(report_path, report)
    # Validate the augmented export with the exact same 401 owner used by 541.
    validator.validate_fixed_boundary_payload(arrays, report, candidate_sha256=prior_sha,
        candidate_points=prior["points_world_m"], candidate_faces=prior["source_O_faces"], npz_path=npz)
    mapped_current = set(surface_mapping["newly_cleared_current_pairs"])
    return npz, report_path, {"surface_mapping": surface_mapping, "prior_guard_pair_count": len(existing_records),
        "augmented_prior_guard_pair_count": len(typed), "newly_cleared_source_pair_count": len(surface_mapping["newly_cleared_source_pairs"]),
        "newly_cleared_current_pairs": mapped_current}


def execute(args) -> dict:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    owners = {"026": OWNER026, "313": OWNER313, "314": OWNER314, "401": OWNER401}
    actual_pins = {key: sha(path) for key, path in owners.items()}
    if actual_pins != PIN:
        raise ValueError(f"frozen 026/313/314/401 owner pin mismatch: {actual_pins}")
    refresh_pin = sha(REFRESH541)
    if refresh_pin != REFRESH541_SHA:
        raise ValueError("frozen 541 refresh owner hash mismatch")
    shim_pin = sha(REFRESH541_SHIM)
    if shim_pin != REFRESH541_SHIM_SHA:
        raise ValueError("multi-parent 546 refresh shim hash mismatch")
    refresh = load_module("refresh541_adapter546", REFRESH541)
    exporter = load_module("fixed_boundary_exporter314_adapter546", OWNER314)
    validator = load_module("fixed_boundary_validator401_adapter546", OWNER401)
    bank_helper = load_module("bank_inputs084_adapter546", exporter.BANK_HELPER)

    cur = load_npz(args.candidate)
    prior = load_npz(args.prior_candidate)
    trial = load_npz(args.failed_seed_candidate)
    currsha = sha(args.candidate)
    priorsha = sha(args.prior_candidate)
    trialsha = sha(args.failed_seed_candidate)
    P, F = cur["points_world_m"], cur["source_O_faces"]
    priorP, priorF = prior["points_world_m"], prior["source_O_faces"]
    trialP, trialF = trial["points_world_m"], trial["source_O_faces"]
    if P.dtype != np.float32 or P.ndim != 2 or P.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("current candidate must contain emitted Float32 points and triangular faces")
    if priorP.dtype != np.float32 or trialP.dtype != np.float32:
        raise ValueError("prior and failed-trial candidates must retain emitted Float32 points")
    if sha(args.prior_guard_candidate) != priorsha:
        raise ValueError("prior guard candidate must be the exact prior-state candidate")

    # Load the exact pinned 039 bank inputs once; all typed IDs and world triangles
    # come from the same 314 transform owner used by the frozen refresh.
    bank_data = bank_helper.read_bank_inputs(exporter.BANK_DIR, exporter.BANK_EVIDENCE)
    bankpins = bank_data[8]
    if len(bank_data[7]) != 185 or len({tuple(map(int, x)) for x in bank_data[7]}) != 185:
        raise ValueError("pinned bank input is not exactly 185 unique typed banks")
    prior_audit, prior_surface_pairs = load_audit_pairs(exporter, args.prior_surface_report,
        args.prior_candidate, priorP, priorF, bank_data[7])
    current_audit, current_surface_pairs = load_audit_pairs(exporter, args.audit159,
        args.candidate, P, F, bank_data[7])
    current_a130, current_exact_self = refresh.validate130(args.audit130,
        args.audit130.parent / "surface-self-pairs.npy", currsha, len(P), len(F))
    if len(current_exact_self) != 0 or int(current_a130.get("exact_PLC_surface_pairs", -1)) != 0:
        raise ValueError("adapter requires current candidate with exact self-pair count zero")
    if current_a130.get("inside_node_bank_pairs") != 0:
        raise ValueError("adapter requires current candidate with zero containment")

    # Existing context seeds were built against the prior candidate; validate
    # their candidate/path/hash receipt before remapping them.
    prior_seed_report_path = Path(args.prior_seed_report)
    prior_seed_report = read_json(prior_seed_report_path)
    prior_seed_path = Path(args.prior_seed_pairs)
    if prior_seed_report.get("candidate_sha256") != priorsha:
        raise ValueError("prior-context self seeds are not bound to prior candidate")
    if Path(prior_seed_report.get("output_pairs_path", "")).resolve() != prior_seed_path.resolve() or prior_seed_report.get("output_pairs_sha256") != sha(prior_seed_path):
        raise ValueError("prior-context self-seed path/hash mismatch")
    prior_seed_pairs = np.load(prior_seed_path, allow_pickle=False)
    if prior_seed_pairs.ndim != 2 or prior_seed_pairs.shape[1] != 2 or prior_seed_pairs.dtype.kind not in "iu":
        raise ValueError("prior-context self-seed payload malformed")
    if len(prior_seed_pairs) and (prior_seed_pairs.min() < 0 or prior_seed_pairs.max() >= len(priorF)):
        raise ValueError("prior-context self-seed face outside prior candidate topology")

    trial_audit, trial_pairs = refresh.validate130(args.failed_seed_audit130,
        args.failed_seed_pairs, trialsha, len(trialP), len(trialF))
    if len(trial_pairs) != int(trial_audit["exact_PLC_surface_pairs"]):
        raise ValueError("failed-trial self pairs do not match its exact audit")

    # Create prior-candidate guard payload augmented only by prior exact-159
    # pairs that uniquely remap to unchanged faces and are absent currently.
    augment_dir = out / "augmentation"
    augment_npz, augment_report, augment_receipt = create_augmented_prior_guard(
        out_dir=augment_dir, prior_candidate=args.prior_candidate,
        prior_surface_report=args.prior_surface_report, prior_guard_npz=args.prior_guard_npz,
        prior_guard_report=args.prior_guard_report, current_candidate=args.candidate,
        current_surface_report=args.audit159, exporter=exporter, validator=validator,
        bank_helper=bank_helper, bank_data=bank_data, bank_pins=bankpins,
        prior_pairs=prior_surface_pairs, current_pairs=current_surface_pairs)

    # Let 541 validate current exact rows and perform the current-domain 313/401
    # barrier rebuild. Supplying current's exact zero-self report makes its own
    # self seed set exact and empty; historical seeds are handled below partially.
    refresh_dir = out / "refresh-541"
    cmd = [str(PYTHON), str(REFRESH541_SHIM), "--candidate", str(args.candidate),
        "--audit130", str(args.audit130), "--audit159", str(args.audit159),
        "--query-dir", str(args.query_dir), "--ancestry-parent-candidate", str(args.ancestry_parent_candidate),
        "--seed-source-candidate", str(args.candidate), "--seed-source-audit130", str(args.audit130),
        "--seed-source-pairs", str(args.audit130.parent / "surface-self-pairs.npy"),
        "--prior-guard-candidate", str(args.prior_candidate), "--prior-guard-report", str(augment_report),
        "--prior-guard-npz", str(augment_npz), "--output-dir", str(refresh_dir),
        "--expected-candidate-sha256", currsha,
        "--expected-audit130-sha256", sha(args.audit130), "--expected-audit159-sha256", sha(args.audit159)]
    launch_path, log_path, execution_path = out / "refresh-541.launch.json", out / "refresh-541.log", out / "refresh-541.execution.json"
    pin_paths = [Path(__file__), REFRESH541, REFRESH541_SHIM, OWNER026, OWNER313, OWNER314, OWNER401,
        Path(exporter.BANK_HELPER), Path(exporter.AUDIT_159), Path(exporter.EXACT_159),
        args.candidate, args.audit130, args.audit159, args.prior_candidate, args.prior_surface_report,
        args.prior_guard_candidate, args.prior_guard_npz, args.prior_guard_report,
        args.prior_seed_pairs, args.prior_seed_report, args.failed_seed_candidate,
        args.failed_seed_audit130, args.failed_seed_pairs]
    pin_paths.extend(exporter.BANK_DIR / name for name in bankpins)
    pins = {str(path): sha(path) for path in pin_paths}
    write_json(launch_path, {"argv": cmd, "source_pins": pins, "slot_contract": "caller must hold geometry slot; no nested slot process",
        "native_admission": False, "accepted_physical_steps": 0})
    start = time.monotonic()
    with log_path.open("x") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    unchanged = all(sha(path) == digest for path, digest in pins.items())
    write_json(execution_path, {"returncode": int(proc.returncode), "seconds": time.monotonic()-start,
        "source_pins_unchanged": unchanged, "native_admission": False, "accepted_physical_steps": 0})
    if proc.returncode != 0 or not unchanged:
        raise RuntimeError(f"frozen 541 refresh failed; see {log_path}")
    rr = read_json(refresh_dir / "refresh-report.json")
    if rr.get("candidate_sha256") != currsha or int(rr.get("current_exact_self_pairs", -1)) != 0:
        raise ValueError("541 refresh output does not bind to exact zero-self current candidate")
    if float(rr.get("fixed_313_max_rhs_mm", float("inf"))) > 2e-8:
        raise ValueError("541 fixed-313 guards did not zero-step certify")

    # Partial exact-identity remap for old pair seeds, retaining only rows where
    # both source faces have unique matches in current source_O_faces.
    seed_dir = out / "self-seeds"
    seed_dir.mkdir(parents=True, exist_ok=False)
    prior_mapping = map_seed_pairs_partial(priorF, F, prior_seed_pairs)
    failed_mapping = map_seed_pairs_partial(trialF, F, trial_pairs)
    prior_map_path = seed_dir / "prior-context-remap.npz"
    failed_map_path = seed_dir / "failed-trial-remap.npz"
    write_mapping_npz(prior_map_path, prior_mapping)
    write_mapping_npz(failed_map_path, failed_mapping)
    current_self_pairs = np.asarray(np.load(rr["seed_pairs_path"], allow_pickle=False), dtype=np.int64).reshape(-1, 2)
    if len(current_self_pairs) != 0:
        raise ValueError("541 current self-seed output is nonempty despite exact-zero current audit")
    all_parts = [current_self_pairs, prior_mapping["mapped_pairs"], failed_mapping["mapped_pairs"]]
    merged = np.unique(np.vstack(all_parts), axis=0) if any(len(p) for p in all_parts) else np.empty((0, 2), np.int64)
    # Rebuild all 026 rows for the mapped set and certify zero-step feasibility.
    pair_owner = refresh.loadmod("self_sep_026_adapter546", OWNER026)
    rows, _records = pair_owner.build_pair_rows(P.astype(np.float64) * 1000.0, F, merged, margin_fraction=.25)
    rhs = max((float(row["lower"]) for row in rows), default=0.0)
    if rhs > 2e-8:
        raise ValueError(f"combined mapped 026 self seeds fail zero-step certificate: {rhs}")
    merged_path = seed_dir / "merged-self-pair-seeds.npy"
    np.save(merged_path, merged)
    packed = refresh.pack(rows)
    rows_path = seed_dir / "merged-self-pair-rows.npz"
    np.savez_compressed(rows_path, **packed)
    status_names = {"0": "mapped_unique", "1": "changed_or_missing_face", "2": "ambiguous_face_identity", "3": "pair_faces_collapsed"}
    seed_report_path = seed_dir / "seed-preflight.json"
    seed_report = {"schema": "numi.human.continuation-self-seeds.v1",
        "status": "PASS_PARTIAL_EXACT_FACE_REMAP_AND_ZERO_STEP_026_CERTIFICATE",
        "candidate_path": str(args.candidate), "candidate_sha256": currsha,
        "current_exact_self_pair_count": 0, "zero_step_feasible": True,
        "zero_step_max_rhs_mm": rhs, "rebuilt_linear_row_count": len(rows),
        "output_pairs_path": str(merged_path), "output_pairs_sha256": sha(merged_path),
        "output_rows_path": str(rows_path), "output_rows_sha256": sha(rows_path),
        "merged_unique_pair_count": len(merged), "current_self_pairs": 0,
        "prior_context_source": {"candidate": str(args.prior_candidate), "candidate_sha256": priorsha,
            "pairs_path": str(prior_seed_path), "pairs_sha256": sha(prior_seed_path),
            "source_pair_count": len(prior_seed_pairs), "mapped_unique_pair_count": len(prior_mapping["mapped_pairs"]),
            "dropped_pair_count": int(np.count_nonzero(prior_mapping["source_status_code"] != 0)),
            "mapping_payload_path": str(prior_map_path), "mapping_payload_sha256": sha(prior_map_path)},
        "failed_trial_source": {"candidate": str(args.failed_seed_candidate), "candidate_sha256": trialsha,
            "audit130_path": str(args.failed_seed_audit130), "audit130_sha256": sha(args.failed_seed_audit130),
            "pairs_path": str(args.failed_seed_pairs), "pairs_sha256": sha(args.failed_seed_pairs),
            "source_pair_count": len(trial_pairs), "mapped_unique_pair_count": len(failed_mapping["mapped_pairs"]),
            "dropped_pair_count": int(np.count_nonzero(failed_mapping["source_status_code"] != 0)),
            "mapping_payload_path": str(failed_map_path), "mapping_payload_sha256": sha(failed_map_path)},
        "mapping_status_codes": status_names,
        "pair_owner_path": str(OWNER026), "pair_owner_sha256": sha(OWNER026),
        "source_pins": {"541": refresh_pin, **{k: v for k, v in actual_pins.items()}},
        "construction_only": True, "native_admission": False, "accepted_physical_steps": 0}
    write_json(seed_report_path, seed_report)

    # Independently confirm the exported current guard payload excludes every
    # current exact crossing, and count the added prior-clear pairs retained.
    current_guard_npz = Path(rr["fixed_guard_npz_path"])
    current_guard_report = Path(rr["fixed_guard_report_path"])
    guard_arrays = load_npz(current_guard_npz)
    guard_report = read_json(current_guard_report)
    guard_records, _ = validator.validate_fixed_boundary_payload(guard_arrays, guard_report,
        candidate_sha256=currsha, candidate_points=P, candidate_faces=F, npz_path=current_guard_npz)
    guard_ids = {(int(r["bank_body"]), int(r["stable_bone_id"]), int(r["skin_face_row"]), int(r["bank_triangle_stable_id"])) for r in guard_records}
    new_current_ids = set(augment_receipt["newly_cleared_current_pairs"])
    if not new_current_ids.issubset(guard_ids):
        raise ValueError("one or more newly cleared prior 159 identities were lost during 541 refresh")
    if guard_ids.intersection(current_surface_pairs):
        raise ValueError("fixed guard export contains an active current exact-159 crossing")

    result = dict(rr)
    result.update({
        "schema": "numi.human.surface_clearance_546_guard_seed_refresh.v1",
        "status": "PASS_TOPOLOGY_AWARE_GUARD_AND_SEED_REFRESH",
        "candidate_path": str(args.candidate), "candidate_sha256": currsha,
        "audit130_path": str(args.audit130), "audit130_sha256": sha(args.audit130),
        "audit159_path": str(args.audit159), "audit159_sha256": sha(args.audit159),
        "prior_candidate_path": str(args.prior_candidate), "prior_candidate_sha256": priorsha,
        "prior_surface_report_path": str(args.prior_surface_report),
        "prior_surface_report_sha256": sha(args.prior_surface_report),
        "prior_exact_159_pair_count": len(prior_surface_pairs),
        "current_exact_159_pair_count": len(current_surface_pairs),
        "newly_cleared_prior_159_pair_count": len(new_current_ids),
        "newly_cleared_prior_159_pairs_retained_in_guard": len(new_current_ids),
        "active_current_crossings_excluded": len(current_surface_pairs),
        "fixed_guard_pair_count": len(guard_ids),
        "fixed_guard_payload_contains_no_current_crossings": True,
        "fixed_313_zero_step_certified": float(rr["fixed_313_max_rhs_mm"]) <= 2e-8,
        "augmented_prior_guard_npz_path": str(augment_npz),
        "augmented_prior_guard_npz_sha256": sha(augment_npz),
        "augmented_prior_guard_report_path": str(augment_report),
        "augmented_prior_guard_report_sha256": sha(augment_report),
        "prior_surface_mapping_payload_path": str(augment_dir / "prior-surface-pair-remap.npz"),
        "prior_surface_mapping_payload_sha256": sha(augment_dir / "prior-surface-pair-remap.npz"),
        "prior_surface_mapping_counts": {
            "newly_cleared": int(np.count_nonzero(augment_receipt["surface_mapping"]["source_status_code"] == 0)),
            "unmapped_modified_face": int(np.count_nonzero(augment_receipt["surface_mapping"]["source_status_code"] == 1)),
            "ambiguous_face_identity": int(np.count_nonzero(augment_receipt["surface_mapping"]["source_status_code"] == 2)),
            "still_crossing": int(np.count_nonzero(augment_receipt["surface_mapping"]["source_status_code"] == 3)),
        },
        "seed_pairs_path": str(merged_path), "seed_pairs_sha256": sha(merged_path),
        "seed_report_path": str(seed_report_path), "seed_report_sha256": sha(seed_report_path),
        "seed_pairs_count": len(merged), "seed_rows_path": str(rows_path), "seed_rows_sha256": sha(rows_path),
        "mapped_prior_seed_count": len(prior_mapping["mapped_pairs"]),
        "dropped_prior_seed_count": int(np.count_nonzero(prior_mapping["source_status_code"] != 0)),
        "mapped_failed_trial_seed_count": len(failed_mapping["mapped_pairs"]),
        "dropped_failed_trial_seed_count": int(np.count_nonzero(failed_mapping["source_status_code"] != 0)),
        "refresh_541_path": str(REFRESH541), "refresh_541_sha256": refresh_pin,
        "refresh_541_invoker_path": str(REFRESH541_SHIM), "refresh_541_invoker_sha256": shim_pin,
        "source_pins": {"541": refresh_pin, "546_multi_parent_shim": shim_pin, **{k: v for k, v in actual_pins.items()},
            "314_bank_reader": sha(exporter.BANK_HELPER), "159_audit": sha(exporter.AUDIT_159),
            "159_exact_predicate": sha(exporter.EXACT_159)},
        "bank_source_pins": bankpins,
        "native_admission": False, "accepted_physical_steps": 0,
    })
    if any(sha(path) != digest for path, digest in pins.items()):
        raise RuntimeError("a pinned source or input changed during guard/seed refresh")
    adapter_report_path = out / "refresh-report.json"
    write_json(adapter_report_path, result)
    context_path = out / "constraint-context.json"
    context = {"candidate": str(args.candidate), "candidate_sha256": currsha,
        "seed_pairs_path": str(merged_path), "seed_report": str(seed_report_path),
        "seed_report_sha256": sha(seed_report_path),
        "guard_npz": str(current_guard_npz), "guard_report": str(current_guard_report),
        "surface_report": str(args.audit159), "surface_report_sha256": sha(args.audit159),
        "point_summary": str(args.audit130), "point_summary_sha256": sha(args.audit130),
        "refresh_report": str(adapter_report_path), "refresh_report_sha256": sha(adapter_report_path),
        "native_admission": False, "accepted_physical_steps": 0}
    write_json(context_path, context)
    return {"refresh_report": str(adapter_report_path), "context": str(context_path),
        "seed_pairs_path": str(merged_path), "guard_npz": str(current_guard_npz),
        "guard_report": str(current_guard_report), "newly_cleared_prior_159_pair_count": len(new_current_ids),
        "mapped_seed_count": len(merged), "dropped_seed_count": int(np.count_nonzero(prior_mapping["source_status_code"] != 0) + np.count_nonzero(failed_mapping["source_status_code"] != 0))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "audit130", "audit159", "query-dir", "ancestry-parent-candidate",
                 "prior-candidate", "prior-surface-report", "prior-guard-candidate",
                 "prior-guard-npz", "prior-guard-report", "prior-seed-pairs", "prior-seed-report",
                 "failed-seed-candidate", "failed-seed-audit130", "failed-seed-pairs", "output-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    result = execute(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
