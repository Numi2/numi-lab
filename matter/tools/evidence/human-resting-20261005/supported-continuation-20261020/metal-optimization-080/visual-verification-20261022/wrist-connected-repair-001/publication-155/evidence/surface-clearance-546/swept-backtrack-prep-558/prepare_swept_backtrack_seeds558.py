#!/usr/bin/env python3
"""Prepare bounded exact-witness self-pair seeds from a 350 receipt.

Consumes a completed 350 receipt; it never runs broadphase, a solver, or an audit.
The emitted seed set is historical certified seeds plus supplied exact witnesses.
It explicitly does not claim coverage of an incomplete swept broadphase.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
SWEEPER = ROOT / "D/swept-self-pair-preseed-350/swept_self_pair_preseed_350.py"
SWEEPER_SHA256 = "54bc7f72724c14eb72eedce73aceca1492ce4abd38771bfc6af780582c9b677b"
EXPECTED_PAIR_OWNER_SHA256 = "c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3"
ZERO_STEP_TOL_MM = 2e-8
SEED_SCHEMA = "numi.human.continuation-self-seeds.v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_pairs(pairs, face_count):
    raw = np.asarray(pairs)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu":
        raise ValueError("pair list must be an integer (N,2) array")
    out = np.sort(raw.astype(np.int64, copy=False), axis=1)
    if np.any(out < 0) or np.any(out >= int(face_count)) or np.any(out[:, 0] == out[:, 1]):
        raise ValueError("pair list contains invalid face rows")
    out = np.unique(out, axis=0)
    return out


def union_seed_pairs(existing, exact_witnesses, face_count):
    old = canonical_pairs(existing, face_count)
    witness = canonical_pairs(exact_witnesses, face_count)
    merged = canonical_pairs(np.vstack((old, witness)), face_count)
    old_set = {tuple(map(int, r)) for r in old}
    witness_set = {tuple(map(int, r)) for r in witness}
    return merged, {
        "existing_unique_count": int(len(old)),
        "exact_witness_unique_count": int(len(witness)),
        "new_exact_witness_count": int(len(witness_set - old_set)),
        "merged_unique_count": int(len(merged)),
        "all_exact_witnesses_retained": witness_set.issubset({tuple(map(int, r)) for r in merged}),
    }


def require_pairs_present(required, available, face_count):
    """Check canonical pair inclusion without constructing Python sets."""
    need = canonical_pairs(required, face_count)
    have = np.asarray(available)
    if have.ndim != 2 or have.shape[1] != 2 or have.dtype.kind not in "iu":
        raise ValueError("available pair list must be an integer (N,2) array")
    have = have.astype(np.int64, copy=False)
    if len(have) and (np.any(have[:, 0] < 0) or np.any(have[:, 0] >= have[:, 1])
                      or np.any(have[:, 1] >= face_count)):
        raise ValueError("available pair list is not canonical/in-range")
    have_keys = have[:, 0] * np.int64(face_count) + have[:, 1]
    if len(have_keys) > 1 and np.any(have_keys[1:] <= have_keys[:-1]):
        raise ValueError("available pair list is not sorted and unique")
    need_keys = need[:, 0] * np.int64(face_count) + need[:, 1]
    positions = np.searchsorted(have_keys, need_keys)
    if len(need_keys) and (np.any(positions >= len(have_keys))
                           or np.any(have_keys[np.minimum(positions, max(0, len(have_keys)-1))] != need_keys)):
        raise ValueError("required pairs are missing from the available pair list")
    return int(len(need))


def load_pinned_owner(path, expected_sha256):
    path = Path(path)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"026 owner hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("swept_seed_026_owner_558", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned 026 owner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_pair_rows", None)):
        raise ValueError("pinned 026 owner lacks build_pair_rows")
    return module


def certify_new_pairs_zero_step(owner, points_mm, faces, pairs, *,
                                timeout_seconds=120.0, max_pairs=5000,
                                batch_size=64, tolerance_mm=ZERO_STEP_TOL_MM):
    """Build only supplied pairs' 026 rows and certify zero-step feasibility."""
    pairs = canonical_pairs(pairs, len(faces))
    if len(pairs) > max_pairs:
        raise ValueError(f"new exact-witness pair cap exceeded: {len(pairs)} > {max_pairs}")
    if batch_size < 1 or timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("invalid bounded row-build limits")
    pts = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("baseline points_mm are invalid")
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("face topology is invalid")
    if f.min(initial=0) < 0 or f.max(initial=-1) >= len(pts):
        raise ValueError("face topology index out of range")
    deadline = time.monotonic() + timeout_seconds
    accepted = []
    rejected = []
    row_count = 0
    owner_calls = 0

    def visit(chunk):
        nonlocal row_count, owner_calls
        if not len(chunk):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("026 zero-step row certification timed out")
        if len(chunk) > batch_size:
            for start in range(0, len(chunk), batch_size):
                visit(chunk[start:start + batch_size])
            return
        owner_calls += 1
        ids = [tuple(map(int, x)) for x in chunk]
        try:
            rows, records = owner.build_pair_rows(pts, f, ids)
            if len(records) != len(ids):
                raise ValueError(f"026 owner returned {len(records)} records for {len(ids)} pairs")
            for row in rows:
                lower = float(row["lower"])
                if not math.isfinite(lower):
                    raise ValueError("026 owner emitted nonfinite row lower")
                if lower > tolerance_mm:
                    raise ValueError(f"026 row is not zero-step feasible: lower={lower}")
                row_count += 1
            accepted.extend(ids)
        except Exception as exc:
            if len(chunk) == 1:
                rejected.append({"pair": ids[0], "reason": f"{type(exc).__name__}: {exc}"})
                return
            mid = len(chunk) // 2
            visit(chunk[:mid])
            visit(chunk[mid:])

    visit(pairs)
    if rejected:
        raise ValueError(f"{len(rejected)} exact witness pairs failed 026 zero-step eligibility")
    if {tuple(x) for x in accepted} != {tuple(map(int, r)) for r in pairs}:
        raise ValueError("026 eligibility result did not cover every requested witness pair")
    return {
        "status": "PASS_SELECTED_EXACT_WITNESS_ROWS_ZERO_STEP_ONLY",
        "pair_count": int(len(pairs)),
        "linear_row_count": int(row_count),
        "owner_calls": int(owner_calls),
        "timeout_seconds": float(timeout_seconds),
        "batch_size": int(batch_size),
        "zero_step_tolerance_mm": float(tolerance_mm),
        "coverage": "exact audit-130 witnesses not already covered by historical 026 seed preflight",
        "swept_pair_coverage": False,
    }


def validate_incomplete_sweep(receipt, *, expected_base_sha, expected_trial_sha,
                              expected_seed_sha, expected_witness_sha):
    if receipt.get("status") != "BLOCKED_CLASSIFICATION_INCOMPLETE":
        raise ValueError("expected the retained incomplete 350 receipt")
    pins = receipt.get("input_pins", {})
    expected = {
        "base_candidate": expected_base_sha,
        "raw_trial_0": expected_trial_sha,
        "seed_pairs_0": expected_seed_sha,
        "witness_pairs_0": expected_witness_sha,
    }
    for key, value in expected.items():
        if pins.get(key) != value:
            raise ValueError(f"350 receipt input pin mismatch: {key}")
    eligibility = receipt.get("pair_row_eligibility", {})
    if eligibility.get("classification_stop_reason") != "candidate_pair_cap":
        raise ValueError("350 receipt was not stopped at the declared candidate cap")
    shape = receipt.get("potential_pair_array_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("350 receipt lacks a valid potential-pair shape")
    potential_count = int(shape[0])
    if (int(eligibility.get("unclassified_pair_count", -2)) != potential_count
            or int(eligibility.get("eligible_pair_count", -1)) != 0
            or int(eligibility.get("rejected_pair_count", -1)) != 0):
        raise ValueError("350 cap receipt counts do not match the retained blocked run")
    return {
        "status": receipt["status"],
        "potential_pair_count": potential_count,
        "swept_aabb_pair_count": int(receipt.get("swept_aabb_pair_count", -1)),
        "unclassified_pair_count": int(eligibility["unclassified_pair_count"]),
        "swept_coverage_complete": False,
        "reason": "candidate_pair_cap",
    }


def make_seed_preflight(*, candidate_path, candidate_sha, seed_path, seed_sha,
                        pair_count, pair_owner_path, pair_owner_sha,
                        audit_path, audit_sha, audit_pairs_sha, audit_pair_count,
                        old_seed_report_path, old_seed_report_sha,
                        sweep_receipt_path, sweep_receipt_sha, certification):
    return {
        "schema": SEED_SCHEMA,
        "status": "PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY",
        "zero_step_feasible": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "output_pairs_path": str(seed_path),
        "output_pairs_sha256": seed_sha,
        "merged_unique_pair_count": int(pair_count),
        "pair_owner_path": str(pair_owner_path),
        "pair_owner_sha256": pair_owner_sha,
        "construction_only": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "old_seed_preflight": {
            "path": str(old_seed_report_path),
            "sha256": old_seed_report_sha,
            "status": "validated against current baseline and pinned 026 owner",
        },
        "exact_witness_source": {
            "audit130_path": str(audit_path),
            "audit130_sha256": audit_sha,
            "pairs_sha256": audit_pairs_sha,
            "pair_count": int(audit_pair_count),
            "all_exact_witnesses_retained": True,
            "new_rows_zero_step_certification": certification,
        },
        "swept_broadphase": {
            "receipt_path": str(sweep_receipt_path),
            "receipt_sha256": sweep_receipt_sha,
            "status": "INCOMPLETE_NOT_USED_AS_COMPLETE_GUARD_SET",
            "full_sweep_guard_coverage_claimed": False,
        },
    }


def prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    pins = {}
    def pin(name, path):
        path = Path(path)
        h = sha256(path)
        pins[name] = {"path": str(path), "sha256": h}
        return h

    if sha256(SWEEPER) != SWEEPER_SHA256:
        raise ValueError("pinned owner-350 source changed")
    base_path, trial_path = Path(args.base_npz), Path(args.trial_npz)
    old_pairs_path, old_seed_report_path = Path(args.existing_pairs), Path(args.existing_seed_report)
    witness_path, audit_path = Path(args.witness_pairs), Path(args.audit_summary)
    context_template_path, sweep_receipt_path = Path(args.context_template), Path(args.sweep_receipt)
    base_sha = pin("base_candidate", base_path)
    trial_sha = pin("trial_candidate", trial_path)
    old_pairs_sha = pin("existing_pairs", old_pairs_path)
    old_seed_sha = pin("existing_seed_preflight", old_seed_report_path)
    witness_sha = pin("exact_witness_pairs", witness_path)
    audit_sha = pin("audit130_summary", audit_path)
    context_sha = pin("context_template", context_template_path)
    sweep_sha = pin("350_receipt", sweep_receipt_path)

    with np.load(base_path, allow_pickle=False) as z:
        base_points = np.asarray(z["points_world_m"])
        base_faces = np.asarray(z["source_O_faces"])
    with np.load(trial_path, allow_pickle=False) as z:
        trial_points = np.asarray(z["points_world_m"])
        trial_faces = np.asarray(z["source_O_faces"])
    if (base_points.dtype != np.float32 or trial_points.dtype != np.float32
            or base_points.shape != trial_points.shape or not np.isfinite(base_points).all()
            or not np.isfinite(trial_points).all()):
        raise ValueError("base/trial points must be same-domain finite Float32")
    if (base_faces.dtype != trial_faces.dtype or base_faces.shape != trial_faces.shape
            or base_faces.tobytes() != trial_faces.tobytes()):
        raise ValueError("base/trial topology bytes differ")
    if base_faces.ndim != 2 or base_faces.shape[1] != 3 or base_faces.dtype.kind not in "iu":
        raise ValueError("base topology is invalid")
    faces = np.asarray(base_faces, dtype=np.int64)
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(base_points):
        raise ValueError("base face indices are out of range")

    old = canonical_pairs(np.load(old_pairs_path, allow_pickle=False), len(faces))
    old_report = json.loads(old_seed_report_path.read_text())
    if (old_report.get("schema") != SEED_SCHEMA
            or old_report.get("status") != "PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY"
            or old_report.get("zero_step_feasible") is not True
            or old_report.get("candidate_sha256") != base_sha
            or old_report.get("output_pairs_path") != str(old_pairs_path)
            or old_report.get("output_pairs_sha256") != old_pairs_sha
            or int(old_report.get("merged_unique_pair_count", -1)) != len(old)
            or old_report.get("pair_owner_sha256") != EXPECTED_PAIR_OWNER_SHA256):
        raise ValueError("historical seeds are not certified for this exact base/owner")

    witnesses = canonical_pairs(np.load(witness_path, allow_pickle=False), len(faces))
    audit_report = json.loads(audit_path.read_text())
    if (audit_report.get("candidate_sha256") != trial_sha
            or audit_report.get("complete_surface_audit") is not True
            or int(audit_report.get("exact_PLC_surface_pairs", -1)) != len(witnesses)):
        raise ValueError("exact 130 witness array does not match its complete audit summary")
    if len(witnesses) > args.max_new_witness_pairs:
        raise ValueError(f"exact witness cap exceeded: {len(witnesses)} > {args.max_new_witness_pairs}")

    sweep = json.loads(sweep_receipt_path.read_text())
    sweep_evidence = validate_incomplete_sweep(
        sweep, expected_base_sha=base_sha, expected_trial_sha=trial_sha,
        expected_seed_sha=old_pairs_sha, expected_witness_sha=witness_sha)
    owner_path = Path(sweep["pair_owner_path"])
    owner_sha = sweep["pair_owner_sha256"]
    if owner_sha != EXPECTED_PAIR_OWNER_SHA256:
        raise ValueError("350 receipt does not pin the expected 026 SAT owner")
    owner_sha_actual = pin("026_pair_owner", owner_path)
    if owner_sha_actual != owner_sha:
        raise ValueError("026 owner path/hash in 350 receipt is stale")

    sweep_output_pins = {}
    potential_path = sweep_receipt_path.parent / "self-pairs-potential.npy"
    unclassified_path = sweep_receipt_path.parent / "self-pairs-unclassified.npy"
    potential_sha = pin("350_potential_pairs", potential_path)
    unclassified_sha = pin("350_unclassified_pairs", unclassified_path)
    if (potential_sha != sweep.get("potential_pair_array_sha256")
            or unclassified_sha != sweep.get("unclassified_pair_array_sha256")):
        raise ValueError("350 output-array hashes do not match its receipt")
    potential = np.load(potential_path, allow_pickle=False)
    unclassified = np.load(unclassified_path, allow_pickle=False)
    if (potential.dtype.kind not in "iu" or potential.ndim != 2 or potential.shape[1] != 2
            or potential.shape != tuple(sweep["potential_pair_array_shape"])
            or unclassified.shape != potential.shape or not np.array_equal(unclassified, potential)):
        raise ValueError("350 cap-blocked output arrays do not match the recorded potential set")
    require_pairs_present(old, potential, len(faces))
    require_pairs_present(witnesses, potential, len(faces))
    sweep_output_pins.update({
        "potential_pair_count": int(len(potential)),
        "potential_pairs_sha256": potential_sha,
        "all_existing_seeds_retained": True,
        "all_exact_witnesses_retained": True,
        "classification_complete": False,
    })

    old_set = {tuple(map(int, p)) for p in old}
    new_witnesses = np.asarray([p for p in witnesses if tuple(map(int, p)) not in old_set], dtype=np.int64).reshape((-1, 2))
    owner = load_pinned_owner(owner_path, owner_sha)
    row_certificate = certify_new_pairs_zero_step(
        owner, base_points.astype(np.float64) * 1000.0, faces, new_witnesses,
        timeout_seconds=args.owner_timeout_seconds,
        max_pairs=args.max_new_witness_pairs,
        batch_size=args.batch_size,
    )
    merged, merge_report = union_seed_pairs(old, witnesses, len(faces))
    if not merge_report["all_exact_witnesses_retained"]:
        raise AssertionError("merged self-pair input dropped an exact witness")

    seed_path = out / "self-pairs-input.npy"
    np.save(seed_path, merged, allow_pickle=False)
    seed_sha = sha256(seed_path)
    preflight = make_seed_preflight(
        candidate_path=base_path, candidate_sha=base_sha,
        seed_path=seed_path, seed_sha=seed_sha, pair_count=len(merged),
        pair_owner_path=owner_path, pair_owner_sha=owner_sha,
        audit_path=audit_path, audit_sha=audit_sha, audit_pairs_sha=witness_sha,
        audit_pair_count=len(witnesses), old_seed_report_path=old_seed_report_path,
        old_seed_report_sha=old_seed_sha, sweep_receipt_path=sweep_receipt_path,
        sweep_receipt_sha=sweep_sha, certification=row_certificate,
    )
    preflight_path = out / "seed-preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    context = json.loads(context_template_path.read_text())
    if context.get("candidate_sha256") != base_sha or context.get("candidate") != str(base_path):
        raise ValueError("constraint context template is stale for this exact base")
    guard_npz = Path(context["guard_npz"])
    guard_report_path = Path(context["guard_report"])
    guard_npz_sha = pin("fixed_guard_npz", guard_npz)
    pin("fixed_guard_report", guard_report_path)
    guard_report = json.loads(guard_report_path.read_text())
    if (guard_report.get("base_candidate_sha256") != base_sha
            or guard_report.get("output_npz_sha256") != guard_npz_sha):
        raise ValueError("fixed guard is stale for the selected base")
    pins["350_builder"] = {"path": str(SWEEPER), "sha256": SWEEPER_SHA256}
    context.update({
        "seed_pairs_path": str(seed_path),
        "seed_pairs_sha256": seed_sha,
        "seed_report": str(preflight_path),
        "seed_report_sha256": sha256(preflight_path),
        "exact_witness_source_path": str(witness_path),
        "exact_witness_source_sha256": witness_sha,
        "swept_receipt_path": str(sweep_receipt_path),
        "swept_receipt_sha256": sweep_sha,
        "swept_pair_coverage_complete": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
    })
    context_out = out / "constraint-context.json"
    context_out.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "PREPARED_SELECTED_WITNESS_SEEDS_ONLY",
        "native_admission": False,
        "accepted_physical_steps": 0,
        "swept_pair_coverage_complete": False,
        "seed_input_path": str(seed_path),
        "seed_input_sha256": seed_sha,
        "seed_input_pair_count": int(len(merged)),
        "merge": merge_report,
        "targeted_witness_row_certification": row_certificate,
        "swept_receipt_disposition": sweep_evidence,
        "swept_output_check": sweep_output_pins,
        "context_path": str(context_out),
        "context_sha256": sha256(context_out),
        "pins": pins,
    }
    manifest_path = out / "preparation-receipt.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not all(sha256(v["path"]) == v["sha256"] for v in pins.values()):
        raise ValueError("an input/source changed while preparing the seed adapter")
    return manifest_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-npz", required=True)
    p.add_argument("--trial-npz", required=True)
    p.add_argument("--existing-pairs", required=True)
    p.add_argument("--existing-seed-report", required=True)
    p.add_argument("--witness-pairs", required=True)
    p.add_argument("--audit-summary", required=True)
    p.add_argument("--context-template", required=True)
    p.add_argument("--sweep-receipt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-witness-pairs", type=int, default=5000)
    p.add_argument("--owner-timeout-seconds", type=float, default=120.0)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_new_witness_pairs < 1 or args.batch_size < 1:
        raise ValueError("pair and batch caps must be positive")
    if not math.isfinite(args.owner_timeout_seconds) or args.owner_timeout_seconds <= 0:
        raise ValueError("owner timeout must be finite and positive")
    path = prepare(args)
    print(json.dumps({"status": "PREPARED_SELECTED_WITNESS_SEEDS_ONLY", "receipt": str(path),
                      "swept_pair_coverage_complete": False,
                      "native_admission": False, "accepted_physical_steps": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

