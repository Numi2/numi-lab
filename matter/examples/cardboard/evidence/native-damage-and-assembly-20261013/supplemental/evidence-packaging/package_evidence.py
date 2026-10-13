#!/usr/bin/env python3
"""Build a content-addressed cardboard evidence record from retained cases.

The output is a derived copy. Source checkouts and case directories are read-only.
On rerun, selected run folders are rediscovered and the output directory is replaced.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path(
    "/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/"
    "native-damage-and-assembly-20261013"
)

CASE_ROOTS = [
    ("bulk/source-point-003", "bulk/source-point-003"),
    ("bulk/source-point-005", "bulk/source-point-005"),
    ("bulk/source-point-006", "bulk/source-point-006"),
    ("bulk/source-point-seeded-1", "bulk/source-point-seeded-1"),
    ("bulk/source-point-seeded-2", "bulk/source-point-seeded-2"),
    ("bulk/source-point-seeded-4", "bulk/source-point-seeded-4"),
    ("bulk/fem-coupling-001", "bulk/fem-coupling-001"),
    ("cohesive/native-003", "cohesive/native-003"),
    ("native-tests-004", "native-tests-004"),
    ("native-tests-005", "native-tests-005"),
    ("native-tests-006", "native-tests-006"),
    ("native-tests-007", "native-tests-007"),
    ("native-tests-008", "native-tests-008"),
    ("box/native-002", "box/native-002"),
    ("solver/armijo-001", "solver/armijo-001"),
    ("solver/armijo-004", "solver/armijo-004"),
    ("solver/krylov1024-prefix-001", "solver/krylov1024-prefix-001"),
    ("solver/mesh-armijo-001", "solver/mesh-armijo-001"),
    ("solver/mesh-armijo-002", "solver/mesh-armijo-002"),
    ("solver/mesh-armijo-003", "solver/mesh-armijo-003"),
    ("solver/converged-positive-noop-001", "solver/converged-positive-noop-001"),
]

# Include later ledgers when present, without requiring every optional case.
OPTIONAL_CASE_ROOTS = [
    ("bulk/source-point-004", "bulk/source-point-004"),
    ("native-tests-009", "native-tests-009"),
    ("native-tests-010/cohesive-reset", "native-tests-010/cohesive-reset"),
    ("native-tests-010/capsule-box", "native-tests-010/capsule-box"),
    ("native-tests-011/cohesive-reset", "native-tests-011/cohesive-reset"),
    ("native-tests-011/capsule-box", "native-tests-011/capsule-box"),
    ("scoring/armijo-001", "scoring/armijo-001"),
    ("scoring/armijo001", "scoring/armijo001"),
    ("scoring/armijo-002", "scoring/armijo-002"),
    ("scoring/armijo-003", "scoring/armijo-003"),
    ("box/native-003", "box/native-003"),
    ("box/native-004", "box/native-004"),
    ("box/native-005", "box/native-005"),
    ("solver/mesh-armijo-004", "solver/mesh-armijo-004"),
    ("solver/operator-fd-baseline-001", "solver/operator-fd-baseline-001"),
    ("solver/operator-fd-001", "solver/operator-fd-001"),
    ("solver/operator-fd-002", "solver/operator-fd-002"),
    ("solver/operator-fd-003", "solver/operator-fd-003"),
    ("solver/operator-fd-scope-034", "solver/operator-fd-scope-034"),
    ("native-tests-012/cohesive-reset", "native-tests-012/cohesive-reset"),
    ("native-tests-012/capsule-box", "native-tests-012/capsule-box"),
    ("native-tests-013/cohesive-reset", "native-tests-013/cohesive-reset"),
    ("native-tests-013/capsule-box", "native-tests-013/capsule-box"),
    ("solver/operator-fd-scope-036", "solver/operator-fd-scope-036"),
    ("solver/local-projection-accuracy-035", "solver/local-projection-accuracy-035"),
]
PACKAGE_LEVEL_ARTIFACTS = [
    "bulk/seeded-sensitivity.json",
    "bulk/seeded-sensitivity-docs.patch",
    "bulk/evidence-summary.json",
    "bulk/evidence-frontier.json",
    "bulk/final-outcomes-delta.patch",
    "docs-patches/robertsson-normalization.patch",
    "evidence-packaging/refresh_frontier.py",
    "evidence-packaging/package_evidence.py",
    "evidence-packaging/plot_native_acceptance.py",
    "evidence-packaging/native-acceptance-coverage.json",
    "evidence-packaging/native-acceptance-coverage.png",
    "host-environment.json",
    "build-integrated-027.log",
    "build-integrated-028.log",
    "build-cohesive-reset-029.log",
    "compiler-regression-build-001.log",
    "compiler-analysis-regression-007.log",
    "cohesive-reset-cpu-029.log",
    "cohesive-reset-cpu-029-invocation-audit.json",
    "build-operator-fd-030.log",
    "build-operator-fd-031.log",
    "build-operator-fd-scale-033.log",
    "build-operator-fd-mass-034.log",
    "build-policy-revision-036.log",
    "build-local-projection-035.log",
    "build-cohesive-fixture-032.log",
    "build_local_accuracy035.py",
    "run_local_accuracy035.py",
    "operator-fd/preexecution-queue-adjustment.json",
    "solver/frozen032-source/README-source-binding.txt",
    "solver/frozen033.json",
    "solver/frozen034.json",
    "solver/frozen036.json",
    "operator-fd/accepted-history-yield-audit.py",
    "operator-fd/accepted-history-yield-audit.json",
    "operator-fd/compare_native_outcomes.py",
    "operator-fd/outcome-comparison.json",
    "operator-fd/compare_scaled_epsilon.py",
    "operator-fd/scaled-epsilon-comparison.json",
    "operator-fd/source-diff-031-vs-033.patch",
    "operator-fd/local-accuracy-build035.json",
    "operator-fd/projection-tolerance-2e-6.patch",
    "operator-fd/projection-tolerance-2e-6-review.md",
    "material-tests/cpu-001.log",
    "material/cohesive-validation/cpu-001.log",
    "robertsson-fem/cpu-001.log",
    "box/controller-cpu-003.log",
]

# These are reviewable run records, data products, and figures. Mesh/OBJ exports,
# compiled packages, runtime executables, and Metal libraries are intentionally
# excluded from this direct-copy layer.
RECORD_EXTENSIONS = {
    ".json", ".csv", ".tsv", ".txt", ".log", ".md", ".png", ".jpg",
    ".jpeg", ".svg", ".pdf", ".patch", ".yaml", ".yml", ".toml",
    ".trace", ".fail", ".failure", ".stderr", ".stdout",
}
EXCLUDED_DIRS = {
    ".git", "inputs", "bound-inputs", "source-inputs", "supplemental-inputs",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {
    ".obj", ".nmatterpack", ".metallib", ".dylib", ".so", ".a", ".o",
    ".air", ".bc", ".exe", ".app", ".zip", ".gz",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def is_text_source(data: bytes) -> bool:
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def candidate_archive_roots(leaf: Path, group_root: Path, supplemental: bool = False) -> list[Path]:
    names = ("supplemental-inputs",) if supplemental else (
        "inputs", "bound-inputs", "source-inputs", "sealed-inputs"
    )
    roots: list[Path] = []
    current = leaf
    while True:
        for name in names:
            p = current / name
            if p.is_dir() and p not in roots:
                roots.append(p)
        if current == group_root or group_root not in current.parents:
            break
        current = current.parent
    return roots


def locate_input(leaf: Path, group_root: Path, rel: str, supplemental: bool = False) -> Path | None:
    relpath = Path(rel)
    if relpath.is_absolute() or ".." in relpath.parts:
        return None
    for root in candidate_archive_roots(leaf, group_root, supplemental=supplemental):
        candidate = root / relpath
        if candidate.is_file():
            return candidate
    return None


def nearest_supplement_manifest(leaf: Path, group_root: Path) -> Path | None:
    current = leaf
    while True:
        p = current / "supplemental-inputs.json"
        if p.is_file():
            return p
        if current == group_root or group_root not in current.parents:
            return None
        current = current.parent


def discover_predictions(group_root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, dirs, files in os.walk(group_root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        if "prediction.json" in files:
            found.append(Path(directory) / "prediction.json")
    return sorted(found)


def copy_reviewable_artifacts(group_id: str, group_root: Path, stage: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    dest_root = stage / "cases" / group_id
    for directory, dirs, files in os.walk(group_root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        source_dir = Path(directory)
        rel_dir = source_dir.relative_to(group_root)
        for name in sorted(files):
            src = source_dir / name
            suffix = src.suffix.lower()
            if suffix in EXCLUDED_SUFFIXES or suffix not in RECORD_EXTENSIONS:
                continue
            # Never include model geometry or packed/runtime bytes under another name.
            lower = name.lower()
            if lower.endswith((".obj", ".nmatterpack", ".metallib", ".dylib", ".so", ".air")):
                continue
            rel = rel_dir / name
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            digest, size = sha_file(dst)
            copied.append({
                "case_group": group_id,
                "source_path": str(src),
                "package_path": dst.relative_to(stage).as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "kind": "plot" if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
                    else "data_or_run_record",
            })
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    case_roots = CASE_ROOTS + [item for item in OPTIONAL_CASE_ROOTS if (source_root / item[1]).is_dir()]
    packaging_dir = Path(__file__).resolve().parent
    readme = packaging_dir / "README.md"
    if not readme.is_file():
        raise SystemExit(f"Evidence README missing: {readme}")
    if source_root == output or source_root in output.parents:
        if output == source_root:
            raise SystemExit("output cannot replace the evidence root")

    stage = output.with_name(output.name + ".staging")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "source-blobs").mkdir()
    (stage / "cases").mkdir()
    shutil.copyfile(readme, stage / "README.md")

    artifact_records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    blob_by_hash: dict[str, dict[str, Any]] = {}
    missing_groups: list[str] = []

    for rel in PACKAGE_LEVEL_ARTIFACTS:
        src = source_root / rel
        if not src.is_file():
            continue
        dst = stage / "supplemental" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest, size = sha_file(dst)
        artifact_records.append({
            "case_group": "package-level",
            "source_path": str(src),
            "package_path": dst.relative_to(stage).as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "kind": "derived_result_or_docs_patch",
        })

    for group_id, rel_root in case_roots:
        group_root = source_root / rel_root
        if not group_root.is_dir():
            missing_groups.append(group_id)
            continue
        artifact_records.extend(copy_reviewable_artifacts(group_id, group_root, stage))
        predictions = discover_predictions(group_root)
        for prediction_path in predictions:
            leaf = prediction_path.parent
            leaf_rel = leaf.relative_to(group_root).as_posix()
            case_id = group_id if leaf_rel == "." else f"{group_id}/{leaf_rel}"
            pred = load_json(prediction_path)
            if not isinstance(pred, dict):
                case_records.append({"case_id": case_id, "prediction_path": str(prediction_path), "status": "invalid_prediction_json"})
                continue
            pred_sha, pred_size = sha_file(prediction_path)
            receipt_path = leaf / "receipt.json"
            receipt = load_json(receipt_path) if receipt_path.is_file() else None
            result_paths = sorted(leaf.rglob("result.json"))
            result_paths = [p for p in result_paths if not any(part in EXCLUDED_DIRS for part in p.relative_to(leaf).parts)]
            results = []
            for rp in result_paths:
                rd = load_json(rp)
                if isinstance(rd, dict):
                    summary = {k: rd[k] for k in ("status", "accepted_steps", "physical_validation", "failure", "stop_reason") if k in rd}
                    if summary:
                        results.append({"path": rp.relative_to(leaf).as_posix(), **summary})
            case_records.append({
                "case_id": case_id,
                "source_case_path": str(leaf),
                "prediction_sha256": pred_sha,
                "prediction_size_bytes": pred_size,
                "prediction_question": pred.get("question"),
                "prediction_text": pred.get("prediction"),
                "receipt": receipt,
                "receipt_present": receipt_path.is_file(),
                "result_summaries": results,
                "status_note": "Included regardless of pass/fail/completion; no outcome filter is applied.",
            })

            inventories: list[tuple[str, dict[str, str], Path, bool]] = []
            sealed = pred.get("inputs_sha256", {})
            if isinstance(sealed, dict):
                inventories.append(("sealed_prediction", sealed, leaf, False))
            supplement_manifest = nearest_supplement_manifest(leaf, group_root)
            if supplement_manifest:
                supplement = load_json(supplement_manifest)
                supp_hashes = supplement.get("inputs_sha256", {}) if isinstance(supplement, dict) else {}
                if isinstance(supp_hashes, dict):
                    inventories.append(("supplemental_not_original_seal", supp_hashes, supplement_manifest.parent, True))

            for seal_kind, hashes, inventory_leaf, supplemental in inventories:
                for rel, expected in sorted(hashes.items()):
                    path = locate_input(inventory_leaf, group_root, rel, supplemental=supplemental)
                    rec: dict[str, Any] = {
                        "case_id": case_id,
                        "seal_kind": seal_kind,
                        "path": rel,
                        "expected_sha256": expected,
                        "retained_archive_path": None,
                        "actual_sha256": None,
                        "size_bytes": None,
                        "verified": False,
                        "content_included": False,
                        "classification": "missing_from_retained_archive",
                        "blob_path": None,
                    }
                    if path is not None:
                        actual, size = sha_file(path)
                        rec.update({"retained_archive_path": str(path), "actual_sha256": actual, "size_bytes": size, "verified": actual == expected})
                        if actual != expected:
                            rec["classification"] = "hash_mismatch"
                        else:
                            data = path.read_bytes()
                            if is_text_source(data):
                                blob_rel = f"source-blobs/{actual[:2]}/{actual}.txt.gz"
                                blob_path = stage / blob_rel
                                if actual not in blob_by_hash:
                                    blob_path.parent.mkdir(parents=True, exist_ok=True)
                                    blob_path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
                                    blob_by_hash[actual] = {"sha256": actual, "raw_size_bytes": len(data), "gzip_size_bytes": blob_path.stat().st_size, "blob_path": blob_rel}
                                rec.update({"classification": "verified_text_source", "content_included": True, "blob_path": blob_rel})
                            else:
                                rec.update({"classification": "verified_binary_or_non_utf8_runtime_input_not_copied", "content_included": False, "rebuild_required": True})
                                runtime_records.append({
                                    "case_id": case_id,
                                    "seal_kind": seal_kind,
                                    "path": rel,
                                    "sha256": actual,
                                    "size_bytes": size,
                                    "retained_archive_path": str(path),
                                    "included": False,
                                    "rebuild_required": True,
                                    "reason": "Runtime executable, Metal library, packed archive, or other non-text input is hash-recorded but intentionally not copied.",
                                })
                    binding_records.append(rec)

    (stage / "README.md").write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")
    binding_manifest = {
        "schema": "numi.cardboard.publication-input-bindings.v1",
        "source_root": str(source_root),
        "verification_rule": "Every hash in each retained prediction inputs_sha256 and supplemental-inputs.json inputs_sha256 is independently recomputed against the retained input archive. Supplemental entries remain explicitly separate from original seals.",
        "expected_entries": len(binding_records),
        "verified_entries": sum(1 for r in binding_records if r["verified"]),
        "all_entries_verified": bool(binding_records) and all(r["verified"] for r in binding_records),
        "text_source_blobs_unique": len(blob_by_hash),
        "text_source_bytes_raw": sum(v["raw_size_bytes"] for v in blob_by_hash.values()),
        "text_source_bytes_gzip": sum(v["gzip_size_bytes"] for v in blob_by_hash.values()),
        "runtime_inputs_not_copied": len(runtime_records),
        "records": binding_records,
        "unique_blobs": sorted(blob_by_hash.values(), key=lambda x: x["sha256"]),
        "runtime_rebuild_requirements": runtime_records,
    }
    (stage / "input-binding-manifest.json").write_text(json.dumps(binding_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_manifest = {
        "schema": "numi.cardboard.publication-artifacts.v1",
        "policy": "Reviewable predictions, receipts, results, logs, CSV/TSV, plots and failure text are copied byte-for-byte. OBJ, compiled packages, executables and Metal binaries are excluded.",
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
    }
    (stage / "artifact-manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    case_index = {
        "schema": "numi.cardboard.publication-case-index.v1",
        "groups": [{"id": gid, "source_path": str(source_root / rel)} for gid, rel in case_roots if (source_root / rel).is_dir()],
        "missing_groups": missing_groups,
        "prediction_case_count": len(case_records),
        "cases": case_records,
        "preservation_note": "Failures, rejected steps, superseded runs, and incomplete cases are included without pass/fail filtering.",
    }
    (stage / "case-index.json").write_text(json.dumps(case_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_entries = len(binding_records)
    verified_entries = sum(1 for r in binding_records if r["verified"])
    build_report = {
        "schema": "numi.cardboard.publication-package-build.v1",
        "source_root": str(source_root),
        "output_path": str(output),
        "case_groups_requested": len(case_roots),
        "case_groups_found": len(case_roots) - len(missing_groups),
        "missing_groups": missing_groups,
        "prediction_cases": len(case_records),
        "artifact_files_copied": len(artifact_records),
        "input_hash_entries": total_entries,
        "input_hash_entries_verified": verified_entries,
        "input_hash_failures": total_entries - verified_entries,
        "runtime_hashes_retained_but_not_copied": len(runtime_records),
        "unique_text_source_blobs": len(blob_by_hash),
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    (stage / "package-build-report.json").write_text(json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Replace the generated candidate as one directory operation. All source evidence
    # remains untouched; this path only contains outputs of this script.
    if output.exists():
        shutil.rmtree(output)
    os.replace(stage, output)
    print(json.dumps(build_report, indent=2, sort_keys=True))
    return 0 if verified_entries == total_entries and not missing_groups else 2


if __name__ == "__main__":
    raise SystemExit(main())
