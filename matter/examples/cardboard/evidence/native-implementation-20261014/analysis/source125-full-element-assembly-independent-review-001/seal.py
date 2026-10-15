#!/usr/bin/env python3
"""Verify upstream manifests and seal independent review 150-002."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/Users/home/cardboard-implementation-20261014")
OUT = BASE / "analysis/source125-full-element-assembly-independent-review-001"
V1 = BASE / "analysis/source125-full-element-assembly-fp64-audit-150-001"
V2 = BASE / "analysis/source125-full-element-assembly-fp64-verification-150-002"
TANGENT = BASE / "analysis/source125-full-tangent-fp64-audit-147-001"
TANGENT_REVIEW = BASE / "analysis/source125-full-tangent-independent-review-001"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_manifest(root: Path, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for rel, expected in data["files"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"manifest file mismatch: {root / rel}")
    for raw, expected in data["inputs"].items():
        if sha(Path(raw)) != expected:
            raise ValueError(f"manifest input mismatch: {raw}")
    return data


def main() -> None:
    v1 = check_manifest(V1, V1 / "manifest.json")
    v2 = check_manifest(V2, V2 / "manifest.json")
    tangent = check_manifest(TANGENT, TANGENT / "manifest.json")
    review = json.loads((TANGENT_REVIEW / "manifest.json").read_text(encoding="utf-8"))
    for raw, expected in review["bound_inputs"].items():
        if sha(Path(raw)) != expected:
            raise ValueError(f"tangent review input mismatch: {raw}")
    review_files = {
        "README.md": review["readme_sha256"],
        "report.json": review["report_sha256"],
        "verification-details.json": review["details_sha256"],
        "verify.py": review["verifier_sha256"],
    }
    for rel, expected in review_files.items():
        if sha(TANGENT_REVIEW / rel) != expected:
            raise ValueError(f"tangent review artifact mismatch: {rel}")
    rerun_manifest = BASE / "analysis/review147-independent-rerun-001/manifest.json"
    if sha(rerun_manifest) != review["rerun_manifest_sha256"]:
        raise ValueError("tangent review rerun manifest mismatch")

    direct_inputs = [
        V1 / "manifest.json", V1 / "report.json", V1 / "audit.py",
        V2 / "manifest.json", V2 / "report.json", V2 / "verify.py",
        TANGENT / "manifest.json", TANGENT / "report.json", TANGENT / "matrices.npz",
        TANGENT_REVIEW / "manifest.json", TANGENT_REVIEW / "report.json",
        TANGENT_REVIEW / "verification-details.json",
        BASE / "snapshots/source-status5-vtccd-diagnostics-125/matter/tools/cardboard_cached_operator_audit.py",
        BASE / "solver/two-stage-first-failure-rich-fd-419/operator-capture",
        BASE / "analysis/captured-base-state-cauchy-419-001/A_free_csr.npz",
        BASE / "analysis/review147-independent-rerun-001/manifest.json",
    ]
    input_rows = [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha(p)}
        for p in direct_inputs
    ]
    input_doc = {
        "schema": "numi.cardboard.full_element_assembly_independent_review_inputs.v1",
        "source_150_files_checked": len(v1["files"]),
        "source_150_inputs_checked": len(v1["inputs"]),
        "verification_150_files_checked": len(v2["files"]),
        "verification_150_inputs_checked": len(v2["inputs"]),
        "tangent_147_files_checked": len(tangent["files"]),
        "tangent_147_inputs_checked": len(tangent["inputs"]),
        "direct_inputs": input_rows,
    }
    (OUT / "inputs.json").write_text(json.dumps(input_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file()):
        if p.name in {"manifest.json", "seal.json"}:
            continue
        if p.is_symlink():
            raise ValueError(f"refusing to seal symlink {p}")
        files.append({"path": str(p.relative_to(OUT)), "size_bytes": p.stat().st_size, "sha256": sha(p)})
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "numi.cardboard.independent_audit_package_manifest.v1",
        "package": "source125-full-element-assembly-independent-review-001",
        "files": files,
        "file_count": len(files),
        "inputs_file_sha256": sha(OUT / "inputs.json"),
        "verdict": "PASS_WITH_SCOPED_LIMITATIONS" if report["scalar_reconstruction"]["stated_gates_reproduced"] else "FAIL",
        "native_or_physical_qualification": False,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = {
        "schema": "numi.cardboard.independent_audit_package_seal.v1",
        "package": manifest["package"],
        "manifest_sha256": sha(manifest_path),
        "inputs_sha256": sha(OUT / "inputs.json"),
        "file_count_excluding_manifest_and_seal": len(files),
    }
    seal_path = OUT / "seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**seal, "seal_sha256": sha(seal_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
