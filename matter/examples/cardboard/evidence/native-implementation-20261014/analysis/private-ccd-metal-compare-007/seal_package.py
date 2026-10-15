#!/usr/bin/env python3
"""Create the reproducible hash manifest for this source-bound diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099")
RESULT = ROOT / "native-result.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.as_posix() == "manifest.json" or "__pycache__" in rel.parts:
            continue
        if rel.as_posix() == "native-result.json":
            continue
        if path.is_symlink():
            raise SystemExit(f"refusing symlink in package: {rel}")
        entries.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})

    case_manifest = ROOT / "captured-input/captured-ee-cases.json"
    case_payload = ROOT / "captured-input/captured-ee-cases.float4.bin"
    build_manifest = ROOT / "build/build-manifest.json"
    failed_native_output = ROOT / "evidence/failed-native-350/native-result.json"
    try:
        json.loads(failed_native_output.read_text())
        raise SystemExit("retained native-350 output unexpectedly parses as JSON")
    except json.JSONDecodeError as exc:
        failed_native_parse = {
            "path": "evidence/failed-native-350/native-result.json",
            "bytes": failed_native_output.stat().st_size,
            "sha256": sha(failed_native_output),
            "json_parse_error": f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
        }
    native_argv = [
        str(ROOT / "build/run_native"),
        "--metallib", str(ROOT / "build/private_ee_ccd_diagnostic.metallib"),
        "--cases", str(case_payload),
        "--cases-manifest", str(case_manifest),
        "--build-manifest", str(build_manifest),
        "--output", str(RESULT),
    ]
    manifest = {
        "schema": "numi.matter.private_ee_ccd_metal_compare_package.v3",
        "classification": "source-bound native diagnostic harness with validated JSON serialization; package preparation does not dispatch GPU work",
        "source": {
            "snapshot": str(SOURCE),
            "parent_revision": "35bfb518c542a306f563cc364ef8496837068b9a",
            "frozen_manifest_sha256": "5fb22c61c280f576f15677c8cf6dbbc174beceaa225973242f6d53f9c7957446",
            "frozen_file_count_verified": 775,
            "contact_source_sha256": "3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132",
            "extracted_edge_edge_ccd_sha256": "a76e6024194848f935521b10ed72fb4b2938a321636bf06faf7ca02e07fb84c9",
        },
        "cases": {
            "native_capture_ids": [333, 334],
            "count": 4,
            "budgets": [16, 17],
            "manifest": "captured-input/captured-ee-cases.json",
            "manifest_sha256": sha(case_manifest),
            "payload": "captured-input/captured-ee-cases.float4.bin",
            "payload_sha256": sha(case_payload),
        },
        "retained_failed_native_attempt": {
            "attempt": 350,
            "receipt": "evidence/failed-native-350/receipt.json",
            "receipt_exit_code": 0,
            "serialization_status": "invalid_json; not admitted as diagnostic result",
            "output": failed_native_parse,
        },
        "validation": {
            "metal_compile": "PASS: generated AIR and metallib with -std=metal4.0 -O3 -fno-fast-math",
            "host_compile": "PASS: Objective-C++ runner compiled with clang++ -std=c++20 -fobjc-arc",
            "python_tests": "PASS: 5 tests after source-bound rebuild",
            "metal_warnings": "Two pre-existing sharedValid/refinementValid uninitialized warnings in frozen source fgmres.metalinc",
            "native_gpu_dispatch": "NOT RUN",
            "solver_acceptance": "NOT EVALUATED",
            "physics_acceptance": "NOT EVALUATED",
            "report_json_validation": "NSJSONSerialization validates the complete four-case document before output; report-only CPU test emits gpu_result=false",
            "retained_failed_native_350": "Malformed JSON retained under evidence/failed-native-350; not admitted",
        },
        "output_contract": {
            "schema": "numi.matter.private_ee_ccd_metal_diagnostic_result.v2",
            "budget16_parity": "raw bitwise comparison against unchanged production kernel",
            "classifications": ["certified_hit", "certified_miss", "uncertified_fail_closed"],
            "json_self_validation": "full output parsed by NSJSONSerialization and required root/case/iteration structure checked before writing or success",
            "armijo_or_transaction_acceptance": "not evaluated by this diagnostic",
        },
        "native_run_argv": native_argv,
        "build_manifest_sha256": sha(build_manifest),
        "files": entries,
    }
    target = ROOT / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(entries)} files; manifest_sha256={sha(target)}")


if __name__ == "__main__":
    main()
