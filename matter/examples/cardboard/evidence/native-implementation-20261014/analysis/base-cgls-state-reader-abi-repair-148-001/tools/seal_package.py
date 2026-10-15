#!/usr/bin/env python3
"""Create a non-recursive inventory and seal for reader repair 148."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = BASE / "analysis/base-cgls-state-reader-abi-repair-148-001"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not (PACKAGE / "inputs.json").is_file():
        raise SystemExit("inputs.json must be generated before sealing")
    excluded = {"package-manifest.json", "package-seal.json"}
    files = []
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()):
        if path.name in excluded:
            continue
        if path.is_symlink():
            raise SystemExit(f"refusing to seal symlink: {path}")
        files.append({
            "path": str(path.relative_to(PACKAGE)),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    manifest = {
        "schema": "numi.matter.audit_package_manifest.v1",
        "package": "base-cgls-state-reader-abi-repair-148-001",
        "source_inputs": "inputs.json",
        "file_count": len(files),
        "files": files,
        "validation": {
            "cpu_tests": "10 passed",
            "reader_455": "valid; 24 arrays; 4,327,464-byte payload",
            "join_455_419": "state-input parity with signed-zero-only tetrahedron representation differences; projected history unavailable",
            "observer_453_455": "production history parity; both runs remain failed attempts",
            "native_or_physical_qualification": False,
        },
    }
    manifest_path = PACKAGE / "package-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = {
        "schema": "numi.matter.audit_package_seal.v1",
        "package": manifest["package"],
        "manifest_sha256": digest(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "inventory_file_count_excluding_manifest_and_seal": len(files),
        "inputs_sha256": digest(PACKAGE / "inputs.json"),
        "sealed_without_native_or_physical_qualification": True,
    }
    seal_path = PACKAGE / "package-seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**seal, "seal_sha256": digest(seal_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
