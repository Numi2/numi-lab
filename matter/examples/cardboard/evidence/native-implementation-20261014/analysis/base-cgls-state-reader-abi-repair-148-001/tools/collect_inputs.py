#!/usr/bin/env python3
"""Hash the immutable source and run evidence used by reader repair 148."""
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


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def main() -> None:
    sources = [
        BASE / "snapshots/source-cgls-state-sidecar-144/frozen-source.json",
        BASE / "snapshots/source-cgls-state-sidecar-144/matter/include/numi/matter/shared.h",
        BASE / "snapshots/source-cgls-state-sidecar-144/matter/src/runtime.mm",
        BASE / "snapshots/source-cgls-primary-policy-143/frozen-source.json",
        BASE / "snapshots/source-cgls-primary-policy-143/matter/src/runtime.mm",
        BASE / "analysis/base-cgls-state-capture-overlay-144-001/package-seal.json",
        BASE / "analysis/base-cgls-state-capture-overlay-144-001/tools/read_state_capture.py",
    ]
    capture_455 = BASE / "solver/cgls-state-sidecar-observer-455"
    capture_453 = BASE / "solver/cgls-primary-after48-control1-453"
    run_files = []
    for label, root in (("453", capture_453), ("455", capture_455)):
        files = sorted(p for p in root.rglob("*") if p.is_file())
        run_files.append({
            "run": label,
            "files": [record(path) for path in files],
        })
    operator = BASE / "solver/two-stage-first-failure-rich-fd-419/operator-capture"
    if not all(path.is_file() for path in sources) or not operator.is_file():
        raise SystemExit("one or more required source/capture inputs are missing")
    result = {
        "schema": "numi.matter.base_cgls_state_reader_repair_inputs.v1",
        "package": str(PACKAGE),
        "source_inputs": [record(path) for path in sources],
        "operator_capture_419": record(operator),
        "native_runs": run_files,
        "native_455_state_sidecar": {
            "manifest": record(capture_455 / "base-cgls.bin.state.json"),
            "payload": record(capture_455 / "base-cgls.bin.state.bin"),
        },
        "scope": "Reader and audit package only; source snapshots, native runs, and payloads are read-only inputs.",
    }
    output = PACKAGE / "inputs.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest(output), "source_count": len(sources),
                      "run_files": {entry["run"]: len(entry["files"]) for entry in run_files}}, sort_keys=True))


if __name__ == "__main__":
    main()
