#!/usr/bin/env python3
"""Generate the public capability matrix from its machine-readable registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_evidence(
    document: dict[str, object],
    root: Path,
) -> dict[str, tuple[str, str, str]]:
    manifests = document.get("evidence_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("capability registry has no evidence manifests")
    evidence: dict[str, tuple[str, str, str]] = {}
    for relative in manifests:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("evidence manifest path is invalid")
        path = root / relative
        raw = json.loads(path.read_text())
        revision = raw.get("source_revision")
        captured = raw.get("captured_at")
        checks = raw.get("checks")
        if (
            raw.get("schema") != 1
            or not isinstance(revision, str)
            or not revision
            or not isinstance(captured, str)
            or not captured
            or not isinstance(checks, dict)
        ):
            raise ValueError(f"invalid evidence manifest: {relative}")
        for check, result in checks.items():
            if not isinstance(check, str) or result != "passed":
                continue
            candidate = (captured, revision, relative)
            if check not in evidence or candidate > evidence[check]:
                evidence[check] = candidate
    return evidence


def render(document: dict[str, object], root: Path) -> str:
    statuses = document.get("statuses")
    capabilities = document.get("capabilities")
    if document.get("schema") != 1 or not isinstance(statuses, list):
        raise ValueError("capability registry schema is invalid")
    if not isinstance(capabilities, list):
        raise ValueError("capability registry has no capability list")
    allowed = set(statuses)
    evidence = load_evidence(document, root)
    ids: set[str] = set()
    rows: list[str] = []
    counts = {status: 0 for status in statuses}
    for raw in capabilities:
        if not isinstance(raw, dict):
            raise ValueError("capability entry is not an object")
        identifier = raw.get("id")
        status = raw.get("status")
        owner = raw.get("owner")
        artifact = raw.get("artifact")
        scope = raw.get("scope")
        if (
            not isinstance(identifier, str)
            or identifier in ids
            or status not in allowed
            or not isinstance(owner, str)
            or not isinstance(artifact, str)
            or not isinstance(scope, str)
            or "|" in scope
        ):
            raise ValueError(f"invalid capability entry: {identifier!r}")
        if status == "qualified" and owner not in evidence:
            raise ValueError(
                f"qualified capability has no passing evidence: {identifier}"
            )
        ids.add(identifier)
        counts[status] += 1
        evidence_label = "none"
        if owner in evidence:
            captured, revision, path = evidence[owner]
            evidence_label = f"`{revision}` at {captured} (`{path}`)"
        rows.append(
            f"| `{identifier}` | **{status}** | `{owner}` | "
            f"`{artifact}` | {evidence_label} | {scope} |"
        )
    summary = ", ".join(
        f"{status}: {counts[status]}" for status in statuses
    )
    return "\n".join(
        [
            "<!-- GENERATED FILE: python/generate_capability_matrix.py -->",
            "# MetalRobo capability matrix",
            "",
            "This file is generated from `schemas/capability_matrix.json`.",
            "A status is a product claim boundary, not a roadmap estimate. A",
            "qualified row is rejected unless an evidence manifest records its",
            "owning check as passed.",
            "",
            f"Current registry: {summary}.",
            "",
            "| Capability | Status | Owning check | Executable | Last evidence | Exact scope |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "Status meanings:",
            "",
            "- **qualified**: an owning check exercises the stated product path.",
            "- **implemented**: code exists, but the full competitive acceptance gate is not published.",
            "- **experimental**: focused research or diagnostic path; not a production promise.",
            "- **unsupported**: compilation or API must not imply this capability exists.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render(
        json.loads(arguments.registry.read_text()),
        arguments.root,
    )
    if arguments.check:
        if not arguments.output.is_file():
            raise SystemExit("generated capability matrix is missing")
        if arguments.output.read_text() != rendered:
            raise SystemExit("generated capability matrix is stale")
        return 0
    arguments.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
