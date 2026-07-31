#!/usr/bin/env python3
"""Generate the public capability matrix from its machine-readable registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(document: dict[str, object]) -> str:
    statuses = document.get("statuses")
    capabilities = document.get("capabilities")
    if document.get("schema") != 1 or not isinstance(statuses, list):
        raise ValueError("capability registry schema is invalid")
    if not isinstance(capabilities, list):
        raise ValueError("capability registry has no capability list")
    allowed = set(statuses)
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
        ids.add(identifier)
        counts[status] += 1
        rows.append(
            f"| `{identifier}` | **{status}** | `{owner}` | "
            f"`{artifact}` | {scope} |"
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
            "A status is a product claim boundary, not a roadmap estimate.",
            "",
            f"Current registry: {summary}.",
            "",
            "| Capability | Status | Owning check | Evidence artifact | Exact scope |",
            "|---|---|---|---|---|",
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
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render(json.loads(arguments.registry.read_text()))
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
