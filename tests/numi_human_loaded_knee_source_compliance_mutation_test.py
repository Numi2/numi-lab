#!/usr/bin/env python3
"""Fail-closed mutations for native loaded-knee source-law admission."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


COMPANION_NAME = "HumanPack.loaded-anatomy-knee.source-compliance.v1.json"
EQUALITY_NAME = "myosim-fullbody-joint-equalities-source-compliance.nheq"
LIMIT_NAME = "myosim-fullbody-joint-limits.nhlim"


def canonical(value: Any, *, newline: bool) -> bytes:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def identity_without(value: dict[str, Any], excluded: str) -> str:
    reduced = dict(value)
    del reduced[excluded]
    return hashlib.sha256(canonical(reduced, newline=False)).hexdigest()


def seal(value: dict[str, Any]) -> bytes:
    value["binding_sha256"] = identity_without(value, "binding_sha256")
    return canonical(value, newline=True)


def run_loader(
    cli: Path,
    manifest: Path,
    binding: Path,
    ownership: Path,
    companion: Path,
    equality: Path,
    limit: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(cli),
            str(manifest),
            str(binding),
            str(ownership),
            str(companion),
            str(equality),
            str(limit),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def require_result(
    name: str, result: subprocess.CompletedProcess[str], expected: int
) -> None:
    if result.returncode != expected:
        raise AssertionError(
            f"{name}: expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    detail = result.stdout.strip() if expected == 0 else result.stderr.strip()
    print(f"{name}=passed: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cli", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("ownership", type=Path)
    parser.add_argument("companion", type=Path)
    parser.add_argument("equality", type=Path)
    parser.add_argument("limit", type=Path)
    args = parser.parse_args()

    companion_bytes = args.companion.read_bytes()
    base_companion = json.loads(companion_bytes)
    equality_bytes = args.equality.read_bytes()
    limit_bytes = args.limit.read_bytes()
    if canonical(base_companion, newline=True) != companion_bytes:
        raise AssertionError("source-compliance fixture is not canonical JSON")
    if identity_without(base_companion, "binding_sha256") != base_companion[
        "binding_sha256"
    ]:
        raise AssertionError("source-compliance fixture identity is invalid")

    require_result(
        "baseline",
        run_loader(
            args.cli,
            args.manifest,
            args.binding,
            args.ownership,
            args.companion,
            args.equality,
            args.limit,
        ),
        0,
    )

    def companion_case(
        name: str, mutation: Callable[[dict[str, Any]], None]
    ) -> None:
        value = copy.deepcopy(base_companion)
        mutation(value)
        with tempfile.TemporaryDirectory(prefix=f"loaded-knee-{name}-") as raw:
            companion = Path(raw) / COMPANION_NAME
            companion.write_bytes(seal(value))
            require_result(
                name,
                run_loader(
                    args.cli,
                    args.manifest,
                    args.binding,
                    args.ownership,
                    companion,
                    args.equality,
                    args.limit,
                ),
                1,
            )

    companion_cases: tuple[
        tuple[str, Callable[[dict[str, Any]], None]], ...
    ] = (
        (
            "base-manifest-identity-forgery",
            lambda value: value["base_manifest"].__setitem__(
                "manifest_sha256", "f" * 64
            ),
        ),
        (
            "base-manifest-file-forgery",
            lambda value: value["base_manifest"].__setitem__(
                "file_sha256", "f" * 64
            ),
        ),
        (
            "source-rigid-forgery",
            lambda value: value["source_model"].__setitem__(
                "source_rigid_payload_sha256", "f" * 64
            ),
        ),
        (
            "source-archive-forgery",
            lambda value: value["source_model"].__setitem__(
                "source_archive_sha256", "f" * 64
            ),
        ),
        (
            "equality-byte-identity-forgery",
            lambda value: value["programs"]["joint_equalities"].__setitem__(
                "file_sha256", "f" * 64
            ),
        ),
        (
            "limit-header-claim-forgery",
            lambda value: value["programs"]["joint_limits"].__setitem__(
                "record_bytes", 81
            ),
        ),
        (
            "cross-program-forgery",
            lambda value: value["cross_program"].__setitem__(
                "same_flags", False
            ),
        ),
        (
            "human-runtime-force-claim",
            lambda value: value["ownership"]["human_source_laws"].__setitem__(
                "owns_runtime_constraint_force", True
            ),
        ),
        (
            "matter-source-law-claim",
            lambda value: value["ownership"][
                "matter_runtime_constraints"
            ].__setitem__("authors_source_laws", True),
        ),
        (
            "prepared-state-forgery",
            lambda value: value["prepared_state"].__setitem__(
                "identity_sha256", "f" * 64
            ),
        ),
        (
            "runtime-execution-overclaim",
            lambda value: value["qualification"].__setitem__(
                "runtime_constraint_force_or_state_executed", True
            ),
        ),
        (
            "production-overclaim",
            lambda value: value["qualification"].__setitem__(
                "production_physical_ownership", True
            ),
        ),
        (
            "boundary-forgery",
            lambda value: value.__setitem__("boundary", "production qualified"),
        ),
        (
            "unknown-root-field",
            lambda value: value.__setitem__("unexpected", False),
        ),
        (
            "unknown-program-field",
            lambda value: value["programs"]["joint_equalities"].__setitem__(
                "unexpected", False
            ),
        ),
    )
    for name, mutation in companion_cases:
        companion_case(name, mutation)

    def program_case(name: str, program: str, replacement: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix=f"loaded-knee-{name}-") as raw:
            directory = Path(raw)
            equality = directory / EQUALITY_NAME
            limit = directory / LIMIT_NAME
            equality.write_bytes(
                replacement if program == "equality" else equality_bytes
            )
            limit.write_bytes(replacement if program == "limit" else limit_bytes)
            require_result(
                name,
                run_loader(
                    args.cli,
                    args.manifest,
                    args.binding,
                    args.ownership,
                    args.companion,
                    equality,
                    limit,
                ),
                1,
            )

    for program, original in (
        ("equality", equality_bytes),
        ("limit", limit_bytes),
    ):
        changed_header = bytearray(original)
        changed_header[0] ^= 1
        program_case(f"{program}-header-mutation", program, bytes(changed_header))
        changed_body = bytearray(original)
        changed_body[80] ^= 1
        program_case(f"{program}-body-mutation", program, bytes(changed_body))
        program_case(f"{program}-truncated", program, original[:-1])
        program_case(f"{program}-trailing", program, original + b"\0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
