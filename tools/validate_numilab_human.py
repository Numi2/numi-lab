#!/usr/bin/env python3
"""Fail-closed NumiLab Human whole-body evidence validator.

The validator deliberately separates structural, prescribed preflight,
one-step, sustained, and independently validated evidence.  It never upgrades
an image, a source manifest, or an unversioned transcript into a stronger
mechanics claim than the manifest assigns to it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST = "docs/numi-human-whole-body-validation.v1.json"


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    level: str
    path: str
    details: tuple[str, ...]


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_values(value: Any, pointer: str) -> list[Any]:
    if pointer == "":
        return [value]
    if not pointer.startswith("/"):
        raise ValidationError(f"JSON pointer must start with '/': {pointer}")
    tokens = [_decode_pointer_token(token) for token in pointer[1:].split("/")]
    current = [value]
    for token in tokens:
        resolved: list[Any] = []
        for candidate in current:
            if token == "*":
                if isinstance(candidate, dict):
                    resolved.extend(candidate.values())
                elif isinstance(candidate, list):
                    resolved.extend(candidate)
                continue
            if isinstance(candidate, dict) and token in candidate:
                resolved.append(candidate[token])
            elif isinstance(candidate, list):
                try:
                    index = int(token)
                except ValueError:
                    continue
                if 0 <= index < len(candidate):
                    resolved.append(candidate[index])
        current = resolved
    return current


def _format_check(check: dict[str, Any]) -> str:
    return f"{check.get('pointer', '<text>')} {check['op']} {check.get('value', '')}".strip()


def evaluate_json_check(document: Any, check: dict[str, Any]) -> tuple[bool, str]:
    pointer = check.get("pointer", "")
    values = resolve_values(document, pointer)
    op = check["op"]
    expected = check.get("value")
    if op == "exists":
        passed = bool(values)
    elif op == "equals":
        passed = bool(values) and all(value == expected for value in values)
    elif op == "not_equals":
        passed = bool(values) and all(value != expected for value in values)
    elif op == "minimum":
        passed = bool(values) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= expected
            for value in values
        )
    elif op == "maximum":
        passed = bool(values) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value <= expected
            for value in values
        )
    elif op == "length_equals":
        passed = len(values) == 1 and hasattr(values[0], "__len__") and len(values[0]) == expected
    elif op == "all_values_equal":
        passed = len(values) == 1 and isinstance(values[0], dict) and bool(values[0]) and all(
            item == expected for item in values[0].values()
        )
    else:
        raise ValidationError(f"unsupported JSON check operation: {op}")
    observed = values if len(values) <= 4 else [*values[:4], f"... {len(values)} values"]
    return passed, f"{_format_check(check)} observed={observed!r}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_evidence(root: Path, evidence: dict[str, Any]) -> EvidenceResult:
    relative_path = evidence["path"]
    path = root / relative_path
    level = evidence["level"]
    if not path.is_file():
        return EvidenceResult("missing", level, relative_path, ("evidence file is missing",))
    failures: list[str] = []
    try:
        if evidence["format"] == "json":
            document = json.loads(path.read_text(encoding="utf-8"))
            for check in evidence.get("checks", []):
                passed, detail = evaluate_json_check(document, check)
                if not passed:
                    failures.append(detail)
        elif evidence["format"] == "text":
            document = path.read_text(encoding="utf-8")
            for pattern in evidence.get("required_regex", []):
                if re.search(pattern, document, flags=re.MULTILINE) is None:
                    failures.append(f"required regex did not match: {pattern}")
            for pattern in evidence.get("forbidden_regex", []):
                if re.search(pattern, document, flags=re.MULTILINE) is not None:
                    failures.append(f"forbidden regex matched: {pattern}")
        else:
            raise ValidationError(f"unsupported evidence format: {evidence['format']}")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        failures.append(f"could not parse evidence: {error}")
    for artifact in evidence.get("artifacts", []):
        artifact_path = root / artifact["path"]
        if not artifact_path.is_file():
            failures.append(f"artifact is missing: {artifact['path']}")
        else:
            observed = sha256(artifact_path)
            if observed != artifact["sha256"]:
                failures.append(
                    f"artifact hash mismatch: {artifact['path']} expected={artifact['sha256']} observed={observed}"
                )
    return EvidenceResult(
        "contradicted" if failures else "verified",
        level,
        relative_path,
        tuple(failures),
    )


def _git_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def requirement_result(
    requirement: dict[str, Any],
    evidence_results: dict[str, EvidenceResult],
    level_rank: dict[str, int],
) -> dict[str, Any]:
    if not requirement.get("required", True):
        return {"status": "not_applicable", "evidence": [], "details": []}
    evidence_ids = requirement.get("evidence", [])
    minimum_level = requirement["minimum_level"]
    if not evidence_ids:
        return {
            "status": "missing",
            "minimum_level": minimum_level,
            "evidence": [],
            "details": [requirement.get("gap", "no evidence is registered")],
        }
    candidates = [(identifier, evidence_results[identifier]) for identifier in evidence_ids]
    mode = requirement.get("evidence_mode", "any")
    if mode not in ("any", "all"):
        raise ValidationError(f"unsupported evidence mode: {mode}")
    if mode == "all":
        contradicted = [
            (identifier, result) for identifier, result in candidates
            if result.status == "contradicted"
        ]
        if contradicted:
            return {
                "status": "contradicted",
                "minimum_level": minimum_level,
                "evidence": [identifier for identifier, _ in contradicted],
                "details": [detail for _, result in contradicted for detail in result.details]
                    + [requirement.get("gap", "registered evidence failed its checks")],
            }
        missing = [
            identifier for identifier, result in candidates if result.status == "missing"
        ]
        if missing:
            return {
                "status": "missing",
                "minimum_level": minimum_level,
                "evidence": missing,
                "details": [requirement.get("gap", "registered evidence files are missing")],
            }
        weak = [
            (identifier, result) for identifier, result in candidates
            if result.status != "verified" or
            level_rank[result.level] < level_rank[minimum_level]
        ]
        if weak:
            return {
                "status": "insufficient",
                "minimum_level": minimum_level,
                "achieved_level": min(
                    (result.level for _, result in candidates),
                    key=lambda level: level_rank[level],
                ),
                "evidence": [identifier for identifier, _ in candidates],
                "details": [requirement.get("gap", "available evidence is below the required level")],
            }
        return {
            "status": "verified",
            "minimum_level": minimum_level,
            "achieved_level": min(
                (result.level for _, result in candidates),
                key=lambda level: level_rank[level],
            ),
            "evidence": [identifier for identifier, _ in candidates],
            "details": [],
        }
    adequate = [
        (identifier, result) for identifier, result in candidates
        if result.status == "verified" and level_rank[result.level] >= level_rank[minimum_level]
    ]
    if adequate:
        identifier, result = max(adequate, key=lambda item: level_rank[item[1].level])
        return {
            "status": "verified",
            "minimum_level": minimum_level,
            "achieved_level": result.level,
            "evidence": [identifier],
            "details": [],
        }
    verified_but_weak = [
        (identifier, result) for identifier, result in candidates if result.status == "verified"
    ]
    if verified_but_weak:
        identifier, result = max(verified_but_weak, key=lambda item: level_rank[item[1].level])
        return {
            "status": "insufficient",
            "minimum_level": minimum_level,
            "achieved_level": result.level,
            "evidence": [identifier],
            "details": [requirement.get("gap", "available evidence is below the required level")],
        }
    contradicted = [
        (identifier, result) for identifier, result in candidates if result.status == "contradicted"
    ]
    if contradicted:
        return {
            "status": "contradicted",
            "minimum_level": minimum_level,
            "evidence": [identifier for identifier, _ in contradicted],
            "details": [detail for _, result in contradicted for detail in result.details]
                + [requirement.get("gap", "registered evidence failed its checks")],
        }
    return {
        "status": "missing",
        "minimum_level": minimum_level,
        "evidence": evidence_ids,
        "details": [requirement.get("gap", "registered evidence files are missing")],
    }


def validate(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "numilab.human.whole-body-validation-manifest.v1":
        raise ValidationError("unsupported whole-body validation manifest schema")
    levels = manifest["evidence_levels"]
    if len(levels) != len(set(levels)) or not levels:
        raise ValidationError("evidence levels must be nonempty and unique")
    level_rank = {level: index for index, level in enumerate(levels)}
    evidence_results: dict[str, EvidenceResult] = {}
    for identifier, evidence in manifest["evidence"].items():
        if evidence["level"] not in level_rank:
            raise ValidationError(f"evidence {identifier} uses an unknown level")
        evidence_results[identifier] = evaluate_evidence(root, evidence)

    scopes: list[dict[str, Any]] = []
    required_counts = {status: 0 for status in ("verified", "insufficient", "contradicted", "missing")}
    for scope_kind in ("regions", "continuum_layers"):
        for scope in manifest[scope_kind]:
            if "profile" in scope:
                profile = manifest.get("profiles", {}).get(scope["profile"])
                if profile is None:
                    raise ValidationError(
                        f"scope {scope['id']} references unknown profile {scope['profile']}"
                    )
                scope_requirements = copy.deepcopy(profile)
                scope_requirements.update(copy.deepcopy(scope.get("requirements", {})))
            else:
                scope_requirements = scope["requirements"]
            requirements: dict[str, Any] = {}
            for name, requirement in scope_requirements.items():
                for identifier in requirement.get("evidence", []):
                    if identifier not in evidence_results:
                        raise ValidationError(
                            f"scope {scope['id']} requirement {name} references unknown evidence {identifier}"
                        )
                result = requirement_result(requirement, evidence_results, level_rank)
                requirements[name] = result
                if result["status"] in required_counts:
                    required_counts[result["status"]] += 1
            required = [item for item in requirements.values() if item["status"] != "not_applicable"]
            scopes.append({
                "kind": scope_kind[:-1],
                "id": scope["id"],
                "status": "qualified" if required and all(
                    item["status"] == "verified" for item in required
                ) else "incomplete",
                "requirements": requirements,
            })
    overall = "qualified" if required_counts["verified"] > 0 and all(
        required_counts[status] == 0 for status in ("insufficient", "contradicted", "missing")
    ) else "incomplete"
    return {
        "schema": "numilab.human.whole-body-validation-report.v1",
        "manifest": str(manifest_path.relative_to(root)),
        "revision": _git_revision(root),
        "status": overall,
        "summary": {
            "scope_count": len(scopes),
            "qualified_scope_count": sum(scope["status"] == "qualified" for scope in scopes),
            "requirements": required_counts,
        },
        "evidence": {
            identifier: {
                "status": result.status,
                "level": result.level,
                "path": result.path,
                "details": list(result.details),
            }
            for identifier, result in evidence_results.items()
        },
        "scopes": scopes,
        "evidence_boundary": manifest["evidence_boundary"],
    }


def _human_summary(report: dict[str, Any]) -> Iterable[str]:
    summary = report["summary"]
    yield (
        f"numilab_human_whole_body={report['status']} revision={report['revision']} "
        f"qualified_scopes={summary['qualified_scope_count']}/{summary['scope_count']}"
    )
    counts = summary["requirements"]
    yield "requirements " + " ".join(f"{name}={value}" for name, value in counts.items())
    for scope in report["scopes"]:
        gaps = [
            f"{name}:{result['status']}"
            for name, result in scope["requirements"].items()
            if result["status"] not in ("verified", "not_applicable")
        ]
        if gaps:
            yield f"{scope['kind']}={scope['id']} gaps=" + ",".join(gaps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    manifest_path = arguments.manifest or root / DEFAULT_MANIFEST
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    try:
        report = validate(root, manifest_path.resolve())
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"numilab_human_whole_body=invalid error={error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    if not arguments.quiet:
        for line in _human_summary(report):
            print(line, file=sys.stderr)
    return 0 if report["status"] == "qualified" or arguments.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
