#!/usr/bin/env python3
"""Convert the native all-DoF Human support trace into stable JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any


SCHEMA = "numi.human.whole-body-support-all-dof-report.v1"
RANK_PATTERN = re.compile(r"residual_rank_(\d+)_(.+)")
MUSCLE_PATTERN = re.compile(r"muscle_(\d+)_(.+)")
CONTACT_PATTERN = re.compile(r"contact_(\d+)_(.+)")


def _value(text: str) -> Any:
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "bitwise":
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        raise ValueError(f"non-finite native report value: {text}")
    if value.is_integer() and not any(character in text for character in ".eE"):
        return int(value)
    return value


def parse(text: str) -> dict[str, Any]:
    marker = "numi_human_whole_body_support_wrench="
    offset = text.find(marker)
    if offset < 0:
        raise ValueError("native trace has no whole-body support record")
    tokens = shlex.split(text[offset:])
    fields: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate native report key: {key}")
        fields[key] = _value(value)
    if fields.get("numi_human_whole_body_support_wrench") != "ok":
        raise ValueError("native support wrench did not report ok")

    residuals: dict[int, dict[str, Any]] = {}
    contacts: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for key, value in fields.items():
        rank_match = RANK_PATTERN.fullmatch(key)
        contact_match = CONTACT_PATTERN.fullmatch(key)
        if rank_match:
            rank = int(rank_match.group(1))
            suffix = rank_match.group(2)
            record = residuals.setdefault(rank, {"rank": rank, "muscles": {}})
            muscle_match = MUSCLE_PATTERN.fullmatch(suffix)
            if muscle_match:
                ordinal = int(muscle_match.group(1))
                record["muscles"].setdefault(ordinal, {"ordinal": ordinal})[
                    muscle_match.group(2)
                ] = value
            else:
                record[suffix] = value
        elif contact_match:
            ordinal = int(contact_match.group(1))
            contacts.setdefault(ordinal, {"ordinal": ordinal})[
                contact_match.group(2)
            ] = value
        else:
            summary[key] = value

    ranks = sorted(residuals)
    if ranks != list(range(len(ranks))):
        raise ValueError("native residual ranks are not contiguous")
    ordered_residuals = []
    seen_dofs: set[int] = set()
    for rank in ranks:
        record = residuals[rank]
        muscles = record.pop("muscles")
        muscle_ordinals = sorted(muscles)
        if muscle_ordinals != list(range(len(muscle_ordinals))):
            raise ValueError(f"rank {rank} muscle contributor ordinals are not contiguous")
        record["muscles"] = [muscles[index] for index in muscle_ordinals]
        dof = record.get("dof")
        if not isinstance(dof, int) or dof in seen_dofs:
            raise ValueError(f"rank {rank} has an invalid or duplicate DoF")
        seen_dofs.add(dof)
        ordered_residuals.append(record)
    if seen_dofs != set(range(len(ordered_residuals))) or "boundary" not in summary:
        raise ValueError("native all-DoF trace is incomplete")
    accelerations = [abs(float(record["acceleration"])) for record in ordered_residuals]
    return {
        "schema": SCHEMA,
        "summary": summary,
        "contacts": [contacts[index] for index in sorted(contacts)],
        "residuals": ordered_residuals,
        "derived": {
            "dof_count": len(ordered_residuals),
            "maximum_absolute_acceleration_rad_s2": max(accelerations),
            "coordinates_over_1_rad_s2": sum(value > 1.0 for value in accelerations),
            "coordinates_over_10_rad_s2": sum(value > 10.0 for value in accelerations),
            "coordinates_over_100_rad_s2": sum(value > 100.0 for value in accelerations),
        },
    }


def enrich(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "numi.human.myosim-fullbody-reference.v1":
        raise ValueError("source manifest is not a MyoSim full-body reference")
    tree = manifest.get("core_tree", {})
    joint_map = {
        int(record["core_v_index"]): record
        for record in tree.get("source_joint_map", [])
    }
    body_order = tree.get("body_order", [])
    muscle_map = {
        int(record["source_actuator_index"]): record
        for record in manifest.get("muscles", [])
    }
    root_names = ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]
    for residual in report["residuals"]:
        dof = int(residual["dof"])
        if dof < len(root_names):
            residual["source_coordinate_name"] = root_names[dof]
        else:
            source = joint_map.get(dof)
            if source is None:
                raise ValueError(f"source manifest has no joint for Core DoF {dof}")
            residual["source_coordinate_name"] = source["source_name"]
            residual["source_joint_id"] = source["source_joint_id"]
        child_body = residual.get("child_body")
        if isinstance(child_body, int) and 0 <= child_body < len(body_order):
            residual["child_body_name"] = body_order[child_body]
        for contributor in residual["muscles"]:
            source = muscle_map.get(int(contributor["index"]))
            if source is None:
                raise ValueError(
                    f"source manifest has no muscle {contributor['index']}"
                )
            contributor["name"] = source["name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", default="macmini")
    parser.add_argument("--revision-parent", required=True)
    parser.add_argument("--command", required=True)
    arguments = parser.parse_args()
    raw = arguments.input.read_bytes()
    report = parse(raw.decode("utf-8"))
    manifest_raw = arguments.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    enrich(report, manifest)
    report["provenance"] = {
        "host": arguments.host,
        "source_revision_parent": arguments.revision_parent,
        "command": arguments.command,
        "raw_trace_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source": manifest["source"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
