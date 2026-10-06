#!/usr/bin/env python3
"""Decode read-only Matter `fem_contact_iterate` trace lines.

The trace is emitted by Runtime's FEM contact line-search diagnostic hook. The
decoder uses only Python's standard library and never opens a GPU or mutates a
native state. It validates the byte widths against the shared ABI-40 records,
then compares the four line-search stages and reports the active rigid-contact
rows that could explain a zero alpha.

Example:
  python3 analyze_contact_trace.py run/stderr.log \
      --receipt run/receipt.json --manifest run/manifest.json \
      --output run/contact-trace-analysis.json

Directional contact diagnostics assume the rigid side has no Newton direction
(appropriate for prescribed kinematic/static cardboard tools). They are an
estimate from the captured sample and FEM solution; deformable-contact records,
rigid generalized state, and object scheduler entries are not in this trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


TRACE_PREFIX = "fem_contact_iterate="
ABI_VERSION_EXPECTED = 40
NODE_BYTES = 80
SAMPLE_BYTES = 160
FGMRES_BYTES = 32
STATUS_BYTES = 48
FLOAT4_BYTES = 16
U32_BYTES = 4
FLOAT_EPSILON = 2.0 ** -23
CONTACT_VALID = 1
STAGE_ORDER = {
    "before_contact_limits": 0,
    "after_deformable_limit": 1,
    "after_rigid_limit": 2,
    "final_line_search": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def f32x4(data: bytes, offset: int) -> list[float | None]:
    return [finite_or_none(x) for x in struct.unpack_from("<4f", data, offset)]


def u32x4(data: bytes, offset: int) -> list[int]:
    return list(struct.unpack_from("<4I", data, offset))


def read_trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            marker = line.find(TRACE_PREFIX)
            if marker < 0:
                continue
            encoded = line[marker + len(TRACE_PREFIX):].strip()
            try:
                item = json.loads(encoded)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid trace JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(item, dict) or not isinstance(item.get("arenas"), dict):
                raise ValueError(f"trace line {line_number} has no arenas object")
            item["_line"] = line_number
            decoded: dict[str, bytes] = {}
            for name, text in item["arenas"].items():
                try:
                    decoded[name] = bytes.fromhex(text)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"trace line {line_number}: invalid hex arena {name!r}"
                    ) from error
            item["_decoded"] = decoded
            records.append(item)
    return records


def verify_shared_header(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    abi = re.search(r"#define\s+NM_MATTER_ABI_VERSION\s+(\d+)u", text)
    if not abi:
        raise ValueError(f"cannot find Matter ABI version in {path}")
    version = int(abi.group(1))
    if version != ABI_VERSION_EXPECTED:
        raise ValueError(
            f"this decoder expects ABI {ABI_VERSION_EXPECTED}, found {version}"
        )
    # Check the traced records' field declarations so a future ABI edit fails
    # closed instead of silently applying old offsets.
    required_fields = {
        "NMFEMNodeStateGPU": [
            "positionAndMass", "velocityAndInverseMass", "restAndFixed",
            "deltaVelocity", "referenceDisplacementAndMode",
        ],
        "NMContactSampleGPU": [
            "identity", "pointAndSeparation", "normalAndVelocity",
            "admissionVelocityAndNormal", "impulseAndNormal",
            "angularImpulseAndTangent", "barrier", "barrierHessianRow0",
            "barrierHessianRow1", "barrierHessianRow2",
        ],
        "NMFGMRESStateGPU": ["diagnostics", "nonlinear"],
        "NMMatterStatusGPU": [
            "code", "environment", "objectIndex", "failingIndex",
            "completedMicrosteps", "fgmresIterations", "contactCount",
            "eventCount", "diagnostics",
        ],
    }
    for struct_name, fields in required_fields.items():
        match = re.search(
            rf"typedef struct NM_ALIGN16 {struct_name}\s*\{{(.*?)\}}\s*{struct_name};",
            text,
            flags=re.S,
        )
        if not match:
            raise ValueError(f"cannot find {struct_name} in {path}")
        for field in fields:
            if not re.search(rf"\b{re.escape(field)}\s*;", match.group(1)):
                raise ValueError(
                    f"{struct_name} no longer declares expected field {field}"
                )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "abi_version": version,
        "record_bytes": {
            "NMFEMNodeStateGPU": NODE_BYTES,
            "NMContactSampleGPU": SAMPLE_BYTES,
            "NMFGMRESStateGPU": FGMRES_BYTES,
            "NMMatterStatusGPU": STATUS_BYTES,
        },
    }


def decode_status(data: bytes, environment: int) -> dict[str, Any]:
    offset = environment * STATUS_BYTES
    words = struct.unpack_from("<8I", data, offset)
    return {
        "code": words[0],
        "environment": words[1],
        "object_index": words[2],
        "failing_index": words[3],
        "completed_microsteps": words[4],
        "fgmres_iterations": words[5],
        "contact_count": words[6],
        "event_count": words[7],
        "diagnostics": f32x4(data, offset + 32),
    }


def arena_dims(arenas: dict[str, bytes]) -> dict[str, int]:
    lengths = {name: len(data) for name, data in arenas.items()}
    for required in ("candidate", "solution", "residual", "alpha",
                     "object_line_search", "fgmres", "status", "samples",
                     "active_pairs", "active_counts"):
        if required not in lengths:
            raise ValueError(f"trace is missing required arena {required!r}")
    envs = lengths["alpha"] // 4
    if envs < 1 or lengths["alpha"] % 4:
        raise ValueError("malformed environment alpha arena")
    expected = {
        "candidate": (envs * NODE_BYTES, "candidate"),
        "solution": (envs * FLOAT4_BYTES, "solution"),
        "residual": (envs * FLOAT4_BYTES, "residual"),
        "fgmres": (envs * FGMRES_BYTES, "fgmres"),
        "status": (envs * STATUS_BYTES, "status"),
        "active_counts": (envs * U32_BYTES, "active_counts"),
    }
    for name, (stride_bytes, label) in expected.items():
        if lengths[name] % stride_bytes:
            raise ValueError(f"malformed {label} arena byte count")
    fem_nodes = lengths["candidate"] // (envs * NODE_BYTES)
    objects = lengths["object_line_search"] // (envs * FLOAT4_BYTES)
    pairs = lengths["samples"] // (envs * SAMPLE_BYTES)
    active_capacity = lengths["active_pairs"] // (envs * U32_BYTES)
    if (fem_nodes == 0 or lengths["candidate"] != envs * fem_nodes * NODE_BYTES or
            lengths["solution"] != envs * fem_nodes * FLOAT4_BYTES or
            lengths["residual"] != envs * fem_nodes * FLOAT4_BYTES or
            lengths["fgmres"] != envs * FGMRES_BYTES or
            lengths["status"] != envs * STATUS_BYTES or
            lengths["object_line_search"] != envs * objects * FLOAT4_BYTES or
            lengths["samples"] != envs * pairs * SAMPLE_BYTES or
            lengths["active_pairs"] != envs * active_capacity * U32_BYTES or
            lengths["active_counts"] != envs * U32_BYTES):
        raise ValueError("inconsistent trace arena strides")
    return {
        "environments": envs,
        "fem_nodes": fem_nodes,
        "objects": objects,
        "contact_pairs": pairs,
        "active_contact_capacity": active_capacity,
    }


def vector_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm3(value: list[float]) -> float:
    return math.sqrt(vector_dot(value, value))


def trace_stage(record: dict[str, Any], environment: int,
                grid_node_count: int, timestep: float | None,
                contact_slop: float | None,
                minimum_separation_ratio: float,
                assume_zero_rigid_newton_direction: bool) -> dict[str, Any]:
    arenas: dict[str, bytes] = record["_decoded"]
    dims = arena_dims(arenas)
    if environment >= dims["environments"]:
        raise ValueError(f"environment {environment} outside trace")
    alpha = struct.unpack_from("<f", arenas["alpha"], environment * 4)[0]
    object_base = environment * dims["objects"] * FLOAT4_BYTES
    object_alpha = [
        f32x4(arenas["object_line_search"], object_base + i * FLOAT4_BYTES)
        for i in range(dims["objects"])
    ]
    fgmres_offset = environment * FGMRES_BYTES
    fgmres_diag = f32x4(arenas["fgmres"], fgmres_offset)
    fgmres_nonlinear = f32x4(arenas["fgmres"], fgmres_offset + 16)
    status = decode_status(arenas["status"], environment)
    active_count = struct.unpack_from("<I", arenas["active_counts"], environment * 4)[0]
    active_count = min(active_count, dims["active_contact_capacity"])
    active_base = environment * dims["active_contact_capacity"]
    active_pairs = struct.unpack_from(
        f"<{dims['active_contact_capacity']}I", arenas["active_pairs"],
        active_base * 4,
    ) if dims["active_contact_capacity"] else ()
    active_rows: list[dict[str, Any]] = []
    for slot, pair_index in enumerate(active_pairs[:active_count]):
        if pair_index >= dims["contact_pairs"]:
            active_rows.append({"active_slot": slot, "pair_index": pair_index,
                                "invalid_pair_index": True})
            continue
        sample_offset = (environment * dims["contact_pairs"] + pair_index) * SAMPLE_BYTES
        identity = u32x4(arenas["samples"], sample_offset)
        point_sep = f32x4(arenas["samples"], sample_offset + 16)
        normal_velocity = f32x4(arenas["samples"], sample_offset + 32)
        impulse = f32x4(arenas["samples"], sample_offset + 64)
        barrier = f32x4(arenas["samples"], sample_offset + 96)
        row: dict[str, Any] = {
            "active_slot": slot,
            "pair_index": pair_index,
            "node_global": identity[0],
            "node_local_fem": identity[0] - grid_node_count,
            "rigid_proxy": identity[1],
            "object_index": identity[2],
            "flags": identity[3],
            "valid": bool(identity[3] & CONTACT_VALID),
            "point_m": point_sep[:3],
            "separation_m": point_sep[3],
            "normal": normal_velocity[:3],
            "normal_velocity": normal_velocity[3],
            "normal_impulse": impulse[3],
            "barrier": {
                "normal_impulse": barrier[0],
                "curvature": barrier[1],
                "thickness_m": barrier[2],
                "stiffness": barrier[3],
            },
        }
        node = identity[0] - grid_node_count
        if 0 <= node < dims["fem_nodes"]:
            candidate_offset = (environment * dims["fem_nodes"] + node) * NODE_BYTES
            row["candidate_position_m"] = f32x4(arenas["candidate"], candidate_offset)[:3]
            row["candidate_velocity_m_s"] = f32x4(arenas["candidate"], candidate_offset + 16)[:3]
            row["candidate_fixed_code"] = f32x4(arenas["candidate"], candidate_offset + 32)[3]
            row["candidate_reference_displacement_mode"] = f32x4(
                arenas["candidate"], candidate_offset + 64
            )
            solution_offset = (environment * dims["fem_nodes"] + node) * FLOAT4_BYTES
            correction = f32x4(arenas["solution"], solution_offset)
            residual = f32x4(arenas["residual"], solution_offset)
            row["newton_correction"] = correction
            row["residual"] = residual
            if timestep is not None and assume_zero_rigid_newton_direction:
                displacement = [timestep * float(correction[i]) for i in range(3)]
                directional = vector_dot(
                    [float(x) for x in normal_velocity[:3]], displacement
                )
                travel = norm3(displacement)
                scale = max(
                    float(barrier[2]),
                    max(abs(float(x)) for x in point_sep[:3]),
                    1.0e-12,
                )
                floor = max(
                    (minimum_separation_ratio * contact_slop)
                    if contact_slop is not None else 0.0,
                    16.0 * FLOAT_EPSILON * scale,
                )
                roundoff = 8.0 * FLOAT_EPSILON * max(floor, scale, 1.0e-12)
                feasible = floor + roundoff
                if point_sep[3] > feasible and travel > 1.0e-12:
                    fraction_bound = max(0.0, min(
                        1.0, 0.9 * (point_sep[3] - feasible) / travel
                    ))
                elif travel > 1.0e-12 and point_sep[3] <= feasible and directional <= 0.0:
                    fraction_bound = 0.0
                else:
                    fraction_bound = None
                row["rigid_limiter_estimate"] = {
                    "assumption": "zero rigid Newton direction; verify proxy is prescribed/static",
                    "timestep_s": timestep,
                    "relative_travel_m": travel,
                    "directional_separation_m": directional,
                    "coordinate_scale_m": scale,
                    "collision_thickness_m": floor,
                    "boundary_roundoff_m": roundoff,
                    "feasible_separation_m": feasible,
                    "estimated_fraction_bound": fraction_bound,
                    "would_zero_by_near_floor_rule": fraction_bound == 0.0,
                }
        active_rows.append(row)
    return {
        "root": record.get("root"),
        "microtick": record.get("microtick"),
        "iteration": record.get("iteration"),
        "stage": record.get("stage"),
        "trace_line": record.get("_line"),
        "alpha": finite_or_none(alpha),
        "object_alpha": object_alpha,
        "status": status,
        "fgmres": {
            "diagnostics": fgmres_diag,
            "nonlinear": fgmres_nonlinear,
            "linear_converged_flag": (
                None if fgmres_diag[2] is None else fgmres_diag[2] > 0.5
            ),
            "nonlinear_converged_flag": (
                None if fgmres_nonlinear[3] is None else fgmres_nonlinear[3] > 0.5
            ),
        },
        "active_count": active_count,
        "active_rows": active_rows,
        "dims": dims,
    }


def manifest_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    solver = value.get("solver", {})
    dt = solver.get("dt_s", value.get("dt_s"))
    contact_slop = solver.get("contact_slop_m", value.get("contact_slop_m"))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "timestep_s": float(dt) if dt is not None else None,
        "contact_slop_m": float(contact_slop) if contact_slop is not None else None,
        "maximum_rate_exponent": solver.get("maximum_rate_exponent"),
        "tooling": value.get("tooling"),
    }


def receipt_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    bindings = value.get("bindings", {})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_patch_sha256": value.get("source_patch_sha256"),
        "base_git_commit": value.get("base_git_commit"),
        "bindings": bindings,
    }


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def run_comparison(current_run: Path, reference_run: Path,
                   current_receipt: Path | None) -> dict[str, Any]:
    current_run = current_run.resolve()
    reference_run = reference_run.resolve()
    current_result_path = current_run / "result.json"
    reference_result_path = reference_run / "result.json"
    current_manifest_path = current_run / "manifest.json"
    reference_manifest_path = reference_run / "manifest.json"
    current_csv_path = current_run / "tool-observations.csv"
    reference_csv_path = reference_run / "tool-observations.csv"
    current_mesh_path = current_run / "mesh.json"
    reference_mesh_path = reference_run / "mesh.json"
    current_result = optional_json(current_result_path)
    reference_result = optional_json(reference_result_path)
    if current_receipt is not None:
        current_receipt_path = current_receipt
    else:
        current_receipt_path = current_run.parent / "receipts" / current_run.name / "receipt.json"
    reference_receipt_path = reference_run.parent / "receipts" / reference_run.name / "receipt.json"
    current_receipt_data = optional_json(current_receipt_path) or {}
    reference_receipt_data = optional_json(reference_receipt_path) or {}

    def binding_for(receipt: dict[str, Any], suffix: str) -> str | None:
        bindings = receipt.get("bindings", {})
        if not isinstance(bindings, dict):
            return None
        for path, digest in bindings.items():
            if path.endswith(suffix):
                return digest
        return None

    current_bindings = current_receipt_data.get("bindings", {})
    reference_bindings = reference_receipt_data.get("bindings", {})
    if not isinstance(current_bindings, dict):
        current_bindings = {}
    if not isinstance(reference_bindings, dict):
        reference_bindings = {}
    exact_build_keys = [
        "numi-matter-cardboard-probe", "NumiMatter.metallib",
        "contact.metalinc", "shared.h",
    ]
    build_binding_comparison: dict[str, Any] = {}
    for suffix in exact_build_keys:
        current_digest = binding_for(current_receipt_data, suffix)
        reference_digest = binding_for(reference_receipt_data, suffix)
        build_binding_comparison[suffix] = {
            "current_sha256": current_digest,
            "reference_sha256": reference_digest,
            "matches": (None if current_digest is None or reference_digest is None
                        else current_digest == reference_digest),
        }

    def same_bytes(left: Path, right: Path) -> bool | None:
        if not left.is_file() or not right.is_file():
            return None
        return sha256_file(left) == sha256_file(right)

    current_step15 = None
    reference_step15 = None
    if current_csv_path.is_file() and reference_csv_path.is_file():
        for path, label in ((current_csv_path, "current"), (reference_csv_path, "reference")):
            with path.open("r", newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rejected = [row for row in rows if row.get("step") == "15" and row.get("step_accepted") == "0"]
            if label == "current":
                current_step15 = rejected
            else:
                reference_step15 = rejected
    return {
        "current_run": str(current_run),
        "reference_run": str(reference_run),
        "current": {
            "result_sha256": sha256_file(current_result_path) if current_result_path.is_file() else None,
            "manifest_sha256": sha256_file(current_manifest_path) if current_manifest_path.is_file() else None,
            "tool_observations_sha256": sha256_file(current_csv_path) if current_csv_path.is_file() else None,
            "mesh_sha256": sha256_file(current_mesh_path) if current_mesh_path.is_file() else None,
            "receipt_sha256": sha256_file(current_receipt_path) if current_receipt_path.is_file() else None,
            "source_patch_sha256": current_receipt_data.get("source_patch_sha256"),
            "compiled_world_fingerprint": current_result.get("compiled_world_fingerprint") if current_result else None,
            "accepted_steps": current_result.get("accepted_steps") if current_result else None,
            "status": current_result.get("status") if current_result else None,
        },
        "reference": {
            "result_sha256": sha256_file(reference_result_path) if reference_result_path.is_file() else None,
            "manifest_sha256": sha256_file(reference_manifest_path) if reference_manifest_path.is_file() else None,
            "tool_observations_sha256": sha256_file(reference_csv_path) if reference_csv_path.is_file() else None,
            "mesh_sha256": sha256_file(reference_mesh_path) if reference_mesh_path.is_file() else None,
            "receipt_sha256": sha256_file(reference_receipt_path) if reference_receipt_path.is_file() else None,
            "source_patch_sha256": reference_receipt_data.get("source_patch_sha256"),
            "compiled_world_fingerprint": reference_result.get("compiled_world_fingerprint") if reference_result else None,
            "accepted_steps": reference_result.get("accepted_steps") if reference_result else None,
            "status": reference_result.get("status") if reference_result else None,
        },
        "same_observation_csv": same_bytes(current_csv_path, reference_csv_path),
        "same_mesh_json": same_bytes(current_mesh_path, reference_mesh_path),
        "same_compiled_world_fingerprint": (
            None if not current_result or not reference_result
            else current_result.get("compiled_world_fingerprint") ==
                 reference_result.get("compiled_world_fingerprint")
        ),
        "step15_rejected_rows_equal": (
            None if current_step15 is None or reference_step15 is None
            else current_step15 == reference_step15
        ),
        "step15_rejected_rows": {
            "current": current_step15,
            "reference": reference_step15,
        },
        "source_build_bindings": build_binding_comparison,
        "exact_solver_build_match": (
            all(value["matches"] is True for value in build_binding_comparison.values())
            if all(value["matches"] is not None for value in build_binding_comparison.values())
            else None
        ),
        "comparison_limit": (
            "Identical mesh and observations plus matching world fingerprint reproduce the recorded outcome; "
            "if executable, metallib, or solver source hashes differ, this is not a bitwise-identical solver-build replay."
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    records = read_trace_records(args.trace)
    if not records:
        raise ValueError(f"no {TRACE_PREFIX} records found in {args.trace}")
    header = verify_shared_header(args.shared_header)
    manifest = manifest_values(args.manifest)
    receipt = receipt_values(args.receipt)
    timestep = args.timestep_s if args.timestep_s is not None else manifest.get("timestep_s")
    contact_slop = (args.contact_slop_m if args.contact_slop_m is not None
                    else manifest.get("contact_slop_m"))
    stages = [trace_stage(
        record,
        args.environment,
        args.grid_node_count,
        timestep,
        contact_slop,
        args.minimum_separation_ratio,
        not args.unknown_rigid_direction,
    ) for record in records]
    stages.sort(key=lambda stage: (
        int(stage["root"] or 0), int(stage["microtick"] or 0),
        int(stage["iteration"] or 0), STAGE_ORDER.get(stage["stage"], 99),
        int(stage["trace_line"] or 0),
    ))
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for stage in stages:
        key = (int(stage["root"] or 0), int(stage["microtick"] or 0),
               int(stage["iteration"] or 0))
        groups.setdefault(key, []).append(stage)
    transitions: list[dict[str, Any]] = []
    object_transitions: list[dict[str, Any]] = []
    first_zero: dict[str, Any] | None = None
    first_zero_kind: str | None = None
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda stage: STAGE_ORDER.get(stage["stage"], 99))
        prev: dict[str, Any] | None = None
        for stage in ordered:
            alpha = stage["alpha"]
            if alpha is not None and alpha <= 0.0:
                if first_zero is None:
                    first_zero = stage
                if prev is None or prev["alpha"] is None or prev["alpha"] > 0.0:
                    transitions.append({
                        "root": key[0], "microtick": key[1], "iteration": key[2],
                        "zero_stage": stage["stage"],
                        "previous_stage": prev["stage"] if prev else None,
                        "previous_alpha": prev["alpha"] if prev else None,
                        "status": stage["status"],
                        "fgmres": stage["fgmres"],
                        "active_rows": stage["active_rows"],
                    })
            object_zeros = [
                index for index, values in enumerate(stage["object_alpha"])
                if values and values[0] is not None and values[0] <= 0.0
            ]
            previous_object_values = (
                prev["object_alpha"] if prev is not None else []
            )
            newly_zero_objects = [
                index for index in object_zeros
                if index >= len(previous_object_values)
                or not previous_object_values[index]
                or previous_object_values[index][0] is None
                or previous_object_values[index][0] > 0.0
            ]
            if object_zeros:
                if first_zero is None:
                    first_zero = stage
                    first_zero_kind = "object_line_search"
                if newly_zero_objects:
                    object_transitions.append({
                        "root": key[0], "microtick": key[1], "iteration": key[2],
                        "zero_stage": stage["stage"],
                        "previous_stage": prev["stage"] if prev else None,
                        "previous_object_alpha": {
                            index: (previous_object_values[index][0]
                                    if index < len(previous_object_values)
                                    and previous_object_values[index]
                                    else None)
                            for index in newly_zero_objects
                        },
                        "zero_object_indices": newly_zero_objects,
                        "zero_object_alpha": {
                            index: stage["object_alpha"][index][0]
                            for index in newly_zero_objects
                        },
                        "environment_alpha_at_stage": alpha,
                        "status": stage["status"],
                        "fgmres": stage["fgmres"],
                        "active_rows": stage["active_rows"],
                    })
            prev = stage
    known_stages = sorted({stage["stage"] for stage in stages},
                          key=lambda name: STAGE_ORDER.get(name, 99))
    limiting_rows = ([
        row for row in first_zero["active_rows"]
        if row.get("rigid_limiter_estimate", {}).get(
            "would_zero_by_near_floor_rule")
    ] if first_zero is not None else [])
    nonlinear_failure_code = 10
    if first_zero is not None:
        zero_key = (
            int(first_zero["root"] or 0), int(first_zero["microtick"] or 0),
            int(first_zero["iteration"] or 0),
            STAGE_ORDER.get(first_zero["stage"], 99),
            int(first_zero["trace_line"] or 0),
        )
        failure_stage = next((
            stage for stage in stages
            if (
                int(stage["root"] or 0), int(stage["microtick"] or 0),
                int(stage["iteration"] or 0),
                STAGE_ORDER.get(stage["stage"], 99),
                int(stage["trace_line"] or 0),
            ) >= zero_key and stage["status"]["code"] != 0
        ), None)
        failure_status = (
            failure_stage["status"] if failure_stage is not None
            else first_zero["status"]
        )
        failure_status_code = failure_status["code"]
        first_fgmres = first_zero["fgmres"]
        first_failure_label = (
            "NM_STATUS_NONLINEAR_SOLVER_FAILURE"
            if failure_status_code == nonlinear_failure_code else None
        )
        if limiting_rows:
            mechanism = (
                "Rigid contact line-search feasibility limiter set an object alpha to zero: "
                "the active sample was at/below feasibleSeparation and its Newton direction "
                "was inward or stationary. The environment alpha was published as zero at "
                "the later final_line_search snapshot."
            )
        else:
            mechanism = (
                "No near-floor rigid sample in the first zero-alpha snapshot met the "
                "decoder's directional zero-step criterion; inspect all recorded limiter "
                "arenas and omitted deformable-contact state."
            )
        numerical_diagnosis = {
            "mechanism": mechanism,
            "first_zero_stage": first_zero["stage"],
            "iteration": first_zero["iteration"],
            "zero_basis": first_zero_kind,
            "failure_stage": failure_stage["stage"] if failure_stage else None,
            "failure_iteration": failure_stage["iteration"] if failure_stage else None,
            "status_code": failure_status_code,
            "status_name": first_failure_label,
            "linear_solve_converged": first_fgmres["linear_converged_flag"],
            "nonlinear_solve_converged": first_fgmres["nonlinear_converged_flag"],
            "active_rows_at_or_below_feasible_floor": limiting_rows,
            "interpretation_limit": (
                "Directional row estimates assume the rigid Newton direction is zero; "
                "the prescribed-tool manifest and zero maximum-rate exponent support "
                "that assumption for this run."
            ),
        }
    else:
        numerical_diagnosis = None
    claimed_shared_binding = None
    for path, digest in receipt.get("bindings", {}).items():
        if path.endswith("/shared.h"):
            claimed_shared_binding = digest
            break
    return {
        "schema": "numi.matter.fem-contact-trace-analysis.v1",
        "source": {
            "trace_path": str(args.trace),
            "trace_sha256": sha256_file(args.trace),
            "decoder_path": str(Path(__file__).resolve()),
            "decoder_sha256": sha256_file(Path(__file__).resolve()),
            "shared_header": header,
            "receipt": receipt,
            "shared_header_matches_receipt": (
                None if claimed_shared_binding is None
                else claimed_shared_binding == header["sha256"]
            ),
            "manifest": manifest,
        },
        "analysis_inputs": {
            "environment": args.environment,
            "grid_node_count": args.grid_node_count,
            "timestep_s": timestep,
            "contact_slop_m": contact_slop,
            "minimum_contact_separation_ratio": args.minimum_separation_ratio,
            "rigid_direction_assumption": (
                "zero rigid Newton direction" if not args.unknown_rigid_direction
                else "unknown; directional estimates suppressed"
            ),
        },
        "trace_record_count": len(records),
        "stages_present": known_stages,
        "numerical_diagnosis": numerical_diagnosis,
        "first_zero_alpha": {
            "basis": first_zero_kind,
            "stage": first_zero,
            "limiting_rows": limiting_rows,
        },
        "zero_alpha_transitions": transitions,
        "zero_object_alpha_transitions": object_transitions,
        "comparison": (
            run_comparison(args.manifest.parent, args.compare_run, args.receipt)
            if args.compare_run is not None else None
        ),
        "all_stages": stages,
        "limits": [
            "Trace omits deformable-contact records, the NMContactPairGPU table, rigid proxy flags/states, and object scheduler records.",
            "A rigid-limiter directional estimate assumes a prescribed/static rigid side; verify the sample rigid proxy from the source-bound world.",
            "Contact floor estimates use the sample's recorded barrier thickness, absolute contact-point coordinates, manifest contact slop, and the supplied/default separation ratio.",
            "This decodes a diagnostic snapshot only; it does not replay the solver or establish physical validation.",
        ],
    }


def main() -> int:
    script_path = Path(__file__).resolve()
    sibling_repo = script_path.parent.parent / "numi-cardboard-20261006"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="stderr or text file containing fem_contact_iterate lines")
    parser.add_argument("--receipt", type=Path, help="source-bound receipt.json from the traced run")
    parser.add_argument("--manifest", type=Path, help="run manifest.json for dt and contact slop")
    parser.add_argument("--compare-run", type=Path,
                        help="prior run directory with result.json and observations to compare")
    parser.add_argument("--shared-header", type=Path,
                        default=sibling_repo / "matter/include/numi/matter/shared.h")
    parser.add_argument("--output", type=Path, help="write JSON here; default is stdout")
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--grid-node-count", type=int, default=0)
    parser.add_argument("--timestep-s", type=float, help="override manifest timestep")
    parser.add_argument("--contact-slop-m", type=float, help="override manifest contact slop")
    parser.add_argument("--minimum-separation-ratio", type=float, default=1.0e-4,
                        help="NM_MixedSolverGPU.contactAcceptance.x; default from Matter source API")
    parser.add_argument("--unknown-rigid-direction", action="store_true",
                        help="suppress rigid-limiter direction estimates if rigid DOFs may be active")
    args = parser.parse_args()
    try:
        result = analyze(args)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"contact trace analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
