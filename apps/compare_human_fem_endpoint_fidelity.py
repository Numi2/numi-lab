#!/usr/bin/env python3
"""Compare two opt-in resting_FEM_endpoint_capture records.

The comparator reports force owners present in the Matter endpoint capture.
Mechanical residual rows remain N*s impulse imbalance; they are not converted
using root dt because the capture has no per-object microtick dt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

NODE_MARKER = "resting_FEM_node,"
TET_MARKER = "resting_FEM_tet,"
BEGIN_MARKER = "resting_FEM_endpoint_capture=begin "
END_MARKER = "resting_FEM_endpoint_capture=end"
FORCE_COMPONENTS = ("x", "y", "z")
FORCE_THRESHOLD = 0.01
STRAIN_THRESHOLD = 0.05


class ComparisonError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(line: str, marker: str) -> str | None:
    at = line.find(marker)
    return line[at:] if at >= 0 else None


def _kv_words(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for word in payload.split():
        if "=" not in word:
            continue
        key, value = word.split("=", 1)
        fields[key] = value
    return fields


def _kv_commas(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for word in payload.split(","):
        if "=" not in word:
            continue
        key, value = word.split("=", 1)
        fields[key] = value
    return fields


def _number(fields: dict[str, str], key: str, context: str) -> float:
    require(key in fields, f"{context} missing {key}")
    try:
        value = float(fields[key])
    except ValueError as error:
        raise ComparisonError(f"{context} has invalid {key}: {fields[key]}") from error
    require(math.isfinite(value), f"{context} has non-finite {key}")
    return value


def _integer(fields: dict[str, str], key: str, context: str) -> int:
    require(key in fields, f"{context} missing {key}")
    try:
        return int(fields[key])
    except ValueError as error:
        raise ComparisonError(f"{context} has invalid {key}: {fields[key]}") from error


def _vector(fields: dict[str, str], prefix: str, unit: str,
            context: str) -> tuple[float, float, float]:
    return tuple(_number(fields, f"{prefix}_{axis}_{unit}", context)
                 for axis in FORCE_COMPONENTS)  # type: ignore[return-value]


def parse_capture(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="strict")
    lines = text.splitlines()
    begin_line = next((p for line in lines if (p := _payload(line, BEGIN_MARKER))), None)
    require(begin_line is not None, f"{path}: missing endpoint capture begin record")
    header = _kv_words(begin_line[len(BEGIN_MARKER):])
    require(header.get("version") == "1",
            f"{path}: unsupported existing endpoint capture version")
    require(header.get("source_region_label_source") ==
                "registered_skin_fat_donor_body_id",
            f"{path}: endpoint capture does not declare registered source-region provenance")
    require(header.get("coordinate_unit") == "m" and
                header.get("mass_unit") == "kg" and
                header.get("force_unit") == "N" and
                header.get("time_unit") == "s" and
                header.get("strain_unit") == "1" and
                header.get("mechanical_residual_unit") == "N_s",
            f"{path}: endpoint capture does not declare required SI units")
    require(any(_payload(line, END_MARKER) is not None for line in lines),
            f"{path}: endpoint capture is incomplete")
    nodes: dict[tuple[str, int], dict[str, Any]] = {}
    tets: dict[tuple[str, int], dict[str, Any]] = {}
    reactions: dict[int, tuple[float, float, float]] = {}
    for line in lines:
        node_line = _payload(line, NODE_MARKER)
        if node_line is not None:
            fields = _kv_commas(node_line)
            context = f"{path}: node record"
            require("object" in fields, f"{context} missing object ID")
            obj = fields["object"]
            local = _integer(fields, "local", context)
            global_node = _integer(fields, "global", context)
            require(local >= 0 and global_node >= 0, f"{context} has negative source ID")
            initial = tuple(_number(fields, f"initial_{axis}", context)
                            for axis in FORCE_COMPONENTS)
            accepted = tuple(_number(fields, f"accepted_{axis}", context)
                             for axis in FORCE_COMPONENTS)
            rest = tuple(_number(fields, f"rest_{axis}", context)
                         for axis in FORCE_COMPONENTS)
            mode = _number(fields, "reference_mode", context)
            require(mode in (0.0, 1.0), f"{context} has unknown coordinate representation")
            hi = tuple(_number(fields, f"reference_u_hi_{axis}", context)
                       for axis in FORCE_COMPONENTS)
            lo = tuple(_number(fields, f"reference_u_lo_{axis}", context)
                       for axis in FORCE_COMPONENTS)
            current = tuple(rest[i] + hi[i] + lo[i] if mode == 1.0 else accepted[i]
                            for i in range(3))
            region_id = (_integer(fields, "anatomical_region_id", context)
                         if "anatomical_region_id" in fields else None)
            region_source = fields.get("anatomical_region_source")
            require((region_id is None) == (region_source is None),
                    f"{context} has incomplete anatomical region provenance")
            require(region_source in (None, "registered_skin_fat_donor_body"),
                    f"{context} has an unsupported anatomical region source")
            node = {
                "object": obj, "local": local, "global": global_node,
                "region_id": region_id, "region_source": region_source,
                "initial": initial, "accepted": accepted, "rest": rest,
                "current": current, "reference_mode": int(mode),
                "reference_hi": hi, "reference_lo": lo,
                "mass_kg": _number(fields, "mass_kg", context),
                "initial_mass_kg": _number(fields, "initial_mass_kg", context),
                "inverse_mass": _number(fields, "inverse_mass", context),
                "anchor_body": _integer(fields, "anchor_body", context),
                "constraint_tag": _number(fields, "constraint_tag", context),
            }
            for channel in ("bulk_reaction", "contact_force"):
                node[channel] = _vector(fields, channel, "N", context)
            for channel in ("free_internal_force", "tag2_attachment_force"):
                keys = tuple(f"{channel}_{axis}_N" for axis in FORCE_COMPONENTS)
                if header.get(channel + "_available") == "1":
                    require(all(key in fields for key in keys),
                            f"{context} missing declared {channel} components")
                node[channel] = (tuple(_number(fields, key, context) for key in keys)
                                  if all(key in fields for key in keys)
                                  else (0.0, 0.0, 0.0))
            node["residual_impulse"] = _vector(fields, "residual_impulse", "Ns", context)
            node["residual_pressure_w"] = _number(fields, "residual_pressure_w", context)
            key = (obj, local)
            require(key not in nodes, f"{path}: duplicate source node {key}")
            nodes[key] = node
            continue
        tet_line = _payload(line, TET_MARKER)
        if tet_line is not None:
            fields = _kv_commas(tet_line)
            context = f"{path}: tetrahedron record"
            obj = fields.get("object")
            require(obj is not None, f"{context} missing object ID")
            local = _integer(fields, "local", context)
            global_tet = _integer(fields, "global", context)
            ns = tuple(_integer(fields, f"n{i}", context) for i in range(4))
            require(len(set(ns)) == 4, f"{context} repeats a source vertex")
            key = (obj, local)
            require(key not in tets, f"{path}: duplicate source tetrahedron {key}")
            region_id = (_integer(fields, "anatomical_region_id", context)
                         if "anatomical_region_id" in fields else None)
            region_source = fields.get("anatomical_region_source")
            require((region_id is None) == (region_source is None),
                    f"{context} has incomplete anatomical region provenance")
            tets[key] = {"object": obj, "local": local, "global": global_tet,
                         "nodes": ns, "region_id": region_id,
                         "region_source": region_source}
            continue
        reaction_line = _payload(line, "resting_FEM_anchor_reaction,")
        if reaction_line is not None:
            fields = _kv_commas(reaction_line)
            step = _integer(fields, "step", f"{path}: anchor reaction")
            reactions[step] = tuple(_number(fields, f"{axis}_N", f"{path}: anchor reaction")
                                    for axis in FORCE_COMPONENTS)
    expected_nodes = _integer(header, "nodes", str(path))
    expected_tets = _integer(header, "tetrahedra", str(path))
    require(len(nodes) == expected_nodes,
            f"{path}: captured {len(nodes)} of {expected_nodes} nodes")
    require(len(tets) == expected_tets,
            f"{path}: captured {len(tets)} of {expected_tets} tetrahedra")
    expected_labeled_nodes = _integer(header, "source_region_labeled_nodes", str(path))
    expected_labeled_tets = _integer(header, "source_region_labeled_tetrahedra", str(path))
    require(_integer(header, "source_region_total_nodes", str(path)) == expected_nodes and
                _integer(header, "source_region_total_tetrahedra", str(path)) == expected_tets,
            f"{path}: anatomical region coverage totals differ from endpoint totals")
    actual_labeled_nodes = sum(node["region_id"] is not None for node in nodes.values())
    actual_labeled_tets = sum(tet["region_id"] is not None for tet in tets.values())
    require(actual_labeled_nodes == expected_labeled_nodes and
                actual_labeled_tets == expected_labeled_tets,
            f"{path}: anatomical region label coverage header differs from source records")
    channels = {
        name: header.get(name + "_available") == "1"
        for name in ("bulk_reaction", "contact_force", "mechanical_residual",
                     "free_internal_force", "tag2_attachment_force")
    }
    for name in ("bulk_reaction", "contact_force", "mechanical_residual"):
        require(header.get(name + "_available") in ("0", "1"),
                f"{path}: missing {name} availability declaration")
    tag2_source_node_count = sum(node["constraint_tag"] == 2.0
                                 for node in nodes.values())
    if tag2_source_node_count:
        require(header.get("tag2_attachment_force_available") == "1",
                f"{path}: active tag-2 source rows are missing their attachment force channel")
    else:
        has_tag2_force = any(any(component != 0.0 for component in node["tag2_attachment_force"])
                             for node in nodes.values())
        require(not has_tag2_force,
                f"{path}: tag-2 attachment force is nonzero but no tag-2 source rows exist")
    return {"path": str(path), "sha256": sha256(path), "header": header,
            "nodes": nodes, "tets": tets, "channels": channels,
            "tag2_source_node_count": tag2_source_node_count,
            "anchor_reactions": reactions}


def _same_float(a: float, b: float) -> bool:
    return a == b


def _decimal_header_number(fields: dict[str, str], key: str, context: str) -> Decimal:
    require(key in fields, f"{context} missing {key}")
    try:
        value = Decimal(fields[key])
    except InvalidOperation as error:
        raise ComparisonError(f"{context} has invalid {key}: {fields[key]}") from error
    require(value.is_finite(), f"{context} has non-finite {key}")
    return value


def _matched_endpoint_header(header: dict[str, str], context: str) -> dict[str, Any]:
    steps = _integer(header, "steps", context)
    require(steps > 0, f"{context} steps must be positive")
    dt = _decimal_header_number(header, "dt_s", context)
    endpoint = _decimal_header_number(header, "endpoint_time_s", context)
    require(dt > 0 and endpoint > 0,
            f"{context} dt_s and endpoint_time_s must be positive")
    with localcontext() as decimal_context:
        decimal_context.prec = max(50, len(dt.as_tuple().digits) + len(str(steps)) + 8)
        elapsed = dt * steps
    require(elapsed == endpoint,
            f"{context} steps * dt_s does not exactly equal endpoint_time_s")

    # Keep the remaining integration metadata validated and visible, while
    # allowing the two captures to use different substep schedules.
    runtime_dt = _decimal_header_number(header, "runtime_dt_s", context)
    multiplier = _integer(header, "timestep_multiplier", context)
    divisor = _integer(header, "timestep_divisor", context)
    microsteps = _integer(header, "completed_microsteps", context)
    return {"steps": steps, "dt_s": str(dt), "endpoint_time_s": str(endpoint),
            "runtime_dt_s": str(runtime_dt), "timestep_multiplier": multiplier,
            "timestep_divisor": divisor, "completed_microsteps": microsteps}


def validate_pair(full: dict[str, Any], reduced: dict[str, Any],
                  matched_physical_time: bool = False) -> None:
    a, b = full["header"], reduced["header"]
    integration_fields = ("steps", "dt_s", "endpoint_time_s", "runtime_dt_s",
                          "timestep_multiplier", "timestep_divisor", "completed_microsteps")
    if matched_physical_time:
        full_endpoint = _matched_endpoint_header(a, "full endpoint")
        reduced_endpoint = _matched_endpoint_header(b, "reduced endpoint")
        require(Decimal(full_endpoint["endpoint_time_s"]) ==
                Decimal(reduced_endpoint["endpoint_time_s"]),
                "full and reduced endpoints differ in endpoint_time_s")
    else:
        for key in integration_fields:
            require(key in a and key in b,
                    f"same-time comparison requires header field {key}")
            if key in ("steps", "timestep_multiplier", "timestep_divisor", "completed_microsteps"):
                require(_integer(a, key, "full endpoint") ==
                        _integer(b, key, "reduced endpoint"),
                        f"full and reduced endpoints differ in {key}")
            else:
                require(_same_float(_number(a, key, "full endpoint"),
                                    _number(b, key, "reduced endpoint")),
                        f"full and reduced endpoints differ in {key}")
    for channel in full["channels"]:
        require(full["channels"][channel] == reduced["channels"][channel],
                f"force channel availability differs for {channel}")
    require(set(full["nodes"]) == set(reduced["nodes"]),
            "full/reduced source object-local node IDs differ")
    require(set(full["tets"]) == set(reduced["tets"]),
            "full/reduced source object-local tetrahedron IDs differ")
    for key, fn in full["nodes"].items():
        rn = reduced["nodes"][key]
        for field in ("global", "anchor_body", "constraint_tag"):
            require(fn[field] == rn[field], f"source node {key} changed {field}")
        for field in ("initial", "rest", "mass_kg", "initial_mass_kg", "inverse_mass"):
            require(fn[field] == rn[field], f"source node {key} changed {field}")
        if fn["region_id"] is not None and rn["region_id"] is not None:
            require(fn["region_id"] == rn["region_id"] and
                        fn["region_source"] == rn["region_source"],
                    f"source node {key} changed registered anatomical region label")
    for key, ft in full["tets"].items():
        rt = reduced["tets"][key]
        require(ft["global"] == rt["global"] and ft["nodes"] == rt["nodes"],
                f"source tetrahedron {key} changed IDs or connectivity")
        if ft["region_id"] is not None and rt["region_id"] is not None:
            require(ft["region_id"] == rt["region_id"] and
                        ft["region_source"] == rt["region_source"],
                    f"source tetrahedron {key} changed registered anatomical region label")
    if matched_physical_time:
        require(bool(full["anchor_reactions"]) == bool(reduced["anchor_reactions"]),
                "accepted articulated-anchor endpoint availability differs")
    else:
        require(set(full["anchor_reactions"]) == set(reduced["anchor_reactions"]),
                "accepted articulated-anchor history indices differ")


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def force_channel_metrics(full: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    objects = sorted({key[0] for key in full["nodes"]})
    result: dict[str, Any] = {}
    any_reference_force = False
    all_channel_pass = True
    channels = ("bulk_reaction", "contact_force", "free_internal_force",
                "tag2_attachment_force", "supported_force_sum")
    for channel in channels:
        per_object: dict[str, Any] = {}
        for obj in objects:
            ref_l1 = cand_l1 = delta_l1 = max_node_delta = 0.0
            max_node: int | None = None
            for key, fn in full["nodes"].items():
                if key[0] != obj:
                    continue
                rn = reduced["nodes"][key]
                if channel == "supported_force_sum":
                    fv = tuple(fn["bulk_reaction"][i] + fn["contact_force"][i]
                               for i in range(3))
                    rv = tuple(rn["bulk_reaction"][i] + rn["contact_force"][i]
                               for i in range(3))
                else:
                    fv, rv = fn[channel], rn[channel]
                ref_l1 += _norm(fv)
                cand_l1 += _norm(rv)
                delta = _norm(_sub(rv, fv))
                delta_l1 += delta
                if delta > max_node_delta:
                    max_node_delta, max_node = delta, key[1]
            has_signal = ref_l1 > 0.0
            any_reference_force = any_reference_force or has_signal
            if ref_l1 > 0.0:
                relative = delta_l1 / ref_l1
                status = "pass" if relative <= FORCE_THRESHOLD else "fail"
                all_channel_pass = all_channel_pass and status == "pass"
            elif cand_l1 == 0.0:
                relative = None
                status = "no_reference_signal"
            else:
                relative = None
                status = "fail_zero_reference"
                all_channel_pass = False
            per_object[obj] = {
                "reference_l1_N": ref_l1,
                "candidate_l1_N": cand_l1,
                "difference_l1_N": delta_l1,
                "relative_l1_error": relative,
                "maximum_node_difference_N": max_node_delta,
                "maximum_node_local_id": max_node,
                "status": status,
            }
        result[channel] = per_object
    matched_region_nodes: dict[int, list[tuple[tuple[str, int], dict[str, Any], dict[str, Any]]]] = {}
    reference_region_nodes = sum(n["region_id"] is not None for n in full["nodes"].values())
    candidate_region_nodes = sum(n["region_id"] is not None for n in reduced["nodes"].values())
    for key, fn in full["nodes"].items():
        rn = reduced["nodes"][key]
        if fn["region_id"] is not None and rn["region_id"] is not None:
            matched_region_nodes.setdefault(fn["region_id"], []).append((key, fn, rn))
    regional_result: dict[str, Any] = {}
    all_regional_pass = bool(matched_region_nodes)
    regional_reference_signal = False
    for channel in channels:
        by_region: dict[str, Any] = {}
        for region, rows in sorted(matched_region_nodes.items()):
            ref_l1 = cand_l1 = delta_l1 = 0.0
            for _, fn, rn in rows:
                if channel == "supported_force_sum":
                    fv = tuple(fn["bulk_reaction"][i] + fn["contact_force"][i]
                               for i in range(3))
                    rv = tuple(rn["bulk_reaction"][i] + rn["contact_force"][i]
                               for i in range(3))
                else:
                    fv, rv = fn[channel], rn[channel]
                ref_l1 += _norm(fv)
                cand_l1 += _norm(rv)
                delta_l1 += _norm(_sub(rv, fv))
            if ref_l1 > 0.0:
                regional_reference_signal = True
                relative = delta_l1 / ref_l1
                status = "pass" if relative <= FORCE_THRESHOLD else "fail"
                all_regional_pass = all_regional_pass and status == "pass"
            elif cand_l1 == 0.0:
                relative, status = None, "no_reference_signal"
            else:
                relative, status = None, "fail_zero_reference"
                all_regional_pass = False
            by_region[str(region)] = {
                "source_label": "registered_skin_fat_donor_body_id",
                "source_nodes": len(rows), "reference_l1_N": ref_l1,
                "candidate_l1_N": cand_l1, "difference_l1_N": delta_l1,
                "relative_l1_error": relative, "status": status,
            }
        regional_result[channel] = by_region
    regional_gate = ("pass" if all_regional_pass and regional_reference_signal else
                     "fail" if not all_regional_pass else "incomplete_or_no_signal")
    tag2_source_nodes = full["tag2_source_node_count"]
    tag2_applicability = ("active_tag2_source_rows" if tag2_source_nodes
                          else "not_applicable_no_tag2_source_nodes")
    if not tag2_source_nodes:
        for entry in result["tag2_attachment_force"].values():
            entry["status"] = "not_applicable_no_tag2_source_nodes"
        for entry in regional_result["tag2_attachment_force"].values():
            entry["status"] = "not_applicable_no_tag2_source_nodes"
    required_channels = ("bulk_reaction", "contact_force", "free_internal_force")
    channels_available = all(full["channels"][name] and reduced["channels"][name]
                             for name in required_channels)
    if tag2_source_nodes:
        channels_available = (channels_available and
                              full["channels"]["tag2_attachment_force"] and
                              reduced["channels"]["tag2_attachment_force"])
    passed = channels_available and any_reference_force and all_channel_pass
    return {
        "units": "N",
        "definition": "per-node channel comparisons keep fixed bulk reaction, contact supplement, free-node bulk element force, and tag-2 attachment reaction separate; supported_force_sum remains bulk reaction plus contact supplement",
        "threshold_relative_l1_error": FORCE_THRESHOLD,
        "channels_available": channels_available,
        "tag2_attachment_force_applicability": tag2_applicability,
        "tag2_attachment_source_node_count": tag2_source_nodes,
        "reference_has_nonzero_signal": any_reference_force,
        "per_channel_per_object": result,
        "per_channel_per_anatomical_region": regional_result,
        "anatomical_region_coverage": {
            "label_source": "registered_skin_fat_donor_body_id",
            "reference_labeled_nodes": reference_region_nodes,
            "candidate_labeled_nodes": candidate_region_nodes,
            "matched_labeled_nodes": sum(len(rows) for rows in matched_region_nodes.values()),
            "total_source_nodes": len(full["nodes"]),
            "complete_for_all_source_nodes": len(matched_region_nodes) > 0 and
                sum(len(rows) for rows in matched_region_nodes.values()) == len(full["nodes"]),
            "coverage_gap": (None if len(matched_region_nodes) > 0 and
                sum(len(rows) for rows in matched_region_nodes.values()) == len(full["nodes"]) else
                "Registered donor-body node labels cover only the verified skin/fat source shell; other source objects have no anatomical-region labels in this capture."),
        },
        "anatomical_region_gate": regional_gate,
        "gate": "pass" if passed and regional_gate == "pass" else
            ("fail" if not all_channel_pass or regional_gate == "fail" else "incomplete_or_no_signal"),
        "coverage_complete_for_all_physical_forces": False,
        "coverage_limit": "free-node element force is constitutive bulk only with field-driven activation; borrowed prescribed tension is not captured; articulated tag-2 reaction is the final accepted-solve linearization; cohesive, MPM, vascular cavity, and other owner channels remain outside this capture",
    }


def _matmul(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _transpose(a: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _det(a: tuple[tuple[float, ...], ...]) -> float:
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _inverse(a: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    d = _det(a)
    require(math.isfinite(d) and abs(d) > 1.0e-30,
            "source tetrahedron rest matrix is singular")
    return (
        ((a[1][1]*a[2][2]-a[1][2]*a[2][1])/d,
         (a[0][2]*a[2][1]-a[0][1]*a[2][2])/d,
         (a[0][1]*a[1][2]-a[0][2]*a[1][1])/d),
        ((a[1][2]*a[2][0]-a[1][0]*a[2][2])/d,
         (a[0][0]*a[2][2]-a[0][2]*a[2][0])/d,
         (a[0][2]*a[1][0]-a[0][0]*a[1][2])/d),
        ((a[1][0]*a[2][1]-a[1][1]*a[2][0])/d,
         (a[0][1]*a[2][0]-a[0][0]*a[2][1])/d,
         (a[0][0]*a[1][1]-a[0][1]*a[1][0])/d),
    )


def _green_strain(nodes: dict[tuple[str, int], dict[str, Any]],
                  tet: dict[str, Any], position_field: str) -> tuple[tuple[float, ...], float, float]:
    obj = tet["object"]
    global_to_key = {n["global"]: key for key, n in nodes.items() if key[0] == obj}
    try:
        vertex_keys = [global_to_key[index] for index in tet["nodes"]]
    except KeyError as error:
        raise ComparisonError(f"tetrahedron {(obj, tet['local'])} references missing node {error}") from error
    x = [nodes[key]["rest"] for key in vertex_keys]
    y = [nodes[key][position_field] for key in vertex_keys]
    dm = tuple(tuple(x[col][row] - x[0][row] for col in range(1, 4))
               for row in range(3))
    ds = tuple(tuple(y[col][row] - y[0][row] for col in range(1, 4))
               for row in range(3))
    dm_det = _det(dm)
    require(math.isfinite(dm_det) and abs(dm_det) > 1.0e-30,
            f"tetrahedron {(obj, tet['local'])} has singular source rest geometry")
    f = _matmul(ds, _inverse(dm))
    c = _matmul(_transpose(f), f)
    strain = tuple(0.5 * (c[i][j] - (1.0 if i == j else 0.0))
                   for i in range(3) for j in range(3))
    j_value = _det(f)
    require(math.isfinite(j_value) and j_value > 0.0,
            f"tetrahedron {(obj, tet['local'])} has nonpositive Green-strain deformation")
    return strain, abs(dm_det) / 6.0, j_value


def _strain_norm(x: tuple[float, ...]) -> float:
    return math.sqrt(sum(v * v for v in x))


def green_strain_metrics(full: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    objects = sorted({key[0] for key in full["tets"]})
    per_object_tets: dict[str, list[dict[str, Any]]] = {obj: [] for obj in objects}
    per_region_tets: dict[str, list[dict[str, Any]]] = {}
    reference_labeled = candidate_labeled = matched_labeled = 0
    unlabeled_by_object: dict[str, int] = {}
    all_tet_metrics: list[dict[str, Any]] = []
    for key, ft in full["tets"].items():
        rt = reduced["tets"][key]
        ef, volume, jf = _green_strain(full["nodes"], ft, "current")
        er, reduced_volume, jr = _green_strain(reduced["nodes"], rt, "current")
        require(volume == reduced_volume,
                f"source rest volume changed for tetrahedron {key}")
        difference = _strain_norm(tuple(er[i] - ef[i] for i in range(9)))
        reference = _strain_norm(ef)
        if reference > 0.0:
            relative: float | None = difference / reference
            unbounded = False
        elif difference == 0.0:
            relative = 0.0
            unbounded = False
        else:
            # A nonzero candidate strain against an exactly unstrained
            # reference has unbounded relative error; keep JSON finite.
            relative = None
            unbounded = True
        metric = {"object": key[0], "local_tet": key[1],
                  "region_id": ft["region_id"], "region_source": ft["region_source"],
                  "volume": volume, "reference_norm": reference,
                  "difference_norm": difference, "relative_error": relative,
                  "unbounded_relative_error": unbounded,
                  "full_J": jf, "reduced_J": jr}
        all_tet_metrics.append(metric)
        per_object_tets[key[0]].append(metric)
        if ft["region_id"] is not None:
            reference_labeled += 1
        if rt["region_id"] is not None:
            candidate_labeled += 1
        if ft["region_id"] is not None and rt["region_id"] is not None:
            # validate_pair already checked equality when both sides carry labels.
            matched_labeled += 1
            label = str(ft["region_id"])
            per_region_tets.setdefault(label, []).append(metric)
        if ft["region_id"] is None or rt["region_id"] is None:
            unlabeled_by_object[key[0]] = unlabeled_by_object.get(key[0], 0) + 1

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        weighted_difference = weighted_reference = total_volume = 0.0
        max_finite_relative = 0.0
        max_absolute = 0.0
        max_witness: dict[str, Any] | None = None
        unbounded_witness: dict[str, Any] | None = None
        minimum_j = math.inf
        at_or_over_threshold = 0
        for item in items:
            weighted_difference += item["volume"] * item["difference_norm"]
            weighted_reference += item["volume"] * item["reference_norm"]
            total_volume += item["volume"]
            minimum_j = min(minimum_j, item["full_J"], item["reduced_J"])
            relative = item["relative_error"]
            if item["unbounded_relative_error"]:
                at_or_over_threshold += 1
                if unbounded_witness is None:
                    unbounded_witness = item
            elif relative is not None and relative >= STRAIN_THRESHOLD:
                at_or_over_threshold += 1
            if relative is not None and relative > max_finite_relative:
                max_finite_relative = relative
            if item["difference_norm"] > max_absolute:
                max_absolute = item["difference_norm"]
                max_witness = item
        weighted_relative = (weighted_difference / weighted_reference
                             if weighted_reference > 0.0 else
                             (0.0 if weighted_difference == 0.0 else None))
        average_pass = weighted_relative is not None and weighted_relative <= STRAIN_THRESHOLD
        # Report and gate individual tetrahedra as well, so a large quiet region
        # cannot hide a localized strain mismatch in a volume average.
        status = "pass" if average_pass and at_or_over_threshold == 0 else "fail"
        selected_witness = unbounded_witness or max_witness
        return {
            "tetrahedra": len(items),
            "reference_rest_volume_m3": total_volume,
            "volume_weighted_reference_strain_l1": weighted_reference,
            "volume_weighted_difference_strain_l1": weighted_difference,
            "volume_weighted_relative_l1_error": weighted_relative,
            "maximum_per_tet_relative_error": (
                None if unbounded_witness is not None else max_finite_relative),
            "maximum_finite_per_tet_relative_error": max_finite_relative,
            "maximum_relative_error_unbounded_due_to_zero_reference":
                unbounded_witness is not None,
            "maximum_per_tet_absolute_frobenius_error": max_absolute,
            "maximum_error_witness": ({
                "object": selected_witness["object"],
                "local_tet": selected_witness["local_tet"],
                "anatomical_region_id": selected_witness["region_id"],
                "absolute_frobenius_error": selected_witness["difference_norm"],
                "relative_error": selected_witness["relative_error"],
                "unbounded_relative_error": selected_witness["unbounded_relative_error"],
            } if selected_witness is not None else None),
            "tetrahedra_error_at_or_over_threshold": at_or_over_threshold,
            "tetrahedra_error_threshold_relative": STRAIN_THRESHOLD,
            "minimum_full_or_reduced_J": minimum_j,
            "gate": status,
        }

    per_object = {obj: summarize(per_object_tets[obj]) for obj in objects}
    per_region = {label: summarize(items) for label, items in sorted(per_region_tets.items())}
    global_metrics = summarize(all_tet_metrics)
    all_pass = global_metrics["gate"] == "pass" and all(
        value["gate"] == "pass" for value in per_object.values())
    total_tets = len(full["tets"])
    region_coverage = {
        "label_source": "registered_skin_fat_donor_body_id",
        "reference_labeled_tetrahedra": reference_labeled,
        "candidate_labeled_tetrahedra": candidate_labeled,
        "matched_labeled_tetrahedra": matched_labeled,
        "total_source_tetrahedra": total_tets,
        "matched_labeled_fraction": matched_labeled / total_tets if total_tets else 0.0,
        "complete_for_all_source_tetrahedra": matched_labeled == total_tets,
        "unlabeled_tetrahedra_by_source_object": unlabeled_by_object,
        "coverage_gap": (None if matched_labeled == total_tets else
            "Registered donor-body labels cover only the verified skin/fat source shell; "
            "other source objects have no anatomical-region labels in this capture."),
    }
    return {
        "units": "dimensionless Green-Lagrange strain tensor; Frobenius norm",
        "threshold_volume_weighted_relative_l1_error": STRAIN_THRESHOLD,
        "per_tet_error_threshold_comparison": ">= threshold counts as failure",
        "per_object_region": per_object,
        "per_anatomical_region": per_region,
        "anatomical_region_coverage": region_coverage,
        "global_volume_weighted_relative_l1_error": global_metrics["volume_weighted_relative_l1_error"],
        "maximum_per_tet_relative_error": global_metrics["maximum_per_tet_relative_error"],
        "maximum_finite_per_tet_relative_error": global_metrics["maximum_finite_per_tet_relative_error"],
        "maximum_relative_error_unbounded_due_to_zero_reference":
            global_metrics["maximum_relative_error_unbounded_due_to_zero_reference"],
        "maximum_absolute_frobenius_error": global_metrics["maximum_per_tet_absolute_frobenius_error"],
        "maximum_error_witness": global_metrics["maximum_error_witness"],
        "tetrahedra_error_at_or_over_threshold":
            global_metrics["tetrahedra_error_at_or_over_threshold"],
        "gate": "pass" if all_pass else "fail",
    }


def residual_impulse_metrics(full: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    if not (full["channels"]["mechanical_residual"] and
            reduced["channels"]["mechanical_residual"]):
        return {"available": False, "units": "N s",
                "meaning": "accepted terminal mechanical residual impulse imbalance; not a physical force field",
                "converted_to_newtons": False, "per_object_region": {}}
    result: dict[str, Any] = {}
    all_ref = all_cand = all_delta = 0.0
    for obj in sorted({key[0] for key in full["nodes"]}):
        ref_l1 = cand_l1 = delta_l1 = 0.0
        for key, fn in full["nodes"].items():
            if key[0] != obj:
                continue
            rv = reduced["nodes"][key]["residual_impulse"]
            fv = fn["residual_impulse"]
            ref_l1 += _norm(fv)
            cand_l1 += _norm(rv)
            delta_l1 += _norm(_sub(rv, fv))
        relative = delta_l1 / ref_l1 if ref_l1 > 0.0 else (0.0 if cand_l1 == 0.0 else None)
        result[obj] = {"reference_l1_N_s": ref_l1,
                       "candidate_l1_N_s": cand_l1,
                       "difference_l1_N_s": delta_l1,
                       "relative_l1_error": relative}
        all_ref += ref_l1
        all_cand += cand_l1
        all_delta += delta_l1
    return {"available": True, "units": "N s", "meaning": "accepted terminal mechanical residual impulse imbalance; not a physical force field",
            "converted_to_newtons": False,
            "per_object_region": result,
            "global_relative_l1_error": all_delta / all_ref if all_ref > 0.0 else (0.0 if all_cand == 0.0 else None)}


def anchor_history_metrics(full: dict[str, Any], reduced: dict[str, Any],
                           matched_physical_time: bool = False) -> dict[str, Any]:
    difference_l1 = reference_l1 = candidate_l1 = 0.0
    if matched_physical_time and full["anchor_reactions"]:
        full_step = max(full["anchor_reactions"])
        reduced_step = max(reduced["anchor_reactions"])
        comparisons = [(full_step, full["anchor_reactions"][full_step],
                       reduced["anchor_reactions"][reduced_step])]
    else:
        full_step = reduced_step = None
        comparisons = [(step, ref, reduced["anchor_reactions"][step])
                       for step, ref in full["anchor_reactions"].items()]
    for _, ref, candidate in comparisons:
        reference_l1 += _norm(ref)
        candidate_l1 += _norm(candidate)
        difference_l1 += _norm(_sub(candidate, ref))
    relative = difference_l1 / reference_l1 if reference_l1 > 0.0 else (
        0.0 if candidate_l1 == 0.0 else None)
    if not full["anchor_reactions"]:
        status = "unavailable_no_accepted_anchor_reaction_steps"
    elif reference_l1 == 0.0 and candidate_l1 == 0.0:
        status = "no_reference_signal"
    elif relative is None or relative > FORCE_THRESHOLD:
        status = "fail"
    else:
        status = "pass"
    result = {"units": "N", "steps": len(comparisons),
              "reference_l1_N": reference_l1, "candidate_l1_N": candidate_l1,
              "difference_l1_N": difference_l1, "relative_l1_error": relative,
              "threshold_relative_l1_error": FORCE_THRESHOLD, "gate": status}
    if matched_physical_time:
        result.update({"comparison_scope": "final_accepted_endpoint_only",
                       "full_final_accepted_step": full_step,
                       "candidate_final_accepted_step": reduced_step})
    return result


def compare(full_path: Path, reduced_path: Path,
            matched_physical_time: bool = False) -> dict[str, Any]:
    full, reduced = parse_capture(full_path), parse_capture(reduced_path)
    validate_pair(full, reduced, matched_physical_time=matched_physical_time)
    force = force_channel_metrics(full, reduced)
    anchor_history = anchor_history_metrics(
        full, reduced, matched_physical_time=matched_physical_time)
    if anchor_history["gate"] == "fail":
        force["gate"] = "fail"
    elif force["gate"] == "pass" and anchor_history["gate"] not in ("pass", "no_reference_signal"):
        force["gate"] = "incomplete_or_no_signal"
    strain = green_strain_metrics(full, reduced)
    residual = residual_impulse_metrics(full, reduced)
    attachment_nodes = sum(1 for n in full["nodes"].values()
                           if n["constraint_tag"] == 2.0)
    report = {
        "source": {"full_log": full["path"], "full_log_sha256": full["sha256"],
                   "reduced_log": reduced["path"], "reduced_log_sha256": reduced["sha256"],
                   "source_node_count": len(full["nodes"]),
                   "source_tetrahedron_count": len(full["tets"]),
                   "source_ids_mass_connectivity_rest_pose_exact": True},
        "matched_endpoint": ({
            "comparison_scope": "matched_physical_time",
            "endpoint_time_s": full["header"]["endpoint_time_s"],
            "full_header": dict(full["header"]),
            "candidate_header": dict(reduced["header"]),
        } if matched_physical_time else {
            key: full["header"][key] for key in
            ("steps", "dt_s", "endpoint_time_s", "runtime_dt_s",
             "timestep_multiplier", "timestep_divisor", "completed_microsteps")
        }),
        "coordinate_projection": {
            "full_endpoint": full["header"].get("runtime_coordinate_projection"),
            "candidate_endpoint": reduced["header"].get("runtime_coordinate_projection"),
            "actual_reduced_coordinate_comparison":
                full["header"].get("runtime_coordinate_projection") == "disabled" and
                reduced["header"].get("runtime_coordinate_projection") == "enabled",
            "qualification_note": "endpoint strain/force deltas qualify reduced coordinates only when the full endpoint used disabled projection and the candidate endpoint explicitly reports enabled projection",
        },
        "force_fidelity": force,
        "accepted_anchor_reaction_history": anchor_history,
        "green_strain_fidelity": strain,
        "mechanical_residual_impulse": residual,
        "owner_coverage": {
            "tag2_articulated_attachment_nodes": attachment_nodes,
            "per_node_tag2_attachment_force_available": full["channels"]["tag2_attachment_force"],
            "per_node_tag2_attachment_force_applicability": force["tag2_attachment_force_applicability"],
            "tag2_attachment_source_node_count": full["tag2_source_node_count"],
            "full_free_node_internal_force_available": full["channels"]["free_internal_force"],
            "anatomical_region_force_coverage": force["anatomical_region_coverage"],
            "anatomical_region_strain_coverage": strain["anatomical_region_coverage"],
            "accepted_aggregate_anchor_reaction_steps": len(full["anchor_reactions"]),
            "accepted_aggregate_anchor_reaction_available": bool(full["anchor_reactions"]),
            "force_gate_is_complete_for_all_physical_owners": False,
        },
        "gates": {
            "supported_reaction_force_relative_l1_le_1_percent": force["gate"] == "pass",
            "source_tet_green_strain_volume_weighted_l1_le_5_percent": strain["gate"] == "pass",
            "source_tet_green_strain_each_tet_below_5_percent":
                strain["tetrahedra_error_at_or_over_threshold"] == 0,
            "measured_channel_fidelity_pass": force["gate"] == "pass" and strain["gate"] == "pass",
            "reduced_coordinate_mode_verified":
                full["header"].get("runtime_coordinate_projection") == "disabled" and
                reduced["header"].get("runtime_coordinate_projection") == "enabled",
            "reduced_coordinate_fidelity_pass":
                full["header"].get("runtime_coordinate_projection") == "disabled" and
                reduced["header"].get("runtime_coordinate_projection") == "enabled" and
                force["gate"] == "pass" and strain["gate"] == "pass",
            "complete_physical_force_fidelity_pass": False,
            "physical_qualification": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_log", type=Path)
    parser.add_argument("reduced_log", type=Path)
    parser.add_argument("--merge-existing-summary", type=Path,
                        help="add endpoint_fidelity to an existing JSON run summary")
    parser.add_argument("--matched-physical-time", action="store_true",
                        help="compare captures at the same exact physical endpoint despite different step schedules")
    args = parser.parse_args()
    try:
        report = compare(args.full_log, args.reduced_log,
                         matched_physical_time=args.matched_physical_time)
        if args.merge_existing_summary is not None:
            summary_path = args.merge_existing_summary
            summary = json.loads(summary_path.read_text())
            require(isinstance(summary, dict), "existing JSON summary root must be an object")
            summary["endpoint_fidelity"] = report
            tmp = summary_path.with_name(summary_path.name + ".tmp")
            tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, summary_path)
        print(json.dumps({"endpoint_fidelity": report}, indent=2, sort_keys=True,
                         allow_nan=False))
        return 0 if report["gates"]["measured_channel_fidelity_pass"] else 2
    except (ComparisonError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"endpoint_fidelity": {"gate": "unavailable",
                         "reason": str(error)}}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
