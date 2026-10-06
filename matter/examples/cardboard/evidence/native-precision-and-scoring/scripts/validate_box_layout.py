#!/usr/bin/env python3
"""Validate exported FEFCO 0201 layout geometry and action references.

This checks metadata and reference closure only. It does not claim that the
unscored blank has been physically folded, bonded, or validated against a box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCORE_ROOT = 0
SCORE_CORNER = 1
SCORE_BOTTOM = 2
SCORE_TOP = 3
ACTION_WALL = 0
ACTION_BOND = 1
ACTION_FLAP = 2
ACTION_SEAL = 3
PANEL_ORDER = [1, 3, 0, 2]


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def close(actual: Any, expected: float, label: str, *, tol: float = 2e-12) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool),
            f"{label}: expected a number, got {actual!r}")
    require(math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, rel_tol=tol, abs_tol=tol),
        f"{label}: got {actual!r}, expected {expected:.17g}")


def integer(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{label}: expected integer, got {value!r}")
    return value


def validate_vec3(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 3,
            f"{label}: expected 3-vector")
    out: list[float] = []
    for axis, item in zip("xyz", value):
        require(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(float(item)), f"{label}.{axis}: invalid coordinate")
        out.append(float(item))
    return out


def validate_bounds(bounds: Any, expected: dict[str, tuple[float, float]], label: str) -> None:
    require(isinstance(bounds, dict), f"{label}: missing bounds_m")
    for axis in "xyz":
        values = bounds.get(axis)
        require(isinstance(values, list) and len(values) == 2,
                f"{label}.{axis}: expected [min,max]")
        close(values[0], expected[axis][0], f"{label}.{axis}[0]")
        close(values[1], expected[axis][1], f"{label}.{axis}[1]")
        require(float(values[1]) > float(values[0]), f"{label}.{axis}: empty bounds")


def dimensions(layout: dict[str, Any]) -> dict[str, float]:
    raw = layout.get("dimensions")
    require(isinstance(raw, dict), "missing dimensions object")
    keys = ("panel_length", "panel_width", "wall_height", "joint_width", "slot_kerf")
    out: dict[str, float] = {}
    for key in keys:
        value = raw.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) > 0.0,
                f"dimensions.{key}: expected finite positive number")
        out[key] = float(value)
    return out


def validate_score_lines(layout: dict[str, Any], d: dict[str, float],
                         *, v2: bool) -> list[dict[str, Any]]:
    lines = layout.get("score_lines")
    require(isinstance(lines, list) and len(lines) == 12,
            "expected four body/root and eight flap score lines")
    L, W, H, J = d["panel_length"], d["panel_width"], d["wall_height"], d["joint_width"]
    flap = 0.5 * W
    bottom_y, top_y = flap, flap + H
    x_edges = [J, J + L, J + L + W, J + 2 * L + W, J + 2 * L + 2 * W]
    for idx, row in enumerate(lines):
        require(isinstance(row, dict), f"score_lines[{idx}]: expected object")
        kind = integer(row.get("kind"), f"score_lines[{idx}].kind")
        panel = integer(row.get("panel"), f"score_lines[{idx}].panel")
        first = validate_vec3(row.get("first"), f"score_lines[{idx}].first")
        second = validate_vec3(row.get("second"), f"score_lines[{idx}].second")
        require(row.get("physically_scored") is False,
                f"score_lines[{idx}] must remain metadata-only/unscored")
        if idx == 0:
            expected = (SCORE_ROOT, 0, [J, bottom_y, 0.0], [J, top_y, 0.0])
        elif idx in (1, 2, 3):
            expected = (SCORE_CORNER, idx - 1,
                        [x_edges[idx], bottom_y, 0.0], [x_edges[idx], top_y, 0.0])
        else:
            flap_idx = idx - 4
            panel_idx, is_top = divmod(flap_idx, 2)
            kind_expected = SCORE_TOP if is_top else SCORE_BOTTOM
            y = top_y if is_top else bottom_y
            expected = (kind_expected, panel_idx,
                        [x_edges[panel_idx], y, 0.0],
                        [x_edges[panel_idx + 1], y, 0.0])
        require((kind, panel) == expected[:2],
                f"score_lines[{idx}] kind/panel does not match generated order")
        for axis, actual, target in zip("xyz", first, expected[2]):
            close(actual, target, f"score_lines[{idx}].first.{axis}")
        for axis, actual, target in zip("xyz", second, expected[3]):
            close(actual, target, f"score_lines[{idx}].second.{axis}")
    return lines


def expected_actions(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for panel in range(4):
        result.append({"kind": ACTION_WALL, "type": "panel", "index": panel,
                       "hinge": panel, "angle": 90.0})
    result.append({"kind": ACTION_BOND, "type": "manufacturer_joint", "index": 0,
                   "hinge": None, "angle": 0.0})
    for is_top in (False, True):
        score_kind = SCORE_TOP if is_top else SCORE_BOTTOM
        for panel in PANEL_ORDER:
            score_index = next((i for i, row in enumerate(lines)
                                if row["kind"] == score_kind and row["panel"] == panel), None)
            require(score_index is not None, f"no score line for panel {panel}, end top={is_top}")
            result.append({"kind": ACTION_FLAP, "type": "score_line", "index": score_index,
                           "hinge": None, "angle": -90.0 if is_top else 90.0})
        result.append({"kind": ACTION_SEAL, "type": "box_end", "index": 1 if is_top else 0,
                       "hinge": None, "angle": 0.0})
    return result


def validate_actions(layout: dict[str, Any], lines: list[dict[str, Any]], *, v2: bool) -> None:
    actions = layout.get("assembly_actions")
    require(isinstance(actions, list) and len(actions) == 15,
            "expected 15 ordered, not-yet-executed assembly targets")
    expected = expected_actions(lines)
    semantics = {
        "panel": "zero-based index into panels",
        "score_line": "zero-based index into score_lines",
        "manufacturer_joint": "0 identifies the manufacturer_joint object",
        "box_end": "zero-based index into box_ends: 0=bottom, 1=top",
    }
    for idx, (row, target) in enumerate(zip(actions, expected)):
        require(isinstance(row, dict), f"assembly_actions[{idx}]: expected object")
        require(integer(row.get("kind"), f"assembly_actions[{idx}].kind") == target["kind"],
                f"assembly_actions[{idx}]: wrong action kind/order")
        close(row.get("target_angle_deg"), target["angle"],
              f"assembly_actions[{idx}].target_angle_deg")
        require(row.get("executed") is False,
                f"assembly_actions[{idx}] must not claim execution")
        require(isinstance(row.get("stage"), str) and row["stage"],
                f"assembly_actions[{idx}].stage: missing stage label")
        if v2:
            hinge = row.get("hinge_score_line_index")
            require(hinge == target["hinge"],
                    f"assembly_actions[{idx}]: hinge reference {hinge!r}, expected {target['hinge']!r}")
            feature = row.get("feature")
            require(isinstance(feature, dict), f"assembly_actions[{idx}].feature missing")
            require(feature.get("type") == target["type"],
                    f"assembly_actions[{idx}]: wrong feature type")
            require(integer(feature.get("index"), f"assembly_actions[{idx}].feature.index") == target["index"],
                    f"assembly_actions[{idx}]: wrong feature index")
            require(feature.get("index_semantics") == semantics[target["type"]],
                    f"assembly_actions[{idx}]: index semantics do not name exported JSON feature")
            if target["type"] == "score_line":
                require(target["index"] < len(lines), f"assembly_actions[{idx}]: score reference out of range")
                require(lines[target["index"]]["kind"] in (SCORE_BOTTOM, SCORE_TOP),
                        f"assembly_actions[{idx}]: flap action points to non-flap score")
        else:
            # Legacy v1 stores an untyped integer, which is intentionally not
            # interpreted as a v2 feature-array index.
            integer(row.get("feature"), f"assembly_actions[{idx}].feature")


def validate_v2(layout: dict[str, Any]) -> dict[str, Any]:
    require(layout.get("schema") == "numi.cardboard.fefco0201-layout.v2",
            "not a supported v2 FEFCO 0201 layout")
    require(layout.get("units") == "metres", "layout units must be metres")
    require(layout.get("state") == "flat_unscored_blank", "unexpected layout state")
    require(layout.get("physical_assembly") is False, "layout must not claim physical assembly")
    d = dimensions(layout)
    ext = layout["dimensions"]
    L, W, H, J = d["panel_length"], d["panel_width"], d["wall_height"], d["joint_width"]
    caliper = ext.get("board_caliper")
    perimeter = J + 2.0 * (L + W)
    height = H + W
    flap = 0.5 * W
    derived = {
        "board_caliper": caliper,
        "perimeter_span": perimeter,
        "blank_height": height,
        "bottom_score_y": flap,
        "top_score_y": flap + H,
        "flap_depth": flap,
    }
    require(isinstance(caliper, (int, float)) and not isinstance(caliper, bool)
            and math.isfinite(float(caliper)) and float(caliper) > 0.0,
            "dimensions.board_caliper must be finite and positive")
    for key, target in derived.items():
        close(ext.get(key), float(target), f"dimensions.{key}")
    lines = validate_score_lines(layout, d, v2=True)

    panels = layout.get("panels")
    require(isinstance(panels, list) and len(panels) == 4,
            "v2 must export all four panels")
    x_edges = [J, J + L, J + L + W, J + 2 * L + W, perimeter]
    for idx, panel in enumerate(panels):
        require(isinstance(panel, dict), f"panels[{idx}]: expected object")
        require(integer(panel.get("index"), f"panels[{idx}].index") == idx,
                f"panels[{idx}]: array/index mismatch")
        require(panel.get("long_wall") is (idx in (0, 2)),
                f"panels[{idx}]: wrong long-wall classification")
        require(panel.get("bounds_kind") == "axis_aligned_envelope",
                f"panels[{idx}]: bounds kind must be explicit")
        validate_bounds(panel.get("bounds_m"),
                        {"x": (x_edges[idx], x_edges[idx + 1]),
                         "y": (flap, flap + H), "z": (0.0, float(caliper))},
                        f"panels[{idx}].bounds_m")

    joint = layout.get("manufacturer_joint")
    require(isinstance(joint, dict), "v2 must export manufacturer_joint bounds")
    require(integer(joint.get("index"), "manufacturer_joint.index") == 0,
            "manufacturer_joint singleton index must be zero")
    require(joint.get("bounds_kind") == "axis_aligned_envelope",
            "manufacturer_joint bounds kind must be explicit")
    validate_bounds(joint.get("bounds_m"),
                    {"x": (0.0, J), "y": (flap, flap + H),
                     "z": (0.0, float(caliper))}, "manufacturer_joint.bounds_m")
    root_index = integer(joint.get("root_score_line_index"),
                         "manufacturer_joint.root_score_line_index")
    require(root_index < len(lines) and lines[root_index]["kind"] == SCORE_ROOT,
            "manufacturer_joint root score-line reference is invalid")

    ends = layout.get("box_ends")
    require(isinstance(ends, list) and len(ends) == 2, "v2 must export bottom and top box_ends")
    for idx, end in enumerate(ends):
        top = idx == 1
        require(isinstance(end, dict), f"box_ends[{idx}]: expected object")
        require(integer(end.get("index"), f"box_ends[{idx}].index") == idx,
                f"box_ends[{idx}]: array/index mismatch")
        require(end.get("end") == ("top" if top else "bottom"),
                f"box_ends[{idx}]: wrong end identity")
        require(end.get("bounds_kind") == "axis_aligned_envelope",
                f"box_ends[{idx}]: bounds kind must be explicit")
        y0, y1 = (flap + H, height) if top else (0.0, flap)
        validate_bounds(end.get("bounds_m"),
                        {"x": (J, perimeter), "y": (y0, y1),
                         "z": (0.0, float(caliper))}, f"box_ends[{idx}].bounds_m")
        panel_indices = end.get("panel_indices")
        require(panel_indices == [0, 1, 2, 3], f"box_ends[{idx}]: panel references incomplete")
        target_kind = SCORE_TOP if top else SCORE_BOTTOM
        target_scores = [i for i, line in enumerate(lines) if line["kind"] == target_kind]
        require(end.get("flap_score_line_indices") == target_scores,
                f"box_ends[{idx}]: flap score-line references do not resolve")
    validate_actions(layout, lines, v2=True)
    return {"version": "v2", "panels": len(panels), "score_lines": len(lines),
            "assembly_actions": len(layout["assembly_actions"]), "physical_assembly": False}


def validate_v1(layout: dict[str, Any]) -> dict[str, Any]:
    require(layout.get("schema") == "numi.cardboard.fefco0201-layout.v1",
            "not a supported v1 FEFCO 0201 layout")
    require(layout.get("units") == "metres", "layout units must be metres")
    require(layout.get("state") == "flat_unscored_blank", "unexpected layout state")
    require(layout.get("physical_assembly") is False, "layout must not claim physical assembly")
    d = dimensions(layout)
    lines = validate_score_lines(layout, d, v2=False)
    validate_actions(layout, lines, v2=False)
    return {"version": "v1", "score_lines": len(lines),
            "assembly_actions": len(layout["assembly_actions"]), "physical_assembly": False}


def compare_v2_to_v1(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    require(baseline.get("schema") == "numi.cardboard.fefco0201-layout.v1",
            "--baseline-v1 must point to a v1 layout")
    new_d = dimensions(current)
    old_d = dimensions(baseline)
    for key in ("panel_length", "panel_width", "wall_height", "joint_width", "slot_kerf"):
        close(new_d[key], old_d[key], f"baseline dimension {key}")
    new_lines = current["score_lines"]
    old_lines = baseline["score_lines"]
    require(len(new_lines) == len(old_lines), "v1/v2 score-line count differs")
    for idx, (new, old) in enumerate(zip(new_lines, old_lines)):
        require(new["kind"] == old["kind"] and new["panel"] == old["panel"],
                f"v1/v2 score line {idx} kind or panel differs")
        require(new["physically_scored"] is old["physically_scored"],
                f"v1/v2 score line {idx} physical-score state differs")
        for endpoint in ("first", "second"):
            for axis, (a, b) in enumerate(zip(new[endpoint], old[endpoint])):
                close(a, float(b), f"v1/v2 score line {idx} {endpoint}[{axis}]")
    new_actions = current["assembly_actions"]
    old_actions = baseline["assembly_actions"]
    require(len(new_actions) == len(old_actions), "v1/v2 action count differs")
    for idx, (new, old) in enumerate(zip(new_actions, old_actions)):
        require(new["stage"] == old["stage"] and new["kind"] == old["kind"]
                and new["executed"] is old["executed"],
                f"v1/v2 action {idx} kind, stage, or execution state differs")
        close(new["target_angle_deg"], float(old["target_angle_deg"]),
              f"v1/v2 action {idx} target angle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path, help="exported box-layout.json")
    parser.add_argument("--baseline-v1", type=Path,
                        help="optional retained v1 geometry/schedule for compatibility comparison")
    args = parser.parse_args()
    raw = args.layout.read_bytes()
    layout = json.loads(raw)
    require(isinstance(layout, dict), "layout root must be a JSON object")
    schema = layout.get("schema")
    if schema == "numi.cardboard.fefco0201-layout.v2":
        result = validate_v2(layout)
        if args.baseline_v1:
            baseline = json.loads(args.baseline_v1.read_text())
            compare_v2_to_v1(layout, baseline)
            result["baseline_v1_geometry_and_schedule"] = "matched"
    elif schema == "numi.cardboard.fefco0201-layout.v1":
        require(args.baseline_v1 is None, "--baseline-v1 applies only to a v2 layout")
        result = validate_v1(layout)
    else:
        fail(f"unsupported schema: {schema!r}")
    result["path"] = str(args.layout)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
