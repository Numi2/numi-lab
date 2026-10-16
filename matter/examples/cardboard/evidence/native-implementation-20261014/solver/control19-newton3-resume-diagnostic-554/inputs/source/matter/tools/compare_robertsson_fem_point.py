#!/usr/bin/env python3
"""Compare native FEM accepted state/reaction against the separate point probe CSV.

The comparison validates FEM coupling and prescribed-reference kinematics for
one synthetic tetrahedron. Both paths use the same authored 27-state material
program, so this is not an independent constitutive implementation or a
published physical validation.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

STATE_COUNT = 27


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no CSV records in {path}")
    return rows


def finite_float(row: dict[str, str], name: str, where: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{where}: missing/non-numeric column {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: nonfinite {name}")
    return value


def compare(args: argparse.Namespace) -> int:
    fem_rows = read_rows(args.fem_csv)
    point_rows = read_rows(args.point_csv)
    if len(fem_rows) != len(point_rows):
        raise ValueError(
            f"row count differs: FEM {len(fem_rows)} vs point {len(point_rows)}"
        )

    max_state_abs = 0.0
    max_state_scaled = 0.0
    max_state_location: tuple[int, int] | None = None
    max_p13_abs = 0.0
    max_p13_rel = 0.0
    max_p13_step = -1
    for index, (fem, point) in enumerate(zip(fem_rows, point_rows, strict=True)):
        fem_step = int(fem["step"])
        point_step = int(point["step"])
        if fem_step != index or point_step != index:
            raise ValueError(f"step indexing mismatch at record {index}")
        if fem.get("accepted") != "1":
            raise ValueError(f"native FEM record {index} is not accepted")
        point_accepted = point.get("algorithmic_response_accepted",
                                   point.get("material_projection_accepted"))
        if point_accepted != "1":
            raise ValueError(f"source-point record {index} is not accepted")

        fem_gamma = finite_float(fem, "command_shear", f"FEM step {index}")
        point_gamma = finite_float(point, "command_shear", f"point step {index}")
        if abs(fem_gamma - point_gamma) > 2e-15:
            raise ValueError(
                f"shear command mismatch at step {index}: {fem_gamma} vs {point_gamma}"
            )
        for state in range(STATE_COUNT):
            fem_value = finite_float(
                fem, f"accepted_state_{state}", f"FEM step {index}"
            )
            point_value = finite_float(
                point, f"candidate_state_{state}", f"point step {index}"
            )
            absolute = abs(fem_value - point_value)
            scaled = absolute / max(1e-30, abs(fem_value), abs(point_value))
            if absolute > max_state_abs:
                max_state_abs = absolute
                max_state_location = (index, state)
            max_state_scaled = max(max_state_scaled, scaled)
            allowed = args.state_atol + args.state_rtol * max(
                abs(fem_value), abs(point_value)
            )
            if absolute > allowed:
                raise ValueError(
                    f"state mismatch step={index} state={state}: "
                    f"FEM {fem_value:.9g}, point {point_value:.9g}, "
                    f"abs {absolute:.4g} > {allowed:.4g}"
                )

        fem_p13 = finite_float(
            fem, "p13_from_reaction_Pa", f"FEM step {index}"
        )
        point_p13 = finite_float(point, "P13_Pa", f"point step {index}")
        p13_absolute = abs(fem_p13 - point_p13)
        p13_relative = p13_absolute / max(1.0, abs(fem_p13), abs(point_p13))
        if p13_absolute > max_p13_abs:
            max_p13_abs = p13_absolute
            max_p13_rel = p13_relative
            max_p13_step = index
        allowed_p13 = args.force_atol + args.force_rtol * max(
            abs(fem_p13), abs(point_p13)
        )
        if p13_absolute > allowed_p13:
            raise ValueError(
                f"reaction-derived P13 mismatch step={index}: "
                f"FEM {fem_p13:.9g} Pa, point {point_p13:.9g} Pa, "
                f"abs {p13_absolute:.4g} > {allowed_p13:.4g}"
            )

    print(f"fem_csv={args.fem_csv}")
    print(f"point_csv={args.point_csv}")
    print(f"matched_steps={len(fem_rows)}")
    print(f"max_state_absolute_error={max_state_abs:.9g}")
    print(f"max_state_relative_error={max_state_scaled:.9g}")
    print(f"max_state_error_location={max_state_location}")
    print(f"max_reaction_P13_absolute_error_Pa={max_p13_abs:.9g}")
    print(f"reaction_P13_error_relative_at_max={max_p13_rel:.9g}")
    print(f"reaction_P13_error_step={max_p13_step}")
    print("claim=synthetic_FEM_state_force_history_coupling_parity_only")
    print("not=independent_material_implementation_or_physical_calibration")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fem_csv", type=Path)
    parser.add_argument("point_csv", type=Path)
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--state-rtol", type=float, default=2e-4)
    parser.add_argument("--force-atol", type=float, default=2.0)
    parser.add_argument("--force-rtol", type=float, default=1e-2)
    args = parser.parse_args()
    if min(args.state_atol, args.state_rtol, args.force_atol, args.force_rtol) < 0:
        parser.error("tolerances must be nonnegative")
    try:
        return compare(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
