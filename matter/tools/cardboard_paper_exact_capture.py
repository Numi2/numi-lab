#!/usr/bin/env python3
"""Export one complete native FEM capture row for the paper Metal instrument.

The v1 TSV keeps F_local's binary32 high and low words separate. Values are
written as widened binary32 numbers with enough decimal digits to round-trip.
This is an instrument-input export, not a second constitutive evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


SCHEMA = "numi.cardboard.paper-exact-capture.v1"
PAPER_PARAMETER_COUNT = 20
PAPER_PARAMETER_NAMES = (
    "density", "c11", "c22", "c33", "c12", "c13", "c23", "g12", "g13", "g23",
    "sigma_ref", "hill_F", "hill_G", "hill_H", "hill_L", "hill_M", "hill_N",
    "q_floor", "fb_epsilon", "residual_scale",
)


def f32(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("capture contains a nonfinite number")
    widened = struct.unpack("<f", struct.pack("<f", number))[0]
    if not math.isfinite(widened):
        raise ValueError("capture value is outside finite binary32 range")
    return widened


def flatten9(value: object, label: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not an array")
    flat = [item for row in value for item in row] if value and isinstance(value[0], list) else value
    if len(flat) != 9:
        raise ValueError(f"{label} must contain nine row-major entries")
    return [f32(item) for item in flat]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_capture(
    source: Path, destination: Path, cases: list[tuple[int, int]], *, require_smooth: bool = False
) -> dict:
    header = None
    selected: dict[tuple[int, int], dict] = {}
    integrity = None
    counts = [0, 0]
    seen: set[tuple[int, int]] = set()
    requested = set(cases)
    if not requested or len(requested) != len(cases):
        raise ValueError("requested case list is empty or contains duplicates")
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record") == "header":
                if header is not None:
                    raise ValueError("capture has duplicate header")
                header = record
            elif record.get("record") == "tetrahedron":
                env = int(record["environment"])
                ident = int(record["tet_id"])
                if env not in (0, 1) or ident < 0 or (env, ident) in seen:
                    raise ValueError("capture has invalid or duplicate tetrahedron identity")
                seen.add((env, ident))
                counts[env] += 1
                if (env, ident) in requested:
                    selected[(env, ident)] = record
            elif record.get("record") == "integrity":
                if integrity is not None:
                    raise ValueError("capture has duplicate integrity record")
                integrity = record

    if header is None or set(selected) != requested or integrity is None:
        raise ValueError("capture is missing its header, one or more requested tets, or terminal integrity record")
    if header.get("schema") not in {"numi.cardboard.fem-newton-capture.v1", "numi.cardboard.fem-newton-capture.v2"}:
        raise ValueError("unsupported native capture schema")
    if not integrity.get("production_inputs_bitwise_unchanged"):
        raise ValueError("native capture reports changed production inputs")
    expected = int(header["tetrahedron_count_per_environment"])
    if int(header.get("environment_count", 0)) != 2 or counts != [expected, expected]:
        raise ValueError("native capture is incomplete")
    if any(ident >= expected for _, ident in seen):
        raise ValueError("native capture has a tetrahedron identity outside its declared range")
    if "high_float32_plus_low_float32" not in header.get("F_local_word_encoding", ""):
        raise ValueError("capture does not declare high-plus-low deformation words")

    materials = header.get("materials", [])
    rows: list[tuple[tuple[int, int], str, str, list[float | str], float, float]] = []
    for environment, tet_id in cases:
        selected_record = selected[(environment, tet_id)]
        material_index = int(selected_record["material_index"])
        material = next((entry for entry in materials if int(entry["index"]) == material_index), None)
        if material is None:
            raise ValueError("selected material is absent from capture header")
        material_name = str(material["name"])
        if "hajali2009_liner" in material_name:
            kind = "liner"
        elif "hajali2009_medium" in material_name:
            kind = "medium"
        else:
            raise ValueError("selected tetrahedron is not a Haj-Ali paper material")
        if selected_record.get("material_name") != material_name or int(selected_record["material_parameter_count"]) != PAPER_PARAMETER_COUNT:
            raise ValueError("selected material identity or parameter count is inconsistent")
        if tuple(material.get("parameter_names", ())) != PAPER_PARAMETER_NAMES:
            raise ValueError("selected capture parameter ordering differs from the instrument contract")
        parameter_offset = int(material["parameter_offset"])
        parameters = [f32(value) for value in header["parameters_by_environment"][environment]
                      [parameter_offset:parameter_offset + PAPER_PARAMETER_COUNT]]
        if len(parameters) != PAPER_PARAMETER_COUNT:
            raise ValueError("captured paper parameter table is incomplete")

        center = selected_record["center"]
        direction = flatten9(center["F_direction_local_per_s"], "F_direction_local_per_s")
        direction_max = max(abs(value) for value in direction)
        projection = [f32(value) for value in center["projection"]]
        if len(projection) != 4:
            raise ValueError("selected projection diagnostics must contain four entries")
        if require_smooth and direction_max == 0.0:
            raise ValueError(f"selected case e{environment}_t{tet_id} has zero deformation direction")
        if require_smooth and projection[3] == 0.0:
            raise ValueError(f"selected case e{environment}_t{tet_id} has no converged captured projection")
        state = [f32(value) for value in center["projected_state"]]
        F, G, H, L, M, N = parameters[11:17]
        s = state[6:12]
        q_squared = (
            F * (s[1] - s[2]) ** 2
            + G * (s[2] - s[0]) ** 2
            + H * (s[0] - s[1]) ** 2
            + 2.0 * L * s[3] ** 2
            + 2.0 * M * s[4] ** 2
            + 2.0 * N * s[5] ** 2
        )
        if q_squared < 0.0:
            raise ValueError("captured paper stress has a negative Hill-equivalent square")
        source_q = math.sqrt(q_squared)
        if require_smooth and abs(1.0 - source_q) < 1.0e-2:
            raise ValueError(f"selected case e{environment}_t{tet_id} is too close to the Hill branch")
        fields = [
            f"e{environment}_t{tet_id}_center",
            kind,
            *flatten9(center["F_local"], "F_local_high"),
            *flatten9(center["F_local_low"], "F_local_low"),
            *direction,
            *[f32(value) for value in center["accepted_state"]],
            *[f32(value) for value in center["projected_state"]],
            *parameters,
            *flatten9(center["P_local_Pa"], "P_local_Pa"),
            *flatten9(center["tangent_local_Pa_per_s"], "tangent_local_Pa_per_s"),
            *projection,
        ]
        if len(center["accepted_state"]) != 13 or len(center["projected_state"]) != 13:
            raise ValueError("selected paper state layout is not 13 components")
        rows.append(((environment, tet_id), material_name, kind, fields, source_q, direction_max))

    source_sha256 = hash_file(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(f"# {SCHEMA}\n")
        output.write(f"# source_capture_sha256={source_sha256}\n")
        output.write(f"# source_world_fingerprint_hex={header['world_fingerprint_hex']}\n")
        output.write(f"# source_execution_fingerprint_hex={header['execution_fingerprint_hex']}\n")
        output.write(f"# source_newton_iteration={header.get('newton_iteration', 'unknown')} source_capture_rows_are_candidate_states_not_accepted_global_states=true\n")
        output.write("# cases=" + ",".join(
            f"e{env}_t{tet}" for (env, tet), _, _, _, _, _ in rows) + "\n")
        output.write("# row: key kind F_high[9] F_low[9] H[9] accepted[13] captured_high[13] parameters[20] captured_P[9] captured_tangent[9] projection_diagnostics[4]\n")
        for _, _, _, fields, _, _ in rows:
            output.write(" ".join(format(value, ".17g") if isinstance(value, float) else str(value)
                                  for value in fields) + "\n")

    return {
        "schema": SCHEMA,
        "source_capture": str(source),
        "source_capture_sha256": source_sha256,
        "source_newton_iteration": header.get("newton_iteration"),
        "source_capture_rows_are_candidate_states_not_accepted_global_states": True,
        "destination": str(destination),
        "cases": [
            {"environment": case[0], "tet_id": case[1], "material": material,
             "material_kind": kind, "fields": len(fields), "captured_hill_q": source_q,
             "hill_branch_margin": abs(1.0 - source_q), "direction_max_per_s": direction_max}
            for case, material, kind, fields, source_q, direction_max in rows
        ],
    }


def parse_case(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("case must be ENVIRONMENT:TET_ID")
    try:
        environment, tet_id = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case environment and tet id must be integers") from error
    if environment not in (0, 1) or tet_id < 0:
        raise argparse.ArgumentTypeError("case environment must be 0 or 1 and tet id nonnegative")
    return environment, tet_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--tet-id", type=int, default=1434)
    parser.add_argument("--case", type=parse_case, action="append", default=None,
                        help="additional ENVIRONMENT:TET_ID row; repeat to export multiple cases")
    args = parser.parse_args()
    if args.environment < 0 or args.tet_id < 0:
        parser.error("environment and tet id must be nonnegative")
    cases = args.case if args.case is not None else [(args.environment, args.tet_id)]
    result = export_capture(args.capture, args.output, cases, require_smooth=args.case is not None)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
