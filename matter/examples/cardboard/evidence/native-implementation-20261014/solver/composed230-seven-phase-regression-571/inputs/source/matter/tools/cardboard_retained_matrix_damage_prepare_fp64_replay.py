#!/usr/bin/env python3
"""Convert a native tangent JSON diagnostic into the exact CPU replay stream."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def f32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def array(values: list[float], expected: int, label: str) -> list[float]:
    if len(values) != expected:
        raise ValueError(f"{label}: expected {expected} scalars, got {len(values)}")
    result = [f32(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label}: nonfinite FP32 input")
    return result


def prepare(raw: dict) -> str:
    if raw.get("schema") != "numi.cardboard.retained-damage-tangent-diagnostic.v1":
        raise ValueError("unsupported native tangent diagnostic schema")
    if raw.get("evidence_class") != "explicit_synthetic_instrument_only_not_calibration":
        raise ValueError("unexpected damage evidence class")
    center_f = array(raw["deformation_F32"], 9, "center F")
    direction = array(raw["direction_F32"], 9, "direction")
    plus_f = array(raw["plus_deformation_F32"], 9, "plus F")
    minus_f = array(raw["minus_deformation_F32"], 9, "minus F")
    state_names = list(raw["center"]["projected_state"])
    if not state_names or state_names != list(raw["accepted_state"]):
        raise ValueError("native accepted/candidate state field order differs")
    accepted = [f32(raw["accepted_state"][name]) for name in state_names]

    def response(name: str) -> list[str]:
        value = raw[name]
        flags = ["1" if value["valid"] else "0",
                 "1" if value["projection_valid"] else "0",
                 format(float(value["projection_residual_inf"]), ".17g")]
        stress = array(value["stress_Pa"], 9, f"{name} stress")
        tangent = array(value["tangent_action_Pa"], 9, f"{name} tangent")
        states = value["projected_state"]
        if list(states) != state_names:
            raise ValueError(f"{name} state fields do not match center ordering")
        candidate = [f32(states[key]) for key in state_names]
        return flags + [format(x, ".9g") for x in stress + tangent + candidate]

    tokens = [format(f32(raw["dt_seconds"]), ".9g"),
              format(float(raw["epsilon"]), ".17g")]
    for values in (center_f, direction, plus_f, minus_f):
        tokens.extend(format(x, ".9g") for x in values)
    tokens.append(str(len(accepted)))
    tokens.extend(format(x, ".9g") for x in accepted)
    for name in ("center", "plus", "minus"):
        tokens.extend(response(name))
    return "\n".join(tokens) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    line = None
    for candidate in args.diagnostic.read_text().splitlines():
        marker = "retained_damage_native_tangent_diagnostic="
        if marker in candidate:
            line = candidate.split(marker, 1)[1]
    if line is None:
        raise SystemExit("no retained_damage_native_tangent_diagnostic JSON found")
    args.output.write_text(prepare(json.loads(line)))
    print(f"prepared_exact_fp32_replay={args.output}")
    print("qualification=diagnostic_only_synthetic_damage_no_calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
