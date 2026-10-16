#!/usr/bin/env python3
"""Compare captured native responses to the exact-widened-input FP64 replay."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        value = json.loads(line)
        key = value["key"]
        if key in rows:
            raise ValueError(f"duplicate replay key: {key}")
        rows[key] = value
    if not rows:
        raise ValueError("empty replay")
    return rows


def metric(actual, reference, noise_floor):
    if len(actual) != len(reference):
        raise ValueError("response shape mismatch")
    error = max(abs(a-b) for a, b in zip(actual, reference))
    scale = max(abs(b) for b in reference)
    relative = error/scale if scale else None
    if noise_floor is not None and scale <= noise_floor:
        verdict = "inconclusive_below_measured_noise_floor"
    elif scale == 0:
        verdict = "inconclusive_zero_reference_without_noise_floor"
    else:
        verdict = "pass" if relative <= 1e-3 else "fail"
    return dict(absolute_error=error, reference_scale=scale,
                relative_error=relative, verdict=verdict)


def compare(reference_path, native_path, stress_floor=None, tangent_floor=None):
    reference, native = read(reference_path), read(native_path)
    if reference.keys() != native.keys():
        raise ValueError("reference/native key sets differ")
    if any(row.get("input_contract") != "exact_widened_fp32" for row in reference.values()):
        raise ValueError("reference does not declare exact widened FP32 inputs")
    invalid, rows = [], []
    for key, actual in native.items():
        expected = reference[key]
        if not actual["valid"] or not expected["valid"]:
            invalid.append(dict(key=key, native_valid=actual["valid"],
                                reference_valid=expected["valid"],
                                native_failure=actual.get("projection_failure"),
                                native_iterations=actual.get("projection_iterations")))
            continue
        rows.append(dict(key=key,
                         stress=metric(actual["first_piola_Pa"], expected["oracle_P_Pa"], stress_floor),
                         tangent=metric(actual["tangent_Pa_per_s"], expected["oracle_tangent_Pa"], tangent_floor),
                         state_max_absolute_error=max(abs(a-b) for a, b in
                             zip(actual["projected_state"], expected["oracle_state"]))))
    fields = {}
    for field in ("stress", "tangent"):
        ordered = sorted(rows, key=lambda row: row[field]["relative_error"] or 0, reverse=True)
        fields[field] = dict(
            counts={verdict: sum(row[field]["verdict"] == verdict for row in rows)
                    for verdict in sorted({row[field]["verdict"] for row in rows})},
            worst=ordered[:10])
    return dict(schema="numi.cardboard.captured_material_comparison.v1",
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                total_rows=len(reference), invalid_rows=len(invalid), invalid=invalid,
                measured_noise_floors=dict(stress_pa=stress_floor, tangent_pa_per_s=tangent_floor),
                layer_gate_relative=1e-3, layers=fields,
                state_max_absolute_error=max((row["state_max_absolute_error"] for row in rows), default=None),
                verdict="fail" if invalid or any(row[field]["verdict"] == "fail"
                    for row in rows for field in fields) else
                    ("inconclusive" if any(row[field]["verdict"] != "pass"
                        for row in rows for field in fields) else "pass"),
                scope="material projection and analytical algorithmic tangent only; no element, FD, convergence or physical qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--measured-stress-noise-floor-pa", type=float)
    parser.add_argument("--measured-tangent-noise-floor-pa-per-s", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output already exists")
    for floor in (args.measured_stress_noise_floor_pa, args.measured_tangent_noise_floor_pa_per_s):
        if floor is not None and (floor < 0 or not __import__("math").isfinite(floor)):
            raise ValueError("noise floor must be measured, finite and nonnegative")
    result = compare(args.reference, args.native, args.measured_stress_noise_floor_pa,
                     args.measured_tangent_noise_floor_pa_per_s)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("total_rows", "invalid_rows", "verdict")}))


if __name__ == "__main__":
    main()
