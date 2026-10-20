#!/usr/bin/env python3
"""Compare retained Human mechanics summaries without promoting them to integrated qualification."""
import argparse
import json
import math
from pathlib import Path

METRICS = (
    "root_xyz_m", "center_of_mass_xyz_m", "root_linear_velocity_xyz_m_s",
    "max_generalized_acceleration", "root_linear_speed_m_s", "penetration_m",
    "contact_count", "min_J", "max_J", "max_displacement_m",
    "max_anchor_reaction_N", "passive_ligament_max_tension_N",
    "passive_ligament_force_residual_N", "passive_ligament_moment_residual_Nm",
    "body_contact_normal_force_N", "body_contact_force_residual_N",
    "body_contact_moment_residual_Nm",
)

def finite_vector(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) for x in values):
        raise ValueError("expected a nonempty finite scalar/vector")
    return values

def compare(baseline, candidate):
    failures = []
    for name, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary.get("exit_code") != 0:
            failures.append(name + ": process did not succeed")
        for gate in ("fem_terminal_passed", "fascia_terminal_passed"):
            if summary.get(gate) is not True:
                failures.append(name + ": " + gate + " did not pass")
    rows = []
    for key in METRICS:
        row = {"metric": key, "passed": False}
        try:
            a = baseline["terminal_metrics"][key]
            b = candidate["terminal_metrics"][key]
            av, bv = finite_vector(a), finite_vector(b)
            if isinstance(a, list) != isinstance(b, list) or len(av) != len(bv):
                raise ValueError("scalar/vector shape mismatch")
            error = max(abs(x-y)/max(abs(x), abs(y), 1e-5)
                        for x, y in zip(av, bv))
            row.update(baseline=a, candidate=b, relative_error=error,
                       passed=error <= 0.01)
        except (KeyError, TypeError, ValueError) as exc:
            row["reason"] = str(exc)
        rows.append(row)
    speedup = None
    try:
        durations = []
        for summary in (baseline, candidate):
            simulated = float(summary["body"]["simulated_s"])
            wall = float(summary["body"]["wall_s"])
            if not all(math.isfinite(v) and v > 0 for v in (simulated, wall)):
                raise ValueError("nonpositive or nonfinite duration")
            durations.append((simulated, wall))
        if not math.isclose(durations[0][0], durations[1][0], rel_tol=1e-7, abs_tol=0):
            raise ValueError("simulated durations differ")
        speedup = durations[0][1]/durations[1][1]
    except (KeyError, TypeError, ValueError) as exc:
        failures.append("timing: " + str(exc))
    return {"passed": not failures and all(row["passed"] for row in rows),
            "speedup": speedup, "metrics": rows, "failures": failures,
            "qualification": "short mechanics comparison only"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()),
                     json.loads(args.candidate.read_text()))
    result.update(baseline=str(args.baseline), candidate=str(args.candidate))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "metrics"}))
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
