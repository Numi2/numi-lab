#!/usr/bin/env python3
"""Summarize paired CCD kernel observations without inferring scene speed."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("observations", nargs="+", type=Path)
args = parser.parse_args()
reports = []
for path in args.observations:
    data = json.loads(path.read_text())
    units = []
    for case in data["cases"]:
        arms = case["arms"]
        control, treatment = arms[:2]
        if control["directional"] or not treatment["directional"]:
            raise ValueError("arm identity mismatch")
        for arm in arms:
            samples = arm.get("gpu_seconds_samples", [])
            if len(samples) != data["repeats"] or not all(t > 0 for t in samples):
                raise ValueError("raw GPU timing samples missing or invalid")
        units.append({
            "unit": case["name"],
            "iterations": [arm["iterations"] for arm in arms],
            "unresolved": [arm["unresolved"] for arm in arms],
            "typed_failure": [arm["status"] != 0 for arm in arms],
            "geometric_or_rejected_gate": [arm["valid"] for arm in arms],
            "gpu_seconds_median": [arm["gpu_seconds_median"] for arm in arms],
        })
    sums = [
        sum(unit["gpu_seconds_median"][i] for unit in units)
        for i in range(len(data["cases"][0]["arms"]))
    ]
    reports.append({
        "observation": path.name,
        "scale": data["scale"],
        "max_iterations": data.get("max_iterations",128),
        "device": data["device"],
        "distinct_sweeps": len(units),
        "replicated_environments_per_sweep": data["environments"],
        "directional_geometry_gate_failures": data["failures"],
        "unresolved_sweeps": [
            sum(unit["unresolved"][i] for unit in units) for i in range(len(data["cases"][0]["arms"]))
        ],
        "typed_failure_sweeps": [
            sum(unit["typed_failure"][i] for unit in units) for i in range(len(data["cases"][0]["arms"]))
        ],
        "sum_of_case_gpu_medians_seconds": sums,
        "directional_to_control_gpu_cost_ratio": sums[1] / sums[0],
        "interval_local_to_control_gpu_cost_ratio": sums[2]/sums[0] if len(sums)>2 else None,
        "units": units,
    })
print(json.dumps({
    "schema": "numi.ccd.bound-analysis.v1",
    "reports": reports,
    "decision": "retain opt-in experiment; do not promote to default",
    "accepted_simulated_time_per_wall_second": None,
    "limitations": [
        "A geometric gate can pass because a sweep is rejected as unresolved.",
        "Neither an unresolved result nor a speculative fallback proves a resolved physical interaction.",
        "A sum of kernel medians is descriptive; it is not accepted simulation throughput.",
        "Replicated environments and timing repeats are not independent geometric units.",
        "These isolated-kernel observations do not qualify force, friction, torque, control or full physical transactions.",
    ],
}, indent=2))
