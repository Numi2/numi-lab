#!/usr/bin/env python3
"""Validate matched native waist-release traces and lower-body free-fall evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

INTERNAL_CONTACT_INDICES = {0, 2, 3, 8, 9, 12}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_row(row, path, line_number):
    """Require every non-label field in these native CSV rows to be numeric and finite."""
    for name, value in row.items():
        if name == "body_name":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric {name}={value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{path}:{line_number}: non-finite {name}={value!r}")


def read_trace(path, expected_masses, expected_body_ids):
    rows = {}
    step_bodies = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "accepted_step", "time_s", "body_index", "mass_kg",
            *(f"root_{axis}_m" for axis in "xyz"),
            *(f"com_relative_root_{axis}_m" for axis in "xyz"),
            *(f"orientation_{axis}" for axis in "xyzw"),
            *(f"linear_velocity_{axis}_m_s" for axis in "xyz"),
            *(f"angular_velocity_{axis}_rad_s" for axis in "xyz"),
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{path}: missing required body-motion columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            finite_row(row, path, line_number)
            step = int(row["accepted_step"])
            body = int(row["body_index"])
            if body not in expected_body_ids:
                raise ValueError(f"{path}:{line_number}: unexpected runtime body {body}")
            key = (step, body)
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate accepted sample {key}")
            mass = float(row["mass_kg"])
            if abs(mass - expected_masses[body]) > 1e-10:
                raise ValueError(
                    f"{path}:{line_number}: body {body} mass {mass} differs from pinned source mass {expected_masses[body]}"
                )
            rows[key] = row
            step_bodies.setdefault(step, set()).add(body)
    if not rows:
        raise ValueError(f"{path}: empty body-motion trace")
    expected = set(expected_body_ids)
    for step, body_ids in sorted(step_bodies.items()):
        if body_ids != expected:
            missing = sorted(expected - body_ids)
            extra = sorted(body_ids - expected)
            raise ValueError(
                f"{path}: accepted step {step} is not a complete {len(expected)}-body sample; missing={missing}, extra={extra}"
            )
    return rows, sorted(step_bodies)


def read_contact_rows(path):
    required = {"accepted_step", "contact_index", "body_index", "normal_impulse_ns"}
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{path}: missing required contact columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["accepted_step"])
                contact_index = int(row["contact_index"])
                body = int(row["body_index"])
                impulse = float(row["normal_impulse_ns"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed contact record") from exc
            if not math.isfinite(impulse):
                raise ValueError(f"{path}:{line_number}: non-finite normal impulse")
            rows.append((step, contact_index, body, impulse))
    return rows


def source_mass_map(mapping):
    body_count = int(mapping["runtime_dimensions"]["bodies"])
    expected = {}
    for record in mapping["source_bodies"]:
        body = int(record["runtime_body_index"])
        mass = float(record["mass_kg"])
        if body in expected or not math.isfinite(mass) or mass < 0:
            raise ValueError("pinned source mass map contains duplicate body ids or invalid masses")
        expected[body] = mass
    carrier_ids = [int(i) for i in mapping["seam"]["carrier_runtime_bodies"]]
    for body in carrier_ids:
        if body in expected:
            raise ValueError(f"carrier runtime body {body} duplicates a source body")
        expected[body] = 0.0
    expected_ids = set(range(body_count))
    if set(expected) != expected_ids:
        raise ValueError(
            f"source plus carrier mass map does not cover runtime body range 0..{body_count - 1}: "
            f"missing={sorted(expected_ids - set(expected))}, extra={sorted(set(expected) - expected_ids)}"
        )
    return expected, expected_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--held", type=Path, required=True)
    ap.add_argument("--released", type=Path, required=True)
    ap.add_argument("--support-impulses", type=Path, required=True)
    ap.add_argument("--mass-map", type=Path, required=True)
    ap.add_argument("--release-step", type=int, required=True)
    ap.add_argument("--fit-first-step", type=int, required=True)
    ap.add_argument("--fit-last-step", type=int, required=True)
    ap.add_argument("--contact-first-step", type=int, default=55)
    ap.add_argument("--contact-last-step", type=int, default=100)
    ap.add_argument("--external-contact-tolerance-ns", type=float, default=0.0,
                    help="positive lower-body external normal impulse above this threshold invalidates free-fall")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.fit_first_step > args.fit_last_step or args.contact_first_step > args.contact_last_step:
        ap.error("step interval lower bound must not exceed upper bound")
    if not math.isfinite(args.external_contact_tolerance_ns) or args.external_contact_tolerance_ns < 0:
        ap.error("external contact tolerance must be finite and nonnegative")

    mapping = json.loads(args.mass_map.read_text())
    expected_masses, expected_body_ids = source_mass_map(mapping)
    source_to_runtime = [int(i) for i in mapping["source_to_runtime_body_index"]]
    lower_source = [int(i) for i in mapping["partition"]["lower_source_body_indices"]]
    if any(i < 0 or i >= len(source_to_runtime) for i in lower_source):
        raise ValueError("lower partition references a source body outside the pinned body map")
    lower = {source_to_runtime[i] for i in lower_source}
    if not lower or not lower.issubset(expected_body_ids):
        raise ValueError("lower partition is empty or references an unmapped runtime body")

    held, held_steps = read_trace(args.held, expected_masses, expected_body_ids)
    released, released_steps = read_trace(args.released, expected_masses, expected_body_ids)
    held_pre_steps = [step for step in held_steps if step < args.release_step]
    released_pre_steps = [step for step in released_steps if step < args.release_step]
    if held_pre_steps != released_pre_steps:
        raise ValueError(
            "held and released traces have different pre-event accepted-step sets: "
            f"held={held_pre_steps}, released={released_pre_steps}"
        )
    sample_steps = held_pre_steps
    if not sample_steps:
        raise ValueError("missing matched pre-event samples")
    columns = [f"root_{a}_m" for a in "xyz"] + [
        f"com_relative_root_{a}_m" for a in "xyz"] + [
        f"linear_velocity_{a}_m_s" for a in "xyz"]
    pre_delta = max(
        abs(float(held[(step, body)][column]) - float(released[(step, body)][column]))
        for step in sample_steps for body in expected_body_ids for column in columns
    )

    fit_steps = [step for step in released_steps
                 if args.fit_first_step <= step <= args.fit_last_step]
    if len(fit_steps) < 2:
        raise ValueError(f"need at least two accepted samples in fit interval; found {fit_steps}")
    times, velocities = [], []
    lower_mass = sum(expected_masses[i] for i in lower)
    if lower_mass <= 0:
        raise ValueError("lower partition has no positive pinned source mass")
    for step in fit_steps:
        rows = [released[(step, i)] for i in sorted(lower)]
        time_values = {float(row["time_s"]) for row in rows}
        if len(time_values) != 1:
            raise ValueError(f"accepted step {step} has inconsistent body times")
        times.append(time_values.pop())
        velocities.append([
            sum(float(row[f"mass_kg"]) * float(row[f"linear_velocity_{axis}_m_s"])
                for row in rows) / lower_mass
            for axis in "xyz"
        ])
    if len(set(times)) != len(times):
        raise ValueError("fit samples contain duplicate timestamps")
    acceleration = np.polyfit(times, np.asarray(velocities), 1)[0]
    finite = bool(np.all(np.isfinite(acceleration)))

    contact_rows = read_contact_rows(args.support_impulses)
    external_lower_rows = [
        (step, contact_index, body, impulse)
        for step, contact_index, body, impulse in contact_rows
        if args.contact_first_step <= step <= args.contact_last_step
        and contact_index not in INTERNAL_CONTACT_INDICES
        and body in lower
        and impulse > args.external_contact_tolerance_ns
    ]
    contact_clear = len(external_lower_rows) == 0
    contact_summary = dict(
        accepted_step_interval=[args.contact_first_step, args.contact_last_step],
        excluded_internal_contact_indices=sorted(INTERNAL_CONTACT_INDICES),
        lower_runtime_body_indices=sorted(lower),
        positive_normal_impulse_tolerance_ns=args.external_contact_tolerance_ns,
        active_lower_external_contact_samples=len(external_lower_rows),
        first_active_step=min((r[0] for r in external_lower_rows), default=None),
        maximum_lower_external_normal_impulse_ns=max((r[3] for r in external_lower_rows), default=None),
        lower_external_contact_body_indices=sorted({r[2] for r in external_lower_rows}),
        free_fall_contact_gate_passed=contact_clear,
    )
    result = dict(
        inputs={str(p): sha256(p) for p in
                (args.held, args.released, args.support_impulses, args.mass_map)},
        runtime_body_count=len(expected_body_ids),
        complete_body_set_per_accepted_step=True,
        matched_pre_event_accepted_steps_identical=True,
        release_step=args.release_step,
        fit_step_interval=[args.fit_first_step, args.fit_last_step],
        fit_accepted_steps=fit_steps,
        pre_event_max_delta=pre_delta,
        lower_mass_kg=lower_mass,
        lower_com_acceleration_m_s2=acceleration.tolist(),
        vertical_gravity_error_m_s2=abs(float(acceleration[2]) + 9.81),
        lateral_acceleration_m_s2=float(np.linalg.norm(acceleration[:2])),
        lower_external_contact=contact_summary,
        criteria=dict(pre_event_max_delta_limit=1e-7,
                      vertical_gravity_error_limit_m_s2=.2,
                      lateral_acceleration_limit_m_s2=.2,
                      lower_external_contact_samples_must_be_zero=True),
        scope="short articulated release; not continuum fracture or injury physiology")
    result["passed"] = bool(finite and pre_delta <= 1e-7 and
        result["vertical_gravity_error_m_s2"] < .2 and
        result["lateral_acceleration_m_s2"] < .2 and contact_clear)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
