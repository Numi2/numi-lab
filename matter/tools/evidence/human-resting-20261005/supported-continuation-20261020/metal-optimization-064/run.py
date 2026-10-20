#!/usr/bin/env python3
"""Reproduce preregistered contact-incidence screen 064 on SSH Numi."""
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
HERE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-064"
AUDIT = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-performance-audit-063"
INVOCATION = AUDIT / "inputs/062/invocation.json"
BINARY_SHA256 = "c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828"
SOURCE_HEAD = "aadcd68e0747540978edd88a23af0cd114932942"
SWITCH = "NM_CONTACT_NODE_INCIDENCE_SORT"
ORDER = [("control-1", False), ("sorted-1", True), ("sorted-2", True), ("control-2", False)]
PAIRING = [("control-1", "sorted-1"), ("control-2", "sorted-2")]
METRICS = [
    "root_xyz_m", "center_of_mass_xyz_m", "root_linear_velocity_xyz_m_s",
    "root_linear_speed_m_s", "max_generalized_acceleration", "penetration_m",
    "contact_count", "min_J", "max_J", "max_displacement_m",
    "max_anchor_reaction_N", "passive_ligament_max_tension_N",
    "passive_ligament_force_residual_N", "passive_ligament_moment_residual_Nm",
    "body_contact_normal_force_N", "body_contact_force_residual_N",
    "body_contact_moment_residual_Nm",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_values(text):
    out = {}
    for line in text.splitlines():
        if "human_standing_progress=accepted" not in line and            "resting_FEM_terminal_audit=" not in line and            "resting_fascia_fem_terminal_audit=" not in line:
            continue
        for key, raw in re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", line):
            if key not in METRICS:
                continue
            try:
                if raw.startswith("[") and raw.endswith("]"):
                    value = [float(x) for x in raw[1:-1].split(",")]
                else:
                    value = float(raw)
                if isinstance(value, list) and not all(math.isfinite(x) for x in value):
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                out[key] = value
            except ValueError:
                pass
    return out


def summarize(label, treatment, base):
    run_dir = HERE / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = list(base["argv"])
    argv[4] = str(run_dir / "output")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("NUMI_", "NM_", "MTL_"))}
    env.update(base.get("environment", {}))
    env.pop(SWITCH, None)
    if treatment:
        env[SWITCH] = "1"
    else:
        env.pop(SWITCH, None)
    effective = {k: v for k, v in sorted(env.items())
                 if k.startswith(("NUMI_", "NM_", "MTL_"))}
    invocation = {
        "label": label,
        "treatment": "sorted" if treatment else "control",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "binary_sha256": sha256(Path(argv[0])),
        "input_invocation_sha256": sha256(INVOCATION),
        "argv": argv,
        "cwd": str(run_dir),
        "simulation_environment": effective,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if invocation["source_head"] != SOURCE_HEAD:
        raise RuntimeError("source HEAD changed from preregistered revision")
    if invocation["binary_sha256"] != BINARY_SHA256:
        raise RuntimeError("binary changed from preregistered executable")
    with open(run_dir / "invocation.json", "w") as f:
        json.dump(invocation, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"START {label} treatment={int(treatment)}", flush=True)
    start = time.perf_counter()
    result = subprocess.run(argv, cwd=run_dir, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "stderr.log").write_text(result.stderr)
    accepted = [int(x) for x in re.findall(
        r"human_standing_progress=accepted step=(\d+)", result.stdout)]
    rejected = [int(x) for x in re.findall(
        r"human_standing_progress=rejected step=(\d+)", result.stdout)]
    body = re.findall(r"human_integrated_body=completed([^\n]*)", result.stdout)
    through = re.findall(r"resting_integrated_throughput([^\n]*)", result.stdout)
    body_fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", body[-1])) if body else {}
    throughput_fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", through[-1])) if through else {}
    profiles = []
    for m in re.finditer(r"resting_native_profile step=(\d+) command_ms=([^\s]+) gpu_ms=([^\s]+) host_copy_ms=([^\s]+)", result.stdout):
        profiles.append({"step": int(m.group(1)), "command_ms": float(m.group(2)),
                         "gpu_ms": float(m.group(3)), "host_copy_ms": float(m.group(4))})
    summary = {
        "label": label, "treatment": treatment, "returncode": result.returncode,
        "process_wall_seconds": elapsed, "accepted_steps": accepted,
        "rejected_steps": rejected, "body": body_fields,
        "integrated_throughput": throughput_fields, "root_profiles": profiles,
        "fem_terminal_passed": "resting_FEM_terminal_audit=passed" in result.stdout,
        "fascia_terminal_passed": "resting_fascia_fem_terminal_audit=passed" in result.stdout,
        "root_assistance_false": "root_assistance=false" in result.stdout and
                                 all(float(x) == 0.0 for x in re.findall(
                                     r"root_assistance_force_n=([^\s]+)", result.stdout)) and
                                 all(float(x) == 0.0 for x in re.findall(
                                     r"root_assistance_torque_nm=([^\s]+)", result.stdout)),
        "terminal_metrics": metric_values(result.stdout),
        "stdout_sha256": sha256(run_dir / "stdout.log"),
        "stderr_sha256": sha256(run_dir / "stderr.log"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    valid = (result.returncode == 0 and accepted == [1, 2, 3, 4] and
             not rejected and body_fields.get("real_time_factor") is not None and
             summary["fem_terminal_passed"] and summary["fascia_terminal_passed"] and
             summary["root_assistance_false"] and len(profiles) == 4)
    summary["valid"] = valid
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"END {label} wall_s={elapsed:.3f} valid={int(valid)} "
          f"accepted={accepted} rejected={rejected} "
          f"body_rtf={body_fields.get('real_time_factor')} "
          f"gpu_ms={[round(p['gpu_ms'], 3) for p in profiles]}", flush=True)
    if not valid:
        raise RuntimeError(f"invalid trial retained; preregistered stop at {label}")
    return summary


def as_float(value):
    return float(value)


def compare_metrics(left, right):
    checks = {}
    for key in METRICS:
        if key not in left or key not in right:
            return False, {"missing": key}
        lv, rv = left[key], right[key]
        ls = lv if isinstance(lv, list) else [lv]
        rs = rv if isinstance(rv, list) else [rv]
        if len(ls) != len(rs):
            return False, {"length": key}
        deltas = []
        for a, b in zip(ls, rs):
            scale = max(abs(a), abs(b))
            allowed = max(1e-7, 0.01 * scale)
            deltas.append({"left": a, "right": b, "abs_delta": abs(a-b),
                           "allowed": allowed})
            if abs(a-b) > allowed:
                checks[key] = deltas
                return False, checks
        checks[key] = deltas
    return True, checks


def main():
    base = json.loads(INVOCATION.read_text())
    results = {}
    for label, treatment in ORDER:
        results[label] = summarize(label, treatment, base)
    pair_results = []
    for control, treated in PAIRING:
        c = results[control]
        t = results[treated]
        valid_metrics, metric_deltas = compare_metrics(
            c["terminal_metrics"], t["terminal_metrics"])
        crtf = as_float(c["body"]["real_time_factor"])
        trtf = as_float(t["body"]["real_time_factor"])
        pair_results.append({
            "control": control, "sorted": treated,
            "control_body_rtf": crtf, "sorted_body_rtf": trtf,
            "speedup_ratio": trtf / crtf,
            "improvement_fraction": trtf / crtf - 1.0,
            "physical_metrics_within_tolerance": valid_metrics,
            "physical_metric_comparison": metric_deltas,
        })
    median_speedup = statistics.median(x["speedup_ratio"] for x in pair_results)
    passed = (all(x["speedup_ratio"] > 1.0 for x in pair_results) and
              median_speedup >= 1.15 and
              all(x["physical_metrics_within_tolerance"] for x in pair_results))
    report = {
        "schema": "numi.matter.contact-incidence-screen-064.v1",
        "preregistered_plan": str(HERE / "plan.md"),
        "order": [x[0] for x in ORDER],
        "pairs": pair_results,
        "median_speedup_ratio": median_speedup,
        "median_improvement_fraction": median_speedup - 1.0,
        "promotion_screen_passed": passed,
        "interpretation": "exploratory mechanics-prefix screen; not a real-time or coupled-physiology qualification",
    }
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ANALYSIS " + json.dumps({
        "pair_speedups": [round(x["speedup_ratio"], 4) for x in pair_results],
        "median_speedup": round(median_speedup, 4),
        "physical_checks": [x["physical_metrics_within_tolerance"] for x in pair_results],
        "screen_passed": passed,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
