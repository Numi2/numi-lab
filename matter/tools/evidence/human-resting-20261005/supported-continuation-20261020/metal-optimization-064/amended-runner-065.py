#!/usr/bin/env python3
"""Amended instrument for preregistered contact screen 064."""
import hashlib, json, math, os, re, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
HERE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-064"
AUDIT = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-performance-audit-063"
BASE_INV = AUDIT / "inputs/062/invocation.json"
BASE_RUN = HERE / "runs/control-1"
BIN_SHA = "c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828"
HEAD = "aadcd68e0747540978edd88a23af0cd114932942"
SWITCH = "NM_CONTACT_NODE_INCIDENCE_SORT"
METRICS = ["root_xyz_m", "center_of_mass_xyz_m", "root_linear_velocity_xyz_m_s",
           "root_linear_speed_m_s", "max_generalized_acceleration", "penetration_m",
           "contact_count", "min_J", "max_J", "max_displacement_m",
           "max_anchor_reaction_N", "passive_ligament_max_tension_N",
           "passive_ligament_force_residual_N", "passive_ligament_moment_residual_Nm",
           "body_contact_normal_force_N", "body_contact_force_residual_N",
           "body_contact_moment_residual_Nm"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if ("human_standing_progress=accepted" not in line and
            "resting_FEM_terminal_audit=" not in line and
            "resting_fascia_fem_terminal_audit=" not in line):
            continue
        for key, raw in re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", line):
            if key not in METRICS:
                continue
            try:
                val = [float(x) for x in raw[1:-1].split(",")] if raw.startswith("[") else float(raw)
                vals = val if isinstance(val, list) else [val]
                if all(math.isfinite(x) for x in vals):
                    out[key] = val
            except ValueError:
                pass
    return out


def parse_native(label, run_dir, returncode, elapsed, stdout, stderr):
    (run_dir / "stdout.log").write_text(stdout)
    (run_dir / "stderr.log").write_text(stderr)
    ends = [int(x) for x in re.findall(
        r"human_execution_stage=authoritative_segment_end[^\n]*stage_step=(\d+)", stdout)]
    rejected = [int(x) for x in re.findall(
        r"human_standing_progress=rejected step=(\d+)", stdout)]
    body_rows = re.findall(r"resting_integrated_body=completed([^\n]*)", stdout)
    throughput_rows = re.findall(r"resting_integrated_throughput([^\n]*)", stdout)
    body = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", body_rows[-1])) if body_rows else {}
    throughput = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", throughput_rows[-1])) if throughput_rows else {}
    profiles = [{"step": int(a), "command_ms": float(b), "gpu_ms": float(c),
                 "host_copy_ms": float(d)} for a,b,c,d in re.findall(
        r"resting_native_profile step=(\d+) command_ms=([^\s]+) gpu_ms=([^\s]+) host_copy_ms=([^\s]+)", stdout)]
    assistance = [float(x) for x in re.findall(r"root_assistance_force_n=([^\s]+)", stdout)]
    torques = [float(x) for x in re.findall(r"root_assistance_torque_nm=([^\s]+)", stdout)]
    summary = {
        "label": label, "returncode": returncode, "process_wall_seconds": elapsed,
        "authoritative_root_steps": ends, "rejected_steps": rejected,
        "body": body, "integrated_throughput": throughput, "root_profiles": profiles,
        "fem_terminal_passed": "resting_FEM_terminal_audit=passed" in stdout,
        "fascia_terminal_passed": "resting_fascia_fem_terminal_audit=passed" in stdout,
        "root_assistance_false": "root_assistance=false" in stdout and
            all(v == 0.0 for v in assistance) and all(v == 0.0 for v in torques),
        "terminal_metrics": parse_metrics(stdout),
        "stdout_sha256": sha(run_dir / "stdout.log"),
        "stderr_sha256": sha(run_dir / "stderr.log"),
    }
    try:
        sim_s = float(body["simulated_s"])
        rtf = float(body["real_time_factor"])
    except (KeyError, ValueError):
        sim_s, rtf = float("nan"), float("nan")
    summary["valid"] = (
        returncode == 0 and ends == [1,2,3,4] and not rejected and
        math.isfinite(sim_s) and abs(sim_s - 0.00050000002374872565) < 1e-12 and
        math.isfinite(rtf) and rtf > 0 and len(profiles) == 4 and
        summary["fem_terminal_passed"] and summary["fascia_terminal_passed"] and
        summary["root_assistance_false"])
    summary["process_wall_seconds"] = float(elapsed)
    return summary


def check_metrics(left, right):
    result = {}
    for key in METRICS:
        if key not in left or key not in right:
            return False, {"missing": key}
        a = left[key] if isinstance(left[key], list) else [left[key]]
        b = right[key] if isinstance(right[key], list) else [right[key]]
        if len(a) != len(b):
            return False, {"length_mismatch": key}
        rows = []
        for x,y in zip(a,b):
            allowed = max(1e-7, 0.01 * max(abs(x), abs(y)))
            rows.append({"control": x, "treatment": y, "absolute_delta": abs(x-y),
                         "allowed": allowed})
            if abs(x-y) > allowed:
                result[key] = rows
                return False, result
        result[key] = rows
    return True, result


def write_summary(run_dir, summary):
    with open(run_dir / "amended-summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def calibrate_prior_control():
    stdout = (BASE_RUN / "stdout.log").read_text()
    stderr = (BASE_RUN / "stderr.log").read_text()
    raw = json.loads((BASE_RUN / "summary.json").read_text())
    summary = parse_native("control-1", BASE_RUN, raw["returncode"],
                           raw["process_wall_seconds"], stdout, stderr)
    summary["source"] = "retained native run from failed screen-064 parser attempt"
    summary["stdout_sha256"] = sha(BASE_RUN / "stdout.log")
    summary["stderr_sha256"] = sha(BASE_RUN / "stderr.log")
    if not summary["valid"]:
        raise RuntimeError("retained control fails calibrated native-schema check")
    write_summary(BASE_RUN, summary)
    return summary


def run_arm(label, treatment, base):
    run_dir = HERE / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = list(base["argv"])
    argv[4] = str(run_dir / "output")
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(("NUMI_", "NM_", "MTL_"))}
    env.update(base.get("environment", {}))
    env.pop(SWITCH, None)
    if treatment:
        env[SWITCH] = "1"
    effective = {k:v for k,v in sorted(env.items())
                 if k.startswith(("NUMI_", "NM_", "MTL_"))}
    invocation = {
        "label": label, "treatment": "sorted" if treatment else "control",
        "source_head": subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO, text=True).strip(),
        "binary_sha256": sha(Path(argv[0])), "base_invocation_sha256": sha(BASE_INV),
        "argv": argv, "cwd": str(run_dir), "simulation_environment": effective,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if invocation["source_head"] != HEAD or invocation["binary_sha256"] != BIN_SHA:
        raise RuntimeError("source or executable identity changed from preregistration")
    (run_dir / "amended-invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True)+"\n")
    print(f"START {label} treatment={int(treatment)}", flush=True)
    start = time.perf_counter()
    p = subprocess.run(argv, cwd=run_dir, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    summary = parse_native(label, run_dir, p.returncode, elapsed, p.stdout, p.stderr)
    summary["treatment"] = treatment
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_summary(run_dir, summary)
    print(f"END {label} wall_s={elapsed:.3f} valid={int(summary['valid'])} "
          f"roots={summary['authoritative_root_steps']} body_rtf={summary['body'].get('real_time_factor')} "
          f"gpu_ms={[round(x['gpu_ms'],3) for x in summary['root_profiles']]}", flush=True)
    if not summary["valid"]:
        raise RuntimeError("native validity failed; retaining result and stopping")
    return summary


def main():
    prior = calibrate_prior_control()
    (HERE / "control-1-calibration.json").write_text(json.dumps(prior, indent=2, sort_keys=True)+"\n")
    base = json.loads(BASE_INV.read_text())
    results = {"control-1": prior}
    for label, treatment in [("sorted-1", True), ("sorted-2", True), ("control-2", False)]:
        results[label] = run_arm(label, treatment, base)
    pairs = []
    for control, sorted_label in [("control-1", "sorted-1"), ("control-2", "sorted-2")]:
        c, s = results[control], results[sorted_label]
        metrics_ok, deltas = check_metrics(c["terminal_metrics"], s["terminal_metrics"])
        crtf, srtf = float(c["body"]["real_time_factor"]), float(s["body"]["real_time_factor"])
        pairs.append({"control": control, "sorted": sorted_label,
                      "control_body_rtf": crtf, "sorted_body_rtf": srtf,
                      "speedup_ratio": srtf/crtf, "improvement_fraction": srtf/crtf-1,
                      "physical_metrics_within_tolerance": metrics_ok,
                      "physical_metric_comparison": deltas})
    median = statistics.median(x["speedup_ratio"] for x in pairs)
    passed = (all(x["speedup_ratio"] > 1 for x in pairs) and median >= 1.15 and
              all(x["physical_metrics_within_tolerance"] for x in pairs))
    analysis = {
        "schema": "numi.matter.contact-incidence-screen-064.v2",
        "amendment": str(HERE / "instrument-amendment-065.md"),
        "pairs": pairs, "median_speedup_ratio": median,
        "median_improvement_fraction": median-1,
        "promotion_screen_passed": passed,
        "interpretation": "exploratory mechanics-prefix screen; not a real-time or coupled-physiology qualification",
    }
    (HERE / "analysis-065.json").write_text(json.dumps(analysis, indent=2, sort_keys=True)+"\n")
    print("ANALYSIS " + json.dumps({
        "pair_speedups": [round(x["speedup_ratio"],4) for x in pairs],
        "median_speedup": round(median,4),
        "physical_checks": [x["physical_metrics_within_tolerance"] for x in pairs],
        "screen_passed": passed,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
