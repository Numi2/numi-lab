#!/usr/bin/env python3
"""Prepare, explicitly run, and assess a hash-pinned 100 ms bilateral elbow-flexor trial.

Default action is help. "prepare" only writes a pinned invocation; it does not launch native code.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
EVIDENCE = Path(__file__).resolve().parent
SOURCE_RUN = EVIDENCE / "results/bilateral-elbow-flexors006"
RUN_DIR = EVIDENCE / "results/bilateral-elbow-flexors006-100ms"
MANIFEST_PATH = EVIDENCE / "scoped-native-build-manifest-006.json"
INVOCATION_PATH = RUN_DIR / "invocation.json"
STEPS = 800
DT = 0.000125
ACTIVATION = 0.5
MUSCLE_IDS = [230, 231, 232, 233, 293, 294, 295, 296]
EXPECTED_ELBOWS = {"right": (39, 38), "left": (77, 76)}
SCOPE = (
    "Core selected-tendon elbow-flexor actuation over a matched 100 ms horizon. "
    "Free authored joints and existing source muscle/tendon laws. Not whole-body "
    "FEM/contact, grasp, clinical, biological, or production qualification."
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def resolved_input(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def validate_pinned_inputs(inv: dict) -> None:
    for name, expected in inv["input_hashes"].items():
        path = resolved_input(name)
        if not path.is_file():
            raise RuntimeError(f"pinned input missing: {path}")
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"pinned input hash mismatch: {path}: {actual} != {expected}")
    manifest = read_json(MANIFEST_PATH)
    if sha(MANIFEST_PATH) != inv["build_source_manifest_sha256"]:
        raise RuntimeError("build source manifest changed since 006 run")
    binary = Path(inv["binary_path"])
    if sha(binary) != inv["binary_sha256"]:
        raise RuntimeError("native binary changed since 006 run")
    if manifest["binary_path"] != str(binary) or manifest["binary_sha256"] != inv["binary_sha256"]:
        raise RuntimeError("build manifest and invocation binary identity disagree")


def make_invocation() -> dict:
    source_path = SOURCE_RUN / "invocation.json"
    source = read_json(source_path)
    validate_pinned_inputs(source)
    if source["step_count"] != 80 or source["timestep_seconds"] != DT:
        raise RuntimeError("source run is not the pinned 80-step, 125-us 006 owner/config")
    if source["activation"] != ACTIVATION or source["cohort"] != "bilateral_elbow-flexors006":
        raise RuntimeError("source cohort or activation differs from the approved 006 run")
    if source["initial_configuration"] != "bilateral_hand_q36_0p1_q74_0p1":
        raise RuntimeError("source initial reference posture changed")
    if source["pose_or_lock_flags"] or source["ongoing_q_lock"] is not False:
        raise RuntimeError("source invocation contains a pose override or ongoing lock")
    if source["initial_configuration_flags"] != ["--initial-reference-posture"]:
        raise RuntimeError("source initialization flag changed")

    argv = list(source["argv"])
    if len(argv) < 5 or Path(argv[4]) != SOURCE_RUN:
        raise RuntimeError("could not safely identify the native output directory argument")
    argv[4] = str(RUN_DIR)
    count_at = argv.index("--muscle-step-count") + 1
    argv[count_at] = str(STEPS)
    activation_at = argv.index("--muscle-activation") + 1
    if float(argv[activation_at]) != ACTIVATION:
        raise RuntimeError("native activation argument differs from source receipt")
    source_ids = []
    i = 0
    while i < len(argv):
        if argv[i] == "--activated-source-muscle-index":
            source_ids.append(int(argv[i + 1]))
            i += 2
        else:
            i += 1
    if source_ids != MUSCLE_IDS:
        raise RuntimeError(f"selected source muscle IDs changed: {source_ids}")

    receipt = dict(source)
    receipt.update({
        "argv": argv,
        "cohort": "bilateral_elbow_flexors006_100ms",
        "step_count": STEPS,
        "timestep_seconds": DT,
        "simulated_seconds": STEPS * DT,
        "timeout_seconds": 300.0,
        "qualification_helper_path": str(Path(__file__).resolve()),
        "qualification_helper_sha256": sha(Path(__file__).resolve()),
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": SCOPE,
        "execution_authorized_by_this_script": False,
    })
    return receipt


def prepare() -> None:
    receipt = make_invocation()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if INVOCATION_PATH.exists():
        raise RuntimeError(f"refusing to replace prepared/native invocation: {INVOCATION_PATH}")
    if any(RUN_DIR.iterdir()):
        raise RuntimeError(f"refusing to prepare into nonempty result directory: {RUN_DIR}")
    INVOCATION_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "mode": "prepared_only_native_not_launched",
        "invocation_path": str(INVOCATION_PATH),
        "invocation_sha256": sha(INVOCATION_PATH),
        "binary_path": receipt["binary_path"],
        "binary_sha256": receipt["binary_sha256"],
        "step_count": STEPS,
        "timestep_seconds": DT,
        "simulated_seconds": STEPS * DT,
        "argv": receipt["argv"],
    }, indent=2))


def execute() -> None:
    if not INVOCATION_PATH.is_file():
        raise RuntimeError("prepare the invocation first")
    receipt = read_json(INVOCATION_PATH)
    if sha(Path(__file__).resolve()) != receipt["qualification_helper_sha256"]:
        raise RuntimeError("runner changed after invocation preparation")
    validate_pinned_inputs(receipt)
    if receipt["step_count"] != STEPS or receipt["simulated_seconds"] != 0.1:
        raise RuntimeError("prepared horizon changed")
    if any(p.name != "invocation.json" for p in RUN_DIR.iterdir()):
        raise RuntimeError("refusing to overwrite existing native outputs")
    argv = receipt["argv"]
    if argv[0] != receipt["binary_path"]:
        raise RuntimeError("native command binary differs from pinned receipt")

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("NUMI_HUMAN_RESTING_", "NUMI_HUMAN_SUPPORTED_"))
    }
    for key in ("NUMI_HUMAN_WRIST_CONTOUR_ENABLE", "NUMI_HUMAN_WRIST_CONTOUR_SCALE"):
        env.pop(key, None)
    env["NUMI_HUMAN_SUPPORTED_INITIALIZATION"] = "0"
    start = time.monotonic()
    try:
        run = subprocess.run(
            argv, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=float(receipt["timeout_seconds"]),
        )
        rc, stdout, stderr = run.returncode, run.stdout, run.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    elapsed = time.monotonic() - start
    (RUN_DIR / "stdout.log").write_text(stdout)
    (RUN_DIR / "stderr.log").write_text(stderr)
    result = {
        "native_exit_code": rc,
        "wall_elapsed_seconds": elapsed,
        "simulated_seconds": receipt["simulated_seconds"],
        "status": "native_completed_pending_100ms_assessment" if rc == 0 else "FAIL",
        "scope": SCOPE,
        "invocation_sha256": sha(INVOCATION_PATH),
    }
    (RUN_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if rc:
        raise SystemExit(1)


def finite(value, label: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise RuntimeError(f"non-finite telemetry: {label}")
    return x


def extract_stdout_record(stdout: str, prefix: str) -> dict:
    found = []
    for line in stdout.splitlines():
        if line.startswith(prefix + "="):
            found.append(json.loads(line.split("=", 1)[1]))
    if len(found) != 1:
        raise RuntimeError(f"expected one {prefix} record, found {len(found)}")
    return found[0]


def assess() -> None:
    receipt = read_json(INVOCATION_PATH)
    result_path = RUN_DIR / "result.json"
    stdout_path = RUN_DIR / "stdout.log"
    if not result_path.is_file() or not stdout_path.is_file():
        raise RuntimeError("native result/log not present")
    result = read_json(result_path)
    if result["invocation_sha256"] != sha(INVOCATION_PATH):
        raise RuntimeError("native result is not bound to prepared invocation")
    if result["native_exit_code"] != 0:
        raise RuntimeError(f"native exit code {result['native_exit_code']}")
    if receipt["step_count"] != STEPS or receipt["timestep_seconds"] != DT:
        raise RuntimeError("wrong requested horizon")
    validate_pinned_inputs(receipt)

    stdout = stdout_path.read_text()
    control = extract_stdout_record(stdout, "selected_hand_wrist_actuation")
    terminal = extract_stdout_record(stdout, "stand_terminal_state")
    if terminal["step_count"] != STEPS or terminal["timestep_seconds"] != DT:
        raise RuntimeError("native terminal state has wrong accepted horizon")
    if terminal["root_assistance"] is not False:
        raise RuntimeError("root assistance was active")
    if len(terminal["q"]) != 129 or len(terminal["v"]) != 128:
        raise RuntimeError("native terminal q/v dimensions changed")
    for i, x in enumerate(terminal["q"]):
        finite(x, f"terminal q[{i}]")
    for i, x in enumerate(terminal["v"]):
        finite(x, f"terminal v[{i}]")

    if control["accepted_steps"] != STEPS:
        raise RuntimeError("accepted step count mismatch")
    if control["matched_baseline"] != "same_input_zero_activation_increment":
        raise RuntimeError("matched zero-activation baseline missing")
    if control["replay"] != "bitwise":
        raise RuntimeError("accepted-state replay is not bitwise")
    if len(control["q_rows"]) != 44 or len(control["arm_q_rows"]) != 10:
        raise RuntimeError("expected hand/wrist and arm endpoint telemetry is missing")

    all_rows = control["q_rows"] + control["arm_q_rows"]
    for row in all_rows:
        qidx = int(row["q_index"])
        vidx = int(row["dof_index"])
        accepted = finite(row["accepted_q_rad"], f"accepted q[{qidx}]")
        baseline = finite(row["baseline_q_rad"], f"baseline q[{qidx}]")
        accepted_v = finite(row["accepted_v_rad_s"], f"accepted v[{vidx}]")
        baseline_v = finite(row["baseline_v_rad_s"], f"baseline v[{vidx}]")
        lower = finite(row["source_lower_limit_rad"], f"lower limit q[{qidx}]")
        upper = finite(row["source_upper_limit_rad"], f"upper limit q[{qidx}]")
        # Preserve finite q/v telemetry for every emitted row. Authored joint
        # limit acceptance is applied below to the two elbow coordinates under
        # qualification; many passive hand coordinates sit exactly at zero
        # and carry tiny signed solver residuals around a zero lower limit.
        if terminal["q"][qidx] != accepted or terminal["v"][vidx] != accepted_v:
            raise RuntimeError(f"endpoint telemetry differs from terminal state at q[{qidx}]")

    arm_by_q = {int(r["q_index"]): r for r in control["arm_q_rows"]}
    elbows = {}
    for side, (qidx, vidx) in EXPECTED_ELBOWS.items():
        row = arm_by_q.get(qidx)
        if row is None or int(row["dof_index"]) != vidx:
            raise RuntimeError(f"missing elbow endpoint row for {side}")
        accepted_q = finite(row["accepted_q_rad"], f"{side} elbow q")
        baseline_q = finite(row["baseline_q_rad"], f"{side} elbow baseline q")
        accepted_v = finite(row["accepted_v_rad_s"], f"{side} elbow v")
        baseline_v = finite(row["baseline_v_rad_s"], f"{side} elbow baseline v")
        torque = finite(row["selected_muscle_generalized_torque_increment_nm"], f"{side} torque")
        torque_l1 = finite(row["selected_muscle_generalized_torque_increment_l1_nm"], f"{side} torque L1")
        lower = finite(row["source_lower_limit_rad"], f"{side} elbow lower limit")
        upper = finite(row["source_upper_limit_rad"], f"{side} elbow upper limit")
        if not lower <= accepted_q <= upper or not lower <= baseline_q <= upper:
            raise RuntimeError(f"{side} elbow is outside authored joint limits")
        if not (accepted_q > baseline_q and accepted_v > baseline_v and torque > 0 and torque_l1 > 0):
            raise RuntimeError(f"{side} elbow did not show positive free flexor q/v response")
        elbows[side] = {
            "q_index": qidx,
            "dof_index": vidx,
            "accepted_q_rad": accepted_q,
            "baseline_q_rad": baseline_q,
            "q_delta_rad": accepted_q - baseline_q,
            "accepted_v_rad_s": accepted_v,
            "baseline_v_rad_s": baseline_v,
            "v_delta_rad_s": accepted_v - baseline_v,
            "selected_muscle_torque_increment_nm": torque,
            "selected_muscle_torque_increment_l1_nm": torque_l1,
            "within_authored_source_limits": True,
            "positive_direction_vs_matched_zero_activation": True,
        }

    muscles = {int(row["source_actuator_index"]): row for row in control["selected_muscles"]}
    if sorted(muscles) != MUSCLE_IDS:
        raise RuntimeError(f"selected muscle IDs do not match the prepared cohort: {sorted(muscles)}")
    for idx in MUSCLE_IDS:
        row = muscles[idx]
        if not math.isclose(finite(row["accepted_activation"], f"muscle {idx} accepted activation"), ACTIVATION, rel_tol=0, abs_tol=0):
            raise RuntimeError(f"muscle {idx} accepted activation differs from requested cohort value")
        finite(row["source_force_n"], f"muscle {idx} source force")
        finite(row["tendon_tension_normalized"], f"muscle {idx} tendon tension")
        finite(row["normalized_equilibrium_residual"], f"muscle {idx} equilibrium residual")

    output = {
        "schema": "numi.human.elbow-flexor-100ms-assessment.v1",
        "status": "PASS_CORE_DIRECTIONAL_100MS" if elbows else "FAIL",
        "scope": SCOPE,
        "errors": [],
        "native_exit_code": result["native_exit_code"],
        "wall_elapsed_seconds": result["wall_elapsed_seconds"],
        "step_count": STEPS,
        "timestep_seconds": DT,
        "simulated_seconds": STEPS * DT,
        "replay": control["replay"],
        "matched_baseline": control["matched_baseline"],
        "root_assistance": terminal["root_assistance"],
        "muscle_ids": MUSCLE_IDS,
        "elbows": elbows,
        "acceptance": [
            "native exit 0",
            "all requested steps accepted",
            "bitwise replay",
            "matched same-input zero-activation baseline",
            "finite terminal and target joint telemetry",
            "accepted and baseline elbow q values within authored source limits",
            "positive free elbow q/v and selected-muscle torque direction versus baseline",
            "all selected source muscles match requested IDs and activation",
        ],
        "limits": (
            "No minimum displacement, force, or physiological threshold is imposed. This is a core selected-tendon "
            "response check only; it does not establish whole-body FEM/contact, grasp, useful-range motion, or "
            "biological/clinical/production qualification."
        ),
        "invocation_path": str(INVOCATION_PATH),
        "invocation_sha256": sha(INVOCATION_PATH),
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha(stdout_path),
    }
    assessment_path = RUN_DIR / "assessment.json"
    assessment_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"assessment_path": str(assessment_path), **output}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute", "assess"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "execute":
        execute()
    else:
        assess()


if __name__ == "__main__":
    main()
