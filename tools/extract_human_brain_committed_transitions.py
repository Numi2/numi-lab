#!/usr/bin/env python3
"""Extract accepted sensory-action-consequence records from a native Human run.

The motor journal is opt-in. This tool only reads a completed, source-bound
launcher run; it never invents actions for missing or rejected steps.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex


HASHED_INPUTS = ("binary ", "lab_library ", "brain_library ",
                 "brain_resource ", "metal_library ", "rigid_source ",
                 "muscle_source ", "support_contact_source ",
                 "joint_equality_source ", "tendon_source ",
                 "bodyparts3d_bone_source ", "bodyparts3d_muscle_surface_source ",
                 "launcher ", "locomotor_program ", "lab_source_file=")
KINDS = ("human_standing_progress=accepted", "human_brain_sensor_audit=accepted",
         "human_brain_motor_audit=accepted", "human_brain_joint_commit=accepted")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token(line, name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if match is None:
        raise ValueError(f"accepted row lacks {name}")
    return match.group(1)


def number(line, name, integer=False):
    raw = token(line, name)
    value = int(raw) if integer else float(raw)
    if not integer and not math.isfinite(value):
        raise ValueError(f"accepted row has nonfinite {name}")
    return value


def vector(line, name, count, integer=False):
    raw = token(line, name)
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError(f"accepted row has malformed {name}")
    values = [int(part) if integer else float(part)
              for part in raw[1:-1].split(",")]
    if len(values) != count or (not integer and not all(
            math.isfinite(value) for value in values)):
        raise ValueError(f"accepted row has invalid {name} values")
    return values


def manifest(run):
    path = run / "launch.manifest"
    entries = path.read_text().splitlines()
    fields = {}
    for line in entries:
        if "=" in line and not line.startswith("command="):
            name, value = line.split("=", 1)
            if " " not in name and name not in fields:
                fields[name] = value
    if fields.get("accepted_sensor_audit") != "1":
        raise ValueError("native launcher did not request accepted sensor audit")
    interval = int(fields.get("accepted_motor_audit_interval", "0"))
    steps = int(fields.get("requested_physical_steps", "0"))
    if not 1 <= interval <= 1000 or not 2 <= steps <= 10000:
        raise ValueError("native run lacks a usable motor interval or horizon")
    retained = {}
    for line in entries:
        if line.startswith("retained_artifact "):
            _, digest, artifact = line.split(None, 2)
            retained[digest] = Path(artifact)
    source_hashes = []
    for line in entries:
        if not line.startswith(HASHED_INPUTS):
            continue
        kind, digest, source = line.split(None, 2)
        candidates = (Path(source), retained.get(digest),
                      run.parent / ".native-artifact-store" / digest)
        if not any(candidate is not None and candidate.is_file() and
                   sha256(candidate) == digest for candidate in candidates):
            raise ValueError(f"native source hash is not retained: {kind}")
        source_hashes.append({"kind": kind, "sha256": digest})
    if not all(any(entry["kind"] == kind for entry in source_hashes) for kind in
               ("binary", "lab_library", "brain_library", "locomotor_program")):
        raise ValueError("native Brain/Human owner inputs are incomplete")
    commands = [line[8:] for line in entries if line.startswith("command=")]
    if len(commands) != 1:
        raise ValueError("native run command is missing or ambiguous")
    argv = shlex.split(commands[0])
    if argv.count("--stand-contact-iterations") != 1 or argv[
            argv.index("--stand-contact-iterations") + 1] != "64":
        raise ValueError("native run did not use 64 contact sweeps")
    if "--stand-root-assistance" in argv or "--stand-remove-assistance" in argv:
        raise ValueError("native run selected assisted standing")
    if (run / "launch.exit").read_text().splitlines() != [
            "binary_exit_code=0", "tee_exit_code=0", "launcher_exit_code=0"]:
        raise ValueError("native launcher did not complete")
    return steps, interval, source_hashes, sha256(path)


def sensor(line, step):
    if (number(line, "brain_generation", True) != step or
            number(line, "receptor_timestamp_us", True) != step * 1000 or
            number(line, "delivery_timestamp_us", True) != (step + 1) * 1000):
        raise ValueError(f"sensor clocks disagree at step {step}")
    result = {
        "touchValidity": vector(line, "touch_validity", 10, True),
        "touchNormalForceN": vector(line, "touch_normal_force_n", 10),
        "headValidity": number(line, "head_validity", True),
        "headQuaternionXYZW": vector(line, "head_quaternion_xyzw", 4),
        "rootPositionValidity": number(line, "root_position_validity", True),
        "rootPositionXYZM": vector(line, "root_position_xyz_m", 3),
        "rootLinearVelocityValidity": number(
            line, "root_linear_velocity_validity", True),
        "rootLinearVelocityXYZMPerS": vector(
            line, "root_linear_velocity_xyz_m_s", 3),
    }
    if (result["rootPositionValidity"] != 7 or
            result["rootLinearVelocityValidity"] != 896 or
            any(value < 0 for value in result["touchNormalForceN"])):
        raise ValueError(f"required native sensory fields are invalid at step {step}")
    return result


def progress(line):
    assistance = (number(line, "root_assistance_force_n"),
                  number(line, "root_assistance_torque_nm"))
    if assistance != (0.0, 0.0):
        raise ValueError("native run used root assistance")
    return {
        "centerOfMassXYZM": vector(line, "center_of_mass_xyz_m", 3),
        "rootXYZM": vector(line, "root_xyz_m", 3),
        "contactCount": number(line, "contact_count", True),
        "penetrationM": number(line, "penetration_m"),
        "supportForceN": number(line, "support_force_n"),
    }


def extract(run):
    steps, interval, hashes, manifest_digest = manifest(run)
    rows = {kind: {} for kind in KINDS}
    log_path = run / "launch.log"
    completed = terminal = False
    for line in log_path.open(errors="replace"):
        if (line.startswith("human_brain_sensor_audit=failed") or
                line.startswith("human_brain_motor_audit=failed") or
                line.startswith("human_brain_completion=failed") or
                line.startswith("human_brain_joint_commit=rejected")):
            raise ValueError("native run contains a rejected Brain/Human receipt")
        for kind in KINDS:
            if line.startswith(kind + " "):
                step = number(line, "step", True)
                if step in rows[kind]:
                    raise ValueError(f"duplicate {kind} step {step}")
                rows[kind][step] = line.rstrip("\n")
                break
        if line.startswith("human_execution_stage=native_horizon_end"):
            completed |= number(line, "stage_step", True) == steps
        terminal |= line.startswith("stand_terminal_state=")
    physical, sensory, motor, joints = (rows[kind] for kind in KINDS)
    sampled = set(range(interval, steps + 1, interval))
    if (list(physical) != list(range(1, steps + 1)) or
            list(sensory) != list(range(1, steps + 1)) or
            list(motor) != sorted(sampled) or not completed or not terminal or
            steps not in joints):
        raise ValueError("native run lacks a complete accepted physical, sensory, or motor trace")
    for step in range(1, steps + 1):
        progress(physical[step])
        sensor(sensory[step], step)
    for step, line in joints.items():
        if (number(line, "brain_generation", True) != step or
                number(line, "joint_commit_fingerprint", True) !=
                number(sensory[step], "joint_commit_fingerprint", True) or
                not all(f"{flag}=true" in line for flag in (
                    "physical_motor_same_command",
                    "accepted_consequence_followup_command",
                    "same_native_owner_queue"))):
            raise ValueError(f"joint ownership witness disagrees at step {step}")
    transitions = []
    for step in sorted(sampled):
        line = motor[step]
        if (number(line, "brain_generation", True) != step or
                number(line, "physical_fingerprint", True) !=
                number(sensory[step], "physical_fingerprint", True) or
                number(line, "joint_commit_fingerprint", True) !=
                number(sensory[step], "joint_commit_fingerprint", True) or
                number(line, "locomotor_program_fingerprint", True) == 0):
            raise ValueError(f"motor journal disagrees with joint commit at step {step}")
        action = vector(line, "excitation", 416)
        activation = vector(line, "activation", 416)
        if any(value < 0 or value > 1 for value in action + activation):
            raise ValueError(f"muscle command or activation is out of range at step {step}")
        if step == 1:
            continue  # There is no accepted sensor frame for state zero.
        transitions.append({
            "step": step,
            "observationStep": step - 1,
            "observation": sensor(sensory[step - 1], step - 1),
            "muscleExcitation": action,
            "nextMuscleActivation": activation,
            "consequence": sensor(sensory[step], step),
            "physicalConsequence": progress(physical[step]),
            "jointCommitFingerprint": number(
                line, "joint_commit_fingerprint", True),
        })
    receipt = {
        "schema": "numi.human.brain.committed-transitions.v1",
        "evidenceBoundary": "accepted native sensor/action/consequence data; no learned skill claim",
        "run": str(run.resolve()), "steps": steps, "motorAuditInterval": interval,
        "transitionCount": len(transitions),
        "launchManifestSHA256": manifest_digest,
        "launchLogSHA256": sha256(log_path), "sourceSHA256": hashes,
    }
    return transitions, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    transitions, receipt = extract(args.run)
    args.output.mkdir(parents=True)
    with (args.output / "transitions.jsonl").open("w") as destination:
        for row in transitions:
            destination.write(json.dumps(row, separators=(",", ":"),
                                         allow_nan=False) + "\n")
    receipt["transitionsSHA256"] = sha256(args.output / "transitions.jsonl")
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run": receipt["run"], "steps": receipt["steps"],
                      "motorAuditInterval": receipt["motorAuditInterval"],
                      "transitionCount": receipt["transitionCount"],
                      "receipt": str(args.output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
