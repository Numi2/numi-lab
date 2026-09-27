#!/usr/bin/env python3
"""Audit paired, accepted native Human recovery episodes.

Input JSON has an ``episodes`` array. Each entry names ``off``, ``on``,
``offReplay``, and ``onReplay`` native launcher directories and the common
``pushStartStep``. This tool reads evidence; it never advances physics.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex


PROGRESS = re.compile(r"^human_standing_progress=accepted step=(\d+)\b")
SENSOR = re.compile(r"^human_brain_sensor_audit=accepted step=(\d+)\b")
JOINT = re.compile(r"^human_brain_joint_commit=accepted step=(\d+)\b")
VECTOR = r"\[([^]]+)\]"
HASHED_INPUTS = ("binary ", "lab_library ", "brain_library ",
                 "brain_resource ", "metal_library ", "rigid_source ",
                 "muscle_source ", "support_contact_source ",
                 "joint_equality_source ", "tendon_source ",
                 "bodyparts3d_bone_source ", "bodyparts3d_muscle_surface_source ",
                 "launcher ", "locomotor_program ", "lab_source_file=")


@lru_cache(maxsize=256)
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field(line, name, vector=False):
    match = re.search(rf"\b{re.escape(name)}=" + (VECTOR if vector else r"([^ ]+)"), line)
    if match is None:
        raise ValueError(f"accepted row lacks {name}")
    if vector:
        values = tuple(float(value) for value in match.group(1).split(","))
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid {name}")
        return values
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {name}")
    return value


def manifest(directory):
    entries = (directory / "launch.manifest").read_text().splitlines()
    retained = {}
    for line in entries:
        if line.startswith("retained_artifact "):
            _, digest, path = line.split(None, 2)
            if digest in retained and retained[digest] != path:
                raise ValueError("retained input hash has conflicting archive paths")
            retained[digest] = path
    for line in entries:
        if line.startswith(HASHED_INPUTS):
            parts = line.split(None, 2)
            archived = directory.parent / ".native-artifact-store" / parts[1] if len(parts) == 3 else None
            if len(parts) != 3 or not (
                (Path(parts[2]).is_file() and sha256(parts[2]) == parts[1])
                or (parts[1] in retained and
                    Path(retained[parts[1]]).is_file() and
                    sha256(retained[parts[1]]) == parts[1])
                or (archived.is_file() and sha256(archived) == parts[1])
            ):
                raise ValueError(f"native input hash disagrees with retained file: {parts[0]}")
    command = next(line[8:] for line in entries if line.startswith("command="))
    argv = shlex.split(command)
    if "--stand-brain-program" not in argv or "--stand-push" not in argv:
        raise ValueError("native command lacks Brain program or timed push")
    if argv.count("--stand-contact-iterations") != 1 or argv[
        argv.index("--stand-contact-iterations") + 1] != "64":
        raise ValueError("native command lacks 64 contact sweeps")
    if argv.count("--muscle-step-seconds") != 1 or float(argv[
        argv.index("--muscle-step-seconds") + 1]) != 0.001:
        raise ValueError("recovery audit requires the 1 ms native task")
    if "--stand-root-assistance" in argv or "--stand-remove-assistance" in argv:
        raise ValueError("native command selected an assisted path")
    input_hashes = sorted(line for line in entries if line.startswith((
        *HASHED_INPUTS,
        "lab_source_file=", "lab_source_revision=", "seed=",
        "native_timestep_microseconds=", "brain_epoch_microseconds=")))
    return argv, input_hashes


def read_run(directory, steps):
    directory = Path(directory)
    argv, hashes = manifest(directory)
    rows, sensors, joints = {}, {}, {}
    progress_digest = hashlib.sha256()
    sensor_digest = hashlib.sha256()
    completed = terminal = False
    for line in (directory / "launch.log").open(errors="replace"):
        if match := PROGRESS.match(line):
            step = int(match.group(1))
            if step in rows:
                raise ValueError(f"duplicate physical step {step}")
            progress_digest.update(line.encode())
            rows[step] = {
                "root": field(line, "root_xyz_m", True),
                "com": field(line, "center_of_mass_xyz_m", True),
                "velocity": field(line, "root_linear_velocity_xyz_m_s", True),
                "contacts": int(field(line, "contact_count")),
                "penetration": field(line, "penetration_m"),
                "supportForce": field(line, "support_force_n"),
                "excitation": field(line, "brain_muscle_excitation_mean"),
                "assistanceForce": field(line, "root_assistance_force_n"),
                "assistanceTorque": field(line, "root_assistance_torque_nm"),
            }
        if match := SENSOR.match(line):
            step = int(match.group(1))
            if step in sensors:
                raise ValueError(f"duplicate sensory step {step}")
            sensor_digest.update(line.encode())
            sensors[step] = line.rstrip("\n")
        if match := JOINT.match(line):
            step = int(match.group(1))
            if step in joints:
                raise ValueError(f"duplicate joint witness {step}")
            if (int(field(line, "brain_generation")) != step or
                    not all(f"{flag}=true" in line for flag in (
                        "physical_motor_same_command",
                        "accepted_consequence_followup_command",
                        "same_native_owner_queue"))):
                raise ValueError(f"joint ownership failed at step {step}")
            joints[step] = line.rstrip("\n")
        completed |= ("human_execution_stage=native_horizon_end" in line
                      and f"stage_step={steps}" in line)
        terminal |= line.startswith("stand_terminal_state=")
        if ("human_brain_completion=failed" in line or
                "human_brain_joint_commit=rejected" in line):
            raise ValueError("native Brain/Human transaction failed")
    if (directory / "launch.exit").read_text().splitlines() != [
        "binary_exit_code=0", "tee_exit_code=0", "launcher_exit_code=0"]:
        raise ValueError("native launcher failed")
    if list(rows) != list(range(1, steps + 1)) or list(sensors) != list(range(1, steps + 1)):
        raise ValueError("physical or sensory accepted steps are missing or unordered")
    if steps not in joints or not completed or not terminal:
        raise ValueError("native terminal or final joint witness is missing")
    if any(row["assistanceForce"] != 0 or row["assistanceTorque"] != 0
           for row in rows.values()):
        raise ValueError("runtime root assistance is nonzero")
    return {"directory": str(directory), "argv": argv, "hashes": hashes,
            "rows": rows, "physicalSHA256": progress_digest.hexdigest(),
            "sensorSHA256": sensor_digest.hexdigest()}


def normalized_command(argv):
    argv = argv.copy()
    argv[4] = "<output>"
    argv[argv.index("--stand-brain-program") + 1] = "<program>"
    return argv


def source_without_program(hashes):
    return [line for line in hashes if not line.startswith("locomotor_program ")]


def outcome(run, push_start, steps):
    rows = run["rows"]
    before = rows[push_start]
    final = [rows[step] for step in range(steps - 499, steps + 1)]
    horizontal = lambda row: math.hypot(row["com"][0] - before["com"][0],
                                        row["com"][1] - before["com"][1])
    speed = lambda step: math.hypot(
        rows[step]["com"][0] - rows[step - 1]["com"][0],
        rows[step]["com"][1] - rows[step - 1]["com"][1]) / 0.001
    minimum_contacts = min(row["contacts"] for row in rows.values())
    peak_penetration = max(row["penetration"] for row in rows.values())
    peak_com_drift = max(horizontal(row) for row in final)
    peak_final_speed = max(speed(step) for step in range(steps - 499, steps + 1))
    minimum_root_height = min(row["root"][2] for row in rows.values())
    mean_excitation = sum(row["excitation"] for row in rows.values()) / steps
    success = (minimum_contacts >= 6 and peak_penetration <= 5e-6
               and minimum_root_height >= before["root"][2] - 0.02
               and peak_com_drift <= 0.01 and peak_final_speed <= 0.01)
    return {"success": success, "minimumContacts": minimum_contacts,
            "peakPenetrationM": peak_penetration,
            "peakFinalCOMDriftM": peak_com_drift,
            "peakFinalHorizontalSpeedMPerS": peak_final_speed,
            "minimumRootHeightM": minimum_root_height,
            "meanMuscleExcitationProxy": mean_excitation,
            "physicalSHA256": run["physicalSHA256"],
            "sensorSHA256": run["sensorSHA256"]}


def audit(data):
    steps = data.get("steps", 5000)
    episodes = data["episodes"]
    if steps < 1500 or len(episodes) < 1:
        raise ValueError("recovery audit needs at least 1500 steps and one episode")
    results = []
    for episode in episodes:
        start = episode["pushStartStep"]
        if not 1 <= start < steps - 500:
            raise ValueError("push must precede the final 500 step window")
        off = read_run(episode["off"], steps)
        on = read_run(episode["on"], steps)
        off_replay = read_run(episode["offReplay"], steps)
        on_replay = read_run(episode["onReplay"], steps)
        if any(normalized_command(run["argv"]) != normalized_command(off["argv"])
               or source_without_program(run["hashes"]) != source_without_program(off["hashes"])
               for run in (on, off_replay, on_replay)):
            raise ValueError("paired arms differ in native task, source, seed, or hardware")
        for primary, replay in ((off, off_replay), (on, on_replay)):
            if (primary["hashes"] != replay["hashes"] or
                    primary["physicalSHA256"] != replay["physicalSHA256"] or
                    primary["sensorSHA256"] != replay["sensorSHA256"]):
                raise ValueError("same-arm exact physical/sensory replay failed")
        if off["hashes"] == on["hashes"]:
            raise ValueError("paired arms use the same motor program")
        push = off["argv"].index("--stand-push")
        if int(off["argv"][push + 1]) != start:
            raise ValueError("push start differs from evaluation manifest")
        results.append({"off": outcome(off, start, steps),
                        "on": outcome(on, start, steps),
                        "pushNewtons": list(map(float, off["argv"][push + 3:push + 6]))})
    gain = sum(int(item["on"]["success"]) - int(item["off"]["success"])
               for item in results) / len(results)
    randomizer = random.Random(0x4E554D49)
    differences = [int(item["on"]["success"]) - int(item["off"]["success"])
                   for item in results]
    samples = sorted(sum(randomizer.choice(differences) for _ in differences)
                     / len(differences) for _ in range(10000))
    lower = samples[249]
    upper = samples[9750]
    metric_names = ("peakPenetrationM", "peakFinalCOMDriftM",
                    "peakFinalHorizontalSpeedMPerS", "meanMuscleExcitationProxy")
    means = {name: {arm: sum(item[arm][name] for item in results) / len(results)
                    for arm in ("off", "on")} for name in metric_names}
    no_regression = (all(value["on"] <= value["off"] + 1e-12
                         for value in means.values()) and
                     all(item["on"]["minimumContacts"] >= item["off"]["minimumContacts"]
                         for item in results))
    passed = len(results) >= 64 and gain >= 0.10 and lower > 0 and no_regression
    return {"schema": "numi.human.brain.recovery-audit.v1",
            "episodes": results, "episodeCount": len(results),
            "successGain": gain, "pairedBootstrap95": [lower, upper],
            "meanMetrics": means, "physicalNoRegression": no_regression,
            "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "episodes"}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
