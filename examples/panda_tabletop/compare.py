#!/usr/bin/env python3
"""Replay a compiled Panda tabletop scene through Numi and, optionally, MuJoCo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import subprocess
from pathlib import Path

DT = 0.02
SUBSTEPS = 4
HOLD_START = 320
MIN_LIFT = 0.060
MAX_OPEN_LIFT = 0.005
MUJOCO_PANDA_SHA256 = "c047a49f3c770ff2393e2a74f7e0fbd9d99080a305407254f8ff53ac053cde3b"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trace(path: Path) -> list[list[float]]:
    with path.open() as source:
        columns = next(source)
        if "nq=9 " not in columns or "scene_bodies=4 " not in columns:
            raise ValueError("Unexpected Panda state trace layout")
        rows = [[float(value) for value in line.split()] for line in source]
    if not rows or not all(
        len(row) == 62
        and row[0] == tick
        and all(math.isfinite(value) for value in row)
        for tick, row in enumerate(rows, start=1)
    ):
        raise ValueError("Panda state trace has missing or invalid samples")
    return rows


def native_run(
    rollout: Path, metallib: Path, scene: Path, clip: Path,
    directory: Path, ticks: int,
) -> tuple[dict, list[list[float]]]:
    directory.mkdir()
    trace = directory / "state.tsv"
    command = [
        str(rollout), "--scene-file", str(scene), "--headless",
        "--metallib", str(metallib), "--steps", str(ticks), "--action-clip",
        str(clip), "--physics-substeps", str(SUBSTEPS), "--state-trace", str(trace),
        "--contact-trace", str(directory / "contacts.tsv"), "--contact-body", "pick_object",
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "runtime.json").open("w") as stdout, (
        directory / "runtime.log"
    ).open("w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    runtime = json.loads((directory / "runtime.json").read_text())
    rows = load_trace(trace)
    if len(rows) != ticks:
        raise ValueError(f"Expected {ticks} native steps, received {len(rows)}")
    return runtime, rows


def measure(rows: list[list[float]]) -> dict[str, float]:
    hold = rows[HOLD_START - 1 :]
    start = rows[0][10:13]
    first_hold = hold[0][10:13]
    return {
        "peak_lift_m": max(row[12] - start[2] for row in rows),
        "minimum_hold_lift_m": min(row[12] - start[2] for row in hold),
        "maximum_hold_drift_m": max(math.dist(first_hold, row[10:13]) for row in hold),
        "final_cube_height_m": rows[-1][12],
    }


def mujoco_reference(
    xml: Path, targets: list[list[float]], output: Path, cube_mass: float, cube_friction: float,
) -> dict:
    import mujoco

    model_hash = sha256(xml)
    if model_hash != MUJOCO_PANDA_SHA256:
        raise ValueError("The MuJoCo reference requires the same pinned stock panda.xml")
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = DT / SUBSTEPS
    # The Panda model's stock joints, actuators, fingertip geometry and
    # constraint parameters stay in the reference. The tabletop is shared.
    spec.worldbody.add_geom(
        name="numi_table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05],
        friction=[0.7, 0.005, 0.0001],
    )
    cube = spec.worldbody.add_body(name="numi_cube", pos=[0.5, 0, 0.025])
    cube.add_freejoint(name="numi_cube_free")
    cube.add_geom(
        name="numi_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025], mass=cube_mass, friction=[cube_friction, 0.005, 0.0001],
    )
    # Keep the unused Numi target fixture and clutter present in the reference.
    target = spec.worldbody.add_body(name="numi_target", pos=[0.45, 0.25, 0])
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.004])
    clutter = spec.worldbody.add_body(name="numi_clutter", pos=[0.35, -0.25, 0.03])
    clutter.add_freejoint()
    clutter.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.03], mass=0.08,
    )
    # Stock Panda's keyframe has nine qpos values. This task adds two free bodies.
    spec.delete(spec.key("home"))
    model = spec.compile()
    data = mujoco.MjData(model)
    joint_names = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    indices = [model.joint(name).qposadr[0] for name in joint_names]
    data.qpos[indices] = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2,
                          math.pi / 4, 0.035, 0.035]
    mujoco.mj_forward(model, data)
    body = model.body("numi_cube").id
    rows = []
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tick", *joint_names, "cube_x_m", "cube_y_m", "cube_z_m"])
        for tick, target in enumerate(targets, 1):
            data.ctrl[:7] = target[:7]
            data.ctrl[7] = 255 * (target[7] + target[8]) / 0.08
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            # mj_step integrates after computing derived body transforms.
            mujoco.mj_kinematics(model, data)
            row = [tick, *map(float, data.qpos[indices]), *map(float, data.xpos[body])]
            if not all(map(math.isfinite, row)):
                raise ValueError(f"MuJoCo state became non-finite at tick {tick}")
            rows.append(row)
            writer.writerow(row)
    return {"version": mujoco.__version__, "source_sha256": model_hash, **measure(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="compiled panda-tabletop.numi-window.json")
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mujoco-panda", type=Path, help="original stock panda.xml")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = args.scene.resolve()
    clip = scene.parent / "tabletop.actions.f32"
    fixture_path = scene.parent / "tabletop.experiment.json"
    fixture = json.loads(fixture_path.read_text()) if fixture_path.exists() else {
        "cube_mass_kg": 0.1, "cube_static_friction": 0.7,
    }
    with (scene.parent / "tabletop.targets.csv").open() as stream:
        targets = [[float(v) for v in row[1:]] for row in list(csv.reader(stream))[1:]]
    ticks = len(targets)
    raw = clip.read_bytes()
    if ticks != 370 or len(raw) != ticks * 9 * 4:
        raise ValueError("This comparison requires the compiled Panda tabletop task")
    open_clip = output / "open-gripper.actions.f32"
    opened = bytearray(raw)
    for tick in range(ticks):
        # Hold both fingers at 39 mm, with the same arm targets as the grasp.
        struct.pack_into("<ff", opened, (tick * 9 + 7) * 4, 0.1, 0.1)
    open_clip.write_bytes(opened)

    runs = {}
    for name, sequence in (("grasp", clip), ("replay", clip), ("open-gripper", open_clip)):
        runtime, rows = native_run(
            args.rollout.resolve(), args.metallib.resolve(), scene, sequence, output / name, ticks
        )
        with (output / name / "contacts.tsv").open() as contact_file:
            contacts = list(csv.DictReader(contact_file, delimiter="\t"))
        if len(contacts) != ticks or any(
            int(contact["tick"]) != tick or
            not math.isfinite(float(contact["peak_normal_force_n"]))
            for tick, contact in enumerate(contacts, 1)
        ):
            raise ValueError(f"Incomplete native contact trace for {name}")
        fingertip_loads = [
            float(contact["peak_normal_force_n"]) for contact in contacts[HOLD_START - 1 :]
            if {"pick_object", "panda_leftfinger"} == {contact["body_a"], contact["body_b"]}
            or {"pick_object", "panda_rightfinger"} == {contact["body_a"], contact["body_b"]}
        ]
        runs[name] = {
            **measure(rows),
            "hold_ticks_with_fingertip_contact": len(fingertip_loads),
            "peak_hold_fingertip_normal_force_n": max(fingertip_loads, default=0),
            "contact_trace_sha256": sha256(output / name / "contacts.tsv"),
            "failed_environment_steps": runtime["failed_environment_steps"],
            "maximum_active_contacts": runtime["maximum_active_contacts"],
            "retained_buffer_bytes": runtime["retained_buffer_bytes"],
            "peak_aliased_bytes": runtime["peak_aliased_bytes"],
            "control_steps_per_second_with_trace": runtime["environment_control_steps_per_second"],
            "run_fingerprint": runtime["run_fingerprint"],
            "trace_sha256": sha256(output / name / "state.tsv"),
        }
    checks = {
        "no_failed_steps": all(run["failed_environment_steps"] == 0 for run in runs.values()),
        "replay_identical": (
            runs["grasp"]["trace_sha256"] == runs["replay"]["trace_sha256"]
            and runs["grasp"]["contact_trace_sha256"] == runs["replay"]["contact_trace_sha256"]
        ),
        "fingertip_contact_during_hold": (
            runs["grasp"]["hold_ticks_with_fingertip_contact"] == ticks - HOLD_START + 1
        ),
        "hold_lift_passed": runs["grasp"]["minimum_hold_lift_m"] >= MIN_LIFT,
        "open_control_passed": runs["open-gripper"]["peak_lift_m"] <= MAX_OPEN_LIFT,
    }
    summary = {
        "scope": "Deterministic tabletop demonstration with distinct native Panda and stock MuJoCo dynamics.",
        "fixture": fixture,
        "host": platform.platform(),
        "control_timestep_s": DT,
        "physics_substeps": SUBSTEPS,
        "control_steps": ticks,
        "hold_start_tick": HOLD_START,
        "minimum_hold_lift_m": MIN_LIFT,
        "maximum_open_control_lift_m": MAX_OPEN_LIFT,
        "rollout_sha256": sha256(args.rollout),
        "metallib_sha256": sha256(args.metallib),
        "worldpack_sha256": sha256(scene.parent / "tabletop.worldpack"),
        "clip_sha256": sha256(clip),
        "numi": runs,
        "checks": checks,
    }
    if args.mujoco_panda is not None:
        reference = mujoco_reference(
            args.mujoco_panda.resolve(), targets, output / "mujoco-state.csv",
            fixture["cube_mass_kg"], fixture["cube_static_friction"],
        )
        summary["mujoco_stock_reference"] = reference
        checks["mujoco_reference_hold_lift_passed"] = reference["minimum_hold_lift_m"] >= MIN_LIFT
    report = json.dumps(summary, indent=2) + "\n"
    (output / "comparison.json").write_text(report)
    print(report, end="")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
