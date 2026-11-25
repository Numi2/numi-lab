#!/usr/bin/env python3
"""Compare the authored workbench fixture with a matched MuJoCo CPU model."""

from argparse import ArgumentParser
import json
from pathlib import Path
import platform
import statistics
import time
import xml.etree.ElementTree as ET

import mujoco
from mujoco.rollout import Rollout
import numpy as np

from metalrobo.experiment import Experiment


def reference_model(project, gravity, substeps):
    """Map the fixture's hinge and inertial fields to the reference model."""
    robot = ET.parse(project / "source/model.urdf").getroot()
    link = robot.find("./link[@name='pendulum']/inertial")
    joint = robot.find("./joint[@name='pivot']")
    diagonal = [link.find("inertia").get(key) for key in ("ixx", "iyy", "izz")]
    damping = joint.find("dynamics").get("damping")
    limits = joint.find("limit")
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco model="bench pendulum">
      <option timestep="{0.02 / substeps}" gravity="0 0 {-gravity}" integrator="Euler"/>
      <worldbody><body pos="{joint.find('origin').get('xyz')}">
        <inertial pos="{link.find('origin').get('xyz')}" mass="{link.find('mass').get('value')}"
          diaginertia="{' '.join(diagonal)}"/>
        <joint name="pivot" axis="{joint.find('axis').get('xyz')}"
          range="{limits.get('lower')} {limits.get('upper')}" damping="{damping}"/>
      </body></worldbody>
      <compiler angle="radian"/>
      <actuator><motor joint="pivot" gear="{limits.get('effort')}"/></actuator>
    </mujoco>
    """)


def timed(call, repeats):
    clock = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        clock.append(time.perf_counter() - start)
    return result, statistics.median(clock)


def measure_reference(model, controls, substeps, environments, repeats):
    workers = min(environments, 8)
    control = np.tile(controls[None, :, None], (environments, 1, 1))
    initial = np.zeros((environments, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
    with Rollout(nthread=workers) as rollout:
        data = [mujoco.MjData(model) for _ in range(workers)]
        def run():
            return rollout.rollout(model, data, initial, control)
        run()
        (state, _), elapsed = timed(run, repeats)
    return state[0, substeps - 1::substeps, 1], elapsed, workers


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environments", type=int, nargs="+", default=(1, 64, 512))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or not all(n > 0 for n in args.environments):
        parser.error("Repeat and environment counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    contents = args.app / "Contents"
    project = contents / "Resources/Scenes/Bench Pendulum.numiproject"
    experiment = Experiment(
        project, rollout=contents / "MacOS/metalrobo_task_rollout",
        metallib=contents / "Resources/MetalRobo.metallib",
    )
    settings = json.loads((project / "project.json").read_text())
    action = np.fromfile(project / "robot.actions.f32", dtype="<f4").astype(float)
    steps = len(action)
    fidelity = []
    for substeps in (4, 10, 20):
        trace = args.output / f"numi-{substeps}.tsv"
        replicate = args.output / f"numi-{substeps}-repeat.tsv"
        sample = experiment.run(steps=steps, physics_substeps=substeps, state_trace=trace,
                                contact_trace=args.output / f"numi-{substeps}-contacts.tsv")
        experiment.run(steps=steps, physics_substeps=substeps, state_trace=replicate)
        model = reference_model(project, settings["gravity"], substeps)
        reference, _, _ = measure_reference(model, np.repeat(action, substeps), substeps, 1, 1)
        data = np.loadtxt(trace)
        error = data[:, 1] - reference
        np.savetxt(args.output / f"matched-{substeps}.csv",
                   np.column_stack((data[:, 0], data[:, 1], reference, error)), delimiter=",",
                   header="control_tick,numi_radians,mujoco_radians,error_radians", comments="")
        fidelity.append({
            "substeps": substeps, "maximum_absolute_error_radians": float(np.max(np.abs(error))),
            "rms_error_radians": float(np.sqrt(np.mean(error ** 2))),
            "deterministic_trace": trace.read_bytes() == replicate.read_bytes(),
            "numi_failed_steps": sample.stats["failed_environment_steps"],
            "numi_maximum_contacts": sample.stats["maximum_active_contacts"],
        })
    performance = []
    model = reference_model(project, settings["gravity"], 4)
    controls = np.repeat(action, 4)
    for environments in args.environments:
        chunk = 1 if environments == 1 else 20
        durations = []
        retained = []
        host = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = experiment.run(steps=steps, environments=environments, chunk=chunk)
            host.append(time.perf_counter() - start)
            stats = result.stats
            durations.append(stats["elapsed_seconds"])
            retained.append(stats["retained_buffer_bytes"])
            if stats["failed_environment_steps"]:
                raise RuntimeError("Numi returned failed benchmark steps")
        _, cpu_time, workers = measure_reference(model, controls, 4, environments, args.repeats)
        transitions = steps * environments
        performance.append({
            "environments": environments, "numi_ticks_per_submission": chunk,
            "numi_internal_ticks_per_second": transitions / statistics.median(durations),
            "numi_process_wall_ticks_per_second": transitions / statistics.median(host),
            "numi_control_tick_ms_amortized": 1000 * statistics.median(durations) / steps,
            "numi_retained_buffer_bytes": max(retained),
            "mujoco_cpu_workers": workers, "mujoco_cpu_ticks_per_second": transitions / cpu_time,
            "mujoco_control_tick_ms_amortized": 1000 * cpu_time / steps,
        })
    report = {
        "device": stats["device"], "system": platform.platform(), "mujoco": mujoco.__version__,
        "control_tick_seconds": 0.02, "ticks_per_run": steps, "repeats": args.repeats,
        "numi_world_fingerprint": stats["world_fingerprint"], "fidelity": fidelity,
        "performance": performance,
        "measurement": "Numi and MuJoCo execute a fixed-base no-contact hinge with matched inertials, "
                       "damping, limits and control. Timed Numi runs report native transactional "
                       "collection. MuJoCo native rollout emits the full physics state buffer.",
    }
    path = args.output / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
