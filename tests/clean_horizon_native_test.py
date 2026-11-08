#!/usr/bin/env python3
"""Native metric regression; run on an Apple host after building MetalRobo.

  python3 tests/clean_horizon_native_test.py --build /path/to/build

This synthetic fixture is a control-flow test, not a trained policy benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"{command!r} exited {result.returncode}:\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--policy-pack", type=Path,
                        help="optional compatible policy; otherwise use zero actions")
    args = parser.parse_args()
    build = args.build.resolve()
    rollout = build / "bin/metalrobo_task_rollout"
    metallib = build / "shaders/MetalRobo.metallib"
    library = build / "lib"
    with tempfile.TemporaryDirectory(prefix="numi-clean-horizon-") as tmp:
        scratch = Path(tmp)
        generator = scratch / "generate"
        run(["xcrun", "clang++", "-std=c++20", "-O0", "-I", str(ROOT / "include"),
             str(ROOT / "tests/fixtures/clean_horizon_fixture.cpp"),
             "-L", str(library), "-lmetalrobo", f"-Wl,-rpath,{library}",
             "-o", str(generator)])
        for name, force in (("normal", False), ("failure", True)):
            case_dir = scratch / name
            (case_dir / "fixture").mkdir(parents=True)
            env = os.environ.copy()
            env.pop("NUMI_HORIZON_FORCE_FAILURE", None)
            if force:
                env["NUMI_HORIZON_FORCE_FAILURE"] = "1"
            run([str(generator)], cwd=case_dir, env=env)

        def evaluate(name, envs, chunk, steps, repeats=1):
            f = scratch / name / "fixture"
            command = [str(rollout), "--world-pack", str(f / "world.mrworld"),
                       "--task-pack", str(f / "task.taskpack"),
                       "--robot-actuator-pack", str(f / "actuators.actuatorpack"),
                       "--sensor-program-pack", str(f / "sensors.sensorpack"),
                       "--reality-program-pack", str(f / "reality.realitypack"),
                       "--envs", str(envs), "--chunk", str(chunk),
                       "--steps", str(steps), "--repeats", str(repeats),
                       "--seed", "4081019", "--metallib", str(metallib)]
            if args.policy_pack and name == "normal":
                command += ["--policy-pack", str(args.policy_pack.resolve())]
            else:
                command += ["--zero-actions"]
            return json.loads(run(command))

        def check(name, envs, chunk, steps, repeats, expected_rate, reasons):
            outcome = evaluate(name, envs, chunk, steps, repeats)
            actual = (outcome["clean_horizon_environment_rate"],
                      outcome["termination_reason_counts"],
                      outcome["failed_environment_steps"])
            expected = (expected_rate, reasons, 0)
            if actual != expected:
                raise AssertionError(f"{name} e={envs} c={chunk} s={steps} "
                                     f"r={repeats}: {actual!r} != {expected!r}")
            print(f"PASS {name} e={envs} c={chunk} s={steps} r={repeats}: {actual}",
                  flush=True)

        for envs in (1, 2):
            for chunk in (1, 8):
                for steps in (255, 256, 257):
                    rate = 0 if steps == 257 else (1 if envs == 1 else 0.5)
                    check("normal", envs, chunk, steps, 1, rate,
                          {} if steps == 255 else {"4": 1})
        for chunk in (1, 8):
            check("normal", 2, chunk, 220, 1, 1, {})
            check("normal", 2, chunk, 221, 1, 0.5, {})
        check("normal", 1, 8, 128, 2, 1, {"4": 1})
        check("normal", 1, 8, 256, 2, 0, {"4": 1})
        for steps in (1, 8):
            check("failure", 1, 1, steps, 1, 0, {"3": steps})


if __name__ == "__main__":
    main()
