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
    policy_options = parser.add_mutually_exclusive_group()
    policy_options.add_argument("--policy-pack", type=Path,
                                help="optional compatible policy")
    policy_options.add_argument("--self-policy", action="store_true",
                                help="generate a runtime-compatible zero policy")
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
        for name in ("normal", "failure", "reset"):
            case_dir = scratch / name
            (case_dir / "fixture").mkdir(parents=True)
            env = os.environ.copy()
            env.pop("NUMI_HORIZON_FORCE_FAILURE", None)
            env.pop("NUMI_HORIZON_AUTHORED_RESET", None)
            if name == "failure":
                env["NUMI_HORIZON_FORCE_FAILURE"] = "1"
            if name == "reset":
                env["NUMI_HORIZON_AUTHORED_RESET"] = "1"
            run([str(generator)], cwd=case_dir, env=env)

        generated_policy = None

        def evaluate(name, envs, chunk, steps, repeats=1, periodic=True, extra=()):
            f = scratch / name / "fixture"
            command = [str(rollout), "--world-pack", str(f / "world.mrworld"),
                       "--task-pack", str(f / "task.taskpack"),
                       "--robot-actuator-pack", str(f / "actuators.actuatorpack"),
                       "--sensor-program-pack", str(f / "sensors.sensorpack"),
                       "--reality-program-pack", str(f / "reality.realitypack"),
                       "--envs", str(envs), "--chunk", str(chunk),
                       "--steps", str(steps), "--repeats", str(repeats),
                       "--seed", "4081019", "--metallib", str(metallib)]
            if not periodic:
                command.append("--no-periodic-resets")
            policy_path = args.policy_pack or generated_policy
            if policy_path and name == "normal":
                command += ["--policy-pack", str(policy_path.resolve())]
            else:
                command += ["--zero-actions"]
            command.extend(extra)
            return json.loads(run(command))

        if args.self_policy:
            # Capture the contract from this executable and this exact fixture.
            contract = evaluate("normal", 1, 1, 1)
            policy_generator = scratch / "generate-policy"
            run(["xcrun", "clang++", "-std=c++20", "-O0", "-I", str(ROOT / "include"),
                 str(ROOT / "tests/fixtures/clean_horizon_policy.cpp"),
                 "-L", str(library), "-lmetalrobo", f"-Wl,-rpath,{library}",
                 "-o", str(policy_generator)])
            generated_policy = scratch / "zero.policypack"
            keys = ("world_fingerprint", "task_fingerprint", "observation_fingerprint",
                    "action_fingerprint", "actor_observation_count", "action_count")
            run([str(policy_generator), str(generated_policy)] +
                [str(contract[key]) for key in keys])

        def check(name, envs, chunk, steps, repeats, expected_rate, reasons,
                  periodic=True):
            outcome = evaluate(name, envs, chunk, steps, repeats, periodic)
            if not periodic and (outcome["host_requested_resets"] != envs or
                                 not outcome["scheduled_resets"] or
                                 outcome["periodic_resets"]):
                raise AssertionError(f"initial or periodic reset mismatch: {outcome!r}")
            actual = (outcome["clean_horizon_environment_rate"],
                      outcome["termination_reason_counts"],
                      outcome["failed_environment_steps"])
            expected = (expected_rate, reasons, 0)
            if actual != expected:
                raise AssertionError(f"{name} e={envs} c={chunk} s={steps} "
                                     f"r={repeats}: {actual!r} != {expected!r}")
            print(f"PASS {name} e={envs} c={chunk} s={steps} r={repeats}: {actual}",
                  flush=True)

        # The authored reset moves the first scene object from x=0.5 to 0.9.
        # Observe actual state, not just the requested-reset counter.
        for enabled in (False, True):
            trace = scratch / f"authored-reset-{enabled}.tsv"
            extra = ["--state-trace", str(trace)]
            if not enabled:
                extra.append("--no-scheduled-resets")
            result = evaluate("reset", 1, 1, 1, periodic=False, extra=extra)
            rows = [line for line in trace.read_text().splitlines()
                    if line and not line.startswith("#")]
            x = float(rows[0].split("\t")[10])  # step, nine q, scene x
            expected_host_resets = 1 if enabled else 0
            if (abs(x - 0.9) > 1e-4 or result["failed_environment_steps"] or
                    result["host_requested_resets"] != expected_host_resets):
                raise AssertionError(f"host reset={enabled}: x={x}, "
                                     f"requests={result['host_requested_resets']}")
            print(f"PASS host reset={enabled}: x={x}, "
                  f"requests={result['host_requested_resets']}", flush=True)

        for envs in (1, 2):
            for chunk in (1, 8):
                for steps in (255, 256, 257):
                    rate = 0 if steps == 257 else (1 if envs == 1 else 0.5)
                    check("normal", envs, chunk, steps, 1, rate,
                          {} if steps == 255 else {"4": 1})
        for chunk in (1, 8):
            check("normal", 2, chunk, 220, 1, 1, {})
            check("normal", 2, chunk, 221, 1, 0.5, {})
        for chunk in (1, 8):
            check("normal", 2, chunk, 221, 1, 1, {}, periodic=False)
            check("normal", 2, chunk, 256, 1, 1, {"4": 2}, periodic=False)
            check("normal", 2, chunk, 257, 1, 0, {"4": 2}, periodic=False)
        check("normal", 1, 8, 128, 2, 1, {"4": 1})
        check("normal", 1, 8, 256, 2, 0, {"4": 1})
        for steps in (1, 8):
            check("failure", 1, 1, steps, 1, 0, {"3": steps})


if __name__ == "__main__":
    main()
