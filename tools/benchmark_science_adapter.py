#!/usr/bin/env python3
"""Adapt one native benchmark invocation for Numi's paired science recorder."""
import json
import math
import pathlib
import shlex
import subprocess
import sys


def parse(line, expected_envs, expected_steps, expected_seed):
    fields = dict(part.split("=", 1) for part in shlex.split(line) if "=" in part)
    required = ("envs", "steps", "seed", "requested_env_steps", "accepted_env_steps",
                "failed_env_steps", "completed", "transition_physics_errors", "nonfinite_rewards")
    if any(k not in fields for k in required):
        raise ValueError("benchmark output is missing required accounting")
    expected = expected_envs * expected_steps
    checks = (int(fields["envs"]) == expected_envs, int(fields["steps"]) == expected_steps,
              int(fields["seed"]) == expected_seed, int(fields["requested_env_steps"]) == expected,
              int(fields["accepted_env_steps"]) == expected, int(fields["failed_env_steps"]) == 0,
              int(fields["completed"]) == 1, int(fields["transition_physics_errors"]) == 0,
              int(fields["nonfinite_rewards"]) == 0)
    rate = float(fields["env_steps_per_s"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid reported throughput")
    return {"seed": expected_seed, "environment_count": expected_envs,
            "steps": expected_steps, "valid": all(checks),
            "env_steps_per_s": rate, "raw_fields": fields}


def main():
    if len(sys.argv) != 6:
        raise ValueError("expected executable, metallib, environments, steps, seed")
    exe, metallib, envs, steps, seed = sys.argv[1:]
    envs, steps, seed = int(envs), int(steps), int(seed)
    process = subprocess.run([exe, "--metallib", metallib, "--envs", str(envs),
                              "--steps", str(steps), "--seed", str(seed)],
                             capture_output=True, text=True)
    pathlib.Path("native.stdout").write_text(process.stdout)
    pathlib.Path("native.stderr").write_text(process.stderr)
    if process.returncode != 0:
        raise RuntimeError("native benchmark failed with exit code " + str(process.returncode))
    output = parse(process.stdout.strip(), envs, steps, seed)
    print(json.dumps(output, allow_nan=False))
    return 0 if output["valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
