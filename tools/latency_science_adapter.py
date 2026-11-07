#!/usr/bin/env python3
"""Retain and validate one persistent native latency series for Numi science."""
import json
import math
import pathlib
import shlex
import subprocess
import sys


def parse(line, expected_steps, expected_seed):
    fields = dict(part.split("=", 1) for part in shlex.split(line) if "=" in part)
    ints = ("seed", "requested_steps", "attempted_steps", "accepted_env_steps",
            "failed_env_steps", "terminal_events", "timeout_events", "first_gpu_status", "completed")
    nums = ("wall_ms", "call_sum_ms", "latency_p50_ms", "latency_p95_ms",
            "latency_max_ms", "accepted_simulated_s_per_wall_s")
    if any(k not in fields for k in (*ints, *nums, "device", "policy")):
        raise ValueError("latency output is missing required accounting")
    checks = (int(fields["seed"]) == expected_seed,
              int(fields["requested_steps"]) == expected_steps,
              int(fields["attempted_steps"]) == expected_steps,
              int(fields["accepted_env_steps"]) == expected_steps,
              int(fields["failed_env_steps"]) == 0,
              int(fields["first_gpu_status"]) == 0,
              int(fields["completed"]) == 1)
    values = {k: float(fields[k]) for k in nums}
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("non-finite or non-positive latency accounting")
    if not (values["latency_p50_ms"] <= values["latency_p95_ms"] <= values["latency_max_ms"] and
            values["call_sum_ms"] <= values["wall_ms"] * 1.00001):
        raise ValueError("inconsistent latency accounting")
    return {"seed": expected_seed, "steps": expected_steps, "valid": all(checks),
            "latency_p95_ms": values["latency_p95_ms"], "raw_fields": fields}


def main():
    if len(sys.argv) != 5:
        raise ValueError("expected executable, metallib, steps, seed")
    exe, metallib, steps, seed = sys.argv[1:]
    steps, seed = int(steps), int(seed)
    process = subprocess.run([exe, "--metallib", metallib, "--steps", str(steps),
                              "--seed", str(seed)], capture_output=True, text=True)
    pathlib.Path("native.stdout").write_text(process.stdout)
    pathlib.Path("native.stderr").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError("native latency process failed with exit code " + str(process.returncode))
    output = parse(process.stdout.strip(), steps, seed)
    print(json.dumps(output, allow_nan=False))
    return 0 if output["valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
