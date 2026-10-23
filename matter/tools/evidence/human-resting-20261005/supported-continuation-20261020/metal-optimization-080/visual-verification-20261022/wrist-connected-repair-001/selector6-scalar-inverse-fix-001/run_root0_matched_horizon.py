#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/selector6-scalar-inverse-fix-001")
BASE = ROOT / "results/root0-activation0-fixed-scalar-001/invocation.json"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, choices=(8, 16), required=True)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    base = json.loads(BASE.read_text())
    argv = list(base["argv"])
    assert argv[0] == base["binary"]["path"]
    binary = Path(argv[0])
    assert sha(binary) == base["binary"]["sha256"]
    met = base["metallib"]
    assert sha(met["path"]) == met["sha256"]
    lib = base["private_runtime_library"]
    assert sha(lib["path"]) == lib["sha256"]
    def single_flag(flag):
        hits = [i for i, x in enumerate(argv) if x == flag]
        assert len(hits) == 1, (flag, hits)
        return hits[0]
    nidx = single_flag("--muscle-step-count")
    assert argv[nidx + 1] == "1"
    argv[nidx + 1] = str(args.steps)
    out_idx = 4
    assert argv[out_idx].endswith("/root0-activation0-fixed-scalar-001/output")
    run_name = f"root0-activation0-fixed-scalar-{args.steps}steps-125us"
    outdir = ROOT / "results" / run_name
    assert not outdir.exists(), f"refusing to overwrite {outdir}"
    outdir.mkdir(parents=True)
    argv[out_idx] = str(outdir / "output")
    env_override = dict(base["simulation_environment_override"])
    failure_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    assert failure_key in env_override
    old_failure = env_override[failure_key]
    assert old_failure.startswith(str(ROOT / "results/root0-activation0-fixed-scalar-001/"))
    env_override[failure_key] = str(outdir / "common-field-failure.json")
    assert [a for a in argv if a == "--muscle-step-seconds"]
    dtidx = single_flag("--muscle-step-seconds")
    assert argv[dtidx + 1] == "0.000125"
    assert "--selected-tendon-control" in argv and "--mechanics-only" in argv
    assert "--stand-root-assistance" not in argv and "--stand-remove-assistance" not in argv
    assert "--muscle-activation" in argv
    actidx = single_flag("--muscle-activation")
    assert argv[actidx + 1] == "0.0"
    rec = {
        "schema": "numi.human.selector6-matched-root0-horizon.v1",
        "base_invocation": str(BASE),
        "base_invocation_sha256": sha(BASE),
        "steps": args.steps,
        "step_seconds": 0.000125,
        "activation": 0.0,
        "scope": "matched fixed-library ordinary shared-owner root-zero horizon extension; not whole-body qualification",
        "argv_changed": {str(out_idx): {"from": base["argv"][out_idx], "to": argv[out_idx]}, str(nidx + 1): {"from": "1", "to": str(args.steps)}},
        "environment_changed": {failure_key: {"from": old_failure, "to": env_override[failure_key]}},
        "binary": base["binary"],
        "metallib": met,
        "private_runtime_library": lib,
        "inputs_verified": base["inputs_verified"],
        "simulation_environment_override": env_override,
        "simulation_environment_override_count": len(env_override),
        "physics_tolerances_iterations_materials_contact_settings_changed": False,
        "started_unix": time.time(),
        "argv": argv,
        "output_directory": str(outdir),
        "timeout_seconds": args.timeout,
    }
    for p, expected in rec["inputs_verified"].items():
        assert sha(p) == expected, f"input hash changed: {p}"
    (outdir / "invocation.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    (outdir / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    effective = os.environ.copy()
    effective.update(env_override)
    start = time.perf_counter()
    try:
        with (outdir / "stdout.log").open("wb") as stdout, (outdir / "stderr.log").open("wb") as stderr:
            proc = subprocess.run(argv, env=effective, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
        timed_out = False
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
    elapsed = time.perf_counter() - start
    rec.update({"finished_unix": time.time(), "elapsed_seconds": elapsed, "exit_code": exit_code, "timed_out": timed_out, "stdout_path": str(outdir / "stdout.log"), "stderr_path": str(outdir / "stderr.log")})
    (outdir / "result.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run": str(outdir), "exit_code": exit_code, "timed_out": timed_out, "elapsed_seconds": elapsed, "stdout": str(outdir / "stdout.log"), "stderr": str(outdir / "stderr.log")}, sort_keys=True))
    return 0 if exit_code == 0 and not timed_out else 1

if __name__ == "__main__":
    raise SystemExit(main())
