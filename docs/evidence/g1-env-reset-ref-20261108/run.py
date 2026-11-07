#!/usr/bin/env python3
# Execute once only after exclusive GPU slot authorization, from this directory.
import datetime
import os
import subprocess
from pathlib import Path

evidence = Path(__file__).resolve().parent
build = Path("/Users/numi/numi-g1-env-reset-ref-build-20261108")
for name in ("gpu-test.stdout", "gpu-test.stderr", "gpu-test.timing", "snapshots.tsv", "snapshots.tsv.partial.tsv", "preflight.sha256-check", "postflight.sha256-check"):
    if (evidence / name).exists():
        raise SystemExit("refusing to overwrite prior output: " + name)
with (evidence / "preflight.sha256-check").open("w") as out:
    check = subprocess.run(["shasum", "-a", "256", "-c", "declared-identities.sha256"], cwd=evidence, stdout=out, stderr=subprocess.STDOUT)
if check.returncode:
    raise SystemExit("preflight hash failure")
start = datetime.datetime.now(datetime.timezone.utc)
try:
    with (evidence / "gpu-test.stdout").open("w") as out, (evidence / "gpu-test.stderr").open("w") as err:
        run = subprocess.run([str(build / "reset_ref_probe"), str(build / "shaders/MetalRobo.metallib"),
                              str(build / "assets"), str(evidence / "snapshots.tsv")],
                             stdout=out, stderr=err, timeout=120, cwd=evidence)
        result = str(run.returncode)
except subprocess.TimeoutExpired:
    result = "timeout"
end = datetime.datetime.now(datetime.timezone.utc)
(evidence / "gpu-test.timing").write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={result}\n")
with (evidence / "postflight.sha256-check").open("w") as out:
    check = subprocess.run(["shasum", "-a", "256", "-c", "declared-identities.sha256"], cwd=evidence, stdout=out, stderr=subprocess.STDOUT)
print(f"exit={result} start_utc={start.isoformat()} end_utc={end.isoformat()} postflight={check.returncode}")

