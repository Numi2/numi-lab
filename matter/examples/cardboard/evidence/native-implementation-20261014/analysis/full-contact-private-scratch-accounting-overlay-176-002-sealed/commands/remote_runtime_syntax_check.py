#!/usr/bin/env python3
"""Source-bound Objective-C++ syntax check for the 176 candidate on macmini."""
import json
import subprocess
from pathlib import Path

SNAP = Path("/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-sparse-private-scan-175-buildfix-003")
CAND = Path("/Users/n/cardboard-implementation-20261014-host-macmini/analysis/full-contact-private-scratch-accounting-overlay-176-002/candidate/matter")
METALLIB = "/Users/n/cardboard-implementation-20261014-host-macmini/build-source175-buildfix-003/shaders/NumiMatter.metallib"
SOURCE = CAND / "src/runtime.mm"
OUT = Path("/Users/n/cardboard-implementation-20261014-host-macmini/analysis/full-contact-private-scratch-accounting-overlay-176-002/validation")
OUT.mkdir(parents=True, exist_ok=True)
common = [
    "/usr/bin/clang++", f'-DNUMI_MATTER_DEFAULT_METALLIB="{METALLIB}"',
    f"-I{CAND / 'include'}", f"-I{SNAP / 'matter/include'}",
    f"-I{SNAP / 'matter/src'}", f"-I{SNAP / 'include'}",
    "-x", "objective-c++", "-std=c++23", "-arch", "arm64", "-fPIC",
    "-fobjc-arc", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    "-fsyntax-only", str(SOURCE),
]
records = []
for label, opts in (("O3", ["-O3", "-DNDEBUG"]), ("O0", ["-O0", "-UNDEBUG"])):
    argv = common[:1] + opts + common[1:]
    run = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    record = {"label": label, "argv": argv, "exit_code": run.returncode, "output": run.stdout}
    (OUT / f"runtime-syntax-{label}.json").write_text(json.dumps(record, indent=2) + "\n")
    records.append(record)
print(json.dumps({"checks": [{"label": r["label"], "exit_code": r["exit_code"], "output": r["output"]} for r in records]}, indent=2))
raise SystemExit(0 if all(r["exit_code"] == 0 for r in records) else 1)
