#!/usr/bin/env python3
"""Build and run the exact CPU scratch-accounting checker."""
import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
cand = root / "candidate/matter"
exe = Path("/tmp/full_contact_armijo_scratch_budget_check_176_002")
compile_argv = [
    "clang++", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    f"-I{cand / 'include'}", f"-I{root / 'base/matter/include'}",
    str(cand / "tools/cardboard_full_contact_armijo_scratch_budget_check.cpp"),
    "-o", str(exe),
]
compiled = subprocess.run(compile_argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
record = {"compile_argv": compile_argv, "compile_exit": compiled.returncode, "compile_output": compiled.stdout}
if compiled.returncode == 0:
    run = subprocess.run([str(exe)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    record.update({"run_argv": [str(exe)], "run_exit": run.returncode, "run_output": run.stdout})
else:
    record.update({"run_argv": None, "run_exit": None, "run_output": "not run after compile failure"})
(root / "validation/cpu-budget-check.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
raise SystemExit(0 if record["compile_exit"] == 0 and record["run_exit"] == 0 else 1)
