#!/usr/bin/env python3
"""Exercise persistent latency accounting using deterministic native API faults."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / "tests/benchmark_accounting_test.py").read_text())
mock = next(ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "mock" for target in node.targets))
mock = mock.replace("calls==2", "calls==21")
mock = mock.replace("a->successful_environment_steps=2;", "a->successful_environment_steps=0;")
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    (temp / "mock.cpp").write_text(mock)
    binary = temp / "latency"
    subprocess.run(["clang++", "-std=c++23", "-I", str(root / "include"),
                    str(root / "apps/benchmark_latency.cpp"), str(temp / "mock.cpp"),
                    "-o", str(binary)], check=True)
    def run(mode="", *args):
        return subprocess.run([str(binary), "--steps", "3", *args],
                              env=dict(os.environ, NUMI_BENCH_TEST_MODE=mode),
                              capture_output=True, text=True)
    ok = run()
    assert ok.returncode == 0, ok.stderr
    fields = dict(part.split("=", 1) for part in shlex.split(ok.stdout))
    assert fields["completed"] == "1" and fields["accepted_env_steps"] == "3"
    assert fields["attempted_steps"] == "3" and float(fields["wall_ms"]) >= float(fields["call_sum_ms"])
    assert 0 < float(fields["latency_p50_ms"]) <= float(fields["latency_p95_ms"]) <= float(fields["latency_max_ms"])
    expected = 3 * (1.0 / 60.0) * 1000.0 / float(fields["wall_ms"])
    assert abs(float(fields["accepted_simulated_s_per_wall_s"]) / expected - 1) < 0.002
    for mode in ("incomplete", "error", "physics_error", "nan"):
        result = run(mode)
        assert result.returncode != 0, (mode, result.stdout)
        f = dict(part.split("=", 1) for part in shlex.split(result.stdout))
        assert f["completed"] == "0" and f["attempted_steps"] == "1", (mode, f)
    assert run("warmup").returncode != 0
    for args in (("--steps", "0"), ("--steps", "4294967296"), ("--seed", " -1")):
        assert run("", *args).returncode != 0, args
print("persistent latency accounting rejects incomplete and invalid steps")
