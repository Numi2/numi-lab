#!/usr/bin/env python3
"""Validate only the native-output accounting adapter, without simulating physics."""
import importlib.util
from pathlib import Path
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adapter", root / "tools/benchmark_science_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
line = (root / "docs/evidence/benchmark-accounting-20261107/native-env32.log").read_text().strip()
result = adapter.parse(line, 32, 10, 7)
assert result["valid"] and result["seed"] == 7
assert not adapter.parse(line.replace("accepted_env_steps=320", "accepted_env_steps=319"), 32, 10, 7)["valid"]
assert not adapter.parse(line.replace("transition_physics_errors=0", "transition_physics_errors=1"), 32, 10, 7)["valid"]
assert not adapter.parse(line, 32, 10, 8)["valid"]
try:
    adapter.parse(line.replace("env_steps_per_s=7304.115714", "env_steps_per_s=nan"), 32, 10, 7)
except ValueError:
    pass
else:
    raise AssertionError("non-finite rate accepted")
print("benchmark science adapter accounting checks passed")
