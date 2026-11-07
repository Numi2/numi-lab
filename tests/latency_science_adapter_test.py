#!/usr/bin/env python3
"""Check latency adapter accounting without executing a physical simulation."""
import importlib.util
from pathlib import Path
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adapter", root / "tools/latency_science_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
line = (root / "docs/evidence/benchmark-accounting-20261107/warmup-smoke/latency.log").read_text().strip()
assert adapter.parse(line, 5, 7)["valid"]
assert not adapter.parse(line.replace("accepted_env_steps=5", "accepted_env_steps=4"), 5, 7)["valid"]
assert not adapter.parse(line.replace("first_gpu_status=0", "first_gpu_status=10"), 5, 7)["valid"]
assert not adapter.parse(line, 5, 8)["valid"]
for malformed in (line.replace("latency_p95_ms=3.658367", "latency_p95_ms=nan"),
                  line.replace("latency_p95_ms=3.658367", "latency_p95_ms=4.0"),
                  line.replace("wall_ms=17.650959", "wall_ms=1")):
    try:
        adapter.parse(malformed, 5, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid latency accounting accepted")
print("latency science adapter rejects inconsistent accounting")
