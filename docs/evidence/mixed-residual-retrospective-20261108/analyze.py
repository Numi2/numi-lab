#!/usr/bin/env python3
"""Retrospective calculation from previously captured mixed-contact rows."""
import hashlib
import json
from pathlib import Path

source = Path(__file__).resolve().parent.parent / "mixed-sustained-contact-20261108" / "rowtrace.json"
raw = source.read_bytes()
expected = "b54edb44daa94aa54e11f8a02960a3befcf08fe0ee2a530b6b8ba46ca5f866fb"
assert hashlib.sha256(raw).hexdigest() == expected
run = json.loads(raw)
assert run["success"] == 7 and run["failed"] == 13
assert run["repeat"] and run["exact_trace"] and run["capture_noninvasive"]
assert len(run["steps"]) == 20
result = []
for index, step in enumerate(run["steps"]):
    if step["code"] != 0:
        continue
    rows = step["rows"]
    assert len(rows) == step["contacts"] and rows
    def residual(row, threshold):
        equation = row["row_trace"][0]
        impulse = row["impulse"][0]
        assert impulse >= 0
        return abs(equation) if impulse > threshold else max(-equation, 0)
    legacy = max(residual(row, 1e-7) for row in rows)
    strict = max(residual(row, 0) for row in rows)
    assert abs(legacy - step["normal_residual"]) < 1e-9
    worst_index = max(range(len(rows)), key=lambda i: residual(rows[i], 0))
    row = rows[worst_index]
    result.append({"step": index, "legacy_status_m_per_s": step["normal_residual"],
                   "legacy_reconstructed_m_per_s": legacy,
                   "strict_reconstructed_m_per_s": strict,
                   "worst_row_index": worst_index, "worst_row_flags": row["flags"],
                   "worst_row_normal_impulse_Ns": row["impulse"][0],
                   "worst_row_normal_equation_m_per_s": row["row_trace"][0],
                   "strict_exceeds_2e-5": strict > 2e-5})
assert len(result) == 7
assert sum(x["strict_exceeds_2e-5"] for x in result) == 6
out = {"source_sha256": expected, "accepted_rows_only": True,
       "threshold_m_per_s": 2e-5, "accepted_steps": result}
print(json.dumps(out, indent=2, sort_keys=True))
