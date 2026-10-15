#!/usr/bin/env python3
"""Source-bound contract test for the common summary/private JSON seam."""
import json
import re
import sys
from pathlib import Path

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "overlay/matter/src/runtime.mm"
text = SOURCE.read_text(encoding="utf-8")
lines = [line.strip() for line in text.splitlines() if 'manifest << ' in line and 'payload' in line]
assert len(lines) == 2, f"expected existing regional and coupled payload writer seams, found {len(lines)}"
line = lines[-1]
match = re.search(r'manifest << ("(?:\\.|[^"\\])*")', line)
assert match, f"payload separator literal not found: {line}"
separator = json.loads(match.group(1))
assert separator == '],"payload":{"file":', repr(separator)
assert ']},\\"payload\\"' not in line, "old extra-brace form returned"

def strict_load(raw):
    def reject(token):
        raise ValueError(f"nonstandard token: {token}")
    return json.loads(raw, parse_constant=reject)

def emit(mode):
    if mode == "summary":
        prefix = {"schema":"numi.matter.coupled_armijo_trial_summary.v1",
          "capture_kind":"observational_all_trial_summary_not_checkpoint",
          "producer_semantics":{"measured_predecision":"after measure, before decide"},
          "phase_order":["entry_global_status","trial_0_predecision"]}
    else:
        prefix = {"schema":"numi.matter.coupled_armijo_trial_capture.v1",
          "capture_kind":"observational_private_trial_not_checkpoint",
          "producer_semantics":{"candidate":"post rebuild"},
          "phase_order":["candidate_before_rebuild","candidate_after_rebuild"]}
    # Emit the prefix exactly through phase_order; use the literal extracted from production.
    before = json.dumps(prefix, separators=(",",":"))
    assert before.endswith("}")
    before = before[:-2]
    payload = json.dumps("trial.bin") + ',"byte_count":4,"sha256":"00000000"}'
    tail = ',"max_payload_bytes":33554432,"max_arrays":192,"array_count":0,"omitted_optional_buffers":[],"arrays":[]}'
    return before + separator + payload + tail

for mode in ("summary", "private"):
    decoded = strict_load(emit(mode))
    assert decoded["capture_kind"].startswith("observational_")
    assert decoded["payload"]["file"] == "trial.bin"
    assert decoded["phase_order"]

# The exact former separator must remain demonstrably invalid at this boundary.
for mode in ("summary", "private"):
    raw = emit(mode).replace(separator, ']},"payload":{"file":', 1)
    try:
        strict_load(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        raise AssertionError(f"old malformed separator unexpectedly parsed in {mode} mode")
print("PASS production-source-bound common manifest separator roundtrips summary and private modes; former extra brace rejected")
