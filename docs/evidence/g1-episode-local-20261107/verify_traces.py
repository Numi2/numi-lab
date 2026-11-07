"""Cross-check retained selected-environment forward deltas against trace rows."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for label, environment in (("policy-4081007", 2), ("policy-4081008", 0), ("zero-4081007", 2)):
    result = json.loads((root / (label + ".stdout")).read_text())
    assert result["forward_progress_available"]
    assert result["failed_environment_steps"] == 0
    step = result["first_termination_step_by_environment"][environment]
    rows = [line.split() for line in (root / (label + ".trace.tsv")).read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 1000 and int(rows[0][0]) == 1 and int(rows[step-1][0]) == step
    measured = float(rows[step-1][1]) - float(rows[0][1])
    reported = result["final_forward_progress_by_environment_m"][environment]
    assert abs(measured - reported) < 1e-7, (label, measured, reported)
    print(f"{label}: trace and reported displacement agree within 1e-7 m")
