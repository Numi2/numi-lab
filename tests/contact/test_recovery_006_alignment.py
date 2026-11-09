#!/usr/bin/env python3
"""Check recovery identities before an expensive, serial study."""
import ast
from pathlib import Path
p = Path(__file__).resolve().parent

def strings(name):
    return {x.value for x in ast.walk(ast.parse((p / name).read_text()))
            if isinstance(x, ast.Constant) and isinstance(x.value, str)}

queue = strings("run_recovery_sequence_005.py")
runner = strings("run_spatial_n64_rebound_006.py")
analysis = strings("spatial_n64_rebound_006_analysis.py")
continuation = strings("run_large_spatial_continuation_006.py")
assert "be-precision-008-n2-h*" in queue
assert any(x.endswith("run_be_compressed_ladder_008.py") for x in queue)
assert any(x.endswith("run_accuracy_adjusted_cost_003.py") for x in queue)
assert any(x.endswith("run_large_spatial_continuation_006.py") for x in queue)
assert "run_spatial_n64_rebound_006.py" in queue
assert "n64-rebound-006-supervisor.json" in queue and "n64-rebound-006-supervisor.json" in runner
assert "0.00000025" in runner and "0.0000005" not in runner
assert "rk4-n64-rebound-005-h050.json" in runner
assert "rk4-n64-rebound-005-h050-execution.json" in runner
assert "005" in analysis and "006" in analysis
assert "spatial-n64-rebound-006-analysis.json" in continuation
assert "n64-rebound-006-supervisor.json" in continuation
assert "0.0000005" in continuation and "0.00000025" in continuation
print("recovery identities and half-step values agree")
