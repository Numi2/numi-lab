#!/usr/bin/env python3
"""Guard the control-ladder identity across the queued timing workflow."""
import ast
from pathlib import Path
p=Path(__file__).resolve().parent
def strings(name):
    return {node.value for node in ast.walk(ast.parse((p/name).read_text()))
            if isinstance(node, ast.Constant) and isinstance(node.value,str)}
cost=strings('run_accuracy_adjusted_cost_003.py')
queue=strings('run_recovery_sequence_004.py')
ladder=strings('run_be_compressed_ladder_008.py')
assert 'be-precision-008-n2-h*' in cost and 'be-precision-008-n2-h*' in queue
assert 'be-precision-007-n2-h*' not in cost and 'be-precision-007-n2-h*' not in queue
assert 'be-precision-008-n2-h' in ladder
assert any(x.endswith('run_accuracy_adjusted_cost_003.py') for x in queue)
assert any(x.endswith('run_be_compressed_ladder_008.py') for x in queue)
assert 'cost-fair-n2-003' in queue
print('queued control, selection and timing identities agree')
