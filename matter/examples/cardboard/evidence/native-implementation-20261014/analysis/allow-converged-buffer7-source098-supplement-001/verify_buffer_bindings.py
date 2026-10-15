#!/usr/bin/env python3
"""Static regression for explicit buffer-7 bindings in direct checker calls."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CANDIDATE = Path('/Users/home/cardboard-implementation-20261014/analysis/center-residual-jdelta-source098-candidate-001/merged/matter/tools')
OVERLAY = ROOT / 'overlay/matter/tools'
FILES = {
    'cardboard_free_box_angular_native_check.mm': [
        'setComputePipelineState:d_.apply];',
        'setComputePipelineState:d_.applyFreeRigid];',
    ],
    'cardboard_coupled_free_box_operator_check.mm': [
        'setComputePipelineState:device.base.applyFreeRigid];',
        'setComputePipelineState:device.base.apply];',
    ],
}

def source(name: str) -> str:
    p = OVERLAY / name
    return (p if p.exists() else CANDIDATE / name).read_text()

for name, markers in FILES.items():
    text = source(name)
    for marker in markers:
        starts = [m.start() for m in re.finditer(re.escape(marker), text)]
        if len(starts) != 1:
            raise SystemExit(f'{name}: expected one call site for {marker}, found {len(starts)}')
        start = starts[0]
        end = text.find('dispatchThreads', start)
        if end < 0:
            raise SystemExit(f'{name}: dispatch missing after {marker}')
        block = text[start:end]
        if not re.search(r'atIndex:\s*7u?', block):
            raise SystemExit(f'{name}: buffer 7 is unbound at {marker}')
        if 'allowConvergedState' not in block:
            raise SystemExit(f'{name}: buffer 7 is not the convergence guard at {marker}')
    if 'const uint allowConvergedState=0u' not in text and 'const uint allowConvergedState = 0u' not in text:
        raise SystemExit(f'{name}: guard is not explicitly disabled')
print('PASS: all four direct production operator dispatches bind allowConvergedState=0 at buffer 7')
