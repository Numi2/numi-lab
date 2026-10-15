#!/usr/bin/env python3
"""Source-bound contract test for the selector 5/6 outcome-dump overlay."""
from pathlib import Path
import hashlib
BASE=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-hybrid-status-includes-129')
CAND=Path(__file__).resolve().parent/'candidate/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm'
EXPECTED='10fdbe2f68d4f31ee767ca3c949d5bee8c7834fe82b73f466d3460827c2fc461'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source=(BASE/'matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm').read_text()
candidate=CAND.read_text()
assert sha(BASE/'matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm')==EXPECTED
assert sha(CAND)!=EXPECTED
assert 'selector_runtime_outcome label=' in candidate
for token in ('statuses_raw_hex=', 'certificates_raw_hex=', 'fem_nodes_raw_hex=',
              'selector_runtime_status label=', 'selector_runtime_certificate label='):
    assert token in candidate, token
start=candidate.index('void checkSelector5And6AcceptanceParity(')
end=candidate.index('\n\n} // namespace',start)
parity=candidate[start:end]
pos5=parity.index('fiveState = runOneAcceptedRelease')
pos6=parity.index('sixState = runOneAcceptedRelease')
posprint5=parity.index('printSelectorRuntimeOutcome("selector5"')
posprint6=parity.index('printSelectorRuntimeOutcome("selector6"')
posgate=parity.index('need(acceptedOutcome(fiveState')
assert pos5 < pos6 < posprint5 < posprint6 < posgate
assert 'catch (const std::exception& error)' in parity
# Preserve the original acceptance checks and their thresholds.
for token in ('state.statuses[0].code == NM_STATUS_SUCCESS',
              'state.solverCertificates[0].validity.w > 0.5f',
              'relativeCorrection > 0.0f', 'restAndFixed.w == 0.0f',
              'restAndFixed.w == 1.0f', 'loadedNodeMotion > 1.0e-8'):
    assert token in candidate, token
# The production runtime and Metal sources are outside this checker-only overlay.
assert 'runOneAcceptedRelease' in source and 'selector_runtime_outcome label=' not in source
print('PASS: frozen checker base; both modes run independently; raw status/certificate/node outcomes emitted before unchanged acceptance gate')
