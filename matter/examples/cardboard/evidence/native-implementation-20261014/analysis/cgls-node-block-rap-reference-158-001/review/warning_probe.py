"""Single-vector finite/warning probe; does not rerun the CGLS study."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'analysis/cgls-node-block-rap-reference-158-001/run.py'
spec = importlib.util.spec_from_file_location('sealed_run158_warning_probe', RUN)
run158 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run158
spec.loader.exec_module(run158)

audit = run158.ref.load_audit()
_, graph_ref = audit.load_capture_readers()
topology, free_mask, potential_mask, _ = audit.parse_graph_input(
    run158.ref.GRAPH_INPUT, run158.ref.GRAPH_BINDINGS)
capture, _ = graph_ref.load_base_view(
    run158.ref.REGIONAL, environment=0, topology_bytes=topology,
    free_mask_bytes=free_mask, potential_mask_bytes=potential_mask,
    request_actual_consumed_view=False)
capture = dataclasses.replace(capture, completed_tail_latch=False)
A, _, free_nodes, _, _ = audit.load_sparse_A(
    run158.ref.CSRPKG / 'A_free_csr.npz',
    run158.ref.CSRPKG / 'manifest.json', free_mask)
rng = np.random.default_rng(5)
input_vector = rng.standard_normal(3 * len(free_nodes))
results = []
for dtype in (np.float64, np.float32):
    mapping = run158.BlockRAP(A, free_nodes, capture, dtype, 1.0)
    x = input_vector.astype(dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        y = mapping.apply(x)
        z = mapping.transpose(x)
    results.append({
        'dtype': str(np.dtype(dtype)),
        'input_linf': float(np.max(np.abs(x))),
        'apply_finite': bool(np.isfinite(y).all()),
        'apply_linf': float(np.max(np.abs(y))),
        'transpose_finite': bool(np.isfinite(z).all()),
        'transpose_linf': float(np.max(np.abs(z))),
        'warnings': dict(Counter(str(w.message) for w in caught)),
    })

# Surface the first warning as an exception too; do not suppress or relabel it.
old = np.seterr(all='raise')
try:
    try:
        _ = mapping.P.T @ x
        raised = None
    except FloatingPointError as exc:
        raised = str(exc)
finally:
    np.seterr(**old)

print(json.dumps({
    'schema': 'numi.cardboard.block_rap_warning_probe.v1',
    'result': results,
    'seterr_raise_probe': raised,
    'warning_cause': 'unresolved; output finiteness does not explain the reported NumPy/BLAS flags',
}, indent=2, allow_nan=False))
