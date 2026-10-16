"""CPU-only captured-state comparison; no native or physical qualification."""
from pathlib import Path
from dataclasses import replace
import hashlib
import importlib.util
import json
import sys
import warnings
from contextlib import nullcontext

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
OUT = Path(__file__).parent
SOURCE = B / 'analysis/native464-cgls-base-preconditioner-audit-002/run_cgls_preconditioners.py'
spec = importlib.util.spec_from_file_location('captured_cgls_ref158', SOURCE)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

import numpy as np
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:
    # The command fixes the BLAS worker environment before NumPy imports.
    threadpool_limits = lambda **kwargs: nullcontext()


class BlockRAP:
    def __init__(self, A, free_nodes, capture, dtype, weight):
        self.dtype = dtype
        self.weight = dtype(weight)
        self.H = ref.RightMap('block3', free_nodes, A, capture, dtype)
        indices = (np.asarray(free_nodes)[:, None] * 3 + np.arange(3)).reshape(-1)
        self.P = np.asarray(capture.P, dtype=dtype)[indices]
        self.AP = np.asarray(capture.AP, dtype=dtype)[indices]
        self.factor = capture.base_factor
        self.Q = self.P - np.stack([self.H.apply(self.AP[:, i])
                                   for i in range(self.AP.shape[1])], axis=1)
        assert self.factor.valid and np.all(np.isfinite(self.Q))

    def apply(self, v):
        h = self.H.apply(v)
        if self.weight == 0:
            return h
        c = ref.solve_lu(self.factor, self.P.T @ v, self.dtype, False)
        return (h + self.weight * (self.Q @ c)).astype(self.dtype, copy=False)

    def transpose(self, u):
        h = self.H.transpose(u)
        if self.weight == 0:
            return h
        c = ref.solve_lu(self.factor, self.Q.T @ u, self.dtype, True)
        return (h + self.weight * (self.P @ c)).astype(self.dtype, copy=False)


def main():
    assert not (OUT / 'results.json').exists(), 'immutable result path already exists'
    bindings = {}
    for path in [SOURCE, B / 'analysis/native464-cgls-base-preconditioner-audit-002/manifest.json',
                 B / 'analysis/native464-regional-capture-audit-003/manifest.json',
                 ref.REGIONAL, ref.GRAPH_INPUT, ref.GRAPH_BINDINGS,
                 ref.CSRPKG / 'manifest.json', ref.CSRPKG / 'A_free_csr.npz']:
        bindings[str(path)] = ref.hfile(path)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        audit = ref.load_audit()
        state_reader = audit.load_state_reader()
        regional_reader, graph_ref = audit.load_capture_readers()
        state_doc = regional_reader.v1._strict_json(ref.REGIONAL)
        topology, mask, potential, graph_meta = audit.parse_graph_input(ref.GRAPH_INPUT, ref.GRAPH_BINDINGS)
        capture, view = graph_ref.load_base_view(
            ref.REGIONAL, environment=0, topology_bytes=topology,
            free_mask_bytes=mask, potential_mask_bytes=potential,
            request_actual_consumed_view=False)
        assert capture.completed_tail_latch, 'expected retained completed-tail capture'
        clone = replace(capture, completed_tail_latch=False)
        state_path = ref.CASE / state_doc['candidate_state_binding']['manifest_file']
        state = audit.read_state(state_path, state_reader)
        A, _, free_nodes, dofs, dof_map = audit.load_sparse_A(
            ref.CSRPKG / 'A_free_csr.npz', ref.CSRPKG / 'manifest.json', mask)
        bfull = np.frombuffer(state['arrays']['base_residual_rhs'], dtype='<f4').reshape(4, 882, 4)[0, :, :3]
        b = bfull[np.asarray(free_nodes)].reshape(-1).astype(np.float64)
        assert A.shape == (len(b), len(b)) and np.isfinite(b).all()
        bindings[str(state_path)] = ref.hfile(state_path)
        state_binary = state['payload_path']
        bindings[str(state_binary)] = ref.hfile(state_binary)
        cases = []
        checks = []
        # Declare this bounded grid before looking at the new recurrence output.
        for dtype in [np.float64, np.float32]:
            for weight in [0.0, 0.1, 1.0]:
                mapping = BlockRAP(A, free_nodes, clone, dtype, weight)
                rng = np.random.default_rng(158464)
                u = rng.standard_normal(len(b)).astype(dtype)
                v = rng.standard_normal(len(b)).astype(dtype)
                sv, stu = mapping.apply(v), mapping.transpose(u)
                assert np.isfinite(sv).all() and np.isfinite(stu).all()
                left = float(u.astype(np.float64) @ sv.astype(np.float64))
                right = float(stu.astype(np.float64) @ v.astype(np.float64))
                scale = float(np.linalg.norm(u.astype(np.float64)) * np.linalg.norm(sv.astype(np.float64)) +
                              np.linalg.norm(v.astype(np.float64)) * np.linalg.norm(stu.astype(np.float64)))
                error = abs(left - right) / max(scale, 1e-300)
                gate = 1e-12 if dtype == np.float64 else 1e-5
                checks.append({'dtype': str(np.dtype(dtype)), 'coarse_weight': weight,
                               'adjoint_absolute_gap': abs(left-right), 'norm_scaled_gap': error,
                               'gate': gate, 'pass': error <= gate})
                result = ref.cgls(A, b, mapping, cap=64, dtype=dtype)
                result.update(dtype=str(np.dtype(dtype)), coarse_weight=weight)
                cases.append(result)
        result = {'schema': 'numi.cardboard.block_rap_cpu_comparison.v1',
                  'map': 'S=H+w(P-H*AP)*R^-1*P^T; H=node-block inverse; exact algebraic transpose',
                  'scope': 'unlatched base formula on clone, not a captured consumed augmented map',
                  'operator_shape': list(A.shape), 'coarse_modes': int(clone.mode_count),
                  'physical_operator_changed': False, 'native_run': False,
                  'complete_nonlinear_or_physical_qualification': False,
                  'bindings': bindings, 'adjoint_checks': checks, 'cases': cases,
                  'warnings': [{'category': w.category.__name__, 'message': str(w.message),
                                'file': w.filename, 'line': w.lineno} for w in seen]}
        (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'cases': [{k: row[k] for k in ['dtype', 'coarse_weight', 'status',
                                                       'iterations', 'final_relative_true_residual']}
                                    for row in cases], 'all_adjoint_checks_pass': all(c['pass'] for c in checks)}, indent=2))


if __name__ == '__main__':
    main()
