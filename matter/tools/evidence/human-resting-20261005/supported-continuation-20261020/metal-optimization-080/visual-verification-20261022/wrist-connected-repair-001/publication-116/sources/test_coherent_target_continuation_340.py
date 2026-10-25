from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_coherent_target_continuation_340.py')
spec = importlib.util.spec_from_file_location('coherent_target_340_tested', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

@pytest.fixture(scope='module')
def context():
    state = mod.read(mod.DEFAULT_STATE)
    candidate = np.load(state['candidate'], allow_pickle=False)
    faces = np.asarray(candidate['source_O_faces'], dtype=np.int64)
    archive, archive_receipt = mod.seed_archive_index()
    return faces, archive, archive_receipt

def test_337_retains_every_complete_exact_audit(context):
    faces, archive, _ = context
    root = mod.FOURTH_SEED_ROOT
    expected_files = sorted(root.rglob('surface-self-pairs.npy'))
    pairs, receipts = mod.collect_pairs(root, faces, archive)
    assert len(expected_files) == len(receipts) == 5
    assert len(pairs) == 7837
    assert all(x['candidate']['mode'] == 'local_candidate_bytes_verified' for x in receipts)
    assert all(x['topology_dimensions'] == {'points': 20889, 'faces': len(faces)} for x in receipts)

def test_union_includes_all_four_pinned_seed_roots(context):
    faces, archive, _ = context
    pairs, receipts = mod.collect_seed_pairs(faces, archive)
    roots = {Path(x['audit_root']) for x in receipts}
    assert roots == {x.resolve() for x in (mod.SEED_ROOT, mod.EXTRA_SEED_ROOT, mod.ADDITIONAL_SEED_ROOT, mod.FOURTH_SEED_ROOT)}
    assert len(receipts) == 39
    assert len(pairs) == 11321
    assert mod.FOURTH_SEED_ROOT.resolve() in roots

def test_337_candidate_topology_mismatch_fails_closed(context):
    faces, archive, _ = context
    wrong = faces.copy()
    wrong[0, [0, 1]] = wrong[0, [1, 0]]
    with pytest.raises(ValueError, match='topology mismatch'):
        mod.collect_pairs(mod.FOURTH_SEED_ROOT, wrong, archive)


def test_initial_trial_gradient_guard_reduces_unsafe_f32_emission():
    base = np.array([[1., 0., 0.], [1.01, 0., 0.], [1., 1., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    scales, emitted, receipt = mod.emit_gradient_checked_trial(
        base, faces, np.array([1], dtype=np.int64), np.array([[0.02, 0., 0.]]),
        np.array([0], dtype=np.int64))
    assert emitted is not None and receipt['candidate_usable'] is True
    assert scales[1] < 1.0
    assert receipt['edge_constraints_checked_after_float32_emission'] is True
    edges = mod.load_module(mod.LOCAL_HELPER_339, 'edge_gradient_test_339').mesh_unique_edges(faces)
    moved = emitted.astype(np.float64) - base.astype(np.float64)
    lengths = np.linalg.norm(base[edges[:, 0]].astype(np.float64) - base[edges[:, 1]].astype(np.float64), axis=1)
    ratios = np.max(np.abs(moved[edges[:, 0]] - moved[edges[:, 1]]), axis=1) / lengths
    assert np.max(ratios) <= 0.5


def test_initial_trial_gradient_guard_preserves_passing_raw_f32_bytes():
    base = np.array([[1., 0., 0.], [1.01, 0., 0.], [1., 1., 0.]], dtype=np.float32)
    raw = base.copy()
    raw[1, 1] = np.float32(0.0005)
    displacement = raw[[1]].astype(np.float64) - base[[1]].astype(np.float64)
    scales, emitted, receipt = mod.emit_gradient_checked_trial(
        base, np.array([[0, 1, 2]], dtype=np.int64), np.array([1], dtype=np.int64),
        displacement, np.array([0], dtype=np.int64))
    assert receipt['candidate_usable'] is True
    assert np.all(scales == 1.0)
    assert np.array_equal(emitted.view(np.uint32), raw.view(np.uint32))
