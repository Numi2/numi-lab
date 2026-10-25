from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("local_witness_backtracking_339_tested", ROOT / "local_witness_backtracking_339.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


@pytest.mark.parametrize(("scales", "expected_ratio"), [
    ([0.125, 1.0], 7.4383389),  # frozen pass-021 scale pair
    ([0.0625, 1.0], 7.9661214), # frozen pass-029 scale pair
])
def test_actual_337_edge_16752_20012_is_repaired_after_f32(scales, expected_ratio):
    # Frozen 337 base/trial rows. Reference edge is 0.0434513932 mm; both
    # historical local scale states violate the 0.5 component-gradient bound.
    base = np.array([
        [-0.32190507650375366, -1.010913610458374, 0.11835474520921707],
        [-0.32191523909568787, -1.0108751058578491, 0.11833736300468445],
    ], dtype=np.float32)
    delta = np.array([
        [-0.00017347931861877441, 0.0003669261932373047, -0.00005360692739486694],
        [-0.00015175342559814453, 0.00036907196044921875, -0.00003188103437423706],
    ], dtype=np.float64)
    scales = np.array(scales, dtype=np.float64)
    edges = np.array([[0, 1]], dtype=np.int64)
    length = np.linalg.norm(base[0].astype(float) - base[1].astype(float))
    before = MOD._edge_ratios(base, MOD._emit(base, delta, scales), edges, np.array([length]))[0]
    assert before == pytest.approx(expected_ratio, rel=2e-4)
    out, candidate, report = MOD.enforce_emitted_edge_gradient(base, delta, scales, edges)
    emitted = candidate.astype(np.float64) - base.astype(np.float64)
    after = np.max(np.abs(emitted[0] - emitted[1])) / length
    assert report["status"] == "EMITTED_F32_EDGE_GRADIENT_PASS"
    assert after <= 0.5
    assert np.all(out <= scales)
    assert report["edge_constraints_checked_after_float32_emission"] is True


def test_scale_propagation_is_local_monotone_and_fixes_free_boundary_neighbor():
    # Node 0 is an unchanged boundary anchor. The 0-1 bound forces node 1 down;
    # the 1-2 edge is rechecked and couples the reduction to node 2. Node 3 is
    # disconnected and must keep its full proposed scale.
    base = np.array([[0,0,0], [1,0,0], [2,0,0], [7,0,0]], dtype=np.float32)
    delta = np.array([[0,0,0], [1,0,0], [1,0,0], [3,0,0]], dtype=float)
    scales = np.ones(4)
    edges = np.array([[0,1], [1,2]], dtype=np.int64)
    out, candidate, report = MOD.enforce_emitted_edge_gradient(base, delta, scales, edges, max_component_gradient=0.4)
    assert candidate is not None
    assert report["status"] == "EMITTED_F32_EDGE_GRADIENT_PASS"
    assert out[1] < 1.0 and out[2] < 1.0
    assert out[3] == 1.0
    assert np.all(out <= scales)
    assert np.array_equal(candidate[0].view(np.uint32), base[0].view(np.uint32))
    assert np.all(MOD._edge_ratios(base, candidate, edges, np.ones(2)) <= 0.4)


def test_183_witness_wrapper_keeps_anchors_and_pinned_behavior():
    faces = np.array([[0,1,2], [1,3,2]], dtype=np.int64)
    base = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=np.float32)
    delta = np.array([[0,0,0], [2,0,0], [0,0,0], [2,0,0]], dtype=float)
    scales = np.ones(4)
    out, candidate, report = MOD.reduce_witness_node_scales_with_edge_gradient(
        faces, scales, np.array([0]), np.empty((0,2), dtype=np.int64),
        base_points_f32=base, proposed_displacements=delta, anchors=np.array([0]),
        max_component_gradient=0.5,
    )
    assert candidate is not None
    assert np.array_equal(candidate[0].view(np.uint32), base[0].view(np.uint32))
    assert out[0] == 0.0
    assert report["pinned_183_helper_sha256"] == MOD._BASE_183_SHA256
    assert report["edge_gradient_update"]["candidate_usable"] is True


def test_bounded_sweeps_fail_closed_when_residual_violation_remains():
    # The first edge is the only initial violation. Reducing its shared node
    # creates a second adjacent-edge violation that is not in the initial set;
    # one sweep must report that propagated constraint rather than accept it.
    base = np.array([[0,0,0], [1,0,0], [2,0,0]], dtype=np.float32)
    delta = np.array([[0,0,0], [2,0,0], [2,0,0]], dtype=float)
    scales = np.ones(3)
    out, candidate, report = MOD.enforce_emitted_edge_gradient(
        base, delta, scales, np.array([[0,1], [1,2]], dtype=np.int64), max_component_gradient=0.4, max_sweeps=1,
    )
    assert candidate is None
    assert report["status"] == "EDGE_GRADIENT_UNRESOLVED_FAIL_CLOSED"
    assert report["sweeps"] == 1
    assert report["unresolved_edge_count"] > 0
    assert np.all(out <= scales)


def test_invalid_edge_and_scale_inputs_reject():
    base = np.array([[0,0,0], [1,0,0]], dtype=np.float32)
    delta = np.zeros((2,3))
    with pytest.raises(ValueError):
        MOD.enforce_emitted_edge_gradient(base.astype(float), delta, np.ones(2), np.array([[0,1]]))
    with pytest.raises(ValueError):
        MOD.enforce_emitted_edge_gradient(base, delta, np.array([1.1,1.0]), np.array([[0,1]]))
    with pytest.raises(ValueError):
        MOD.enforce_emitted_edge_gradient(base, delta, np.ones(2), np.array([[0,2]]))
