from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
SOLVER = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-local-witness-proposal-315/propose_joint_clearance_witnesses_315.py")
SPEC = importlib.util.spec_from_file_location("fixed_support_existing_315", SOLVER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_fixed_prefix_zero_disables_only_contiguous_prefix_bounds():
    support=np.array([0,2,4,6],dtype=np.int64)
    lo=np.full(12,-1.0); hi=np.full(12,1.0)
    got_lo,got_hi,locked=MOD.apply_fixed_prefix_zero_bounds(lo,hi,support,0,8)
    assert locked.size==0
    np.testing.assert_array_equal(got_lo,lo)
    np.testing.assert_array_equal(got_hi,hi)
    old_lo,old_hi,old_locked=MOD.apply_fixed_prefix_zero_bounds(lo,hi,support,5,8)
    np.testing.assert_array_equal(old_locked,[0,2,4])
    np.testing.assert_array_equal(old_lo[:9],np.zeros(9))
    np.testing.assert_array_equal(old_hi[:9],np.zeros(9))

def test_support_emitter_moves_supported_old_row_but_locks_anchor_and_outside():
    points=np.arange(24,dtype=np.float32).reshape(8,3)/100.0
    support=np.array([0,2,4,6],dtype=np.int64)
    anchors=np.array([2],dtype=np.int64)
    delta=np.array([[.25,0,0],[9,9,9],[-.5,0,0],[0,.75,0]],dtype=np.float64)
    emitted=MOD.emit_raw_lp_points(points,points.astype(np.float64)*1000.0,support,delta,anchors)
    assert not np.array_equal(emitted[0].view(np.uint32),points[0].view(np.uint32))
    assert not np.array_equal(emitted[4].view(np.uint32),points[4].view(np.uint32))
    np.testing.assert_array_equal(emitted[2].view(np.uint32),points[2].view(np.uint32))
    outside=np.setdiff1d(np.arange(len(points)),support)
    np.testing.assert_array_equal(emitted[outside].view(np.uint32),points[outside].view(np.uint32))
    changed=set(np.flatnonzero(np.any(emitted!=points,axis=1)).tolist())
    assert changed=={0,4,6}
    assert changed.issubset(set(support))

def test_anchor_reset_does_not_depend_on_prefix_bounds():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]],dtype=np.float32)
    support=np.array([0,1,2],dtype=np.int64)
    anchors=np.array([1],dtype=np.int64)
    delta=np.ones((3,3),dtype=np.float64)
    emitted=MOD.emit_raw_lp_points(points,points.astype(np.float64)*1000.0,support,delta,anchors)
    np.testing.assert_array_equal(emitted[1].view(np.uint32),points[1].view(np.uint32))
    np.testing.assert_array_equal(emitted[3].view(np.uint32),points[3].view(np.uint32))
