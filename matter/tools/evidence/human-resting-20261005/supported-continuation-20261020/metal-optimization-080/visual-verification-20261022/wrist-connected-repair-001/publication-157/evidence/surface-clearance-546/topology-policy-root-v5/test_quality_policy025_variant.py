import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
p=HERE/"quality_policy025_variant.py"
s=importlib.util.spec_from_file_location("quality_policy025_variant_test",p)
policy=importlib.util.module_from_spec(s);s.loader.exec_module(policy)
class FakeQuality:
    MIN_AREA_RATIO=.1
    MIN_PATCH_ANGLE_DEG=2.
    def __init__(self,current,new):self.current=current;self.new=new
    def face_quality(self,points_mm,ref_mm,faces,ref_faces):
        faces=np.asarray(faces)
        if np.array_equal(faces,np.array([[0,1,2],[0,2,3]])):return self.current
        return self.new
def qvals(area=(.5,.6),cos=(.4,.5),angle=(10.,12.)):
    return {"area_ratio":np.asarray(area,dtype=float),"normal_cos":np.asarray(cos,dtype=float),"min_angle_deg":np.asarray(angle,dtype=float)}
def inputs(current=None,new=None):
    current=qvals() if current is None else current;new=qvals() if new is None else new
    q=FakeQuality(current,new);faces=np.array([[0,1,2],[0,2,3]],dtype=np.int64)
    c={"source_O_faces":faces,"area_reference_037_source_O_faces":faces}
    op={"face_rows":[0,1],"new_faces":[[0,1,3],[1,2,3]]}
    return q,c,op
def both(current,new):
    q,c,op=inputs(current,new)
    before=q.face_quality(None,None,c["source_O_faces"],c["area_reference_037_source_O_faces"])
    fast=policy.evaluate_op_local(q,c,op,before,None,None)
    parity=policy.evaluate_op_parity(q,c,op,before,None,None)
    assert fast==parity
    return fast
def test_zero_quality_gain_is_eligible_until_exact_185_reduction_gate():
    result,reason=both(qvals(),qvals())
    assert reason=="eligible"
    assert result["deltas"]=={"area_ratio":0.,"normal_cosine":0.,"min_angle_deg":0.}
    assert result["strict_quality_gain_required_before_exact_pairs"] is False
    assert result["exact_185_strict_reduction_still_required_downstream"] is True
def test_absolute_floors_remain():
    assert both(qvals(angle=(8,9)),qvals(angle=(1.999,3)))[1]=="angle_loss_bound"
    assert both(qvals(),qvals(area=(.099,.7)))[1]=="fixed037_area_floor"
    assert both(qvals(),qvals(cos=(0.,.7)))[1]=="new_face_not_aligned_to_fixed037"
def test_angle_loss_is_bounded_by_minimum_of_five_and_half_old_minimum():
    result,reason=both(qvals(angle=(20,22)),qvals(angle=(15,18)))
    assert reason=="eligible" and result["angle_loss_limit_deg"]==5.
    assert both(qvals(angle=(20,22)),qvals(angle=(14.9,18)))[1]=="angle_loss_bound"
    result,reason=both(qvals(angle=(7,9)),qvals(angle=(3.5,5)))
    assert reason=="eligible" and result["angle_loss_limit_deg"]==3.5
    assert both(qvals(angle=(7,9)),qvals(angle=(3.49,5)))[1]=="angle_loss_bound"
def test_nonfinite_metrics_fail_closed_in_both_evaluators():
    assert both(qvals(),qvals(area=(np.nan,.5)))[1]=="nonfinite_quality_metrics"
def test_fast_and_parity_hooks_share_exact_policy():
    q_path=Path("/pinned/541.py");q=SimpleNamespace(evaluate_op=lambda *args:("old","old"))
    topology=SimpleNamespace(evaluate_op_local=lambda *args:("old-local","old-local"),loadmod=lambda name,path:q)
    installed=policy.install_policy_hooks(topology,policy,q_path)
    assert topology.evaluate_op_local is policy.evaluate_op_local
    assert topology.loadmod("541",q_path).evaluate_op is policy.evaluate_op_parity
    assert installed==[q]
