import numpy as np,pytest
import build_joint_outward_targets597_v2 as m
from test_joint_outward_targets597 import depth
def test_joint_selection(depth):
 d,p=m.select_with_fallback(depth,np.array([0.,0.,1.]),np.array([0.,0.,1.]))
 assert p["joint_source_current_feasible"] and d[2]==1
def test_explicit_fold_exception_preserves_goal(depth):
 s=np.array([0.,0.,1.]);d,p=m.select_with_fallback(depth,s,-s)
 assert not p["joint_source_current_feasible"] and np.array_equal(d,s)and p["selected_depth_cosine"]==-1
def test_unbounded_free_interval_is_explicit():
 r=m.strict_records({"target_selection":{"free_gap_m":float("inf")},"other":1.})
 assert r["target_selection"]=={"free_gap_m":None,"free_gap_is_unbounded":True}
def test_unexpected_nonfinite_fails():
 for r in [{"target_distance_m":float("inf")},{"target_selection":{"free_gap_m":float("nan")}}]:
  with pytest.raises(ValueError):m.strict_records(r)
