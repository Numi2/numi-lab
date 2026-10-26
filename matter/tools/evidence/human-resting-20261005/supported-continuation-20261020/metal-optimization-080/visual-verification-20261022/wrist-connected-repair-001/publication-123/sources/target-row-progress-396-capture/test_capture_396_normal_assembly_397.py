import importlib.util,inspect
from pathlib import Path
import numpy as np
from scipy import sparse
S=importlib.util.spec_from_file_location("capture396_test",Path(__file__).with_name("capture_396_normal_assembly_397.py"));m=importlib.util.module_from_spec(S);S.loader.exec_module(m)
def test_strip_phase_and_redirect_output():
 a=["--base-candidate","/tmp/b","--precision-phase-one-report","/tmp/r","--precision-phase-one-rows-npz=/tmp/n","--precision-safety-factor","4","--precision-cert-tolerance-mm","2e-7","--output","/tmp/o","--immediate-parent-basis"]
 out,removed=m.prepare_argv(a,Path("/tmp/c"))
 assert removed==["--precision-phase-one-report","/tmp/r","--precision-phase-one-rows-npz","/tmp/n","--precision-safety-factor","4","--precision-cert-tolerance-mm","2e-7"]
 assert "--immediate-parent-basis" in out and out[out.index("--output")+1]=="/tmp/c/owner-output-never-created"
def test_target_row_mapping():
 assert m.target_index(np.array([9,18706,8]),18706,3)==1
 for q,n,msg in [(np.array([1]),18706,"occurs 0"),(np.array([18706,18706]),18706,"occurs 2"),(np.array([18706]),18706,"outside")]:
  try:m.target_index(q,n,0 if msg=="outside" else 3)
  except ValueError as e:assert msg in str(e)
  else:raise AssertionError("expected target mapping failure")
def test_extract_basis_solver_call_without_solving():
 def f(A,lower_rows,target_output,lower_output,upper_output,C,h,weights,output_map,time_limit_s,slack_tolerance=1e-9):pass
 A=sparse.csr_matrix([[1.,0.],[0.,2.]]);C=sparse.csr_matrix([[1.,0.]]);M=sparse.csr_matrix([[1.],[.5]])
 p=m.extract_solver((A,np.array([.2,-.1]),np.array([1.,2.]),np.array([-3.,-4.]),np.array([3.,4.]),C,np.array([2.]),np.array([1.]),M,30.),{},inspect.signature(f),True)
 assert p["A"].shape==(2,2) and p["M"].shape==(2,1) and p["lo"].tolist()==[-3.,-4.]
def test_extract_unreduced_solver_call():
 def f(A,lower_rows,target,lower,upper,C,h,weights,time_limit_s,slack_tolerance=1e-9):pass
 p=m.extract_solver((sparse.eye(1),np.array([0.]),np.array([.5]),np.array([-1.]),np.array([1.]),sparse.eye(1),np.array([1.]),np.array([1.]),30.),{},inspect.signature(f),False)
 assert p["M"] is None and p["t"].tolist()==[.5]
def test_dual_summary_maps_typed_row_identity():
 r={"hard_A_ranges":{"pair_and_point_barriers":[0,1],"fixed_037_area":[1,2],"edge_gradient":[2,3]},"pair_and_point_barrier_row_metadata":[{"kind":"fixed_bone_triangle","skin_face_row":3,"bank_body":114}],"area_face_rows":[16]}
 out=m.dual_summary([-2.,0.,0.],[3],["hard_A"],r)
 assert out["top_nonzero_hard_rows"][0]["identity"]["bank_body"]==114
