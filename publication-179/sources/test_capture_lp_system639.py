import importlib.util, json, sys, types
from pathlib import Path
import numpy as np
from scipy import sparse

P=Path(__file__).with_name("capture_lp_system639.py")
spec=importlib.util.spec_from_file_location("capture_lp_system639_test",P)
cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)

def test_capture_exact_lp_inputs_and_passthrough(tmp_path):
    class Owner:
        def build_immediate_parent_basis(self,base,support,anchors):
            return sparse.eye(2,format="csr"),np.asarray([0,1]),{"output_scope_rows":[4,5],"control_parent_rows":[4,5]}
        def build_clearance_rows(self,*a):
            return sparse.csr_matrix([[1.,0.]]),np.array([.25]),np.array([1.]),[{"kind":"face","typed_ids":[1,2,3]}]
        def _owner_matrix(self,pair_rows,area_rows,area_rhs,support):
            return sparse.csr_matrix([[1.,0.],[0.,1.]]),np.array([0.,0.])
    owner=Owner()
    out=tmp_path/"capture"
    np.savez(tmp_path/"base.npz",x=np.array([1]))
    c=cap.Capture(out,source_pins={},argv=[],base_candidate=tmp_path/"base.npz")
    c.wrap_owner(owner)
    basis=owner.build_immediate_parent_basis(np.zeros(2),np.array([4,5]),np.array([],dtype=int))
    C,h,w,meta=owner.build_clearance_rows()
    A,b=owner._owner_matrix([{"kind":"self"}],sparse.csr_matrix((0,2)),np.array([]),np.array([4,5]))
    m=types.ModuleType(cap.COMPONENT_MODULE)
    def solve(original,A,b,t,lo,hi,C,h,w,M,tlim,**kw):
        assert np.array_equal(A.toarray(),[[1,0],[0,1]])
        return {"success":True,"status":0,"delta":np.array([0.,0.])}
    m.solve_by_components=solve
    c.wrap_component_solver(m)
    result=m.solve_by_components(lambda:None,A,b,np.array([0.,0.]),np.array([-1.,-1.]),np.array([1.,1.]),C,h,w,sparse.eye(2,format="csr"),1.)
    assert result["success"] is True
    report=json.loads((out/"capture.json").read_text())
    assert report["dimensions"]["hard_rows"]==2
    assert report["metadata_coverage"]["basis"] is True
    assert report["metadata_coverage"]["clearance_rows"] is True
    assert report["metadata_coverage"]["owner_matrix"] is True
    assert json.loads((out/"solver-outcome.json").read_text())["outcome"]["success"] is True
    A2=sparse.load_npz(out/"hard_A.npz")
    assert np.array_equal(A2.toarray(),A.toarray())

def test_capture_does_not_create_output_until_solver(tmp_path):
    base=tmp_path/"base.npz";np.savez(base,x=np.array([1]))
    out=tmp_path/"capture"
    c=cap.Capture(out,source_pins={},argv=[],base_candidate=base)
    m=types.ModuleType(cap.COMPONENT_MODULE)
    m.solve_by_components=lambda *a,**k:{"success":True}
    c.wrap_component_solver(m)
    assert not out.exists()

def test_capture_rejects_dimension_mismatch_before_solver(tmp_path):
    base=tmp_path/"base.npz";np.savez(base,x=np.array([1]))
    c=cap.Capture(tmp_path/"capture",source_pins={},argv=[],base_candidate=base)
    m=types.ModuleType(cap.COMPONENT_MODULE);m.solve_by_components=lambda *a,**k:{"success":True}
    c.wrap_component_solver(m)
    A=sparse.csr_matrix([[1.,0.,0.]])
    with np.testing.assert_raises(ValueError):
        m.solve_by_components(None,A,np.array([0.]),np.zeros(2),-np.ones(2),np.ones(2),
           sparse.csr_matrix((0,2)),np.array([]),np.array([]),sparse.eye(2),1.)
    assert not c.output_dir.exists()

def test_hook_wraps_original_install_first(tmp_path):
    base=tmp_path/"base.npz";np.savez(base,x=np.array([1]))
    c=cap.Capture(tmp_path/"capture",source_pins={},argv=[],base_candidate=base)
    m=types.ModuleType(cap.COMPONENT_MODULE);m.solve_by_components=lambda *a,**k:{"success":True}
    order=[]
    def original(owner,obj,cross):order.append("original");return 17
    wrapped=cap.install_capture_after_axis_adapter(original,c,module_lookup=lambda:m)
    owner=types.SimpleNamespace(build_immediate_parent_basis=lambda *a:None,
        build_clearance_rows=lambda *a:None,_owner_matrix=lambda *a:None)
    assert wrapped(owner,None,None)==17
    assert order==["original"]
    assert c._installed and getattr(m,"_capture639_wrapped",False)
