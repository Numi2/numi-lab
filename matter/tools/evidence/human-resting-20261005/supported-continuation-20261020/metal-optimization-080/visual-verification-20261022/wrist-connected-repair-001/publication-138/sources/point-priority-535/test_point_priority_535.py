from pathlib import Path
import importlib.util, sys
import numpy as np
from scipy import sparse

D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
OWNER=D/"surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
SOFT=D/"D/soft-target-nonregression-470/soft_target_nonregression_470.py"
DECOMP=D/"D/component-lp-decomposition-485/component_lp_decomposition_485.py"
HELPER=Path(__file__).with_name("point_priority_535.py")

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

owner=load("owner408_priority535_test",OWNER)
soft=load("soft470_priority535_test",SOFT)
helper=load("point_priority_535_test",HELPER)
base_solver=soft.make_solver(owner)

def toy():
    # x+y <= 1. The pooled weighted objective prefers y; priority C[0] must
    # first maximize x, then retain 95% of its feasible optimum.
    A=sparse.csr_matrix([[-1.,-1.]])
    b=np.array([-1.])
    C=sparse.eye(2,format="csr")
    h=np.ones(2); w=np.array([.1,.9]); M=sparse.eye(2,format="csr")
    target=np.zeros(2); lo=np.zeros(2); hi=np.ones(2)
    return A,b,target,lo,hi,C,h,w,M

def test_metadata_identifies_only_query_inside_point_prefix():
    metadata=[{"kind":"point","query_row":101},{"kind":"point","query_row":19226},
              {"kind":"face","face_row":9}]
    assert helper.identify_priority_point_row(metadata,2,19226)==1
    for bad in ((metadata,1,19226),(metadata,3,19226)):
        try: helper.identify_priority_point_row(*bad)
        except ValueError: pass
        else: raise AssertionError("invalid point prefix/query was accepted")

def test_priority_floor_preserves_original_lp_and_basis_contract():
    A,b,t,lo,hi,C,h,w,M=toy()
    baseline=base_solver(A,b,t,lo,hi,C,h,w,M,3.,retained_gain_fraction=1.)
    seen={}
    def recording_solver(*args,**kwargs):
        seen["args"]=args
        return base_solver(*args,**kwargs)
    result=helper.solve_with_point_priority(recording_solver,A,b,t,lo,hi,C,h,w,M,3.,
        point_prefix_count=1,priority_row_index=0,priority_c_functional=C.getrow(0),
        priority_query_row=19226,retention_fraction=.95,retained_gain_fraction=1.)
    assert baseline["success"] and result["success"],result
    assert baseline["delta"][0]<1e-6
    assert result["delta"][0]>=.95-2e-7
    call=seen["args"]; A2=sparse.csr_matrix(call[0]); b2=np.asarray(call[1])
    assert A2.shape==(A.shape[0]+1,A.shape[1])
    assert (A2[:A.shape[0]]-A).nnz==0 and np.array_equal(b2[:len(b)],b)
    assert (A2.getrow(A.shape[0])-C.getrow(0)).nnz==0
    assert abs(b2[-1]-.95)<2e-7
    for got,want in zip(call[2:9],(t,lo,hi,C,h,w,M)):
        if sparse.issparse(want): assert (sparse.csr_matrix(got)-want).nnz==0
        else: np.testing.assert_array_equal(got,want)
    r=result["point_priority_535"]
    assert r["status"]=="PRIORITY_AND_POOLED_CERTIFICATES_PASSED"
    assert r["original_A_b_preserved"] and r["bounds_C_h_weights_M_preserved"]
    assert r["basis_coordinates_preserved"] and r["native_admission"] is False
    assert np.min(A@result["delta"]-b)>=-2e-7
    np.testing.assert_allclose(M@result["basis_coordinates"],result["delta"],atol=2e-9)

def test_missing_original_stage1_certificate_fails_closed():
    A,b,t,lo,hi,C,h,w,M=toy()
    def uncertified_solver(*args,**kwargs):
        out=base_solver(*args,**kwargs)
        out.pop("stage1_output_certificate",None)
        return out
    result=helper.solve_with_point_priority(uncertified_solver,A,b,t,lo,hi,C,h,w,M,3.,
        point_prefix_count=1,priority_row_index=0,priority_c_functional=C.getrow(0),
        priority_query_row=19226)
    assert not result["success"]
    assert result["point_priority_535"]["status"]=="POOLED_CERTIFICATE_FAILED"

def test_component_hook_prioritizes_only_component_with_selected_C_row():
    A=sparse.csr_matrix([[-1.,-1.,0.,0.],[0.,0.,-1.,-1.]])
    b=np.array([-1.,-1.]); C=sparse.eye(4,format="csr"); h=np.ones(4)
    w=np.array([.1,.9,.9,.1]); M=sparse.eye(4,format="csr")
    t=np.zeros(4); lo=np.zeros(4); hi=np.ones(4)
    state={"capture":{"query_row":19226,"point_prefix_count":2,"priority_row_index":0,
        "priority_c_functional":C.getrow(0),"priority_target_mm":1.,
        "metadata":{"kind":"point","query_row":19226},"clearance_row_count":4}}
    decomp=load("decomposition485_priority535_test",DECOMP)
    receipt=helper.patch_component_decomposition(decomp,state)
    try:
        result=decomp.solve_by_components(base_solver,A,b,t,lo,hi,C,h,w,M,3.,
            retained_gain_fraction=1.,allow_soft_target_regression=True,max_total_wall_s=30.)
        assert result["success"],result
        assert result.get("success"), result
        assert "component_decomposition_475" in result, result
        assert result["component_decomposition_475"]["component_count"]==2
        assert result["delta"][0]>=.95-2e-7
        assert result["delta"][2]>.99
        rr=result["point_priority_535"]
        assert len(rr["component_receipts"])==1 and rr["priority_component"]==0
        assert rr["component_receipts"][0]["status"]=="PRIORITY_AND_POOLED_CERTIFICATES_PASSED"
        assert result["component_decomposition_475"]["global_original_space_certificate_pass"]
    finally:
        receipt["restore"]()

def test_owner_builder_metadata_capture_pins_19226_functional():
    class BaryToy:
        @staticmethod
        def build_barycentric_exit_system(points, faces, rows, bary, normals, targets, *, anchor_rows=()):
            tri=np.asarray(faces,dtype=np.int64)[rows]
            B=sparse.coo_matrix((np.asarray(bary).ravel(),
                (np.repeat(np.arange(len(rows)),3),tri.ravel())),shape=(len(rows),len(points))).tocsr()
            delta=np.asarray(targets)-np.asarray(B@points)
            lower=np.einsum("ij,ij->i",normals,delta)
            grad=np.asarray(bary)[:,:,None]*np.asarray(normals)[:,None,:]
            return B,tri,grad,lower,delta
    points=np.zeros((19227,3),dtype=np.float64)
    points[19224]=[0.,0.,0.]; points[19225]=[1.,0.,0.]; points[19226]=[0.,1.,0.]
    faces=np.array([[19224,19225,19226]],dtype=np.int64)
    pt={"point_rows":np.array([19226]),"point_targets_f32_m":np.array([[.001,.001,0.]],np.float32),
        "point_directions":np.array([[1.,0.,0.]])}
    sample=np.array([[1/3/1000,1/3/1000,0.]],dtype=np.float64)
    ft={"face_rows":np.array([0]),"barycentric_weights":np.array([[1/3,1/3,1/3]]),
        "face_targets_f32_m":np.array([[1/3/1000,1/3/1000,.001]],np.float32),
        "face_directions":np.array([[0.,0.,1.]]),"face_samples_m":sample,
        "face_npz":{"bank_body":np.array([114]),"bank_stable_bone_id":np.array([117])}}
    state={}
    hook=helper.patch_owner_clearance_builder(owner,state,19226)
    try:
        C,h,w,meta=owner.build_clearance_rows(points,faces,np.array([19224,19225,19226]),
            np.empty(0,dtype=np.int64),pt,ft,BaryToy)
        assert meta[0]["kind"]=="point" and meta[0]["query_row"]==19226
        assert state["capture"]["priority_row_index"]==0
        assert state["capture"]["point_prefix_count"]==1
        assert state["capture"]["priority_target_mm"]==h[0]
        assert (state["capture"]["priority_c_functional"]-C.getrow(0)).nnz==0
    finally:
        hook["restore"]()


if __name__=="__main__":
    test_owner_builder_metadata_capture_pins_19226_functional()
    test_metadata_identifies_only_query_inside_point_prefix()
    test_priority_floor_preserves_original_lp_and_basis_contract()
    test_component_hook_prioritizes_only_component_with_selected_C_row()
    test_missing_original_stage1_certificate_fails_closed()
    print("5 point-priority 535 tests passed")

