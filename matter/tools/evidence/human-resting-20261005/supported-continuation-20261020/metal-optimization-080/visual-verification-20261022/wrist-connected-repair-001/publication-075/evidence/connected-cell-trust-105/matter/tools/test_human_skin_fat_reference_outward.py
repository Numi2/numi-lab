import numpy as np
import pytest
import human_skin_fat_bone_clearance as owner

def fixture():
    # Two source columns, three layers; identity labels deliberately nonlocal.
    r=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,-.1],[1.,0.,-.1],[0.,0.,-.3],[1.,0.,-.4],[.5,0.,-.2]])
    rows=np.arange(7,dtype=np.int64);rows[-1]=-1
    kw=dict(source_global_node_ids=np.array([800,91,700,302,421,9]),
            output_source_input_row=rows,
            native_parent_tetrahedra=np.array([[0,1,4,5]],dtype=np.int64),
            generated_output_rows=np.array([6],dtype=np.int64),
            generated_parent_tet_rows=np.array([0],dtype=np.int64),
            generated_parent_barycentric=np.array([[0.,.5,.5,0.]]),column_count=2)
    r[6]=kw['generated_parent_barycentric'][0]@r[kw['native_parent_tetrahedra'][0]]
    return r,kw

def test_reference_frame_rigid_transform_equivariance():
    r,kw=fixture();f,_=owner.build_reference_outward_field(r,**kw)
    angle=.73;rot=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    moved=r@rot.T+np.array([2.,-3.,1.])
    g,report=owner.build_reference_outward_field(moved,**kw)
    np.testing.assert_allclose(g,f@rot.T,atol=2e-14,rtol=0)
    assert report['generated_reference_ancestry_max_residual_m']<1e-14
    assert not report['legacy_layer_coordinate_arrays_used']

def test_reference_output_permutation_preserves_source_domain():
    r,kw=fixture();f,_=owner.build_reference_outward_field(r,**kw)
    p=np.array([3,6,1,5,0,4,2]);inv=np.argsort(p)
    kw['output_source_input_row']=kw['output_source_input_row'][p]
    kw['generated_output_rows']=inv[kw['generated_output_rows']]
    g,_=owner.build_reference_outward_field(r[p],**kw)
    np.testing.assert_allclose(g,f[p],atol=0,rtol=0)

def test_wrong_generated_reference_and_duplicate_rows_fail_closed():
    r,kw=fixture();r[-1,1]=1e-4
    with pytest.raises(owner.CertificationError,match="disagrees with declared ancestry"):
        owner.build_reference_outward_field(r,**kw)
    r,kw=fixture();kw['output_source_input_row'][1]=0
    with pytest.raises(owner.CertificationError,match="exactly once"):
        owner.build_reference_outward_field(r,**kw)

def test_degenerate_reference_column_rejects():
    r,kw=fixture();r[4]=r[0]
    with pytest.raises(owner.CertificationError,match="zero or invalid"):
        owner.build_reference_outward_field(r,**kw)
