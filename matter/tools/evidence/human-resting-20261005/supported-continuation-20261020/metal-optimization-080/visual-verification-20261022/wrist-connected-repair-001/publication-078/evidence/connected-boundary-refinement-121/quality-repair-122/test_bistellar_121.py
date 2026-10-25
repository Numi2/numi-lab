import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).with_name("bistellar_connected_116.py")
spec=importlib.util.spec_from_file_location("bistellar116",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_three_to_two_unit_bipyramid_exact_boundary_and_positive_cells():
    X=np.array([[0.,0.,1.],[0.,0.,-1.],
                [1.,0.,0.],[-.5,.8660254038,0.],[-.5,-.8660254038,0.]])
    edge=(0,1); cycle=(2,3,4)
    old=m.orient_ref(np.array([[0,1,2,3],[0,1,3,4],[0,1,4,2]]),X)
    new=m.orient_ref(m.raw_new_tets("3to2",edge,cycle),X)
    assert np.all(m.dets(X,new)>0)
    assert m.boundary_chain(old,X)==m.boundary_chain(new,X)
    assert m.line_crosses_face(X,0,1,cycle)


def octahedron():
    X=np.array([[0.,0.,1.],[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],
                [-1.,0.,0.],[0.,-1.,0.]])
    edge=(0,1);cycle=(2,3,4,5)
    old=np.array([[0,1,2,3],[0,1,3,4],[0,1,4,5],[0,1,5,2]])
    return X,edge,cycle,m.orient_ref(old,X)


def test_four_to_four_both_alternate_diagonals_match_boundary():
    X,e,cyc,old=octahedron()
    for alt in (0,1):
        new=m.orient_ref(m.raw_new_tets("4to4",e,cyc,alt),X)
        assert np.all(m.dets(X,new)>0)
        assert m.boundary_chain(old,X)==m.boundary_chain(new,X)


def test_four_to_four_diagonals_cross_in_reference():
    X,e,cyc,_=octahedron()
    assert m.segment_intersection_interior(X,0,1,2,4)
    assert m.segment_intersection_interior(X,0,1,3,5)



def test_four_to_four_accepts_valid_skew_diagonals():
    # Perturbing an octahedron makes opposite diagonals skew, while both
    # tetrahedralizations remain positive and have the exact same boundary.
    X=np.array([
        [.11188165130823095,.11056499208399465,1.1932520011929264],
        [-.36137932176343673,.20526217261675625,-1.0207648441874573],
        [.45570967654014216,.08791668979038793,-.03377371667509913],
        [.00869682736292679,.4735990172452682,.11859110920961406],
        [-.9100111131568986,.3692851532494967,.09490075757623799],
        [-.16258623598495162,-.8189020224910302,-.007247301443526141],
    ])
    Xc=np.array([
        [.21816550195217133,.23374097049236298,1.1754405498504639],
        [-.3671160042285919,.412713885307312,-1.1108237504959106],
        [.5000909566879272,-.01097472757101059,-.11792658269405365],
        [-.10842201858758926,.29008978605270386,-.03591076284646988],
        [-.86627596616745,.359885662794113,.11766773462295532],
        [-.13685065507888794,-.8641992211341858,.028975369408726692],
    ])
    old_raw=np.array([[0,1,2,3],[0,1,3,4],[0,1,4,5],[0,1,5,2]])
    old=m.orient_ref(old_raw,X)
    new=m.orient_ref(m.raw_new_tets("4to4",(0,1),(2,3,4,5),0),X)
    assert np.all(m.dets(X,old)>0) and np.all(m.dets(X,new)>0)
    assert m.boundary_chain(old,X)==m.boundary_chain(new,X)
    assert not m.segment_intersection_interior(X,0,1,2,4)
    face_inc,edge_inc,cell_owners=m.make_topology(old,X)
    counts=m.boundary_face_counts(old)
    result,reason=m.eval_cavity(
        "4to4",range(4),(0,1),new,old,np.zeros(4,dtype=int),X,
        X.astype(np.float32),Xc.astype(np.float32),edge_inc,cell_owners,counts)
    assert reason is None and result is not None
    assert np.all(result["ref_det32"]>0) and np.all(result["cur_det32"]>0)

def test_bistellar_move_may_remove_old_internal_face():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [.2,.2,1.],[.2,.2,-1.]])
    old=m.orient_ref(np.array([[0,1,2,3],[0,2,1,4]]),X)
    inc=m.make_topology(old,X)[0]
    f,entries=[(f,e) for f,e in inc.items() if len(e)==2][0]
    i,_,u,_=entries[0];j,_,v,_=entries[1]
    new=m.orient_ref(m.raw_new_tets("2to3",f,(u,v)),X)
    counts=m.boundary_face_counts(old)
    # The old shared face disappears; the cavity boundary still matches exactly.
    assert counts[f]==2
    assert m.boundary_chain(old,X)==m.boundary_chain(new,X)
    assert m.face_count_compatible((i,j),new,old,counts)


def test_reference_oriented_replacement_does_not_use_current_pose():
    X,e,cyc,_=octahedron()
    new=m.orient_ref(m.raw_new_tets("4to4",e,cyc,0),X)
    current=X.copy();current[1,2]=2.0
    assert np.all(m.dets(X,new)>0)
    assert np.any(m.dets(current,new)<=0)


def test_canonical_owner_q_is_det_over_longest_edge_cubed():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    T=np.array([[0,1,2,3]],dtype=np.int64)
    _,_,q,_=m.repair.cell_values_gradients(X,X,T)
    assert abs(float(q[0])-1.0/(2.0**1.5)) < 1e-14
