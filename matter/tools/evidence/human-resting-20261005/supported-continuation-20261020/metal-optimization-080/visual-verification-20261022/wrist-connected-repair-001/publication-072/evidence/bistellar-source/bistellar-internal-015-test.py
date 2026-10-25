import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).with_name("bistellar_connected_015.py")
spec=importlib.util.spec_from_file_location("bistellar015",P)
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


def test_reference_oriented_replacement_does_not_use_current_pose():
    X,e,cyc,_=octahedron()
    new=m.orient_ref(m.raw_new_tets("4to4",e,cyc,0),X)
    current=X.copy();current[1,2]=2.0
    assert np.all(m.dets(X,new)>0)
    assert np.any(m.dets(current,new)<=0)
