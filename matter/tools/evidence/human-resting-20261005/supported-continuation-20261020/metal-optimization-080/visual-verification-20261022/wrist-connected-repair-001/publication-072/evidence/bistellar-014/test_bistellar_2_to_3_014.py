import importlib.util
from pathlib import Path
import numpy as np

P = Path(__file__).with_name("bistellar_2_to_3_014.py")
spec = importlib.util.spec_from_file_location("bistellar014", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def bipyramid():
    X = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                  [.2,.2,1.],[.2,.2,-1.]], dtype=np.float64)
    # Two cells sharing (0,1,2); orientation is normalized from reference.
    T = m.orient_reference_positive(np.array([[0,1,2,3],[0,2,1,4]]), X)
    face_inc = m.shared_face_entries(T, X)
    shared = [x for x in face_inc.items() if len(x[1]) == 2][0]
    return X, T, shared


def test_unit_bipyramid_has_positive_three_cell_flip_and_same_boundary():
    X, T, (face, inc) = bipyramid()
    (i, _, u, _), (j, _, v, _) = inc
    N = m.orient_reference_positive(m.two_to_three(face, u, v), X)
    assert np.all(m.dets(X, N) > 0)
    assert m.tet_boundary_chain(T[[i,j]], X) == m.tet_boundary_chain(N, X)
    assert m.line_crosses_face_interior(X, u, v, face)
    assert abs(np.sum(m.dets(X,N))-np.sum(m.dets(X,T[[i,j]]))) < 1e-14


def test_same_side_apices_are_not_a_convex_bipyramid():
    X = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                  [.2,.2,1.],[.2,.2,2.]], dtype=np.float64)
    assert not m.line_crosses_face_interior(X, 3, 4, (0,1,2))


def test_reference_orientation_does_not_use_current_geometry():
    X, T, (face, inc) = bipyramid()
    (i, _, u, _), (j, _, v, _) = inc
    candidate = m.orient_reference_positive(m.two_to_three(face,u,v), X)
    current = X.copy()
    current[4,2] = 1.0  # current geometry collapses a new tet; orientation remains reference-owned
    assert np.all(m.dets(X,candidate) > 0)
    assert np.any(m.dets(current,candidate) <= 0)


def test_face_incidence_uses_opposite_oriented_reference_sides():
    X, T, (face, inc) = bipyramid()
    assert len(inc) == 2
    assert inc[0][3] == -inc[1][3]
    (i, _, u, _), (j, _, v, _) = inc
    N = m.orient_reference_positive(m.two_to_three(face,u,v), X)
    assert m.tet_boundary_chain(T[[i,j]],X) == m.tet_boundary_chain(N,X)
