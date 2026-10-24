import numpy as np
from human_skin_p1_level import extract_level_exact

def test_open_strict_triangle():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    r=extract_level_exact(V,T,[-1,1,1,1],0.0)
    assert r["faces"].shape==(1,3) and r["edge_hist"]=={1:3}
    boundary_edges=[tuple(sorted((int(a),int(b)))) for a,b,c in r["faces"] for a,b in ((a,b),(b,c),(c,a))]
    from collections import Counter
    degree=Counter(v for e in boundary_edges for v in e)
    assert degree==Counter({0:2,1:2,2:2})
    assert np.all(r["point_kind"]==0)

def test_strict_quad_uses_two_triangles_and_one_shared_diagonal():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    r=extract_level_exact(V,T,[-1,-1,1,1],0.0)
    assert r["faces"].shape==(2,3) and r["edge_hist"]=={1:4,2:1}
    assert np.all(r["point_kind"]==0)

def test_exact_vertex_and_strict_edges():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    r=extract_level_exact(V,T,[0,-1,1,1],0.0)
    assert len(r["points"])==3 and len(r["faces"])==1 and int(np.sum(r["point_kind"]==1))==1
    assert set(r["parent_node_id"][r["point_kind"]==1])=={0} and r["edge_hist"]=={1:3}

def test_exact_edge_plus_strict_crossing():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    r=extract_level_exact(V,T,[-1,0,0,1],0.0)
    assert len(r["points"])==3 and len(r["faces"])==1 and int(np.sum(r["point_kind"]==1))==2
    assert r["edge_hist"]=={1:3}

def test_exact_face_emitted_from_negative_tet_only():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,-1],[0,0,1]],np.float32);T=np.array([[0,1,2,3],[0,2,1,4]])
    r=extract_level_exact(V,T,[0,0,0,-1,1],0.0)
    assert r["faces"].shape==(1,3) and np.all(r["point_kind"]==1)
    assert r["parent_node_id"].tolist()==[0,1,2]

def test_opposite_exact_face_cancels_between_negative_tets():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,-1],[0,0,1]],np.float32);T=np.array([[0,1,2,3],[0,2,1,4]])
    r=extract_level_exact(V,T,[0,0,0,-1,-1],0.0,allow_empty=True)
    assert r["faces"].shape==(0,3) and r["cancelled_internal_exact_faces"]==1

def test_positive_exact_face_does_not_bound_sublevel():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    r=extract_level_exact(V,T,[0,0,0,1],0.0,allow_empty=True)
    assert r["faces"].shape==(0,3)

def test_empty_positive_and_fullnegative_tets_emit_nothing():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    for phi in ([1,1,1,1],[-1,-1,-1,-1]):
        r=extract_level_exact(V,T,phi,0.0,allow_empty=True)
        assert r["faces"].shape==(0,3)

def test_all_level_tet_fails_closed():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]])
    try: extract_level_exact(V,T,[0,0,0,0],0.0,allow_empty=True)
    except ValueError as e: assert "ambiguous plateau" in str(e)
    else: raise AssertionError("plateau accepted")

def test_closed_surface_and_shared_global_edge_ids():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.2,.2,.2]],np.float32)
    T=np.array([[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]])
    r=extract_level_exact(V,T,[1,1,1,1,-1],0.0)
    assert r["faces"].shape[0]==4 and r["edge_hist"]=={2:6}
    assert len(set(map(tuple,r["parent_edge"])))==len(r["points"])

def test_stable_float64_crossing_interpolation():
    V=np.array([[.1124855578,-1.051805377,.2020953447],[.1106144935,-1.051974535,.2004037797],[0,0,0],[0,0,1]],np.float32)
    T=np.array([[0,2,3,1]]);r=extract_level_exact(V,T,[-1,1,1,-1],0.0)
    assert np.isfinite(r["points"]).all() and np.isfinite(r["edge_t"]).all()

def test_noninteger_node_ids_rejected():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    try: extract_level_exact(V,[[0,1,2,3.2]],[-1,1,1,1],0.)
    except ValueError as e: assert "IDs must be integers" in str(e)
    else: raise AssertionError("noninteger source ancestry accepted")

def test_bad_area_floor_rejected():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    for floor in (-1.,np.nan):
        try: extract_level_exact(V,[[0,1,2,3]],[-1,1,1,1],0.,area2_floor=floor)
        except ValueError as e: assert "area floor" in str(e)
        else: raise AssertionError("invalid emission area policy accepted")
