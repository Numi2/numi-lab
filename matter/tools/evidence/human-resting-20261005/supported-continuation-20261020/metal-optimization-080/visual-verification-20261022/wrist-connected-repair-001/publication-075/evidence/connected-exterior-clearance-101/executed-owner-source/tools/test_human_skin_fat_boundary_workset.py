import numpy as np
import pytest
from scipy import sparse
from human_skin_fat_boundary_workset import select_boundary_witness_workset

FACES=np.array([[0,1,2],[1,3,2],[4,5,6]],dtype=np.int64)
def matrix(rows,n=10):
    r=[];c=[];v=[]
    for k,nodes in enumerate(rows):
        for node in nodes:r.append(k);c.append(3*node);v.append(1.)
    return sparse.coo_matrix((v,(r,c)),shape=(len(rows),3*n)).tocsr()
def call(*,edges=None,inside=None,rows=((0,7),(3,8)),rings=0):
    return select_boundary_witness_workset(10,np.arange(10,dtype=np.int64),FACES,
        np.empty((0,2),dtype=np.int64) if edges is None else np.array(edges,dtype=np.int64).reshape(-1,2),
        np.empty(0,dtype=np.int64) if inside is None else np.array(inside,dtype=np.int64),
        matrix(rows),patch_rings=rings)

def test_mixed_layer_contacts_do_not_move_clear_outer_skin():
    free,r=call()
    assert free.tolist()==[7,8,9]
    assert r['anchored_proposed_outer_nodes']==7
    assert r['unaddressed_contact_row_ids']==[]
    assert r['runtime_constraints_added'] is False

def test_actual_boundary_edge_seeds_connected_patch_not_unrelated_component():
    free,r=call(edges=[[0,1]],rings=1)
    assert free.tolist()==[0,1,2,3,7,8,9]
    assert r['crossing_boundary_edge_witnesses']==1
    assert r['outer_patch_nodes']==4

def test_inside_outer_node_and_all_outer_rows_are_not_anchored():
    free,r=call(inside=[4],rows=((0,3),(4,7)))
    assert free.tolist()==[0,3,4,7,8,9]
    assert r['all_outer_contact_rows']==1

def test_nonboundary_tet_edge_is_not_misclassified_as_boundary():
    free,r=call(edges=[[0,3],[0,7]])
    assert free.tolist()==[7,8,9]
    assert r['crossing_boundary_edge_witnesses']==0

def test_generated_outer_vertex_is_classified_by_faces_not_source_row_offset():
    faces=np.array([[0,1,8]],dtype=np.int64)
    free,r=select_boundary_witness_workset(10,np.arange(10,dtype=np.int64),
        faces,np.array([[1,8]],dtype=np.int64),np.empty(0,dtype=np.int64),matrix(((8,7),)),patch_rings=0)
    assert 8 in free and 0 not in free

@pytest.mark.parametrize('edges',[np.array([[0,10]]),np.array([[0.,1.]]),np.array([0,1])])
def test_invalid_edge_domain_fails_closed(edges):
    with pytest.raises(ValueError,match='crossing edge'):
        select_boundary_witness_workset(10,np.arange(10),FACES,edges,
            np.empty(0,dtype=np.int64),matrix(((0,7),)))

def test_unaddressed_contact_rows_fail_instead_of_silent_fixed_contacts():
    with pytest.raises(ValueError,match='without any free node'):
        select_boundary_witness_workset(10,np.array([0,1,9]),FACES,
            np.empty((0,2),dtype=np.int64),np.empty(0,dtype=np.int64),matrix(((0,7),)),patch_rings=0)
