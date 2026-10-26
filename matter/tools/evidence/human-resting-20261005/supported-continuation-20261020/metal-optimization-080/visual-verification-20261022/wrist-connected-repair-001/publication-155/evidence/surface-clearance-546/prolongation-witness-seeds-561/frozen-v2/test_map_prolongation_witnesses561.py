from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('m561',Path(__file__).with_name('map_prolongation_witnesses561.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def mesh(order=None):
 n=6;order=np.arange(n)if order is None else np.array(order)
 inv=np.argsort(order)
 return {'source_prolongation_indptr':np.arange(n+1,dtype=np.int64),'source_prolongation_node_rows':order.astype(np.int64),'source_prolongation_weights':np.ones(n,np.float64),'source_O_faces':inv[np.array([[0,1,2],[3,4,5]])]}
def test_permuted_nodes_match_exactly():
 a=mesh();b=mesh([5,3,1,4,0,2]);r=m.map_exact(a,b,np.array([[0,1]]));assert r['mapped_pairs'].tolist()==[[0,1]]
def test_ambiguous_node_not_guessed():
 a=mesh();b=mesh();b['source_prolongation_node_rows'][5]=4
 r=m.map_exact(a,b,np.array([[0,1]]));assert len(r['mapped_pairs'])==0;assert r['unmapped_source_indices'].tolist()==[0]
def test_missing_face_not_guessed():
 a=mesh();b=mesh();b['source_O_faces'][1]=[1,3,5]
 assert len(m.map_exact(a,b,np.array([[0,1]]))['mapped_pairs'])==0
def test_ambiguous_source_face_not_guessed():
 a=mesh();b=mesh();a['source_O_faces']=np.vstack([a['source_O_faces'],a['source_O_faces'][0]])
 assert len(m.map_exact(a,b,np.array([[0,1]]))['mapped_pairs'])==0
def test_empty_pairs():
 assert m.map_exact(mesh(),mesh(),np.empty((0,2),np.int64))['mapped_pairs'].shape==(0,2)
def test_invalid_pair_fails():
 with pytest.raises(ValueError,match='pairs'):m.map_exact(mesh(),mesh(),np.array([[0,0]]))
def test_nonaffine_csr_fails():
 a=mesh();a['source_prolongation_weights'][0]=.9
 with pytest.raises(ValueError,match='nonaffine'):m.map_exact(a,mesh(),np.array([[0,1]]))
def test_different_domain_fails():
 a=mesh();b=mesh()
 with pytest.raises(ValueError,match='source domain'):m.validate_common_domain(a,b)
