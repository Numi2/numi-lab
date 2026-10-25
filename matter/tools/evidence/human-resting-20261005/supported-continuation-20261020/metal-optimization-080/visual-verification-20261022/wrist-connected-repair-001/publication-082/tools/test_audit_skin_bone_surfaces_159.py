import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).with_name("audit_skin_bone_surfaces_159.py")
sp=importlib.util.spec_from_file_location("surface_audit_159",p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
@pytest.fixture(scope="module")
def exact(): return m.exact_module()
def run(P,F,B,exact): return m.intersect_bank(np.asarray(P,float),np.asarray(F,int),np.asarray(B,float),np.arange(len(B))+73,exact)
def test_separated_complete_aabb_exclusion(exact):
 r=run([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],[[[2,2,-1],[3,2,1],[2,3,1]]],exact)
 assert r["skin_faces"]==r["bank_faces"]==1 and r["skin_aabb_active"]==0 and r["result"] is None
def test_crossing_without_coincident_vertices(exact):
 r=run([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2]],[[[0,-.5,-1],[0,-.5,1],[0,.5,0]]],exact)
 assert r["result"]["triangle_pairs"]==[[0,73]]
def test_aabb_overlap_without_intersection(exact):
 r=run([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],[[[.8,.8,-1],[.9,.8,1],[.8,.9,1]]],exact)
 assert r["skin_aabb_active"]==1 and r["result"]["count"]==0
def test_face_rows_survive_culling(exact):
 r=run([[4,4,0],[5,4,0],[4,5,0],[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2],[3,4,5]],[[[0,-.5,-1],[0,-.5,1],[0,.5,0]]],exact)
 assert r["result"]["triangle_pairs"]==[[1,73]]
def test_no_second_float32_cast(exact):
 dz=2**-30
 P=np.array([[-1,-1,1+dz],[1,-1,1+dz],[0,1,1+dz]])
 B=np.array([[[-1,-1,1],[1,-1,1],[0,1,1]]])
 r=run(P,[[0,1,2]],B,exact);assert r["result"] is None
def test_degenerate_candidate_rejects(exact):
 with pytest.raises(Exception):
  run([[0,0,0],[1,0,0],[2,0,0]],[[0,1,2]],[[[0,0,-1],[0,0,1],[1,0,0]]],exact)
