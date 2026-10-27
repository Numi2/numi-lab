import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse

P=Path(__file__).with_name("component_scp_backtracking603.py")
s=importlib.util.spec_from_file_location("scp603",P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_owner_audit_accepts_local_rows_and_rejects_cross_component():
 labels=np.array([0,0,1,1])
 local=sparse.csr_matrix(([1.,-1.],([0,1],[0,3])),shape=(2,12))
 assert m.row_owner_audit(local,labels,"A")["single_owner_rows"]==2
 crossing=sparse.csr_matrix(([1.,1.],([0,0],[0,6])),shape=(1,12))
 try:m.row_owner_audit(crossing,labels,"A")
 except ValueError as e:assert "crosses components" in str(e)
 else:raise AssertionError("cross-component row must reject")

def test_one_hot_map_accepts_identity_and_rejects_mixed_parent():
 M=sparse.kron(sparse.eye(2,format="csr"),sparse.eye(3,format="csr"),format="csr")
 assert m.one_hot_map(M,2,np.array([0,1])).shape==(2,2)
 Q=sparse.csr_matrix([[.5,.5],[0,1.]])
 try:m.one_hot_map(sparse.kron(Q,sparse.eye(3,format="csr"),format="csr"),2,np.array([0,1]))
 except ValueError:pass
 else:raise AssertionError("mixed parent map must reject")

def test_independent_regions_keep_distinct_408_alphas():
 base=np.zeros((4,3),dtype=np.float32);support=np.arange(4)
 groups=[np.array([0,1]),np.array([2,3])]
 current_delta=np.zeros((4,3));target=np.ones((4,3))
 def gate(points,proposal):
  proposal=np.asarray(proposal).reshape(4,3)
  old=points.astype(float)*1000
  changed=np.any(proposal!=old,axis=1);comp0=bool(changed[:2].any())
  for alpha in (1.,.5,.25,.125):
   out=points.copy();ix=np.flatnonzero(changed)
   out[ix]=((old[ix]+alpha*(proposal[ix]-old[ix]))/1000).astype(np.float32)
   if not comp0 or alpha<=.25:
    return out,(out.astype(float)*1000).ravel(),{"accepted_alpha":alpha,"trials":[{"alpha":alpha,"accepted":True}]}
  return None,None,{"status":"NO_AREA_FEASIBLE_STEP","trials":[{"alpha":a,"accepted":False} for a in (1.,.5,.25,.125)]}
 out,d,events,n=m.componentwise_line_search(base,current_delta,target,support,groups,gate)
 assert n==2 and [e["selected_alpha"] for e in events]==[.25,1.]
 assert np.all(out[:2]==np.float32(.00025)) and np.all(out[2:]==np.float32(.001))

def test_failed_component_stays_at_current_while_other_advances():
 base=np.zeros((2,3),dtype=np.float32);groups=[np.array([0]),np.array([1])]
 def gate(points,proposal):
  proposal=np.asarray(proposal).reshape(2,3);old=points.astype(float)*1000
  changed=np.any(proposal!=old,axis=1)
  if changed[0]:return None,None,{"status":"NO_AREA_FEASIBLE_STEP","trials":[{"alpha":1.,"accepted":False}]}
  out=points.copy();out[1]=(proposal[1]/1000).astype(np.float32)
  return out,(out.astype(float)*1000).ravel(),{"accepted_alpha":1.,"trials":[{"alpha":1.,"accepted":True}]}
 out,d,events,n=m.componentwise_line_search(base,np.zeros((2,3)),np.ones((2,3)),np.array([0,1]),groups,gate)
 assert n==1 and not events[0]["accepted"] and events[1]["accepted"]
 assert np.array_equal(out[0],base[0]) and np.all(out[1]==.001)



def test_component_search_matches_uniform_408_when_alphas_match():
 base=np.zeros((2,3),dtype=np.float32);support=np.array([0,1]);groups=[np.array([0]),np.array([1])]
 def gate(points,proposal):
  proposal=np.asarray(proposal).reshape(2,3);old=points.astype(float)*1000
  ix=np.flatnonzero(np.any(proposal!=old,axis=1));out=points.copy()
  out[ix]=((old[ix]+.5*(proposal[ix]-old[ix]))/1000).astype(np.float32)
  return out,(out.astype(float)*1000).ravel(),{"accepted_alpha":.5,"trials":[{"alpha":.5,"accepted":True}]}
 out,delta,events,n=m.componentwise_line_search(base,np.zeros((2,3)),np.ones((2,3)),support,groups,gate)
 expected=np.full((2,3),.0005,dtype=np.float32)
 assert n==2 and [x["selected_alpha"] for x in events]==[.5,.5]
 assert np.array_equal(out,expected)

def test_603_entry_pins_564_parent_and_installs_only_new_hook():
 from pathlib import Path
 import hashlib
 d=Path(__file__).parent; old=d.parent/"partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"
 entry=d/"frozen-v1/run_joint_full_coordinate_fit603.py"
 def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 assert digest(old)=="bce28ca873ba9fdc75427e16443fdf047587742cfbef1f83e48a31df28caa4d9"
 source=entry.read_text()
 assert "component_scp_backtracking603" in source and "_component_scp603.install(owner" in source
 assert "EXPECTED_COMPONENT_SCP_SHA" in source and "COMPONENT_SCP_HELPER" in source
 assert digest(d/"frozen-v1/component_scp_backtracking603.py")=="f02aed5e462779e85a5d4df46a38a68bcf7eb3972c9dbf565830c4919b387be6"
