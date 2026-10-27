"""Exercise the component line search through the actual unchanged 408 F32 gates."""
from pathlib import Path
import importlib.util,numpy as np
from scipy import sparse
N=Path(__file__).resolve().parent.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_actual_408_area_and_gradient_gates_do_not_throttle_other_patch():
 helper=load("component603_actual",N/"component-scp-area-backtrack-603/component_scp_backtracking603.py")
 owner=load("owner408_actual604",N.parent/"surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
 x=np.array([[0,0,0],[.001,0,0],[0,.001,0],[.01,0,0],[.011,0,0],[.01,.001,0]],dtype=np.float32)
 faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64);rows=np.arange(6);mm=x.astype(float)*1000
 target=np.zeros((6,3));target[2,1]=-1;target[3:,2]=.4
 C=sparse.csr_matrix((0,18));EG,rhs,_=owner.build_edge_gradient_constraints(mm,faces,rows,.5)
 def gate(current,goal):
  return owner.backtrack_area_feasible_scp_delta(x,current,mm,mm,faces,rows,np.empty(0,dtype=np.int64),0,goal,np.full(18,-4.),np.full(18,4.),C,x.astype(float),4.,12.,.1,EG,rhs,defer_clearance_proxy=True)
 uniform,_,u=gate(x,target.ravel())
 assert u["accepted_alpha"]==.5
 result,delta,events,count=helper.componentwise_line_search(x,np.zeros((6,3)),target,rows,[np.array([0,1,2]),np.array([3,4,5])],gate)
 assert count==2 and [e["selected_alpha"]for e in events]==[.5,1.]
 assert np.all(result[3:,2]==np.float32(.0004))
 assert np.all(uniform[3:,2]==np.float32(.0002))
 assert owner.full_f32_area_gate(result,mm,mm,faces,.1)["pass"]
 assert np.min(EG@delta-rhs)>=-1e-7
 assert np.max(np.linalg.norm(result.astype(float)-x,axis=1)*1000)<4
 print("actual408 gates: globalalpha.5, componentalphas[.5,1], fullarea/edge/trustpass")
if __name__=="__main__":test_actual_408_area_and_gradient_gates_do_not_throttle_other_patch()
