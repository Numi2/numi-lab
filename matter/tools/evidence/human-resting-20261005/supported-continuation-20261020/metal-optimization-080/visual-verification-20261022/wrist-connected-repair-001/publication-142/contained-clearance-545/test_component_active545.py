import numpy as np
from scipy import sparse
from component_lp_active545 import solve_by_components

def test_zero_goal_components_are_certified_without_solver_calls():
 def bad(*args,**kw):raise AssertionError('empty component must not invoke LP')
 r=solve_by_components(bad,sparse.eye(3),-np.ones(3),np.zeros(3),-np.ones(3),np.ones(3),sparse.csr_matrix((0,3)),np.empty(0),np.empty(0),sparse.eye(3),10.)
 assert r['success'] and np.array_equal(r['delta'],np.zeros(3))
 assert r['component_decomposition_475']['analytic_zero_component_count']==3
 assert r['component_decomposition_475']['global_original_space_certificate_pass']

def test_nonzero_l1_goal_is_not_skipped():
 called=[]
 def solve(A,b,t,lo,hi,C,h,w,M,seconds,**kw):
  called.append((t.copy(),seconds))
  return dict(success=True,status=0,nit=0,delta=t.copy(),basis_coordinates=t.copy(),normalized_slack=np.empty(0),stage1={'objective':0.},wall_seconds=0.)
 r=solve_by_components(solve,sparse.eye(3),-np.ones(3),np.array([.1,0,0]),-np.ones(3),np.ones(3),sparse.csr_matrix((0,3)),np.empty(0),np.empty(0),sparse.eye(3),10.)
 assert r['success'] and len(called)==1
 assert np.allclose(r['delta'],[.1,0,0])
 assert r['component_decomposition_475']['analytic_zero_component_count']==2
