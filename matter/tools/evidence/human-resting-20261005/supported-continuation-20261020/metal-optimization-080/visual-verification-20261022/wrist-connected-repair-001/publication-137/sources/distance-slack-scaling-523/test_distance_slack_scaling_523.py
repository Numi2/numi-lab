from pathlib import Path
import importlib.util,sys
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
OWNER=D/'surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
SOLVER470=D/'D/soft-target-nonregression-470/soft_target_nonregression_470.py'
HELPER523=D/'D/distance-slack-scaling-523/distance_slack_scaling_523.py'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
owner=load('owner408_distance_slack_523_test',OWNER)
solver=load('soft_solver_470_distance_slack_523_test',SOLVER470)
scale=load('distance_slack_scaling_523_test',HELPER523)

def args(h):
    # Two output rows share one reduced control; the first C row cancels
    # exactly after projection, while the latter pair is linearly dependent.
    M=sparse.csr_matrix([[1.,0.],[1.,0.],[0.,1.]])
    C=sparse.csr_matrix([[1.,-1.,0.],[1.,0.,0.],[2.,0.,0.]])
    lo=np.array([.08,.08,0.]); hi=lo.copy()
    return (sparse.csr_matrix((0,3)),np.zeros(0),lo.copy(),lo,hi,C,
            np.asarray(h,dtype=np.float64),np.array([.2,.4,.4]),M,5.,
            1e-9,False,'highs',1.0)

def test_equivalent_solver_with_dependent_rows_and_cancellation():
    raw=solver.make_solver(owner)(*args([1e-6,.1,.2]))
    transformed=scale.make_solver(owner,solver)(*args([1e-6,.1,.2]))
    assert raw['success'],raw
    assert transformed['success'],transformed
    np.testing.assert_allclose(transformed['stage1']['objective'],raw['stage1']['objective'],rtol=0,atol=2e-8)
    np.testing.assert_allclose(transformed['normalized_slack_objective'],raw['normalized_slack_objective'],rtol=0,atol=2e-8)
    np.testing.assert_allclose(transformed['delta'],raw['delta'],rtol=0,atol=2e-7)
    np.testing.assert_allclose(transformed['normalized_slack'],raw['normalized_slack'],rtol=0,atol=2e-7)
    assert transformed['stage1_output_certificate']['passed']
    assert transformed['distance_slack_scaling_523']['stage1_and_stage2_budget_algebraically_equivalent']
    assert transformed['distance_slack_scaling_523']['minimum_lp_slack_scale']==1e-6

def test_tiny_h_linprog_variable_map_preserves_primal_and_objective():
    from types import SimpleNamespace
    h=np.array([1e-14,1e-9]); seen={}
    ctx=__import__('contextvars').ContextVar('test_h',default=h)
    def fake(c,A_ub=None,b_ub=None,A_eq=None,b_eq=None,bounds=None,method='highs',callback=None,options=None,x0=None,integrality=None):
        seen['c']=np.asarray(c); seen['A']=A_ub.toarray(); seen['b']=np.asarray(b_ub); seen['bounds']=bounds
        # Return the exact feasible distance slack t=h. The wrapper must
        # restore the frozen 470 normalized-slack contract s=t/h.
        return SimpleNamespace(x=h.copy(),fun=1.0,success=True)
    scaled_lp=scale._scaled_linprog(fake,ctx)
    res=scaled_lp(np.array([.4,.6]),A_ub=sparse.diags(-h),b_ub=-h,
                  bounds=[(0,None),(0,None)],method='highs')
    np.testing.assert_allclose(seen['A'].toarray() if sparse.issparse(seen['A']) else seen['A'],-np.eye(2),rtol=0,atol=2e-16)
    np.testing.assert_allclose(seen['b'],-h,rtol=0,atol=0)
    np.testing.assert_allclose(seen['c'],np.array([.4,.6])/h,rtol=1e-15,atol=0)
    np.testing.assert_allclose(res.x,np.ones(2),rtol=0,atol=2e-7)
    assert abs(res.fun-1.0)<2e-7

def test_bounds_and_matrix_scaling_are_exact_for_small_direct_lp():
    h=np.array([.25,2.])
    seen={}
    class Result:
        x=np.array([.25,2.])
        fun=1.0
        success=True
    def fake(c, A_ub=None,b_ub=None,bounds=None,**kwargs):
        seen['c']=np.asarray(c);seen['A']=A_ub.toarray();seen['bounds']=bounds
        return Result()
    ctx=__import__('contextvars').ContextVar('test_h2',default=h)
    wrapped=scale._scaled_linprog(fake,ctx)
    out=wrapped(np.array([2.,4.]),A_ub=sparse.csr_matrix([[3.,5.]]),b_ub=np.array([1.]),
                bounds=[(0,None),(0,3.)])
    np.testing.assert_allclose(seen['c'],[8.,2.])
    np.testing.assert_allclose(seen['A'],[[12.,2.5]])
    assert seen['bounds']==[(0.0,None),(0.0,6.0)]
    np.testing.assert_allclose(out.x,[1.,1.])

def test_runtime_install_hook_scales_installed_470_before_outer_component_wrap():
    previous_solver=owner.solve_sparse_l1_clearance_in_basis
    previous_installer=scale.patch_install_solver(solver)
    try:
        receipt=solver.install_solver(owner)
        assert receipt['distance_slack_scaling_523']['parameterization']=='t=h_map*s, where h_map is 470 row-normalized h'
        installed=owner.solve_sparse_l1_clearance_in_basis
        assert getattr(installed,'__distance_slack_scaling_523__',False)
        result=installed(*args([1e-6,.1,.2]))
        assert result['success'],result
        assert result['stage1_output_certificate']['passed']
        assert result['distance_slack_scaling_523']['soft_target_row_count']==3
    finally:
        solver.install_solver=previous_installer
        owner.solve_sparse_l1_clearance_in_basis=previous_solver

if __name__=='__main__':
    test_equivalent_solver_with_dependent_rows_and_cancellation()
    test_tiny_h_linprog_variable_map_preserves_primal_and_objective()
    test_bounds_and_matrix_scaling_are_exact_for_small_direct_lp()
    test_runtime_install_hook_scales_installed_470_before_outer_component_wrap()
    print('4 distance-slack scaling 523 tests passed')
