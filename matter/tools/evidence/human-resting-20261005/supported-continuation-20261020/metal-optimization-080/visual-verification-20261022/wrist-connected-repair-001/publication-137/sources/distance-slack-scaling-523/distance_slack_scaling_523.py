"""Exact variable scaling for the frozen 470 normalized-clearance LP.

The LP's normalized slack s is replaced internally by distance slack t=h*s.
This is an algebraic variable substitution only: normalized slacks are restored
before the frozen 470 output-space certificates run, and the weighted objective
remains sum(w*s) = sum((w/h)*t).
"""
from __future__ import annotations
import copy
import hashlib
import inspect
from functools import partial
from pathlib import Path
from types import FunctionType
from contextvars import ContextVar
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds

EXPECTED_470_PATH = Path('/Users/numi/Work/numi-human-connected-surface-069/D/D/soft-target-nonregression-470/soft_target_nonregression_470.py')
EXPECTED_470_SHA256 = '990e126932186e33c0a5363a6ac34fa967e211f4bc55977567582e2b89b8bb79'
EXPECTED_OWNER_SHA256 = '5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2'

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def _right_scale(matrix, scales):
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        return matrix @ sparse.diags(scales,format='csc')
    arr=np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[1] != len(scales):
        raise ValueError('LP matrix column count differs from variable scaling')
    return arr * scales[None,:]

def _scaled_bounds(bounds, n, h):
    m=len(h)
    if bounds is None:
        return None
    if isinstance(bounds,Bounds):
        lo=np.broadcast_to(np.asarray(bounds.lb,dtype=np.float64),(n,)).copy()
        hi=np.broadcast_to(np.asarray(bounds.ub,dtype=np.float64),(n,)).copy()
        lo[-m:]=lo[-m:]*h
        hi[-m:]=hi[-m:]*h
        return Bounds(lo,hi,keep_feasible=bounds.keep_feasible)
    seq=list(bounds)
    if len(seq)==2 and not (len(seq)==n and all(isinstance(x,(tuple,list)) and len(x)==2 for x in seq)):
        # scipy's shared pair form, e.g. (0, None), applies to every variable.
        seq=[(seq[0],seq[1]) for _ in range(n)]
    if len(seq)!=n or any(not isinstance(pair,(tuple,list)) or len(pair)!=2 for pair in seq):
        raise ValueError('linprog bounds must be a pair or one pair per variable')
    out=[]
    for i,(lower,upper) in enumerate(seq):
        factor=float(h[i-(n-m)]) if i>=n-m else 1.0
        lo=None if lower is None else float(lower)*factor
        hi=None if upper is None else float(upper)*factor
        out.append((lo,hi))
    return out

def _scaled_linprog(original_linprog, h_context):
    signature=inspect.signature(original_linprog)
    def call(*args,**kwargs):
        h=h_context.get()
        if h is None or len(h)==0:
            return original_linprog(*args,**kwargs)
        bound=signature.bind_partial(*args,**kwargs)
        bound.apply_defaults()
        c=np.asarray(bound.arguments['c'],dtype=np.float64).reshape(-1)
        m=len(h); n=len(c)
        if m>n:
            raise ValueError('soft slack count exceeds LP variable count')
        if not np.isfinite(h).all() or np.any(h<=0):
            raise ValueError('distance slack scaling requires positive finite h')
        if not np.isfinite(c).all():
            raise ValueError('linprog objective contains nonfinite values')
        scale=np.ones(n,dtype=np.float64)
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            scale[-m:]=1.0/h
            c_scaled=c*scale
        if not np.isfinite(scale).all() or not np.isfinite(c_scaled).all():
            raise ValueError('distance slack scaling overflows the LP variable/objective')
        bound.arguments['c']=c_scaled
        for key in ('A_ub','A_eq'):
            mat=bound.arguments.get(key)
            if mat is not None:
                bound.arguments[key]=_right_scale(mat,scale)
        bound.arguments['bounds']=_scaled_bounds(bound.arguments.get('bounds'),n,h)
        # x_new stores distance slack t. Return the original solver contract
        # with normalized slack s=t/h so 470 certificates remain unchanged.
        solved=original_linprog(**bound.arguments)
        x=getattr(solved,'x',None)
        if x is None:
            return solved
        x=np.asarray(x,dtype=np.float64).reshape(-1)
        if x.shape!=(n,):
            raise ValueError('transformed linprog returned an unexpected primal shape')
        normalized=x.copy()
        normalized[-m:]=normalized[-m:]/h
        out=copy.copy(solved)
        out.x=normalized
        return out
    return call

def _effective_slack_scale_h(arguments):
    """Mirror 470's row normalization: its slack coefficient is h_map."""
    C=sparse.csr_matrix(arguments['C'],dtype=np.float64)
    M=sparse.csr_matrix(arguments['output_map'],dtype=np.float64)
    h=np.asarray(arguments['h'],dtype=np.float64).reshape(-1)
    cm=(C@M).tocsr()
    norms=np.sqrt(np.asarray(cm.multiply(cm).sum(axis=1)).reshape(-1))
    if norms.shape!=h.shape:
        raise ValueError('523 could not align reduced clearance rows and slack scales')
    scale=np.ones(len(h),dtype=np.float64)
    nz=norms>0.0
    scale[nz]=1.0/norms[nz]
    h_map=h*scale
    if not np.isfinite(h_map).all() or np.any(h_map<=0):
        raise ValueError('523 derived invalid normalized LP slack scales')
    return h_map

def make_solver(owner, solver_module):
    """Clone only the frozen solver function's globals to scale its linprog call."""
    path=Path(getattr(solver_module,'__file__','')).resolve()
    if path!=EXPECTED_470_PATH.resolve() or sha256_file(path)!=EXPECTED_470_SHA256:
        raise ValueError('523 refuses an unexpected or changed frozen 470 module')
    owner_path=Path(getattr(owner,'__file__','')).resolve()
    if owner_path!=solver_module.EXPECTED_OWNER_PATH.resolve() or sha256_file(owner_path)!=EXPECTED_OWNER_SHA256:
        raise ValueError('523 refuses an unexpected or changed frozen 408 owner')
    source=solver_module._solve_sparse_l1_clearance_in_basis_without_soft_noregression
    h_context=ContextVar('distance_slack_h_523',default=None)
    namespace=dict(source.__globals__)
    original_linprog=namespace.get('linprog')
    if original_linprog is None:
        raise ValueError('frozen 470 solver has no linprog global')
    namespace['linprog']=_scaled_linprog(original_linprog,h_context)
    cloned=FunctionType(source.__code__,namespace,source.__name__,source.__defaults__,source.__closure__)
    cloned.__kwdefaults__=source.__kwdefaults__
    cloned.__annotations__=dict(getattr(source,'__annotations__',{}))
    base=partial(cloned,owner)
    sig=inspect.signature(source)
    def scaled(*args,**kwargs):
        bound=sig.bind(owner,*args,**kwargs)
        h=_effective_slack_scale_h(bound.arguments)
        token=h_context.set(h)
        try:
            result=base(*args,**kwargs)
        finally:
            h_context.reset(token)
        if isinstance(result,dict):
            result['distance_slack_scaling_523']={
                'parameterization':'t_i = h_i * s_i; solver receives t, output certificate receives s=t/h',
                'objective_identity':'sum(w_i*s_i) == sum((w_i/h_i)*t_i)',
                'stage1_and_stage2_budget_algebraically_equivalent':True,
                'soft_target_row_count':int(len(h)),
                'minimum_lp_slack_scale':float(np.min(h)) if len(h) else None,
                'maximum_lp_slack_scale':float(np.max(h)) if len(h) else None,
                'frozen_470_source_sha256':EXPECTED_470_SHA256,
                'construction_only':True,
                'native_admission':False,
            }
        return result
    scaled.__name__='solve_sparse_l1_clearance_in_basis_distance_slack_523'
    scaled.__doc__='Frozen 470 solver with exact t=h*s LP variable scaling.'
    scaled.__distance_slack_scaling_523__=True
    scaled.__soft_target_no_regression_470__=True
    return scaled

def install_after_470(owner, solver_module):
    """Replace the just-installed 470 solver with its exact variable-scaled clone."""
    previous=getattr(owner,'solve_sparse_l1_clearance_in_basis',None)
    if previous is None or not getattr(previous,'__soft_target_no_regression_470__',False):
        raise ValueError('523 must install immediately after the opt-in 470 solver')
    replacement=make_solver(owner,solver_module)
    owner.solve_sparse_l1_clearance_in_basis=replacement
    return {
        'variant':'distance_slack_variable_scaling_523',
        'owner_path':str(Path(owner.__file__).resolve()),
        'owner_sha256':EXPECTED_OWNER_SHA256,
        'frozen_470_path':str(EXPECTED_470_PATH),
        'frozen_470_sha256':EXPECTED_470_SHA256,
        'parameterization':'t=h_map*s, where h_map is 470 row-normalized h',
        'returned_slack':'normalized s=t/h_map for unchanged 470 output-space certificates',
        'objective_and_stage2_budget_equivalent':True,
        'physical_hard_rows_and_bounds_unchanged':True,
        'native_admission':False,
    }

def patch_install_solver(solver_module):
    """Wrap 470.install_solver so 523 is installed before 475 captures the solver."""
    original=solver_module.install_solver
    def install(owner,*args,**kwargs):
        base_receipt=original(owner,*args,**kwargs)
        scaled_receipt=install_after_470(owner,solver_module)
        return {'soft_target_solver_470':base_receipt,'distance_slack_scaling_523':scaled_receipt}
    solver_module.install_solver=install
    return original
