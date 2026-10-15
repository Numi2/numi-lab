#!/usr/bin/env python3
"""Bounded CPU-only FP64 SVD audit of captured source125 Newton-48 operator."""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import load_npz
from scipy.linalg import svd

SRC = Path('/Users/home/cardboard-implementation-20261014/analysis/captured-base-state-cauchy-419-001')
OUT = Path(__file__).resolve().parent
INPUTS = [
    'A_free_csr.npz', 'base_rhs.float64le.bin', 'captured_newton_delta.float64le.bin',
    'post_newton_linear_remainder.float64le.bin',
    'base_column_equilibrated_cauchy.free-delta-v.float64le.bin',
]
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def norm(x): return float(np.linalg.norm(x))
def main():
    started=time.time()
    src_report=json.loads((SRC/'report.json').read_text())
    src_manifest=json.loads((SRC/'manifest.json').read_text())
    for n in INPUTS:
        rec=next(x for x in src_manifest['files'] if x['path']==n)
        actual=sha(SRC/n)
        if actual != rec['sha256']:
            raise SystemExit(f'input hash mismatch: {n}')
    A_sparse=load_npz(SRC/'A_free_csr.npz').astype(np.float64)
    A=A_sparse.toarray().astype(np.float64,copy=False)
    b=np.fromfile(SRC/'base_rhs.float64le.bin',dtype='<f8')
    delta=np.fromfile(SRC/'captured_newton_delta.float64le.bin',dtype='<f8')
    if A.shape != (2538,2538) or b.shape != (2538,) or delta.shape != b.shape:
        raise SystemExit('unexpected capture shape')
    print(f'loaded exact bound inputs: A={A.shape}, bytes={A.nbytes}; ||b||={norm(b):.9g}; ||delta||={norm(delta):.9g}', flush=True)
    # Full economy SVD retains factors only in process memory; output is compact.
    t=time.time()
    A_svd=A.copy()
    U,s,Vh=svd(A_svd,full_matrices=False,overwrite_a=True,check_finite=True,lapack_driver='gesdd')
    print(f'FP64 SVD done in {time.time()-t:.1f}s; sigma_min={s[-1]:.9e}; sigma_max={s[0]:.9e}', flush=True)
    # einsum avoids Accelerate's spurious floating-point warnings on the
    # very small RHS products; sparse matvec preserves the original CSR A.
    ub=np.einsum('ij,i->j',U,b,optimize=False)
    vd=np.einsum('ij,j->i',Vh,delta,optimize=False)
    residual=b-A_sparse@delta
    eps=np.finfo(np.float64).eps
    tol=eps*max(A.shape)*s[0]
    cutoffs=[tol, s[0]*1e-14,s[0]*1e-12,s[0]*1e-10,s[0]*1e-8,s[0]*1e-6,s[0]*1e-4,s[0]*1e-2]
    spectrum={
        'sigma_max':float(s[0]),'sigma_min':float(s[-1]),'condition_2':float(s[0]/s[-1]),
        'numpy_rank_tolerance':float(tol),
        'rank_at_numpy_tolerance':int(np.count_nonzero(s>tol)),
        'relative_singular_value_quantiles':{str(q):float(np.quantile(s/s[0],q)) for q in [0,.001,.01,.05,.1,.25,.5,.75,.9,.99,1]},
        'cutoff_counts':{format(c/s[0],'.3g'):int(np.count_nonzero(s<=c)) for c in cutoffs},
    }
    weak=[]
    for k in [1,2,4,8,16,32,64,128,256]:
        idx=s.size-k
        weak.append({
            'weakest_k':k,
            'sigma_range':[float(s[-k]),float(s[-1])],
            'rhs_left_projection_norm':norm(ub[-k:]),
            'rhs_left_projection_fraction':norm(ub[-k:])/norm(b),
            'delta_right_projection_norm':norm(vd[-k:]),
            'delta_right_projection_fraction':norm(vd[-k:])/norm(delta),
            'captured_delta_max_mode_amplitude':float(np.max(np.abs(vd[-k:]))),
            'weak_mode_amplification_ratio_norm':norm(vd[-k:])/max(norm(ub[-k:]),np.finfo(float).tiny),
        })
    weak_mode_details=[]
    for j in range(s.size-1,max(-1,s.size-17),-1):
        weak_mode_details.append({
            'singular_index_from_weakest':int(s.size-1-j),
            'sigma':float(s[j]),
            'rhs_left_coefficient':float(ub[j]),
            'captured_delta_right_coefficient':float(vd[j]),
            'sigma_times_delta_coefficient':float(s[j]*vd[j]),
            'residual_left_coefficient':float(ub[j]-s[j]*vd[j]),
        })
    x_lstsq=np.einsum('ji,j->i',Vh,ub/s,optimize=False)
    ls_residual=b-A_sparse@x_lstsq
    captured_delta_difference=x_lstsq-delta
    candidates=[]
    def add(name,x,detail):
        r=b-A_sparse@x
        candidates.append({'name':name,'details':detail,'correction_l2':norm(x),'correction_linf':float(np.max(np.abs(x)),),'predicted_residual_l2':norm(r),'residual_ratio_to_base':norm(r)/norm(b),'predicted_merit_decrease_fraction':1.0-(norm(r)/norm(b))**2})
    add('captured_newton_delta',delta,{'kind':'captured production FGMRES correction; not recomputed'})
    add('full_svd_minimum_norm_least_squares',x_lstsq,{'kind':'unregularized FP64 SVD least-squares solution; diagnostic reference only','retained_rank':int(np.count_nonzero(s>tol))})
    # Spectral regularization on the exact nonsymmetric A via its SVD.
    for rel in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
        lam=rel*s[0]
        coeff=(s/(s*s+lam*lam))*ub
        x=np.einsum('ji,j->i',Vh,coeff,optimize=False)
        add(f'tikhonov_lambda_sigma_max_{rel:g}',x,{'kind':'diagnostic Tikhonov on full unsymmetrized A','lambda':float(lam),'lambda_over_sigma_max':rel})
    # Truncated SVD diagnostic; explicit thresholds are not solver recommendations.
    for rel in [1e-4,1e-6,1e-8,1e-10,1e-12,0.0]:
        cutoff=rel*s[0] if rel else tol
        keep=s>cutoff
        coeff=np.zeros_like(ub)
        coeff[keep]=ub[keep]/s[keep]
        x=np.einsum('ji,j->i',Vh,coeff,optimize=False)
        add(f'truncated_svd_sigma_max_{rel:g}' if rel else 'truncated_svd_numpy_rank_threshold',x,{'kind':'diagnostic truncated SVD on exact A','cutoff':float(cutoff),'retained_rank':int(np.count_nonzero(keep))})
    # Euclidean steepest descent for phi(x)=0.5||b-Ax||^2 from x=0.
    g=A_sparse.T@b
    Ag=A_sparse@g
    g2=float(np.sum(g*g)); Ag2=float(np.sum(Ag*Ag))
    beta=g2/Ag2 if Ag2>0 else 0.0
    add('steepest_descent_from_zero',beta*g,{'kind':'exact FP64 Euclidean steepest-descent line minimizer','step_length':beta})
    # Equilibrated Cauchy vector from sealed original package, read as comparison only.
    p=np.fromfile(SRC/'base_column_equilibrated_cauchy.free-delta-v.float64le.bin',dtype='<f8')
    add('sealed_column_equilibrated_cauchy',p,{'kind':'existing exact-capture offline diagnostic candidate; not solver output'})
    # Retain only compact singular values and modal projections, not large U/V factors.
    np.savez_compressed(OUT/'spectral_compact.npz',singular_values=s, rhs_left_coefficients=ub, captured_delta_right_coefficients=vd)
    report={
        'schema':'numi.cardboard.captured419_linear_conditioning.v1',
        'classification':'CPU-only FP64 diagnostic; no production solver/operator/tolerance changes; candidates are predictions from captured linear A only',
        'selection':src_report['selection'],
        'source_capture_bindings':{
            'capture_sha256':src_report['input_bindings']['capture_sha256'],
            'source_snapshot':src_report['input_bindings']['source_snapshot'],
            'assembly_module_sha256':src_report['input_bindings']['assembly_module_sha256'],
            'input_hashes':{n:sha(SRC/n) for n in INPUTS},
            'source_package_manifest_sha256':sha(SRC/'manifest.json'),
            'source_report_sha256':sha(SRC/'report.json'),
        },
        'matrix':{
            'shape':list(A.shape),'nnz':int(load_npz(SRC/'A_free_csr.npz').nnz),
            'matrix_definition':src_report['operator']['assembly'],
            'matrix_layout':src_report['operator']['matrix_layout'],
            'relative_skew_norm':src_report['operator']['relative_skew_norm'],
            'dt_seconds':src_report['operator']['dt_seconds'],
        },
        'input_norms':{'rhs_l2':norm(b),'rhs_linf':float(np.max(np.abs(b))),'captured_delta_l2':norm(delta),'captured_delta_linf':float(np.max(np.abs(delta))),'A_delta_l2':norm(A_sparse@delta),'captured_remainder_l2':norm(residual),'captured_remainder_file_l2':norm(np.fromfile(SRC/'post_newton_linear_remainder.float64le.bin',dtype='<f8'))},
        'spectrum':spectrum,
        'weak_mode_projection':weak,
        'weakest_16_mode_coefficients':weak_mode_details,
        'captured_delta_vs_full_svd_least_squares':{
            'difference_l2':norm(captured_delta_difference),
            'relative_difference_to_captured_delta':norm(captured_delta_difference)/norm(delta),
            'difference_linf':float(np.max(np.abs(captured_delta_difference))),
            'full_svd_lstsq_residual_l2':norm(ls_residual),
            'full_svd_lstsq_residual_ratio':norm(ls_residual)/norm(b),
            'captured_delta_backward_residual_ratio':norm(residual)/(np.linalg.norm(A,2)*norm(delta)+norm(b)),
        },
        'linear_candidates':candidates,
        'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'blas_thread_limit':'VECLIB_MAXIMUM_THREADS=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1'},
        'runtime_seconds':time.time()-started,
        'limits':['Singular values are for the captured 2538x2538 FP64 CSR matrix, not a nonlinear Jacobian accuracy proof.','Regularized/truncated candidates alter the linear solve diagnostically and are not physical/solver recommendations.','Candidate residuals are linear predictions b-Ax, not an Armijo trial or accepted transaction.','The capture contains one environment/operator and a rejected candidate with zero physical transactions.']
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('rank tol',spectrum['rank_at_numpy_tolerance'],'condition',spectrum['condition_2'],'runtime',report['runtime_seconds'],flush=True)
    for row in candidates: print(row['name'], '||x||',row['correction_l2'],'||r||',row['predicted_residual_l2'],'ratio',row['residual_ratio_to_base'],flush=True)
if __name__=='__main__': main()
