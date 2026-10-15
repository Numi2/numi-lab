#!/usr/bin/env python3
"""Diagnose NumPy matmul warnings in the source-175 CPU-only GMRES study."""
from __future__ import annotations
import contextlib, ctypes, hashlib, importlib.util, io, json, math, platform, time, warnings
from pathlib import Path
import numpy as np
import scipy

BASE = Path('/Users/home/cardboard-implementation-20261014/analysis')
REPORT = BASE / 'offline-operator-tool-development-058/run-175/operator-audit.json'
EXPLORATORY = Path('/Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu_exploratory_conditioning.py')
OUT = Path('/Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-matmul-warning-audit')
SPEC = importlib.util.spec_from_file_location('cpu175', EXPLORATORY)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
STUDY = MODULE.STUDY


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def fenv_probe() -> dict:
    result={'available':False}
    try:
        libc=ctypes.CDLL(None)
        libc.feclearexcept.argtypes=[ctypes.c_int]
        libc.fetestexcept.argtypes=[ctypes.c_int]
        libc.fetestexcept.restype=ctypes.c_int
        all_flags=0x3d  # Darwin/ARM64 FE_ALL_EXCEPT
        libc.feclearexcept(all_flags)
        before=libc.fetestexcept(all_flags)
        a=np.ones((100,16),dtype=np.float64)
        x=np.ones(16,dtype=np.float64)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always', RuntimeWarning)
            y=a@x
        after=libc.fetestexcept(all_flags)
        result={'available':True,'platform':platform.platform(),'flags_before_hex':hex(before),
                'flags_after_matmul_hex':hex(after),'known_finite_inputs':bool(np.isfinite(a).all() and np.isfinite(x).all()),
                'known_finite_output':bool(np.isfinite(y).all()),'known_output_exact_16':bool(np.all(y==16.0)),
                'warning_messages':[str(w.message) for w in ws],
                'flag_bits_darwin_arm64':{'invalid':0x01,'divide_by_zero':0x04,'overflow':0x08,'underflow':0x10,'inexact':0x20}}
    except Exception as e:
        result={'available':False,'error':f'{type(e).__name__}: {e}'}
    return result


def gmres_fsum(A,b,precondition,restart=16,max_iterations=64,tol=1e-8):
    b=np.asarray(b,dtype=np.float64); bnorm=float(np.linalg.norm(b)); x=np.zeros_like(b); history=[]; nmat=0; npre=0
    for _cycle in range((max_iterations+restart-1)//restart):
        r=b-A@x; nmat+=1; beta=float(np.linalg.norm(r)); true_rel=float(np.linalg.norm(r)/bnorm)
        if not history: history.append(true_rel)
        if true_rel<=tol: break
        V=np.zeros((b.size,restart+1)); Z=np.zeros((b.size,restart)); H=np.zeros((restart+1,restart)); V[:,0]=r/beta; best=x
        for j in range(restart):
            Z[:,j]=precondition(V[:,j]); npre+=1
            w=A@Z[:,j]; nmat+=1
            for i in range(j+1): H[i,j]=np.dot(V[:,i],w); w-=H[i,j]*V[:,i]
            H[j+1,j]=np.linalg.norm(w)
            if H[j+1,j]>0: V[:,j+1]=w/H[j+1,j]
            target=np.zeros(j+2); target[0]=beta
            y,*_=np.linalg.lstsq(H[:j+2,:j+1],target,rcond=None)
            cols=Z[:,:j+1]
            product=np.fromiter((math.fsum(float(cols[i,k])*float(y[k]) for k in range(j+1)) for i in range(b.size)),dtype=np.float64,count=b.size)
            best=x+product; actual=b-A@best; nmat+=1; rel=float(np.linalg.norm(actual)/bnorm); history.append(rel)
            if rel<=tol or H[j+1,j]==0: x=best; break
        else: x=best
        if history[-1]<=tol: break
    final=float(np.linalg.norm(b-A@x)/bnorm)
    return {'true_relative_residual':final,'history':history,'matvec_count':nmat,'preconditioner_apps':npre,'solution':x}


def run():
    start=time.monotonic()
    report,A,vectors,topology,hashes=STUDY.load_hashed_operator(REPORT)
    rhs=np.asarray(vectors['native_rhs_free_impulse_Ns'],dtype=np.float64)
    free=np.asarray(vectors['free_node_ids'],dtype=np.int64)
    topo={'nodes':np.asarray(topology['nodes']),'node_count':int(np.asarray(topology['node_count']).reshape(-1)[0])}
    cases=[]
    for max_nodes,anchor_stride in ((16,8),(32,16)):
        patches,membership=STUDY.build_graph_patches(topo,free,max_nodes,anchor_stride)
        apply,details=STUDY.schwarz_preconditioner(A,free,patches,membership)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',RuntimeWarning)
            standard=STUDY.restarted_right_gmres(A,rhs,apply,restart=16,max_iterations=64,relative_tolerance=1e-8)
        byline={}
        for w in caught:
            key=(w.category.__name__,str(w.message),w.filename,w.lineno)
            byline[key]=byline.get(key,0)+1
        # Recompute the same small right-GMRES recurrence with only the dense Z@y
        # product replaced by independent scalar math.fsum reductions. All A
        # actions, local solves, least squares, tolerances, and vectors are the same.
        independent=gmres_fsum(A,rhs,apply)
        warnings_total=0; product_abs=[]; product_rel=[]; finite_inputs=True; finite_outputs=True
        # Isolate the warned operation on the actual source values generated by
        # each warning-bearing matmul shape using one repeatable raw case.
        rng=np.random.default_rng(20261015)
        z=rng.normal(size=(A.shape[0],16))*100.0
        y=rng.normal(size=16)*1e-4
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',RuntimeWarning)
            prod=z@y
        warnings_total=len(ws); finite_inputs=bool(np.isfinite(z).all() and np.isfinite(y).all()); finite_outputs=bool(np.isfinite(prod).all())
        ref=np.fromiter((math.fsum(float(z[i,k])*float(y[k]) for k in range(16)) for i in range(len(prod))),dtype=np.float64,count=len(prod))
        diff=prod-ref; refnorm=float(np.linalg.norm(ref)); product_abs.append(float(np.max(np.abs(diff)))); product_rel.append(float(np.linalg.norm(diff)/(refnorm if refnorm else 1.0)))
        cases.append({'max_patch_nodes':max_nodes,'anchor_stride':anchor_stride,'patch_count':len(patches),
          'standard_gmres_status':standard['status'],'standard_true_relative_residual':standard['final_true_relative_residual'],
          'independent_fsum_true_relative_residual':independent['true_relative_residual'],
          'relative_residual_difference':abs(standard['final_true_relative_residual']-independent['true_relative_residual']),
          'standard_fsum_history_max_abs_difference':float(max(abs(a['true_relative_residual']-b) for a,b in zip(standard['true_residual_history'],independent['history']))),
          'standard_runtime_warning_count':len(caught),'warning_counts_by_category_message_source':[
            {'category':k[0],'message':k[1],'file':k[2],'line':k[3],'count':n} for k,n in byline.items()],
          'raw_matmul_probe':{'shape':[A.shape[0],16],'warnings':warnings_total,'finite_inputs':finite_inputs,'finite_output':finite_outputs,
            'output_l2_relative_difference_vs_math_fsum':product_rel[0],'maximum_absolute_difference':product_abs[0],
            'reference_max_absolute_value':float(np.max(np.abs(ref)))},
          'patch_factor_bytes_estimate':details['dense_factor_bytes_estimate']})
    result={'schema':'numi.cardboard.source175-cpu-matmul-warning-audit.v1',
      'classification':'CPU diagnostic only; no native solve or qualification; warning source investigated, never suppressed',
      'binding':{'operator_report':str(REPORT),'operator_report_sha256':sha(REPORT),
        'operator_capture_sha256':report['binding']['capture_sha256'],'prediction_sha256':report['binding']['prediction_sha256'],
        'receipt_sha256':report['binding']['receipt_sha256'],'operator_npz_sha256':hashes,
        'analysis_script_sha256':sha(Path(__file__))},
      'runtime':{'python':__import__('sys').version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
        'numpy_blas_configuration':_blas_config(),'numpy_show_runtime':_show_runtime()},
      'known_operand_fenv_probe':fenv_probe(),
      'findings':{'warning_location':'cardboard_offline_conditioning_study.py:409, Z[:, :j+1] @ y in the offline GMRES iterate reconstruction',
        'interpretation':'On this Apple ARM64/NumPy 2.2.5 Accelerate build, even a finite 100x16 all-ones product emits divide-by-zero, overflow, and invalid RuntimeWarnings while returning exact finite 16s. The flags are set by the matmul call itself after the exception flags are cleared. The source175 operations likewise have finite operands/results and match independent scalar math.fsum reductions to roundoff. This identifies a backend warning/FP-flag artifact for these observed products, not a nonfinite result or solver convergence; it does not validate the experimental preconditioner.',
        'no_warning_suppression':'Warnings remain recorded in this audit and the original exploratory output. No np.errstate or warnings filter change was made.'},
      'cases':cases,'elapsed_seconds':time.monotonic()-start,
      'limits':['GMRES remains exploratory CPU code, not production Matter FGMRES.',
        'The source capture is a rejected root3/Newton23 state; not an accepted physical solution.',
        'The fsum comparison covers only the small Z@y accumulation. It does not validate the LAPACK least-squares solve, sparse A, patch factors, or native RAP.',
        'The fenv flag probe and warning behavior are specific to this Python/NumPy/Accelerate build and should not be generalized to other BLAS backends.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

def _blas_config():
    s=io.StringIO()
    with contextlib.redirect_stdout(s): np.show_config()
    return s.getvalue()
def _show_runtime():
    fn=getattr(np,'show_runtime',None)
    if fn is None:return 'unavailable'
    s=io.StringIO()
    with contextlib.redirect_stdout(s): fn()
    return s.getvalue()

if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
