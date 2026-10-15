#!/usr/bin/env python3
"""Compare two independently regenerated source-bound Cauchy bundles."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy import sparse

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def walk(a,b,path=''):
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys(): raise AssertionError(f'key mismatch at {path}')
        for k in a: walk(a[k],b[k],f'{path}/{k}')
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b): raise AssertionError(f'list mismatch at {path}')
        for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f'{path}/{i}')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)) or not math.isclose(float(a),float(b),rel_tol=2e-14,abs_tol=1e-300):
            raise AssertionError(f'numeric mismatch at {path}: {a} vs {b}')
    elif a!=b: raise AssertionError(f'value mismatch at {path}: {a!r} vs {b!r}')

def validate(d):
    d=Path(d); r=json.loads((d/'report.json').read_text())
    if r.get('schema')!='numi.cardboard.captured-base-state-cauchy-experiment.v1': raise AssertionError('wrong report schema')
    if r.get('verdict')!='BASE_STATE_LINEAR_CANDIDATE_READY_FOR_NATIVE_RESIDUAL_REPLAY': raise AssertionError('unexpected report verdict')
    for item in r['artifacts']:
        p=d/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise AssertionError(f'artifact inventory mismatch: {item["path"]}')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('recomputed',type=Path);a=p.parse_args()
    x=validate(a.original);y=validate(a.recomputed)
    for key in ('schema','verdict','selection','input_bindings','source_input_binding_result','operator','residuals','base_state_gradient','base_state_directions','total_newton_plus_base_cauchy','post_newton_refinement_supplement','limitations'):
        walk(x[key],y[key],key)
    bx={Path(v['path']).name for v in x['artifacts'] if v.get('dtype','').endswith('f4') or v.get('dtype','').endswith('f8')}
    by={Path(v['path']).name for v in y['artifacts'] if v.get('dtype','').endswith('f4') or v.get('dtype','').endswith('f8')}
    if bx!=by: raise AssertionError('binary artifact name sets differ')
    for name in sorted(bx):
        if (a.original/name).read_bytes()!=(a.recomputed/name).read_bytes(): raise AssertionError(f'binary bytes differ: {name}')
    A=sparse.load_npz(a.original/'A_free_csr.npz').tocsr();B=sparse.load_npz(a.recomputed/'A_free_csr.npz').tocsr()
    D=(A-B).tocsr();D.eliminate_zeros()
    if D.nnz or A.shape!=B.shape: raise AssertionError('CSR operator differs')
    print(json.dumps({'result':'PASS','binary_files_bitwise_equal':len(bx),'csr_shape':list(A.shape),'csr_nnz':int(A.nnz),'input_capture_sha256':x['input_bindings']['capture_sha256']},indent=2))
if __name__=='__main__':main()
