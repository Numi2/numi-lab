#!/usr/bin/env python3
"""Separate selector-5 patch LU reconstruction from triangular solve rounding."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
BUNDLE_DIR = BASE / 'analysis/selector5-native-kernel-check-001'
INPUT_PATH = BUNDLE_DIR / 'canonical182-input/selector5-kernel-input.bin'
OUTPUT_PATH = BASE / 'native/selector5-coarse-patch-kernels-fresh067-canonical182-210/native-output.bin'
VERIFIER = BUNDLE_DIR / 'verify_kernel_output.py'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def rel_norm(actual, expected):
    a=np.asarray(actual,dtype=np.float64).ravel(); e=np.asarray(expected,dtype=np.float64).ravel()
    return float(np.linalg.norm(a-e)/max(np.linalg.norm(e),1e-300))

def fsum_residual(A, x, b):
    return np.asarray([math.fsum([float(A[i,j])*float(x[j]) for j in range(A.shape[1])] + [-float(b[i])])
                       for i in range(A.shape[0])], dtype=np.float64)

def solve_lu64(lu, pivots, rhs):
    n=len(rhs); L=np.tril(lu.astype(np.float64),-1)+np.eye(n); U=np.triu(lu.astype(np.float64))
    permuted=np.asarray(rhs,dtype=np.float64)[pivots.astype(np.int64)]
    y=np.empty(n,dtype=np.float64)
    for i in range(n):
        y[i]=permuted[i]-math.fsum(float(L[i,j])*float(y[j]) for j in range(i))
    x=np.empty(n,dtype=np.float64)
    for i in range(n-1,-1,-1):
        x[i]=(y[i]-math.fsum(float(U[i,j])*float(x[j]) for j in range(i+1,n)))/U[i,i]
    return x

def solve_lu32(lu, pivots, rhs, mode):
    n=len(rhs); rhs=np.asarray(rhs,dtype=np.float32); pivots=pivots.astype(np.int64)
    y=np.asarray(rhs[pivots],dtype=np.float32).copy()
    for i in range(n):
        v=np.float32(y[i])
        if mode=='fma':
            for j in range(i): v=np.float32(float(v)-float(lu[i,j])*float(y[j]))
        elif mode=='fsum':
            v=np.float32(float(y[i])-math.fsum(float(lu[i,j])*float(y[j]) for j in range(i)))
        else:
            for j in range(i): v=np.float32(v-np.float32(lu[i,j]*y[j]))
        y[i]=v
    x=np.empty(n,dtype=np.float32)
    for i in range(n-1,-1,-1):
        v=np.float32(y[i])
        if mode=='fma':
            for j in range(i+1,n): v=np.float32(float(v)-float(lu[i,j])*float(x[j]))
        elif mode=='fsum':
            v=np.float32(float(y[i])-math.fsum(float(lu[i,j])*float(x[j]) for j in range(i+1,n)))
        else:
            for j in range(i+1,n): v=np.float32(v-np.float32(lu[i,j]*x[j]))
        x[i]=np.float32(v/lu[i,i])
    return x.astype(np.float64)

def refine_lu32_fsum(lu, pivots, A, rhs, initial_x, iterations=4):
    """Offline refinement: FP64 residual, FP32 correction solve, FP32 update."""
    x=np.asarray(initial_x,dtype=np.float32).copy()
    b=np.asarray(rhs,dtype=np.float32)
    residuals=[]
    for _ in range(iterations):
        residual64=-fsum_residual(A,x.astype(np.float64),b.astype(np.float64))
        correction=solve_lu32(lu,pivots,residual64.astype(np.float32),'fsum').astype(np.float32)
        x=np.asarray(x+correction,dtype=np.float32)
        residuals.append(float(np.linalg.norm(fsum_residual(A,x.astype(np.float64),b.astype(np.float64))) /
                              max(np.linalg.norm(b.astype(np.float64)),1e-300)))
    return x.astype(np.float64),residuals

def rhs_variants(nodes, patch_node_ids, basis, AP, coeff, mode_count):
    result={k:np.empty(len(patch_node_ids)*3,dtype=np.float64) for k in ('fp64_fsum','fp32_seq','fp32_fma','fp32_fsum')}
    for loc,node in enumerate(patch_node_ids):
        for axis in range(3):
            terms=[float(coeff[m])*float(AP[m,node,axis]) for m in range(mode_count)]
            exact=float(basis[node,axis])-math.fsum(terms)
            result['fp64_fsum'][loc*3+axis]=exact
            value=np.float32(basis[node,axis])
            for m in range(mode_count): value=np.float32(value-np.float32(coeff[m]*AP[m,node,axis]))
            result['fp32_seq'][loc*3+axis]=float(value)
            value=np.float32(basis[node,axis])
            for m in range(mode_count): value=np.float32(float(value)-float(coeff[m])*float(AP[m,node,axis]))
            result['fp32_fma'][loc*3+axis]=float(value)
            result['fp32_fsum'][loc*3+axis]=float(np.float32(float(basis[node,axis])-math.fsum(terms)))
    return result

def main():
    spec=importlib.util.spec_from_file_location('verifier',VERIFIER)
    verifier=importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    raw=INPUT_PATH.read_bytes(); out_raw=OUTPUT_PATH.read_bytes(); h=struct.unpack_from('<11I',raw,8)
    _,N,T,M,patch_count,patch_node_count,dof_count,matrix_count,pivot_count,source_env_count,selected_env=h
    cursor=8+struct.calcsize(verifier.HEADER_FMT); arrays={}
    for name,dtype,count_fn in verifier.ARRAY_SPECS:
        count=int(count_fn(h)); arrays[name]=np.frombuffer(raw,dtype=dtype,count=count,offset=cursor).copy(); cursor+=count*np.dtype(dtype).itemsize
    cursor=8+struct.calcsize(verifier.HEADER_FMT); outputs={}
    specs=(('coarse_rhs','<f4',M),('coefficients','<f4',M),('patch_lu','<f4',matrix_count),('patch_pivots','<u4',pivot_count),('factor_valid','<u4',patch_count),('corrections','<f4',patch_node_count*4),('preconditioned','<f4',N*4),('status','<u4',12))
    for name,dtype,count in specs:
        outputs[name]=np.frombuffer(out_raw,dtype=dtype,count=count,offset=cursor).copy(); cursor+=count*np.dtype(dtype).itemsize
    patches=arrays['patches'].reshape(patch_count,4).astype(np.int64); patch_nodes=arrays['patch_nodes'].astype(np.int64)
    matrices=arrays['patch_matrices'].astype(np.float32); basis=arrays['basis'].reshape(N,4)[:,:3].astype(np.float32)
    AP=arrays['regional_ap'].reshape(M,N,4)[:,:,:3].astype(np.float32); coefficients=outputs['coefficients'].astype(np.float32)
    native_correction=outputs['corrections'].reshape(patch_node_count,4)[:,:3].astype(np.float64)
    rows=[]
    for p,(node_offset,node_count,matrix_offset,pivot_offset) in enumerate(patches):
        node_offset=int(node_offset); n=int(node_count); k=3*n; matrix_offset=int(matrix_offset); pivot_offset=int(pivot_offset)
        A=matrices[matrix_offset:matrix_offset+k*k].reshape(k,k).astype(np.float64)
        lu=outputs['patch_lu'][matrix_offset:matrix_offset+k*k].reshape(k,k).astype(np.float32)
        pivots=outputs['patch_pivots'][pivot_offset:pivot_offset+k].astype(np.int64)
        local_nodes=patch_nodes[node_offset:node_offset+n]
        native_x=native_correction[node_offset:node_offset+n].reshape(-1)
        rhs=rhs_variants(local_nodes,local_nodes,basis,AP,coefficients,M)
        L=np.tril(lu.astype(np.float64),-1)+np.eye(k); U=np.triu(lu.astype(np.float64))
        rec=L@U; Arec=np.empty_like(rec); Arec[pivots,:]=rec
        factor_rel=rel_norm(Arec,A)
        x64={name:solve_lu64(lu,pivots,b) for name,b in rhs.items()}
        x32={mode:solve_lu32(lu,pivots,rhs['fp32_seq'],mode) for mode in ('seq','fma','fsum')}
        exact_x=np.linalg.solve(A,rhs['fp64_fsum'])
        rec_res=lambda x,b: float(np.linalg.norm(fsum_residual(A,x,b))/max(np.linalg.norm(b),1e-300))
        refined32_x,refine32_res=refine_lu32_fsum(
            lu,pivots,A,rhs['fp32_seq'],x32['fsum'])
        # Refine using the preserved source matrix with the same FP32 LU as a correction solver.
        refined=x64['fp64_fsum'].copy(); refine_res=[]
        for _ in range(4):
            rr=-fsum_residual(A,refined,rhs['fp64_fsum'])
            dx=solve_lu64(lu,pivots,rr)
            refined+=dx
            refine_res.append(rec_res(refined,rhs['fp64_fsum']))
        row={
            'patch':p,'nodes':n,'dofs':k,'condition_number_2':float(np.linalg.cond(A)),
            'lu_reconstruction_relative_frobenius':factor_rel,
            'rhs_fp32seq_vs_fp64_relative':rel_norm(rhs['fp32_seq'],rhs['fp64_fsum']),
            'rhs_fp32fma_vs_fp64_relative':rel_norm(rhs['fp32_fma'],rhs['fp64_fsum']),
            'native_vs_fp32_sequential_solution_relative':rel_norm(native_x,x32['seq']),
            'native_vs_fp64_fsum_lu_solution_relative':rel_norm(native_x,x64['fp64_fsum']),
            'native_residual_Ax_minus_b_fp64_rhs':rec_res(native_x,rhs['fp64_fsum']),
            'native_residual_Ax_minus_b_fp32seq_rhs':rec_res(native_x,rhs['fp32_seq']),
            'native_residual_Ax_minus_b_fp32fma_rhs':rec_res(native_x,rhs['fp32_fma']),
            'fp32seq_residual_Ax_minus_b':rec_res(x32['seq'],rhs['fp32_seq']),
            'fma_residual_Ax_minus_b':rec_res(x32['fma'],rhs['fp32_seq']),
            'fsum_residual_Ax_minus_b':rec_res(x32['fsum'],rhs['fp32_seq']),
            'fp64_triangular_lu_residual_Ax_minus_b':rec_res(x64['fp64_fsum'],rhs['fp64_fsum']),
            'fp64_direct_solve_residual_Ax_minus_b':rec_res(exact_x,rhs['fp64_fsum']),
            'fp64_triangular_lu_vs_direct_solution_relative':rel_norm(x64['fp64_fsum'],exact_x),
            'iterative_refinement_residuals':refine_res,
            'iterative_refinement_converged_under_1e_3':bool(refine_res[-1] <= 1e-3),
            'fp32_fsum_iterative_refinement_residuals':refine32_res,
            'fp32_fsum_iterative_refinement_converged_under_1e_3':bool(refine32_res[-1] <= 1e-3),
        }
        rows.append(row)
    report={
      'schema':'numi.matter.selector5-triangle-roundoff-audit.v1',
      'status':'cpu_replay_diagnostic_not_native_qualification',
      'source_snapshot':'source-angular-and-graph-schwarz-integrated-065',
      'native_run':'selector5-coarse-patch-kernels-fresh067-canonical182-210',
      'input_sha256':sha(raw),'native_output_sha256':sha(out_raw),'verifier_sha256':sha(VERIFIER.read_bytes()),
      'dimensions':{'nodes':N,'tets':T,'modes':M,'patches':patch_count,'patch_nodes':patch_node_count,'patch_dofs':dof_count},
      'method':'All metrics use captured float32 source patch matrices/LU/pivots and native correction; FP64 fsum RHS is explicitly distinguished from float32 sequential/fused RHS. Iterative refinement is an offline diagnostic using original source A, not a production change.',
      'threshold_relative_residual':1e-3,
      'all_patches_factor_reconstruction_pass':max(r['lu_reconstruction_relative_frobenius'] for r in rows)<=1e-3,
      'all_patches_fp64_lu_solve_pass':max(r['fp64_triangular_lu_residual_Ax_minus_b'] for r in rows)<=1e-3,
      'all_patches_native_residual_pass':max(r['native_residual_Ax_minus_b_fp64_rhs'] for r in rows)<=1e-3,
      'all_patches_iterative_refinement_below_threshold':all(r['iterative_refinement_converged_under_1e_3'] for r in rows),
      'all_patches_fp32_fsum_iterative_refinement_below_threshold':all(
          r['fp32_fsum_iterative_refinement_converged_under_1e_3'] for r in rows),
      'fp32_fsum_iterative_refinement_first_correction_max_residual':max(
          r['fp32_fsum_iterative_refinement_residuals'][0] for r in rows),
      'per_patch':rows,
    }
    outdir=Path(__file__).resolve().parent
    (outdir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with (outdir/'per_patch.csv').open('w') as f:
        import csv
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k!='per_patch'},indent=2))

if __name__=='__main__': main()
