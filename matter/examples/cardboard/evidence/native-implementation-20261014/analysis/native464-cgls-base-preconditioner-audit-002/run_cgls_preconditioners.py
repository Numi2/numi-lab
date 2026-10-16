#!/usr/bin/env python3
"""Offline bounded CGLS comparison using the exact 464/419 captured base state.

This evaluates identity, per-node A_ii^-1 (block3), and the source-151 *base*
selector map S/S^T. It is a CPU numerical experiment, not native CGLS or proof
that selector 5's augmented map was consumed.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time
import warnings
from collections import Counter

BASE=Path('/Users/home/cardboard-implementation-20261014')
AUDIT=BASE/'analysis/native464-regional-capture-audit-003/audit_native464_v4.py'
CSRPKG=BASE/'analysis/captured-base-state-cauchy-419-001'
CASE=BASE/'solver/cgls-regional-state-capture-464'
GRAPH_INPUT=BASE/'analysis/native464-regional-capture-command-001/preflight/actual-graph-input.bin'
GRAPH_BINDINGS=BASE/'analysis/native464-regional-capture-command-001/source151-regional-capture-bindings.json'
REGIONAL=CASE/'base-cgls.bin.regional.json'
OPCAP=BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'

class ExperimentError(RuntimeError): pass

def hfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def load_audit():
    spec=importlib.util.spec_from_file_location('audit_native464_v4',AUDIT)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def solve_lu(factor, rhs, dtype, trans=False):
    import numpy as np
    from scipy.linalg import solve_triangular
    if not factor.valid: raise ExperimentError('base RAP factor invalid')
    lu=np.asarray(factor.lu,dtype=dtype); piv=np.asarray(factor.pivots,dtype=np.int64)
    b=np.asarray(rhs,dtype=dtype)
    if not trans:
        y=solve_triangular(np.tril(lu,-1)+np.eye(len(lu),dtype=dtype),b[piv],lower=True,unit_diagonal=True,check_finite=False)
        return solve_triangular(np.triu(lu),y,lower=False,check_finite=False).astype(dtype,copy=False)
    z=solve_triangular(np.triu(lu).T,b,lower=True,check_finite=False)
    w=solve_triangular((np.tril(lu,-1)+np.eye(len(lu),dtype=dtype)).T,z,lower=False,unit_diagonal=True,check_finite=False)
    out=np.empty_like(w);out[piv]=w
    return out

class GraphBaseMap:
    def __init__(self,capture,dtype):
        import numpy as np
        self.n=int(capture.node_count); self.nd=3*self.n; self.dtype=dtype
        self.free=np.asarray(capture.free_mask,dtype=bool)
        self.cover=np.asarray(capture.node_coverage,dtype=np.float64)
        self.P=np.asarray(capture.P,dtype=dtype); self.AP=np.asarray(capture.AP,dtype=dtype)
        self.factor=capture.base_factor
        self.patches=[]
        for patch in capture.patches:
            A=np.asarray(patch.original,dtype=dtype)
            lu=np.asarray(patch.factor.lu,dtype=dtype)
            piv=np.asarray(patch.factor.pivots,dtype=np.int64)
            nlocal=A.shape[0]
            if not patch.factor.valid or A.shape!=(3*len(patch.nodes),3*len(patch.nodes)):
                raise ExperimentError('invalid source patch factor or dimensions')
            L=np.tril(lu,-1)+np.eye(nlocal,dtype=dtype); U=np.triu(lu)
            nodes=np.asarray(patch.nodes,dtype=np.int64)
            free_local=np.repeat(self.free[nodes],3)
            self.patches.append((A,L,U,piv,nodes,free_local))
        if not self.factor.valid: raise ExperimentError('base RAP factor-invalid')

    def solve_patch(self,patch,rhs,trans=False):
        import numpy as np
        from scipy.linalg import solve_triangular
        A,L,U,piv,nodes,free_local=patch
        rhs=np.asarray(rhs,dtype=self.dtype)
        if not trans:
            y=solve_triangular(L,rhs[piv],lower=True,unit_diagonal=True,check_finite=False)
            return solve_triangular(U,y,lower=False,check_finite=False).astype(self.dtype,copy=False)
        z=solve_triangular(U.T,rhs,lower=True,check_finite=False)
        w=solve_triangular(L.T,z,lower=False,unit_diagonal=True,check_finite=False)
        out=np.empty_like(w);out[piv]=w
        return out

    def H(self,x):
        import numpy as np
        x=np.asarray(x,dtype=self.dtype)
        if x.shape!=(self.nd,): raise ExperimentError('H input shape')
        sums=np.zeros(self.nd,dtype=self.dtype)
        for patch in self.patches:
            A,L,U,piv,nodes,fl=patch
            rhs=np.zeros(3*len(nodes),dtype=self.dtype)
            for local,node in enumerate(nodes):
                if self.free[node]: rhs[3*local:3*local+3]=x[3*node:3*node+3]
            x0=self.solve_patch(patch,rhs)
            residual=(rhs-A@x0).astype(self.dtype,copy=False)
            correction=self.solve_patch(patch,residual)
            local_out=(x0+correction).astype(self.dtype,copy=False)
            for local,node in enumerate(nodes):
                if self.free[node]: sums[3*node:3*node+3]+=local_out[3*local:3*local+3]
        out=np.zeros(self.nd,dtype=self.dtype)
        for node in range(self.n):
            if self.free[node] and self.cover[node]>0:
                out[3*node:3*node+3]=sums[3*node:3*node+3]/self.dtype(self.cover[node])
        return out

    def HT(self,y):
        import numpy as np
        y=np.asarray(y,dtype=self.dtype)
        if y.shape!=(self.nd,): raise ExperimentError('HT input shape')
        out=np.zeros(self.nd,dtype=self.dtype)
        for patch in self.patches:
            A,L,U,piv,nodes,fl=patch
            rhs=np.zeros(3*len(nodes),dtype=self.dtype)
            for local,node in enumerate(nodes):
                if self.free[node] and self.cover[node]>0:
                    rhs[3*local:3*local+3]=y[3*node:3*node+3]/self.dtype(self.cover[node])
            z=self.solve_patch(patch,rhs,trans=True)
            atz=(A.T@z).astype(self.dtype,copy=False)
            w=self.solve_patch(patch,atz,trans=True)
            local_out=(self.dtype(2)*z-w).astype(self.dtype,copy=False)
            for local,node in enumerate(nodes):
                if self.free[node]: out[3*node:3*node+3]+=local_out[3*local:3*local+3]
        return out

    def apply_full(self,rhs):
        import numpy as np
        v=np.asarray(rhs,dtype=self.dtype)
        coeff=solve_lu(self.factor,self.P.T@v,self.dtype,False)
        residual=(v-self.AP@coeff).astype(self.dtype,copy=False)
        fine=self.H(residual)
        return (fine+self.P@coeff).astype(self.dtype,copy=False)

    def apply_transpose_full(self,y):
        import numpy as np
        v=np.asarray(y,dtype=self.dtype)
        hty=self.HT(v)
        coeff=solve_lu(self.factor,self.P.T@v-self.AP.T@hty,self.dtype,True)
        return (hty+self.P@coeff).astype(self.dtype,copy=False)

class RightMap:
    def __init__(self,kind,free_nodes,A,capture,dtype):
        import numpy as np
        self.kind,self.dtype=kind,dtype
        self.free_nodes=np.asarray(free_nodes,dtype=np.int64)
        self.nfree=len(self.free_nodes); self.N=3*self.nfree
        self.full_n=int(capture.node_count) if capture is not None else max(free_nodes)+1
        self.graph=GraphBaseMap(capture,dtype) if kind=='base_graph' else None
        self.blocks=None
        if kind=='block3':
            self.blocks=[]
            for j in range(self.nfree):
                sl=slice(3*j,3*j+3)
                block=A[sl,sl].toarray().astype(dtype)
                if not np.all(np.isfinite(block)) or abs(float(np.linalg.det(block.astype(np.float64))))<1e-30:
                    raise ExperimentError(f'singular/nonfinite diagonal block at free node {j}')
                self.blocks.append(np.linalg.inv(block).astype(dtype))
            self.blocks=np.stack(self.blocks)
        elif kind not in ('identity','base_graph'):
            raise ExperimentError('unknown right map')
    def _expand(self,x):
        import numpy as np
        full=np.zeros(3*self.full_n,dtype=self.dtype)
        full.reshape(self.full_n,3)[self.free_nodes]=np.asarray(x,dtype=self.dtype).reshape(self.nfree,3)
        return full
    def apply(self,x):
        import numpy as np
        x=np.asarray(x,dtype=self.dtype)
        if self.kind=='identity': return x.copy()
        if self.kind=='block3': return np.einsum('nij,nj->ni',self.blocks,x.reshape(self.nfree,3)).reshape(-1).astype(self.dtype,copy=False)
        return self.graph.apply_full(self._expand(x)).reshape(self.full_n,3)[self.free_nodes].reshape(-1).astype(self.dtype,copy=False)
    def transpose(self,y):
        import numpy as np
        y=np.asarray(y,dtype=self.dtype)
        if self.kind=='identity': return y.copy()
        if self.kind=='block3': return np.einsum('nji,nj->ni',self.blocks,y.reshape(self.nfree,3)).reshape(-1).astype(self.dtype,copy=False)
        return self.graph.apply_transpose_full(self._expand(y)).reshape(self.full_n,3)[self.free_nodes].reshape(-1).astype(self.dtype,copy=False)

def cgls(A,b,right_map,cap=64,dtype=None):
    import numpy as np
    dtype=dtype or b.dtype
    Aop=A.astype(dtype) if A.dtype!=dtype else A
    b=np.asarray(b,dtype=dtype)
    y=np.zeros_like(b)
    r=b.copy()
    g=(Aop.T@r).astype(dtype,copy=False)
    s=right_map.transpose(g)
    p=s.copy()
    gamma=dtype(np.dot(s,s))
    bnorm=float(np.linalg.norm(b.astype(np.float64)))
    rows=[]; status='iteration_cap'
    checkpoints={8,16,32,64}
    for it in range(1,cap+1):
        Sp=right_map.apply(p)
        if not np.all(np.isfinite(Sp)):
            status='nonfinite_preconditioned_direction';break
        q=(Aop@Sp).astype(dtype,copy=False)
        if not np.all(np.isfinite(q)):
            status='nonfinite_operator_action';break
        den=dtype(np.dot(q,q))
        if not np.isfinite(den) or den<=0 or not np.isfinite(gamma) or gamma<=0:
            status='nonpositive_or_nonfinite_cgls_denominator';break
        alpha=dtype(gamma/den)
        if not np.isfinite(alpha):
            status='nonfinite_step_length';break
        y=(y+alpha*p).astype(dtype,copy=False)
        r=(r-alpha*q).astype(dtype,copy=False)
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(r)):
            status='nonfinite_iterate';break
        if it in checkpoints or it==cap:
            x=right_map.apply(y)
            fresh_r=(b-(Aop@x)).astype(dtype,copy=False)
            true_norm=float(np.linalg.norm(fresh_r.astype(np.float64)))
            rows.append({'iteration':it,'recursive_residual_l2':float(np.linalg.norm(r.astype(np.float64))),
                         'true_residual_l2':true_norm,'relative_true_residual':true_norm/bnorm if bnorm else None,
                         'mapped_direction_l2':float(np.linalg.norm(x.astype(np.float64))),
                         'b_dot_Ax':float(np.dot(b.astype(np.float64),(Aop@x).astype(np.float64)))})
        gnew=(Aop.T@r).astype(dtype,copy=False)
        snew=right_map.transpose(gnew)
        if not np.all(np.isfinite(snew)):
            status='nonfinite_preconditioned_normal_residual';break
        gamma_new=dtype(np.dot(snew,snew))
        if not np.isfinite(gamma_new) or gamma_new<0:
            status='nonfinite_normal_residual';break
        if gamma_new==0:
            status='normal_residual_zero';y=(y+0*p).astype(dtype,copy=False);break
        beta=dtype(gamma_new/gamma)
        p=(snew+beta*p).astype(dtype,copy=False)
        if not np.all(np.isfinite(p)):
            status='nonfinite_search_direction';break
        s=snew;gamma=gamma_new
    x=right_map.apply(y)
    rtrue=(b-(Aop@x)).astype(dtype,copy=False)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(rtrue)):
        status='nonfinite_final_candidate_or_true_residual'
    true_norm=float(np.linalg.norm(rtrue.astype(np.float64)))
    normal=(Aop.T@rtrue).astype(dtype,copy=False)
    return {'status':status,'iterations':it if 'it' in locals() else 0,
            'b_l2':bnorm,'final_true_residual_l2':true_norm,
            'final_relative_true_residual':true_norm/bnorm if bnorm else None,
            'final_normal_residual_l2':float(np.linalg.norm(normal.astype(np.float64))),
            'mapped_direction_l2':float(np.linalg.norm(x.astype(np.float64))),
            'b_dot_Ax':float(np.dot(b.astype(np.float64),(Aop@x).astype(np.float64))),
            'checkpoints':rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('cgls-results.json'))
    args=parser.parse_args()
    import numpy as np
    try:
        from threadpoolctl import threadpool_limits
        lim=threadpool_limits(limits=1)
    except Exception: lim=None
    audit=load_audit()
    state_reader=audit.load_state_reader(); regional_reader,graph_ref=audit.load_capture_readers()
    state_doc=regional_reader.v1._strict_json(REGIONAL)
    topology,free_mask,potential_mask,graph_meta=audit.parse_graph_input(GRAPH_INPUT,GRAPH_BINDINGS)
    capture,view=graph_ref.load_base_view(REGIONAL,environment=0,topology_bytes=topology,
        free_mask_bytes=free_mask,potential_mask_bytes=potential_mask,request_actual_consumed_view=False)
    # Capture is after the tail-latch transition. Clone only to evaluate the
    # underlying unlatched base selector formula as a conditioning candidate.
    if not capture.completed_tail_latch:
        raise ExperimentError('expected completed-tail latch for this retained capture')
    c_active=replace(capture,completed_tail_latch=False)
    state_path=CASE/state_doc['candidate_state_binding']['manifest_file']
    state=audit.read_state(state_path,state_reader)
    A,np,free_nodes,dofs,dof_map=audit.load_sparse_A(CSRPKG/'A_free_csr.npz',CSRPKG/'manifest.json',free_mask)
    bfull=np.frombuffer(state['arrays']['base_residual_rhs'],dtype='<f4').reshape(4,882,4)[0,:,:3].astype(np.float64)
    b=bfull[np.asarray(free_nodes)].reshape(-1)
    if A.shape!=(len(b),len(b)) or not np.all(np.isfinite(b)):
        raise ExperimentError('captured RHS and free CSR shape/nonfinite mismatch')
    # Validate fast base S/S^T implementations against source151's pure-Python
    # equations on two deterministic directions before running recurrence.
    rng=np.random.default_rng(464151)
    refchecks=[]
    maps64=GraphBaseMap(c_active,np.float64)
    for seed in [0,1]:
        v=rng.standard_normal(2646);v[~np.repeat(np.asarray(c_active.free_mask),3)]=0.0
        fastS=maps64.apply_full(v); refS=np.asarray(graph_ref.apply_base_S(c_active,v))
        fastT=maps64.apply_transpose_full(v); refT=np.asarray(graph_ref.apply_base_S_transpose(c_active,v))
        refchecks.append({'direction':seed,'S_max_abs':float(np.max(np.abs(fastS-refS))),
                          'S_rel_l2':float(np.linalg.norm(fastS-refS)/max(np.linalg.norm(refS),1e-300)),
                          'ST_max_abs':float(np.max(np.abs(fastT-refT))),
                          'ST_rel_l2':float(np.linalg.norm(fastT-refT)/max(np.linalg.norm(refT),1e-300))})
    u=rng.standard_normal(2646);v=rng.standard_normal(2646)
    u[~np.repeat(np.asarray(c_active.free_mask),3)]=0.0
    v[~np.repeat(np.asarray(c_active.free_mask),3)]=0.0
    sv=maps64.apply_full(v); stu=maps64.apply_transpose_full(u)
    transpose_dot={'u_dot_Sv':float(np.dot(u,sv)),'STu_dot_v':float(np.dot(stu,v)),
                   'absolute_gap':float(abs(np.dot(u,sv)-np.dot(stu,v))),
                   'relative_gap':float(abs(np.dot(u,sv)-np.dot(stu,v))/max(abs(np.dot(u,sv)),abs(np.dot(stu,v)),1e-300))}
    strategies={}
    # Preserve host warning evidence. NumPy 2.2.5 on this host emits BLAS
    # RuntimeWarnings for a benign finite 2646x26 matmul; we record every
    # warning and independently check every CGLS iterate/action for finiteness.
    start=time.time()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always',RuntimeWarning)
        for precision,dtype in [('fp64',np.float64),('fp32_cpu_arithmetic',np.float32)]:
            result_by_map={}
            for kind in ['identity','block3','base_graph']:
                m=RightMap(kind,free_nodes,A,c_active,dtype)
                result_by_map[kind]=cgls(A,b,m,64,dtype)
            strategies[precision]=result_by_map
        warning_summary=[]
        counts=Counter((type(w.message).__name__,str(w.message)) for w in caught_warnings)
        for (category,message),count in sorted(counts.items()):
            warning_summary.append({'category':category,'message':message,'count':count})
    elapsed=time.time()-start
    case_report=json.loads((BASE/'analysis/native464-regional-capture-audit-003/native464-audit-v4.json').read_text())
    out={
      'schema':'numi.cardboard.native464_cgls_preconditioner_cpu_experiment.v1',
      'scope':'offline descriptive CGLS on captured env0 A,b; maps are identity, exact captured A_nn inverse 3x3 blocks, and algebraic unlatched source-151 BASE S/S^T only',
      'not_claimed':['native CGLS trajectory or FP32 Metal action parity','that selector-5 augmented map was consumed','solver convergence/acceptance or physical/shipping qualification','runtime speedup or complete-coupon improvement'],
      'lineage':{'464_receipt_sha256':hfile(CASE/'receipt.json'),'464_regional_manifest_sha256':hfile(REGIONAL),
                 '464_regional_payload_sha256':hfile(CASE/'base-cgls.bin.regional.bin'),
                 '464_state_manifest_sha256':hfile(state_path),'464_state_payload_sha256':hfile(CASE/'base-cgls.bin.state.bin'),
                 '464_base_cgls_sha256':hfile(CASE/'base-cgls.bin'),
                 '419_operator_capture_sha256':hfile(OPCAP),'419_A_free_csr_sha256':hfile(CSRPKG/'A_free_csr.npz'),
                 'source151_runtime_sha256':hfile(BASE/'snapshots/source-cgls-regional-state-capture-151/matter/src/runtime.mm'),
                 'source151_graph_kernel_sha256':hfile(BASE/'snapshots/source-cgls-regional-state-capture-151/matter/src/metal/base_cgls_runtime.metalinc'),
                 'base_capture_view_label':view['mode_view'],'actual_selected_view_admissible':view['actual_selected_view_admissible'],
                 'completed_tail_latch_captured':capture.completed_tail_latch,
                 'benchmark_map_latch_override':'reference clone only: cleared completed-tail output latch to evaluate source-defined underlying base S/S^T formulas; this is not proof of what FGMRES consumed'},
      'dimensions':{'environment':0,'environment_count':2,'nodes':882,'free_nodes':len(free_nodes),'fixed_nodes':882-len(free_nodes),'unknowns':int(A.shape[0]),'csr_nnz':int(A.nnz),'rhs_l2':float(np.linalg.norm(b)),'base_mode_count':capture.mode_count,'patch_count':len(capture.patches)},
      'patch_matrix_gate_context':{'per_entry_atol':case_report['operator_comparison']['patch_matrices']['atol'],'within_atol':case_report['operator_comparison']['patch_matrices']['within_atol'],'max_entry_error':case_report['operator_comparison']['patch_matrices']['free_free_max_abs']},
      'fast_map_vs_source_reference':refchecks,
      'source_map_transpose_dot_identity':transpose_dot,
      'captured_runtime_warnings_during_six_runs':warning_summary,
      'finite_result_guard':'every CGLS preconditioned direction, A action, alpha, y/r iterate, preconditioned normal residual, search direction, and final candidate/residual is checked finite; nonfinite values stop the case with a named status',
      'cgls_cap':64,'precision_methods':strategies,'elapsed_cpu_seconds_all_six_runs':elapsed,
      'method_notes':[
        'CGLS solves min_y ||A S y-b||_2 with recurrence r=b-A S y, s=S^T A^T r; physical correction reported is x=S y.',
        'block3 is inverse of captured 419 A diagonal 3x3 free-node blocks; the transpose action uses inverse-block transpose, not an SPD assumption.',
        'base_graph is the source-151 unaugmented captured base map with captured graph patch factors and base RAP. Its underlying formula is evaluated with the graph reference latch cleared on a clone; the actual consuming restart-augmented view is unadmitted.',
        'FP32 CPU arithmetic is an approximate recurrence with scipy/NumPy CPU operation order; it is not shader emulation or native parity.',
        'CGLS uses a fixed maximum of 64 with no solver tolerance or physics/acceptance change. Breakdown/nonfinite states are reported, not converted into passes.'
      ]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(args.output.resolve()),'elapsed_cpu_seconds':elapsed,'refchecks':refchecks,'final':{p:{k:v['final_relative_true_residual'] for k,v in d.items()} for p,d in strategies.items()}},indent=2))
if __name__=='__main__':main()
