#!/usr/bin/env python3
"""Offline basis-equilibration study for same-Newton q augmentation, capture 304.
No production source, physical operator, tolerance, or GPU path is modified.
"""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'solver/rationalized-paper-control2-newton63-capture-304'
AN=ROOT/'analysis/rationalized-paper-control2-newton63-operator-304'
RAP=RUN/'output/rejected-conditioning-000003'
LAY=ROOT/'analysis/regional-preconditioner-binding-304-002'
OUT=ROOT/'analysis/conditioning-equilibrated-augmentation-304-005'
A64=sparse.load_npz(AN/'operator-report.operator.npz').tocsr().astype(np.float64)
with np.load(AN/'operator-report.vectors.npz',allow_pickle=False) as z: vec={k:np.array(z[k]) for k in z.files}
manifest=json.load(open(RAP/'manifest.json')); E,N,M=manifest['environment_count'],manifest['fem_node_count'],manifest['mode_count']
assert (E,N,M)==(2,882,26)
free=vec['free_node_ids'].astype(np.int64); n=A64.shape[0]
Praw=np.fromfile(RAP/'effective_P.bin',dtype='<f4').reshape(E,M,N,4)[0][:,free,:3].astype(np.float64).transpose(1,2,0).reshape(n,M)
b=np.asarray(vec['native_rhs_free_impulse_Ns'],dtype=np.float64); xnative=np.asarray(vec['direction'],dtype=np.float64); bnorm=float(np.linalg.norm(b))
assert A64.shape==(2538,2538) and b.shape==(n,) and np.isfinite(A64.data).all()
# Reconstruct source091 balanced-Schwarz topology and float32 patch factors exactly as previous offline runs.
text=__import__('subprocess').check_output([str(LAY/'layout'),'882','1920','2',str(LAY/'topology.u32'),str(LAY/'free_mask.u8')],text=True)
lines=text.splitlines(); layout_header=lines[0]; desc=[]
for ln in lines[1:]:
    if ln=='nodes': break
    desc.append(tuple(map(int,ln.split())))
all_nodes=list(map(int,lines[2+len(desc)].split())); covline=3+len(desc)
coverage=np.asarray(list(map(int,lines[covline+1].split())),dtype=np.int64)
patches=[np.asarray(all_nodes[o:o+c],dtype=np.int64) for o,c,_,_ in desc]
freeidx={int(v):i for i,v in enumerate(free)}; patchf=[]
for nodes in patches:
    fn=[int(v) for v in nodes if int(v) in freeidx]
    if not fn: patchf.append((np.empty(0,dtype=np.int64),None,None)); continue
    dofs=np.concatenate([np.arange(3*freeidx[v],3*freeidx[v]+3,dtype=np.int64) for v in fn])
    local=A64[dofs][:,dofs].toarray().astype(np.float32)
    fact,piv=linalg.lu_factor(local,overwrite_a=True,check_finite=True)
    patchf.append((dofs,fact,piv))

def metrics(R, tag):
    R64=np.asarray(R,dtype=np.float64); s=np.linalg.svd(R64,compute_uv=False)
    eig=np.linalg.eigvalsh((R64+R64.T)*0.5)
    r32=R64.astype(np.float32); sf=np.linalg.svd(r32.astype(np.float64),compute_uv=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fact,piv=linalg.lu_factor(r32,overwrite_a=False,check_finite=True)
    pv=np.abs(np.diag(fact)); det={
      'dimension':int(R64.shape[0]),'rank_fp64_rtol1e-7':int(np.sum(s>s[0]*1e-7)),
      'rank_fp32stored_rtol1e-7':int(np.sum(sf>sf[0]*1e-7)),
      'singular_min_fp64':float(s[-1]),'singular_max_fp64':float(s[0]),
      'condition2_fp64':float(s[0]/s[-1]),'singular_min_fp32stored':float(sf[-1]),
      'condition2_fp32stored':float(sf[0]/sf[-1]),
      'symmetric_min_eigenvalue_fp64':float(eig[0]),'symmetric_max_eigenvalue_fp64':float(eig[-1]),
      'lu_min_abs_pivot_fp32':float(pv.min()),'lu_max_abs_pivot_fp32':float(pv.max()),
      'lu_warning_count':len(caught),'lu_warning_text':[str(x.message) for x in caught]}
    return det, r32, fact, piv

def build_basis(q,scale,precision):
    P=np.column_stack((Praw,q*scale))
    if precision=='fp64':
        AP=A64@P; R=P.T@AP
    else:
        A32=A64.astype(np.float32); P32=P.astype(np.float32)
        AP32=A32@P32; R=P32.T@AP32
        P=P32.astype(np.float64); AP=AP32.astype(np.float64); R=R.astype(np.float64)
    return P,AP,R

def residual(x):
    r=b-A64@x; nr=float(np.linalg.norm(r))
    return {'norm_Ns':nr,'relative_rhs':nr/bnorm,'target_absolute_for_relative_1e-6_Ns':bnorm*1e-6}

def make_apply(P,AP,R,precision):
    md,r32,fact32,piv32=metrics(R,'')
    if precision=='fp64':
        fact,piv=linalg.lu_factor(np.asarray(R,dtype=np.float64),check_finite=True)
    else: fact,piv=fact32,piv32
    def coarse(rhs):
        if precision=='fp64': return linalg.lu_solve((fact,piv),np.asarray(rhs,dtype=np.float64),check_finite=False)
        # Match the FP32 coarse input, factor and solve path; widen result only after solve.
        return linalg.lu_solve((fact,piv),np.asarray(rhs,dtype=np.float32),check_finite=False).astype(np.float64)
    def apply(r):
        if precision=='fp64': c=coarse(P.T@r)
        else: c=coarse((P.astype(np.float32).T@np.asarray(r,dtype=np.float32)))
        out=P@c
        fine=np.asarray(r,dtype=np.float64)-AP@c
        graph=np.zeros(n,dtype=np.float64)
        for nodes,(dofs,pf,pp) in zip(patches,patchf):
            if not len(dofs): continue
            local=linalg.lu_solve((pf,pp),fine[dofs].astype(np.float32),check_finite=False).astype(np.float64)
            for j,d in enumerate(dofs): graph[d]+=local[j]/coverage[int(nodes[j//3])]
        return out+graph
    return apply,md

def gmres_restart(P,AP,R,x0,steps,precision):
    apply,_=make_apply(P,AP,R,precision)
    x=np.asarray(x0,dtype=np.float64).copy(); r=b-A64@x; beta=float(np.linalg.norm(r))
    if beta==0: return x,[]
    V=np.zeros((n,steps+1)); Z=np.zeros((n,steps)); H=np.zeros((steps+1,steps)); V[:,0]=r/beta; rows=[]
    for j in range(steps):
        Z[:,j]=apply(V[:,j]); w=A64@Z[:,j]
        for _ in range(2):
            for i in range(j+1):
                h=float(V[:,i]@w); H[i,j]+=h; w-=h*V[:,i]
        H[j+1,j]=np.linalg.norm(w)
        if H[j+1,j]>0: V[:,j+1]=w/H[j+1,j]
        e=np.zeros(j+2);e[0]=beta; y,*_=np.linalg.lstsq(H[:j+2,:j+1],e,rcond=None)
        xk=x+Z[:,:j+1]@y; rr=b-A64@xk
        rows.append({'within_restart':j+1,'absolute_true_residual_Ns':float(np.linalg.norm(rr)),'relative_true_residual':float(np.linalg.norm(rr)/bnorm)})
    return xk,rows

# Base 26-mode same-Newton restart, then derive the online-feasible q exactly as earlier study.
def base_rap():
    return Praw.T@(A64@Praw)
Rbase=base_rap(); base_svd=np.linalg.svd(Rbase,compute_uv=False); base_diag=np.abs(np.diag(Rbase))
Q,_=np.linalg.qr(Praw,mode='reduced')
# Direct and native q from same failed solve. Q projection removes existing coarse subspace.
def orth(v):
    q=v-Q@(Q.T@v); qn=float(np.linalg.norm(q)); return q/qn,qn
q_native,norm_native=orth(xnative)
xdirect=spsolve(A64,b); q_direct,norm_direct=orth(np.asarray(xdirect))
# Online restart q from the first 16 base-preconditioned Arnoldi steps.
APbase=A64@Praw
x16,history16=gmres_restart(Praw,APbase,Rbase,np.zeros(n),16,'fp32')
q_restart,norm_restart=orth(x16)
# A-orthogonalize the same online q against the existing coarse space. This is a
# basis change within span([P,q]); it leaves physical A and the exact coarse space unchanged.
Rbase32=Praw.astype(np.float32).T@(A64.astype(np.float32)@Praw.astype(np.float32))
qAraw64=q_restart-Praw@np.linalg.solve(Rbase,Praw.T@(A64@q_restart))
qA64_norm=float(np.linalg.norm(qAraw64)); qA64=qAraw64/qA64_norm
P32=Praw.astype(np.float32); A32=A64.astype(np.float32); q32=q_restart.astype(np.float32)
qA_rhs32=P32.T@(A32@q32)
qA_fact32,qA_piv32=linalg.lu_factor(Rbase32,check_finite=True)
qA_coeff32=linalg.lu_solve((qA_fact32,qA_piv32),qA_rhs32,check_finite=True).astype(np.float32)
qAraw32=q32-P32@qA_coeff32; qA32_norm=float(np.linalg.norm(qAraw32.astype(np.float64)))
qA32=(qAraw32/np.float32(qA32_norm)).astype(np.float32).astype(np.float64)
qA_build_meta={'q_Aorthogonalized_fp64_raw_norm':qA64_norm,'q_Aorthogonalized_fp32_raw_norm':qA32_norm,
 'qA64_PtAq_maxabs':float(np.max(np.abs(Praw.T@(A64@qA64)))),
 'qA64_transpose_cross_qTAP_maxabs':float(np.max(np.abs(qA64@(A64@Praw)))),
 'qA32_PtAq_maxabs':float(np.max(np.abs(P32.T@(A32@qA32.astype(np.float32))))),
 'qA32_transpose_cross_qTAP_maxabs':float(np.max(np.abs(qA32@(A32@P32.astype(np.float64))))),
 'qA64_Aenergy':float(qA64@(A64@qA64)),'qA32_Aenergy':float(qA32@(A32@qA32.astype(np.float32)).astype(np.float64))}
# Compute Schur complement in FP64 and FP32, using same base coarse block.
def schur_report(q):
    ap=A64@q; rqp=Praw.T@ap; rpq=q@(A64@Praw); rqq=float(q@ap)
    schur64=float(rqq-rqp@np.linalg.solve(Rbase,rpq))
    A32=A64.astype(np.float32); P32=Praw.astype(np.float32); q32=q.astype(np.float32)
    AP32=A32@P32; aq32=A32@q32; Rbase32=P32.T@AP32
    rqp32=P32.T@aq32; rpq32=q32@AP32; rqq32=float(q32@aq32)
    schur32=float(np.float64(rqq32)-np.float64(rqp32)@np.linalg.solve(Rbase32.astype(np.float64),rpq32.astype(np.float64)))
    return {'raw_qTAq_fp64':rqq,'raw_qTAq_fp32':rqq32,'schur_pivot_fp64':schur64,'schur_pivot_fp32_roundtrip':schur32,
            'schur_sign_agrees':bool(np.sign(schur64)==np.sign(schur32)),
            'schur_abs_diff':abs(schur64-schur32),'base_R_diag_abs_median':float(np.median(base_diag)),
            'base_R_diag_abs_mean':float(np.mean(base_diag)),
            'equilibration_scale_to_median':float(np.sqrt(np.median(base_diag)/abs(schur64))) if schur64 else None,
            'equilibration_scale_to_mean':float(np.sqrt(np.mean(base_diag)/abs(schur64))) if schur64 else None,
            'float32_product_qTAq_abs_diff':abs(float(rqq32)-rqq)}

qsets={'online_restart16':(q_restart,norm_restart),
       'online_restart16_Aorth_fp64':(qA64,qA64_norm),
       'online_restart16_Aorth_fp32':(qA32,qA32_norm),
       'native_captured_retrospective':(q_native,norm_native),
       'direct_solution_retrospective':(q_direct,norm_direct)}
results={}
scales_to_test=['unit','median','mean']
for name,(q,qn) in qsets.items():
    sr=schur_report(q)
    scales={'unit':1.0,'schur_median':sr['equilibration_scale_to_median'],'schur_mean':sr['equilibration_scale_to_mean'],'raw_diagonal_median':float(np.sqrt(sr['base_R_diag_abs_median']/abs(sr['raw_qTAq_fp64'])))}
    entry={'projected_q_norm_before_unit':qn,'schur':sr,'scales':{}}
    for sn,scale in scales.items():
        if scale is None or not np.isfinite(scale): continue
        for prec in ('fp64','fp32'):
            P,AP,R=build_basis(q,float(scale),prec)
            met,R32,fact32,piv32=metrics(R,prec)
            rap_exact=P.T@(A64@P)
            rap_cast=rap_exact.astype(np.float32).astype(np.float64)
            Rarr=np.asarray(R,dtype=np.float64)
            schur_built=float(Rarr[-1,-1]-Rarr[-1,:-1]@np.linalg.solve(Rarr[:-1,:-1],Rarr[:-1,-1]))
            schur_ref=float(rap_exact[-1,-1]-rap_exact[-1,:-1]@np.linalg.solve(rap_exact[:-1,:-1],rap_exact[:-1,-1]))
            scaleentry={'basis_column_scale':float(scale),'q_scaled_l2':float(np.linalg.norm(P[:,-1])),
                        'basis_column_max_abs':float(np.max(np.abs(P[:,-1]))),'RAP_metrics':met,
                        'built_RAP_schur_pivot':schur_built,'same_basis_FP64_RAP_schur_pivot':schur_ref,
                        'built_RAP_schur_sign_agrees_with_FP64':bool(np.sign(schur_built)==np.sign(schur_ref)),
                        'RAP_maxabs_difference_vs_same_basis_FP64':float(np.max(np.abs(R-rap_exact))),
                        'RAP_frobenius_difference_vs_same_basis_FP64':float(np.linalg.norm(R-rap_exact)),
                        'RAP_fp32_storage_only_metrics':metrics(rap_cast,'cast')[0]}
            if name.startswith('online_restart16'):
                # Continue same Newton from the online first-restart iterate with augmented preconditioner.
                x=x16.copy(); prev=residual(x); cycle=[]
                # Rebuild augmented R from exact source-precision path and run 3 more restart-16 cycles (to total64).
                for k in range(1,8):
                    x,rows=gmres_restart(P,AP,R,x,16,prec)
                    res=residual(x); cycle.append({'total_iterations':16+16*k,'absolute_true_residual_Ns':res['norm_Ns'],'relative_true_residual':res['relative_rhs'],'within_restart':rows})
                scaleentry.update({'online_start_after_base_restart16':prev,'online_after_restart16_cycles':cycle,
                                   'R32_abs_gate_for_1e-6':bnorm*1e-6})
            entry['scales'][sn+'_'+prec]=scaleentry
    results[name]=entry
report={
 'schema':'numi.cardboard.conditioning-basis-equilibration-304.v2',
 'binding':{'source_run':str(RUN),'prediction_sha256':hashlib.sha256((RUN/'prediction.json').read_bytes()).hexdigest(),
            'receipt_sha256':hashlib.sha256((RUN/'receipt.json').read_bytes()).hexdigest(),
            'operator_capture_sha256':hashlib.sha256((RUN/'operator-capture').read_bytes()).hexdigest(),
            'operator_report_sha256':hashlib.sha256((AN/'operator-report.json').read_bytes()).hexdigest(),
            'regional_manifest_sha256':hashlib.sha256((RAP/'manifest.json').read_bytes()).hexdigest()},
 'operator':{'shape':list(A64.shape),'nnz':int(A64.nnz),'rhs_norm_Ns':bnorm,'relative_1e-6_absolute_gate_Ns':bnorm*1e-6,
             'base26_RAP_diag_abs_min_median_mean_max':[float(base_diag.min()),float(np.median(base_diag)),float(np.mean(base_diag)),float(base_diag.max())],
             'base26_RAP_condition2':float(base_svd[0]/base_svd[-1]),
             'A_relative_frobenius_asymmetry':float(np.linalg.norm((A64-A64.T).data)/np.linalg.norm(A64.data)),
             'A_max_abs_asymmetry':float(np.max(np.abs((A64-A64.T).data))),
             'graph_layout_header':layout_header,
             'base_restart16_relative_residual':float(np.linalg.norm(b-A64@x16)/bnorm),
             'online_q_basis_construction':qA_build_meta},
 'augmentation':results,
 'limits':['A is the same captured offline FP64 operator; FP32 variant casts A/P and uses NumPy/SciPy float32 products, not Metal arithmetic parity.',
           'FP64 RAP is an offline upper-precision control; it does not change the production operator.',
           'Only the online_first_restart q and its within-span A-orthogonalized variants are available at iteration16; captured-native and direct-solution q are retrospective controls.',
           'A-orthogonalization and column scaling change only the coarse basis representation; all RAPs use the same captured A and the augmented column remains in span([P,q]).',
           'For an A-orthogonalized unit q, q^T A q equals the small Schur complement; for the Euclidean q, raw q^T A q is not that Schur pivot.',
           'Balanced-Schwarz patch factors are reconstructed from captured A and are stored/solved FP32 offline, not captured native factors.',
           'No physical operator regularization, tolerance changes, rank-threshold changes, native qualification, or accepted physical step is claimed.']}
out=OUT/'report-v2.json'; out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(out,hashlib.sha256(out.read_bytes()).hexdigest())
for name,e in results.items():
  print('\n',name,'q raw norm',e['projected_q_norm_before_unit'],'schur',e['schur'])
  for k,v in e['scales'].items():
    print(k,'scale',v['basis_column_scale'],'rank',v['RAP_metrics']['rank_fp64_rtol1e-7'],v['RAP_metrics']['rank_fp32stored_rtol1e-7'],'cond',v['RAP_metrics']['condition2_fp64'],v['RAP_metrics']['condition2_fp32stored'],'symmin',v['RAP_metrics']['symmetric_min_eigenvalue_fp64'],'pivotmin',v['RAP_metrics']['lu_min_abs_pivot_fp32'])
    if 'online_after_restart16_cycles' in v:
      print(' residuals',[(x['total_iterations'],x['absolute_true_residual_Ns'],x['relative_true_residual']) for x in v['online_after_restart16_cycles']])
