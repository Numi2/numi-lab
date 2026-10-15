#!/usr/bin/env python3
"""Offline same-Newton coarse-mode augmentation candidates for exact capture 304.
No native source/runtime or physics is modified. Graph patch factors are reconstructed from A.
"""
from pathlib import Path
import json, subprocess, hashlib, time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'solver/rationalized-paper-control2-newton63-capture-304'
AN=ROOT/'analysis/rationalized-paper-control2-newton63-operator-304'
RAP=RUN/'output/rejected-conditioning-000003'
LAY=ROOT/'analysis/regional-preconditioner-binding-304-002'
OUT=ROOT/'analysis/conditioning-augmentation-304-004'
A=sparse.load_npz(AN/'operator-report.operator.npz').tocsr()
with np.load(AN/'operator-report.vectors.npz',allow_pickle=False) as z: vec={k:np.array(z[k]) for k in z.files}
manifest=json.load(open(RAP/'manifest.json')); E,N,M=manifest['environment_count'],manifest['fem_node_count'],manifest['mode_count']
assert (E,N,M)==(2,882,26)
free=vec['free_node_ids'].astype(np.int64); n=A.shape[0]
Praw=np.fromfile(RAP/'effective_P.bin',dtype='<f4').reshape(E,M,N,4)[0][:,free,:3].astype(np.float64).transpose(1,2,0).reshape(n,M)
APnative=np.fromfile(RAP/'AP.bin',dtype='<f4').reshape(E,M,N,4)[0][:,free,:3].astype(np.float64).transpose(1,2,0).reshape(n,M)
b=np.asarray(vec['native_rhs_free_impulse_Ns'],dtype=np.float64); xnative=np.asarray(vec['direction'],dtype=np.float64); bnorm=np.linalg.norm(b)
assert A.shape==(2538,2538) and b.shape==(n,) and np.isfinite(A.data).all()
# Use a rank-revealing QR for robust projection. Keep original P columns in P_aug to
# preserve the production authored basis; q is residualized against its orthonormal Q.
Qp,Rp=np.linalg.qr(Praw,mode='reduced')
p_s=np.linalg.svd(Rp,compute_uv=False)
assert len(p_s)==M and p_s[-1]>1e-10*p_s[0]
def orthogonal_extra(v):
    q=np.asarray(v,dtype=np.float64)-Qp@(Qp.T@v)
    raw=float(np.linalg.norm(q))
    if not np.isfinite(raw) or raw<=1e-10*max(float(np.linalg.norm(v)),1e-300):
        raise ValueError(f'added mode is dependent/ambiguous: qnorm={raw}, vnorm={np.linalg.norm(v)}')
    q=q/raw
    defect=float(np.linalg.norm(Praw.T@q))
    return q,{'projected_norm_before_normalize':raw,'source_vector_norm':float(np.linalg.norm(v)),'normalized_q_norm':float(np.linalg.norm(q)),'P_transpose_q_l2':defect,'P_condition_2':float(p_s[0]/p_s[-1])}
q_native,q_native_meta=orthogonal_extra(xnative)
# Retrospective direct solve is strictly an upper control, unavailable before this same solve.
xdirect=spsolve(A,b)
q_direct,q_direct_meta=orthogonal_extra(xdirect)
# Captured same-step RAP data are checked against the independently formed exact A action.
APbase=A@Praw
assert np.linalg.norm(APbase-APnative)/max(np.linalg.norm(APnative),1e-300)<2e-6
# Recreate exact source091 bounded graph layout.
text=subprocess.check_output([str(LAY/'layout'),'882','1920','2',str(LAY/'topology.u32'),str(LAY/'free_mask.u8')],text=True)
lines=text.splitlines(); layout_header=lines[0]; desc=[]
for ln in lines[1:]:
    if ln=='nodes': break
    desc.append(tuple(map(int,ln.split())))
assert lines[1+len(desc)]=='nodes'
all_nodes=list(map(int,lines[2+len(desc)].split())); covline=3+len(desc); assert lines[covline]=='coverage'
coverage=np.asarray(list(map(int,lines[covline+1].split())),dtype=np.int64)
patches=[np.asarray(all_nodes[o:o+c],dtype=np.int64) for o,c,_,_ in desc]
freeidx={int(a):i for i,a in enumerate(free)}
patchf=[]
for nodes in patches:
    fn=[int(a) for a in nodes if int(a) in freeidx]
    if not fn: patchf.append((np.empty(0,dtype=np.int64),None,None)); continue
    dofs=np.concatenate([np.arange(3*freeidx[a],3*freeidx[a]+3,dtype=np.int64) for a in fn])
    mat=A[dofs][:,dofs].toarray().astype(np.float32)
    fact,piv=linalg.lu_factor(mat,overwrite_a=True,check_finite=True)
    patchf.append((dofs,fact,piv))
def build_coarse(P):
    AP=A@P
    R=P.T@AP
    # Mimic native single-precision coarse storage/factorization. A and projections remain
    # captured FP64 for offline true residual reporting; this is not byte-parity with Metal.
    rf=R.astype(np.float32)
    fact,piv=linalg.lu_factor(rf,overwrite_a=True,check_finite=True)
    sing=np.linalg.svd(R,compute_uv=False)
    singf=np.linalg.svd(rf.astype(np.float64),compute_uv=False)
    return AP,fact,piv,R,{
      'dimension':int(P.shape[1]),'rank_fp64':int(np.linalg.matrix_rank(R)),
      'rank_float32_svd_rtol1e-7':int(np.sum(singf>singf[0]*1e-7)),
      'singular_min_fp64':float(sing[-1]),'singular_max_fp64':float(sing[0]),
      'condition_2_fp64':float(sing[0]/sing[-1]),
      'singular_min_float32stored':float(singf[-1]),'condition_2_float32stored':float(singf[0]/singf[-1]),
      'symmetric_part_min_eigenvalue':float(np.linalg.eigvalsh((R+R.T)*.5)[0]),
      'symmetric_part_max_eigenvalue':float(np.linalg.eigvalsh((R+R.T)*.5)[-1]),
      'pivot_abs_min_float32LU':float(np.min(np.abs(np.diag(fact)))),
      'pivot_abs_max_float32LU':float(np.max(np.abs(np.diag(fact)))),
    }
def coarse_solve(fact,piv,rhs):
    return linalg.lu_solve((fact,piv),np.asarray(rhs,dtype=np.float32),check_finite=False).astype(np.float64)
def apply(P,AP,fact,piv,r):
    c=coarse_solve(fact,piv,P.T@r)
    out=P@c
    fine=np.asarray(r,dtype=np.float64)-AP@c
    graph=np.zeros(n,dtype=np.float64)
    for nodes,(dofs,pf,pp) in zip(patches,patchf):
        if not len(dofs): continue
        local=linalg.lu_solve((pf,pp),fine[dofs].astype(np.float32),check_finite=False).astype(np.float64)
        for j,d in enumerate(dofs): graph[d]+=local[j]/coverage[int(nodes[j//3])]
    return out+graph
def gmres(P,AP,fact,piv,x0,steps=16):
    x=np.asarray(x0,dtype=np.float64).copy(); r=b-A@x; beta=float(np.linalg.norm(r)); V=np.zeros((n,steps+1)); Z=np.zeros((n,steps)); H=np.zeros((steps+1,steps)); V[:,0]=r/beta; records=[]
    for j in range(steps):
        Z[:,j]=apply(P,AP,fact,piv,V[:,j]); w=A@Z[:,j]
        for _ in range(2):
            for i in range(j+1):
                h=float(V[:,i]@w); H[i,j]+=h; w-=h*V[:,i]
        H[j+1,j]=np.linalg.norm(w)
        if H[j+1,j]>0: V[:,j+1]=w/H[j+1,j]
        e=np.zeros(j+2);e[0]=beta;y,*_=np.linalg.lstsq(H[:j+2,:j+1],e,rcond=None)
        xk=x+Z[:,:j+1]@y; rr=b-A@xk
        records.append({'step':j+1,'residual_Ns':float(np.linalg.norm(rr)),'relative_residual':float(np.linalg.norm(rr)/bnorm)})
    return xk,records
def residual(x):
    r=b-A@x
    return {'norm_Ns':float(np.linalg.norm(r)),'relative_rhs':float(np.linalg.norm(r)/bnorm)}

def evaluate_aug(name,q,qmeta,mode):
    P=np.column_stack((Praw,q)); AP,fact,piv,R,rmeta=build_coarse(P)
    # Projection and rank checks for new mode, in both FP64 and actual stored-FP32 range.
    gram=P.T@P
    rmeta.update({'P_aug_gram_rank':int(np.linalg.matrix_rank(gram)),'P_aug_gram_condition_2':float(np.linalg.cond(gram)),'extra_mode':qmeta})
    coarse_from_zero=coarse_solve(fact,piv,P.T@b)
    x0=P@coarse_from_zero
    d={'mode':mode,'RAP':rmeta,'coarse_only_from_zero':residual(x0),'native_x_before':residual(xnative),'native_extra_q_cosine_direct_extra_q':float(q@q_direct),'native_extra_q_cosine_native_x':float(q@xnative)}
    # For captured native q, 32-step/64-step zero-start replay is retrospective only.
    if mode!='online_first_restart':
        x16,h16=gmres(P,AP,fact,piv,np.zeros(n),16)
        x32,h32=gmres(P,AP,fact,piv,x16,16)
        x64,h64=gmres(P,AP,fact,piv,x32,32)
        d['from_zero_restart16']={'coarse_only':residual(x0),'R16':residual(x16),'R32':residual(x32),'R64':residual(x64),'history_16':h16,'history_32':h32,'history_64':h64}
        xn0=xnative+P@coarse_solve(fact,piv,P.T@(b-A@xnative))
        xn16,hn16=gmres(P,AP,fact,piv,xn0,16)
        xn32,hn32=gmres(P,AP,fact,piv,xn16,16)
        d['seeded_from_native_x']={'coarse_augmented':residual(xn0),'after16':residual(xn16),'after32':residual(xn32),'history':hn16+hn32}
    return d
# Base 26-mode preconditioner first cycle forms a same-Newton restart correction q1.
AP0,F0,PV0,R0,m0=build_coarse(Praw)
x16base,h16base=gmres(Praw,AP0,F0,PV0,np.zeros(n),16)
q_restart,q_restart_meta=orthogonal_extra(x16base)
# Same-Newton online-feasible update: q_restart exists at the cycle-1 restart, then recompute RAP.
P_restart=np.column_stack((Praw,q_restart)); APr,Fr,PVr,Rr,mr=build_coarse(P_restart)
x16aug=x16base+P_restart@coarse_solve(Fr,PVr,P_restart.T@(b-A@x16base))
x32aug,h32aug=gmres(P_restart,APr,Fr,PVr,x16aug,16)
x64aug,h64aug=gmres(P_restart,APr,Fr,PVr,x32aug,32)
# Exact direct solution candidate is retrospective upper control, never admissible online.
# Verify its residual before using its projected direction as a basis.
directres=residual(xdirect)
assert directres['relative_rhs']<1e-8
result={
 'schema':'numi.cardboard.same-newton-extra-coarse-mode-304.v2',
 'source_run':str(RUN),'prediction_sha256':hashlib.sha256((RUN/'prediction.json').read_bytes()).hexdigest(),
 'receipt_sha256':hashlib.sha256((RUN/'receipt.json').read_bytes()).hexdigest(),
 'operator_report_sha256':hashlib.sha256((AN/'operator-report.json').read_bytes()).hexdigest(),
 'operator_capture_sha256':hashlib.sha256((RUN/'operator-capture').read_bytes()).hexdigest(),
 'regional_manifest_sha256':hashlib.sha256((RAP/'manifest.json').read_bytes()).hexdigest(),
 'operator_shape':list(A.shape),'rhs_norm_Ns':float(bnorm),'graph_layout_header':layout_header,
 'baseline_sameA_restart16':{'zero_R16':.9529048737118563,'zero_R32':.952248820319242,'zero_R64':.9522241466101745,'native_capture_x_R32':float(np.linalg.norm(b-A@xnative)/bnorm),'native_capture_x_R64_after32_more':.9522652014386638,'source':'conditioning-candidates-304-003; reconstructed patch factors, approximate CPU replay'},
 'base26_first_restart_correction':{'RAP':m0,'R16_correction_residual':residual(x16base),'q_extra':q_restart_meta},
 'online_same_newton_restart_mode':{
    'description':'First 16-step correction from this same Newton/operator, Euclidean-residualized against P; append q and rebuild AP/RAP before continuing the same Newton solve. This is forward-online as a restart augmentation, though this replay uses reconstructed graph factors.',
    'RAP':mr,'after_coarse_augmentation_at_restart16':residual(x16aug),'R32_after_augmented_restart':residual(x32aug),'R64_after_augmented_restart':residual(x64aug),
    'step_histories_restart2':h32aug,'step_histories_restart3plus':h64aug,
    'mode_count':27,'q_cosine_with_direct_q':float(q_restart@q_direct),'q_cosine_with_native_captured_q':float(q_restart@q_native)},
 'retrospective_native_captured_extra_mode':evaluate_aug('captured_native_delta',q_native,q_native_meta,'a_posteriori_from_same_failed_newton_solve_not_online_at_solve_start'),
 'retrospective_direct_solution_upper_control':{'direct_solution_residual':directres,'extra_mode':q_direct_meta,'mode':'unavailable_online_upper_control_from_spsolve_same_captured_A_b','candidate':evaluate_aug('direct_solution_delta',q_direct,q_direct_meta,'retrospective_unavailable_online_upper_control')},
 'limits':['all candidates use same captured A/RHS and same-step 304 history; no cross-Newton 284 recycle','captured-native q is a posteriori from the end of the same failed solve; only first-restart q is online-feasible','direct q uses the direct solution and is only a retrospective upper control','graph patch factors are rebuilt from captured A and cast to FP32, not native factors; coarse and Arnoldi arithmetic are offline, not native parity','native source solve and certificate remain unchanged; no candidate result is accepted physical evidence','full operator indefiniteness is not excluded; residual plateau is far above numerical noise levels demonstrated in double offline arithmetic']}
out=OUT/'report-v2.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(out,hashlib.sha256(out.read_bytes()).hexdigest())
print('q native meta',q_native_meta,'q_restart',q_restart_meta)
print('online R',residual(x16aug),residual(x32aug),residual(x64aug))
for name in ['retrospective_native_captured_extra_mode','retrospective_direct_solution_upper_control']:
 v=result[name]
 candidate=v if name.endswith('upper_control') else v
 print(name,json.dumps(candidate,indent=2)[:1000])
