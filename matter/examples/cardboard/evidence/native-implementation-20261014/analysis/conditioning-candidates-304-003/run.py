#!/usr/bin/env python3
"""Offline same-Newton conditioning candidates for exact source091 capture 304.

The source A/RHS and native accepted-in-transaction correction are immutable inputs.
Graph patch factors are reconstructed from captured A in FP32, so this is not native
preconditioner parity or a solver qualification.
"""
from pathlib import Path
import json, subprocess, hashlib, time
import numpy as np
from scipy import sparse, linalg
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'solver/rationalized-paper-control2-newton63-capture-304'
AN=ROOT/'analysis/rationalized-paper-control2-newton63-operator-304'
RAP=RUN/'output/rejected-conditioning-000003'
LAY=ROOT/'analysis/regional-preconditioner-binding-304-002'
OUT=ROOT/'analysis/conditioning-candidates-304-003'
source_report=json.load(open(AN/'operator-report.json'))
A=sparse.load_npz(AN/'operator-report.operator.npz').tocsr()
with np.load(AN/'operator-report.vectors.npz',allow_pickle=False) as z:
    vec={k:np.array(z[k]) for k in z.files}
manifest=json.load(open(RAP/'manifest.json'))
E,N,M=manifest['environment_count'],manifest['fem_node_count'],manifest['mode_count']
assert E==2 and N==882 and M==26
free=vec['free_node_ids'].astype(np.int64)
ndof=A.shape[0]
P=np.fromfile(RAP/'P.bin',dtype='<f4').reshape(M,N,4)
eff=np.fromfile(RAP/'effective_P.bin',dtype='<f4').reshape(E,M,N,4)
AP=np.fromfile(RAP/'AP.bin',dtype='<f4').reshape(E,M,N,4)
lu_native=np.fromfile(RAP/'pivoted_RAP.bin',dtype='<f4').reshape(E,M,M)[0]
piv_native=np.fromfile(RAP/'pivots.bin',dtype='<u4').reshape(E,M)[0].astype(int)
Pfree=eff[0][:,free,:3].astype(np.float64).transpose(1,2,0).reshape(ndof,M)
APfree=AP[0][:,free,:3].astype(np.float64).transpose(1,2,0).reshape(ndof,M)
b=np.asarray(vec['native_rhs_free_impulse_Ns'],dtype=np.float64)
x_native=np.asarray(vec['direction'],dtype=np.float64)
bnorm=np.linalg.norm(b)
assert A.shape==(ndof,ndof) and b.shape==(ndof,) and x_native.shape==(ndof,)
# Reconstruct exact source091 budgeted layout from the captured topology and free mask.
text=subprocess.check_output([str(LAY/'layout'),'882','1920','2',str(LAY/'topology.u32'),str(LAY/'free_mask.u8')],text=True)
lines=text.splitlines(); summary=lines[0]; desc=[]
for line in lines[1:]:
    if line=='nodes': break
    desc.append(tuple(map(int,line.split())))
assert lines[1+len(desc)]=='nodes'
flat_nodes=list(map(int,lines[2+len(desc)].split()))
coverage_line=3+len(desc); assert lines[coverage_line]=='coverage'
coverage=np.array(list(map(int,lines[coverage_line+1].split())),dtype=np.int64)
patches=[]
for off,count,_,_ in desc: patches.append(np.asarray(flat_nodes[off:off+count],dtype=np.int64))
free_index={int(node):i for i,node in enumerate(free)}
# Captured native coarse LU reconstruction sanity: pivoted row order then triangular solve.
def native_coarse_solve(rhs):
    y=np.asarray(rhs,dtype=np.float32)[piv_native].copy()
    for i in range(M):
        if i: y[i]=np.float32(y[i]-np.dot(lu_native[i,:i],y[:i]))
    x=np.empty_like(y)
    for i in range(M-1,-1,-1):
        if i+1<M: y[i]=np.float32(y[i]-np.dot(lu_native[i,i+1:],x[i+1:]))
        x[i]=np.float32(y[i]/lu_native[i,i])
    return x
# Exact double RAP formed from the exact captured A and exact captured effective P.
R=Pfree.T@(A@Pfree)
Rcond=2047.8035909918326  # bound to independently reported exact-capture RAP SVD in regional-preconditioner-binding-304-002/report.json
# Native LU pivot reconstruction from raw arena, original factor validity is already recorded in 002.
Rrec=np.zeros_like(R)
for i in range(M):
    for j in range(M):
        s=lu_native[i,j]
        if i>j: s += sum(lu_native[i,k]*Rrec[k,j] for k in range(i))
        else: s += sum(lu_native[i,k]*Rrec[k,j] for k in range(j))
        # This recurrence is not the right general LU reconstruction for upper/lower overlap;
        # use the standard factors below for a reliable independent check instead.
        Rrec[i,j]=s
# Build patch factors from same captured A. Fixed-only patches are excluded from free solve.
factors=[]; patch_summary=[]
for patch_nodes in patches:
    nodes=[int(n) for n in patch_nodes if int(n) in free_index]
    if not nodes:
        factors.append((np.empty(0,dtype=np.int64),None,None)); patch_summary.append(len(patch_nodes)); continue
    dofs=np.concatenate([np.arange(3*free_index[n],3*free_index[n]+3,dtype=np.int64) for n in nodes])
    local=A[dofs][:,dofs].toarray().astype(np.float32)
    fact,piv=linalg.lu_factor(local,overwrite_a=True,check_finite=True)
    factors.append((dofs,fact,piv)); patch_summary.append(len(patch_nodes))
# This matches the offline 002 additive native selector5 structure: coarse residual solve,
# local patch solve of r-AP*c, and overlap averaging. All are approximate FP32 reconstructions.
def coarse_coeff(r):
    return np.linalg.solve(R, Pfree.T@r)
def apply_additive(r):
    c=coarse_coeff(r)
    out=Pfree@c
    fine=np.asarray(r,dtype=np.float64)-APfree@c
    accum=np.zeros(ndof,dtype=np.float64)
    for nodes,(dofs,fact,piv) in zip(patches,factors):
        if not len(dofs): continue
        local=linalg.lu_solve((fact,piv),fine[dofs].astype(np.float32),check_finite=False).astype(np.float64)
        for j,d in enumerate(dofs):
            node=int(nodes[j//3])
            accum[d]+=local[j]/coverage[node]
    return out+accum
# Candidate two-level projection: retain the same coarse correction, but project the patch
# correction into the left A-orthogonal complement of exact effective P: P^T A z_perp=0.
def apply_a_orthogonal(r):
    c=coarse_coeff(r)
    out=Pfree@c
    fine=np.asarray(r,dtype=np.float64)-APfree@c
    z=np.zeros(ndof,dtype=np.float64)
    for nodes,(dofs,fact,piv) in zip(patches,factors):
        if not len(dofs): continue
        local=linalg.lu_solve((fact,piv),fine[dofs].astype(np.float32),check_finite=False).astype(np.float64)
        for j,d in enumerate(dofs):
            node=int(nodes[j//3])
            z[d]+=local[j]/coverage[node]
    zc=np.linalg.solve(R,Pfree.T@(A@z))
    zperp=z-Pfree@zc
    return out+zperp
# Flexible right-preconditioned, restarted GMRES; true residual is recomputed from captured A each iter.
def gmres(apply, x0, restart=16, total=256):
    x=np.asarray(x0,dtype=np.float64).copy(); records=[]; n=len(x)
    for cycle in range((total+restart-1)//restart):
        r=b-A@x; beta=float(np.linalg.norm(r)); rel=beta/bnorm
        if not np.isfinite(rel): raise RuntimeError('nonfinite residual')
        V=np.zeros((n,restart+1),dtype=np.float64); Z=np.zeros((n,restart),dtype=np.float64)
        H=np.zeros((restart+1,restart),dtype=np.float64)
        V[:,0]=r/beta
        best=x.copy(); cycle_records=[]
        steps=min(restart,total-cycle*restart)
        for j in range(steps):
            Z[:,j]=apply(V[:,j]); w=A@Z[:,j]
            # Two-pass MGS, used solely for this offline replay; native uses FP32 GPU kernels.
            for _ in range(2):
                for i in range(j+1):
                    hij=float(np.dot(V[:,i],w)); H[i,j]+=hij; w-=hij*V[:,i]
            H[j+1,j]=np.linalg.norm(w)
            if H[j+1,j]>0 and np.isfinite(H[j+1,j]): V[:,j+1]=w/H[j+1,j]
            e=np.zeros(j+2); e[0]=beta
            y,*_=np.linalg.lstsq(H[:j+2,:j+1],e,rcond=None)
            best=x+Z[:,:j+1]@y
            true_r=float(np.linalg.norm(b-A@best)); reltrue=true_r/bnorm
            cycle_records.append({'iteration':cycle*restart+j+1,'true_residual_Ns':true_r,'true_relative_residual':reltrue,'arnoldi_subdiagonal':float(H[j+1,j])})
            if not np.isfinite(reltrue): raise RuntimeError('nonfinite true residual')
        x=best; records.extend(cycle_records)
    return x,records
# Candidate starts: from zero, from the measured native same-Newton 32-iteration correction,
# and from that correction plus exact captured RAP residual projection.
def residual_summary(label,x):
    r=b-A@x
    return {'label':label,'residual_l2_Ns':float(np.linalg.norm(r)),'relative_to_rhs':float(np.linalg.norm(r)/bnorm),'P_transpose_r_l2':float(np.linalg.norm(Pfree.T@r)),'x_l2':float(np.linalg.norm(x)),'native_correction_cosine':float(np.dot(x,x_native)/(max(np.linalg.norm(x)*np.linalg.norm(x_native),1e-300)))}
base_r=b-A@x_native
coarse_c=np.linalg.solve(R,Pfree.T@base_r)
x_aug=x_native+Pfree@coarse_c
# Also project native correction itself to remove its RAP component by A-orthogonalization,
# then restore the exact same coarse component from residual correction (a decomposition diagnostic).
coarse_x=np.linalg.solve(R,Pfree.T@(A@x_native))
x_native_orth=x_native-Pfree@coarse_x
x_decomp=x_native_orth+Pfree@(coarse_x+coarse_c)
# Bounded deterministic Rayleigh probes. These can reveal negative curvature but cannot establish SPD.
S=(A+A.T)*0.5
rng=np.random.default_rng(304)
probe_vectors=[b-A@x_native]
probe_vectors.extend(rng.normal(size=ndof) for _ in range(32))
rayleigh=[]
for v in probe_vectors:
    denom=float(np.dot(v,v))
    if denom>0:
        rayleigh.append(float(np.dot(v,S@v)/denom))
sym_spec={'rayleigh_probe_count':len(rayleigh),'minimum_sampled_rayleigh':min(rayleigh),'maximum_sampled_rayleigh':max(rayleigh),'method':'captured residual plus 32 fixed-seed random vectors; detects but cannot exclude negative curvature'}
result={
 'schema':'numi.cardboard.selector5-conditioning-candidates-304.v1',
 'source_run':str(RUN),
 'prediction_sha256':hashlib.sha256((RUN/'prediction.json').read_bytes()).hexdigest() if (RUN/'prediction.json').exists() else None,
 'operator_capture_sha256':hashlib.sha256((RUN/'operator-capture').read_bytes()).hexdigest(),
 'operator_report_sha256':hashlib.sha256((AN/'operator-report.json').read_bytes()).hexdigest(),
 'rap_manifest_sha256':hashlib.sha256((RAP/'manifest.json').read_bytes()).hexdigest(),
 'same_operator_rhs':True,
 'operator_shape':list(A.shape),'rhs_norm_Ns':float(bnorm),'native_x_norm':float(np.linalg.norm(x_native)),
 'native_captured_x_true_residual':residual_summary('native captured 32-iteration correction',x_native),
 'same_newton_exact_RAP':{'mode_count':M,'condition_2':Rcond,'min_singular_value':3.529858325766682e-05,'max_singular_value':0.0722845655519743,'native_factor_valid':[1,1],'coarse_coeff_l2_for_residual':float(np.linalg.norm(coarse_c)),'residual_after_exact_RAP_augmentation':residual_summary('native correction plus same-Newton RAP correction',x_aug),'P_transpose_residual_after_augmentation':float(np.linalg.norm(Pfree.T@(b-A@x_aug))),'A_orthogonalized_native_decomposition_residual':residual_summary('native correction decomposed against P^T A then reconstructed with same residual RAP correction',x_decomp)},
 'full_operator_symmetric_part':sym_spec,
 'preconditioners':{},
 'cautions':['offline graph patch matrices are reconstructed from same captured A and factored in FP32, not exported native factors','offline MGS uses two-pass FP64 reorthogonalization and least-squares updates; not bitwise native FGMRES','residual decay candidates are conditioning experiments, not acceptance/certificate evidence','no tolerances, A, RHS, constitutive law, or source physics were changed']
}
for label,apply in [('reconstructed_native_additive_graph_plus_RAP',apply_additive),('same_RAP_plus_A_orthogonalized_graph',apply_a_orthogonal)]:
    print('running',label,flush=True); t=time.time()
    starts={
      'zero':np.zeros_like(x_native),
      'native_x_then_continue':x_native,
      'native_x_plus_RAP_residual_correction':x_aug,
    }
    d={}
    for start_name,x0 in starts.items():
        x=np.asarray(x0,dtype=np.float64).copy(); hist=[]; checkpoints=[]
        add=32 if start_name!='zero' else 0
        requested={16,32,64,96,128,224,256}
        checkpoints.append({'offline_steps_after_start':0,'total_steps_including_native_if_seeded':add,**residual_summary(start_name+' start',x)})
        for cycle in range(16):
            r=b-A@x; beta=float(np.linalg.norm(r)); V=np.zeros((ndof,17)); Z=np.zeros((ndof,16)); H=np.zeros((17,16)); V[:,0]=r/beta; best=x.copy()
            for j in range(16):
                Z[:,j]=apply(V[:,j]); w=A@Z[:,j]
                for _ in range(2):
                    for i in range(j+1):
                        hij=float(np.dot(V[:,i],w)); H[i,j]+=hij; w-=hij*V[:,i]
                H[j+1,j]=np.linalg.norm(w)
                if H[j+1,j]>0 and np.isfinite(H[j+1,j]): V[:,j+1]=w/H[j+1,j]
                e=np.zeros(j+2); e[0]=beta
                y,*_=np.linalg.lstsq(H[:j+2,:j+1],e,rcond=None)
                best=x+Z[:,:j+1]@y
                rr=b-A@best; rn=float(np.linalg.norm(rr)); rel=rn/bnorm
                hist.append({'iteration':cycle*16+j+1,'true_residual_Ns':rn,'true_relative_residual':rel,'arnoldi_subdiagonal':float(H[j+1,j])})
                if not np.isfinite(rel): raise RuntimeError('nonfinite true residual')
                if cycle*16+j+1 in requested:
                    checkpoints.append({'offline_steps_after_start':cycle*16+j+1,'total_steps_including_native_if_seeded':add+cycle*16+j+1,**residual_summary(start_name+' continuation',best)})
            x=best
        d[start_name]={'initial':residual_summary(start_name+' initial',x0),'offline_history':hist,'checkpoints':checkpoints,'final':residual_summary(start_name+' final',x),'seconds':round(time.time()-t,3)}
        print(label,start_name,'r0',d[start_name]['initial']['relative_to_rhs'],'r256',d[start_name]['final']['relative_to_rhs'],flush=True)
    result['preconditioners'][label]=d
out=OUT/'report.json'; out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('wrote',out,'sha256',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
