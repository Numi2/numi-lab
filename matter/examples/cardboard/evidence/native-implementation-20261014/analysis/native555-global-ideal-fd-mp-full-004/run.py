#!/usr/bin/env python3
"""Full-mesh 90-digit center-IFT and ideal endpoint directional replay for capture555."""
from __future__ import annotations
import os, sys, pathlib, json, hashlib, time, math, warnings, traceback
from decimal import Decimal, localcontext
import numpy as np

Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
ARENA=Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz'
OLD=Q/'analysis/native555-global-ideal-fd-002/assembled_fd.npz'
OLDREPORT=Q/'analysis/native555-global-ideal-fd-002/report.json'
GLUE=Q/'analysis/native555-global-ideal-fd-glue-corrected-001'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools'
MPDIR=Q/'analysis/native531-all23-corrected-hill-highprecision-013-001/implementation'
OUT=Q/'analysis/native555-global-ideal-fd-mp-full-004'
EXPECTED={
 str(CAP):'c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53',
 str(ARENA):'28f69ede03167ba56a9af76c00ea33b332fd8cf6ccac51179057c67e15d0a0e2',
 str(OLD):'9de7bfcad1a612f562c49eb2eff7fd17d3a5b5e3e1edb02b9b7c6ec0f751f95c',
 str(MPDIR/'audit222_corrected.py'):'f103fa58001c092197d46e7b2365aa2fb708abf7610c2fb6b87bafbd698d606e',
 str(PREP/'strict_layers.py'):'45a986d399afc3c36770ef65eb4a2bf9abbef138813e9353d4c223089863252b',
 str(REF/'cardboard_bilinear_fp64_reference.py'):'8064ba89f0c4b10413937cbf2a5886a9f7de0e4b02358988294620ebae51788f',
 str(REF/'cardboard_compensated_fp64_reference.py'):'760ea825449857d565c78e754c755845e4547134b381fa167d02aab2ae63720c',
 str(GLUE/'report.json'):'97ae36341b08ba25758d3016410fd681c378a627f7fae9fd05e546a662139187',
 str(GLUE/'corrected_actions.npz'):'60742abf930df07eb80173f34e35626642ce819455a11d937c3487cc1f4a93eb',
}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p,hx in EXPECTED.items():
    if sha(p)!=hx: raise SystemExit('input hash mismatch: '+p)
if OUT.exists(): raise SystemExit('fresh output required; existing evidence must remain immutable')
OUT.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(PREP),str(REF),str(MPDIR)]
import strict_layers
import cardboard_bilinear_fp64_reference as ref
import cardboard_compensated_fp64_reference as strict
import audit222_corrected as mp

a=np.load(ARENA)
old=np.load(OLD)
header,nodes0,tets0,integrity=strict_layers.geometry.capture_records(CAP)
tets=sorted((r for r in tets0 if r['environment']==0),key=lambda r:r['tet_id'])
nodes=sorted((r for r in nodes0 if r['environment']==0),key=lambda r:r['node_id'])
if len(tets)!=1920 or len(nodes)!=882: raise SystemExit('capture counts changed')
if [x['tet_id'] for x in tets]!=list(range(1920)) or [x['node_id'] for x in nodes]!=list(range(882)): raise SystemExit('noncanonical capture ids')
material=np.asarray(a['material'],dtype=np.int32)
paper=np.flatnonzero(material!=2)
glue=np.flatnonzero(material==2)
if len(paper)!=1728 or len(glue)!=192: raise SystemExit('material cohort changed')
rowmap={int(tid):i for i,tid in enumerate(paper)}
spec={x['index']:x for x in header['materials']}
params_by_env=np.asarray(header['parameters_by_environment'][0],dtype=np.float32)
params_by_env64=params_by_env.astype(np.float64)
dtf=float(np.float32(header['dt_seconds']))
dtD=Decimal.from_float(dtf)
f32=strict_layers.f32
F0=np.stack([strict_layers.pair(t['center'],'F_local') for t in tets])
H=np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])
if not np.array_equal(F0,np.asarray(a['F_center'])): raise SystemExit('center F does not bind arena')
hist=np.stack([f32(tets[int(tid)]['center']['accepted_state']) for tid in paper])
if not np.array_equal(hist,np.asarray(a['accepted_paper_history'])): raise SystemExit('paper accepted histories do not bind arena')
p_rows=np.stack([params_by_env64[spec[int(material[tid])]['parameter_offset']:
                                     spec[int(material[tid])]['parameter_offset']+28] for tid in paper])
p_rows=np.asarray(p_rows,dtype=np.float32).astype(np.float64)
if p_rows.shape!=(1728,28): raise SystemExit('paper parameter shape mismatch')
rotation=np.stack([
 strict_layers.geometry.rotation_from_xyzw(f32(t['material_frame_quaternion_xyzw']))
 if t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1 else np.eye(3)
 for t in tets])
inverse=np.stack([f32(t['inverse_rest_rows_per_m']) for t in tets])
volume=f32([t['rest_volume_m3'] for t in tets])
tet_nodes=np.asarray([t['nodes'] for t in tets],dtype=np.int64)
mass=f32([n['center']['mass_kg'] for n in nodes])
dv=np.stack([f32(n['correction_delta_v_m_s']) for n in nodes])
free=np.asarray(a['free_node_mask'],dtype=bool)
if free.shape!=(882,): raise SystemExit('free mask shape mismatch')
dt=float(np.float32(header['dt_seconds']))
powers=(26,28)
epsilons=[2.0**(-p) for p in powers]
fpref=strict.Reference(ref)

def dmat(x):
    return [[mp.dfloat(v) for v in row] for row in np.asarray(x,dtype=np.float64)]
def dvec(x):
    return [mp.dfloat(v) for v in np.asarray(x,dtype=np.float64).reshape(-1)]
def fmat(x):
    return np.asarray([[float(v) for v in row] for row in x],dtype=np.float64)
def dnorm(x):
    return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))

def determinant3(A):
    a,b,c=A[0];d,e,f=A[1];g,h,i=A[2]
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
def inverse3(A):
    a,b,c=A[0];d,e,f=A[1];g,h,i=A[2]
    det=determinant3(A)
    if det<=0: raise ArithmeticError('nonpositive glue determinant')
    return [[(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det],
            [(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det],
            [(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]]
def trans(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def mm(A,B):
    return [[sum((A[i][k]*B[k][j] for k in range(3)),Decimal(0)) for j in range(3)] for i in range(3)]
def glue_stress_mp(F,mu,lam):
    det=determinant3(F)
    if det<=0: raise ArithmeticError('nonpositive glue determinant')
    invt=trans(inverse3(F))
    logj=det.ln()
    return [[mu*(F[i][j]-invt[i][j])+lam*logj*invt[i][j] for j in range(3)] for i in range(3)]
def glue_tangent_mp(F,Hd,mu,lam):
    inv=inverse3(F); invt=trans(inv); logj=determinant3(F).ln()
    tr=sum((inv[i][j]*Hd[j][i] for i in range(3) for j in range(3)),Decimal(0))
    term=mm(mm(invt,trans(Hd)),invt)
    return [[mu*Hd[i][j]+(mu-lam*logj)*term[i][j]+lam*tr*invt[i][j] for j in range(3)] for i in range(3)]
def worldify(local):
    return np.asarray(local,dtype=np.float64) @ np.swapaxes(rotation,-1,-2)
def assemble(dP_local):
    world=worldify(dP_local)
    elem=strict_layers.forces(world,inverse,volume)
    return strict_layers.gather(elem,tet_nodes,len(nodes))
def action_stats(x,target):
    xx=np.asarray(x,dtype=np.float64); tt=np.asarray(target,dtype=np.float64)
    bad=~np.isfinite(xx)
    if np.any(bad):
        return {'complete':False,'nonfinite_action_components':int(bad.sum()),
                'error_l2_free_Ns_per_epsilon':None,'target_l2_free_Ns_per_epsilon':float(np.linalg.norm(tt[free])),
                'relative_error_free':None,'max_abs_component_free':None}
    e=xx[free]-tt[free]; t=tt[free]
    xn=float(np.linalg.norm(e));tn=float(np.linalg.norm(t))
    return {'complete':True,'nonfinite_action_components':0,'error_l2_free_Ns_per_epsilon':xn,
            'target_l2_free_Ns_per_epsilon':tn,'relative_error_free':xn/tn if tn else None,
            'max_abs_component_free':float(np.max(np.abs(e))) if e.size else 0.0}

# Allocate exact output rows. Each paper row has an independent accepted history and FP32 material row.
MP_center_state=np.full((len(paper),15),np.nan)
MP_center_tangent=np.full((len(tets),3,3),np.nan)
MP_center_projection_iterations=np.full(len(paper),-1,dtype=np.int32)
MP_center_tangent_relative=np.full(len(paper),np.nan)
MP_center_state_delta=np.full(len(paper),np.nan)
MP_branch_center=np.full((len(paper),3),np.nan)
MP_secant={p:np.full((len(tets),3,3),np.nan) for p in powers}
MP_endpoint_state={p:np.full((2,len(paper),15),np.nan) for p in powers}
MP_endpoint_iterations={p:np.full((2,len(paper)),-1,dtype=np.int32) for p in powers}
MP_branch={p:np.full((2,len(paper),3),np.nan) for p in powers}
FP_secant={p:np.full((len(tets),3,3),np.nan) for p in powers}
FP_endpoint_state={p:np.full((2,len(paper),15),np.nan) for p in powers}
FP_endpoint_ok={p:np.zeros((2,len(paper)),dtype=bool) for p in powers}
FP_endpoint_iterations={p:np.full((2,len(paper)),-1,dtype=np.int32) for p in powers}
FP_endpoint_residual={p:np.full((2,len(paper)),np.nan) for p in powers}
FP_endpoint_correction={p:np.full((2,len(paper)),np.nan) for p in powers}
FP_endpoint_branch={p:np.full((2,len(paper),3),np.nan) for p in powers}
FP_eval_noise={p:np.full((2,len(paper)),np.nan) for p in powers}
MP_endpoint_failures=[]
FP_endpoint_failures=[]
warnings_seen={}
started=time.monotonic()
# One process/one BLAS worker. 90 digits keeps MP residual/IFT arithmetic well below requested 50+ digits.
with localcontext() as ctx:
    ctx.prec=90
    # Center MP projection and IFT tangent for all 1728 paper rows.
    for start in range(0,len(paper),64):
        stop=min(start+64,len(paper))
        for pi in range(start,stop):
            tid=int(paper[pi]);m=int(material[tid]);off=spec[m]['parameter_offset']
            p=[mp.dfloat(v) for v in params_by_env64[off:off+28]]
            Fd=dmat(F0[tid]);oldD=dvec(hist[pi]);Hd=dmat(H[tid])
            try:
                zc,ci=mp.project(p,Fd,oldD,maxiter=80)
                MP_center_projection_iterations[pi]=int(ci['iterations'])
                MP_center_state[pi]=np.asarray([float(x) for x in zc])
                MP_center_state_delta[pi]=float(np.linalg.norm(MP_center_state[pi]-a['s0_reference_state'][pi]))
                _,cm=mp.residual(p,Fd,oldD,zc)
                MP_branch_center[pi]=[float(cm['dl']),float(cm['y']),float(cm['den'])]
                J=mp.jacobian(p,Fd,oldD,zc)
                ht=Decimal('1e-25')
                Fp=[[Fd[i][j]+ht*Hd[i][j] for j in range(3)] for i in range(3)]
                Fm=[[Fd[i][j]-ht*Hd[i][j] for j in range(3)] for i in range(3)]
                rp,_=mp.residual(p,Fp,oldD,zc);rm,_=mp.residual(p,Fm,oldD,zc)
                rf=[(rp[k]-rm[k])/(2*ht) for k in range(15)]
                dz=mp.solve(J,[-x for x in rf])
                zplus=[zc[k]+ht*dz[k] for k in range(15)]
                zminus=[zc[k]-ht*dz[k] for k in range(15)]
                sp=mp.stress(p,Fp,zplus);sm=mp.stress(p,Fm,zminus)
                tangent=[[ (sp[i][j]-sm[i][j])/(2*ht) for j in range(3)] for i in range(3)]
                MP_center_tangent[tid]=fmat(tangent)
                C=np.asarray(a['reference_C'][tid],dtype=np.float64)
                target=(C@H[tid].reshape(9)).reshape(3,3)
                MP_center_tangent_relative[pi]=np.linalg.norm(MP_center_tangent[tid]-target)/max(np.linalg.norm(MP_center_tangent[tid]),1e-300)
            except Exception as e:
                MP_endpoint_failures.append({'tet_id':tid,'stage':'center_IFT','error':type(e).__name__+': '+str(e)})
        # checkpoint center prefix without rewriting earlier evidence
        (OUT/'progress.json').write_text(json.dumps({'phase':'center_IFT','completed_paper_rows':stop,'started_monotonic':started},sort_keys=True)+'\n')
    # High precision paper endpoints, two epsilons and both sides; subtraction is Decimal before float conversion.
    for start in range(0,len(paper),64):
        stop=min(start+64,len(paper))
        for pi in range(start,stop):
            tid=int(paper[pi]);m=int(material[tid]);off=spec[m]['parameter_offset']
            p=[mp.dfloat(v) for v in params_by_env64[off:off+28]]
            Fd=dmat(F0[tid]);oldD=dvec(hist[pi]);Hd=dmat(H[tid])
            side_stresses={pw:[None,None] for pw in powers}
            for ei,pw in enumerate(powers):
                eps=Decimal(1)/(Decimal(2)**pw);delta=eps*dtD
                for si,sgn in enumerate((-1,1)):
                    Fside=[[Fd[i][j]+Decimal(sgn)*delta*Hd[i][j] for j in range(3)] for i in range(3)]
                    try:
                        z,info=mp.project(p,Fside,oldD,maxiter=80)
                        MP_endpoint_state[pw][si,pi]=np.asarray([float(x) for x in z])
                        MP_endpoint_iterations[pw][si,pi]=int(info['iterations'])
                        _,bm=mp.residual(p,Fside,oldD,z)
                        MP_branch[pw][si,pi]=[float(bm['dl']),float(bm['y']),float(bm['den'])]
                        side_stresses[pw][si]=mp.stress(p,Fside,z)
                    except Exception as e:
                        MP_endpoint_failures.append({'tet_id':tid,'power':pw,'side':sgn,'error':type(e).__name__+': '+str(e)})
                if all(x is not None for x in side_stresses[pw]):
                    sec=[[ (side_stresses[pw][1][i][j]-side_stresses[pw][0][i][j])/(2*eps) for j in range(3)] for i in range(3)]
                    MP_secant[pw][tid]=fmat(sec)
        (OUT/'progress.json').write_text(json.dumps({'phase':'MP_endpoints','completed_paper_rows':stop,'started_monotonic':started},sort_keys=True)+'\n')
    # Exact MP glue endpoints and analytic tangent, same local F/H and same material parameters.
    glueoff=spec[2]['parameter_offset'];mu=mp.dfloat(params_by_env64[glueoff+1]);lam=mp.dfloat(params_by_env64[glueoff+2])
    for tid in glue:
        Fd=dmat(F0[tid]);Hd=dmat(H[tid])
        try:
            MP_center_tangent[tid]=fmat(glue_tangent_mp(Fd,Hd,mu,lam))
            for pw in powers:
                eps=Decimal(1)/(Decimal(2)**pw);delta=eps*dtD
                fp=[[Fd[i][j]+delta*Hd[i][j] for j in range(3)] for i in range(3)]
                fm=[[Fd[i][j]-delta*Hd[i][j] for j in range(3)] for i in range(3)]
                sp=glue_stress_mp(fp,mu,lam);sm=glue_stress_mp(fm,mu,lam)
                sec=[[ (sp[i][j]-sm[i][j])/(2*eps) for j in range(3)] for i in range(3)]
                MP_secant[pw][tid]=fmat(sec)
        except Exception as e:
            MP_endpoint_failures.append({'tet_id':int(tid),'stage':'glue','error':type(e).__name__+': '+str(e)})
    (OUT/'progress.json').write_text(json.dumps({'phase':'MP_complete','completed_paper_rows':len(paper),'started_monotonic':started},sort_keys=True)+'\n')

# FP64 endpoints using the strict compensated reference and unchanged 2e-12/2e-11 gates.
with warnings.catch_warnings(record=True) as warnlog:
    warnings.simplefilter('always')
    for start in range(0,len(paper),64):
        stop=min(start+64,len(paper));sl=slice(start,stop)
        pp=p_rows[sl]; oo=hist[sl]; fbase=F0[paper[sl]]; hbase=H[paper[sl]]
        for pw in powers:
            delta=(2.0**(-pw))*dt
            for si,sgn in enumerate((-1,1)):
                Fside=fbase+float(sgn)*delta*hbase
                try:
                    z,it,rn,cn,ok,reasons,prov=strict.project_with_metrics(fpref,pp,Fside,oo,maxiter=80)
                    FP_endpoint_state[pw][si,sl]=z
                    FP_endpoint_ok[pw][si,sl]=ok
                    FP_endpoint_iterations[pw][si,sl]=it
                    FP_endpoint_residual[pw][si,sl]=rn
                    FP_endpoint_correction[pw][si,sl]=cn
                    if np.any(ok):
                        branchv=strict_layers.branch_values(pp[ok],oo[ok],z[ok])
                        FP_endpoint_branch[pw][si,sl][ok]=branchv
                        Pfp=ref.stress(pp[ok],Fside[ok],z[ok])
                        # Same-state MP stress: isolates FP64 stress arithmetic from root/state mismatch.
                        with localcontext() as mpc:
                            mpc.prec=90
                            for local_index in np.flatnonzero(ok):
                                pi=start+int(local_index);tid=int(paper[pi]);m=int(material[tid]);off=spec[m]['parameter_offset']
                                pD=[mp.dfloat(v) for v in params_by_env64[off:off+28]]
                                FD=dmat(Fside[local_index]);zD=dvec(z[local_index])
                                Pmp=fmat(mp.stress(pD,FD,zD))
                                FP_eval_noise[pw][si,pi]=np.linalg.norm(Pfp[local_index]-Pmp)
                    for local_index in np.flatnonzero(~ok):
                        FP_endpoint_failures.append({'tet_id':int(paper[start+int(local_index)]),'power':pw,'side':sgn,
                            'scaled_residual':float(rn[local_index]),'dynamic_correction':float(cn[local_index]),
                            'iterations':int(it[local_index]),'reason':str(reasons[local_index])})
                except Exception as e:
                    for pi in range(start,stop):
                        FP_endpoint_failures.append({'tet_id':int(paper[pi]),'power':pw,'side':sgn,'error':type(e).__name__+': '+str(e)})
        # completed chunks are atomically retained as a compact diagnostic checkpoint.
        chunk={'paper_tet_ids':paper[sl],'mp_center_state':MP_center_state[sl],
               'mp_center_tangent':MP_center_tangent[paper[sl]],'mp_center_iterations':MP_center_projection_iterations[sl],
               'mp_center_tangent_relative_to_C_H':MP_center_tangent_relative[sl],
               'mp_center_state_delta_vs_saved_FP64':MP_center_state_delta[sl],
               'mp_center_branch_dl_y_den':MP_branch_center[sl]}
        for pw in powers:
            chunk[f'mp_secant_p{pw}']=MP_secant[pw][paper[sl]]
            chunk[f'mp_endpoint_state_p{pw}']=MP_endpoint_state[pw][:,sl,:]
            chunk[f'mp_endpoint_iterations_p{pw}']=MP_endpoint_iterations[pw][:,sl]
            chunk[f'mp_endpoint_branch_p{pw}']=MP_branch[pw][:,sl,:]
            chunk[f'fp64_endpoint_state_p{pw}']=FP_endpoint_state[pw][:,sl,:]
            chunk[f'fp64_endpoint_ok_p{pw}']=FP_endpoint_ok[pw][:,sl]
            chunk[f'fp64_endpoint_iterations_p{pw}']=FP_endpoint_iterations[pw][:,sl]
            chunk[f'fp64_endpoint_residual_p{pw}']=FP_endpoint_residual[pw][:,sl]
            chunk[f'fp64_endpoint_correction_p{pw}']=FP_endpoint_correction[pw][:,sl]
            chunk[f'fp64_endpoint_branch_p{pw}']=FP_endpoint_branch[pw][:,sl,:]
            chunk[f'fp64_same_state_stress_error_p{pw}']=FP_eval_noise[pw][:,sl]
        np.savez_compressed(OUT/f'checkpoint_{start:04d}_{stop:04d}.npz',**chunk)
        (OUT/'progress.json').write_text(json.dumps({'phase':'FP64_endpoints','completed_paper_rows':stop,'started_monotonic':started},sort_keys=True)+'\n')
    for w in warnlog:
        warnings_seen[str(w.message)]=warnings_seen.get(str(w.message),0)+1

# Store complete-tet FP64 secants where both strict endpoints passed.
for pw in powers:
    ok=FP_endpoint_ok[pw][0]&FP_endpoint_ok[pw][1]
    local_rows=np.flatnonzero(ok)
    if local_rows.size:
        # Stress values are recomputed from the exact strict states; all arithmetic here is binary64.
        psub=p_rows[local_rows]
        fplus=F0[paper[local_rows]]+(2.0**(-pw))*dt*H[paper[local_rows]]
        fminus=F0[paper[local_rows]]-(2.0**(-pw))*dt*H[paper[local_rows]]
        zp=FP_endpoint_state[pw][1,local_rows]; zm=FP_endpoint_state[pw][0,local_rows]
        Pplus=ref.stress(psub,fplus,zp);Pminus=ref.stress(psub,fminus,zm)
        FP_secant[pw][paper[local_rows]]=(Pplus-Pminus)/(2.0**(-pw))
    # FP64 glue stress secants use the same binary64 endpoint path.
    for tid in glue:
        mu64=float(params_by_env64[glueoff+1]);lam64=float(params_by_env64[glueoff+2])
        ep=2.0**(-pw);fp=F0[tid]+ep*dt*H[tid];fm=F0[tid]-ep*dt*H[tid]
        try:
            FP_secant[pw][tid]=(ref.glue_stress(fp[None],np.array([mu64]),np.array([lam64]))[0]-
                                ref.glue_stress(fm[None],np.array([mu64]),np.array([lam64]))[0])/(2.0**(-pw))
        except Exception:
            pass

# Compare local material tangent and full assembled residual derivative, with every tet including glue.
target_dR=np.asarray(old['linear_residual_derivative'],dtype=np.float64)
mp_dR={}
fp_dR={}
for pw in powers:
    mp_world=worldify(MP_secant[pw])
    mp_force=strict_layers.forces(mp_world,inverse,volume)
    mp_internal=strict_layers.gather(mp_force,tet_nodes,len(nodes))
    mp_dR[pw]=dt*mp_internal - mass[:,None]*dv
    fp_complete=bool(np.isfinite(FP_secant[pw]).all())
    if fp_complete:
        fp_world=worldify(FP_secant[pw])
        fp_force=strict_layers.forces(fp_world,inverse,volume)
        fp_internal=strict_layers.gather(fp_force,tet_nodes,len(nodes))
        fp_dR[pw]=dt*fp_internal - mass[:,None]*dv
    else:
        fp_dR[pw]=np.full_like(target_dR,np.nan)

tangent_relative=[]
for tid in paper:
    C=np.asarray(a['reference_C'][tid],dtype=np.float64)
    CH=(C@H[tid].reshape(9)).reshape(3,3)
    mt=MP_center_tangent[tid]
    tangent_relative.append(float(np.linalg.norm(CH-mt)/max(np.linalg.norm(mt),1e-300)))
tangent_relative=np.asarray(tangent_relative)
# Validate mass formula separately against recorded operator mass action.
mass_action=mass[:,None]*dv
recorded_mass=np.asarray(old['mass_action'],dtype=np.float64)
mass_identity_error=mass_action-recorded_mass
percentile=lambda x,q:float(np.quantile(np.asarray(x)[np.isfinite(x)],q)) if np.isfinite(x).any() else None
def local_stats(diff,target):
    d=np.asarray(diff,dtype=np.float64);t=np.asarray(target,dtype=np.float64)
    dn=np.linalg.norm(d.reshape(len(d),-1),axis=1);tn=np.linalg.norm(t.reshape(len(t),-1),axis=1)
    valid=np.isfinite(dn)&np.isfinite(tn)&(tn>0)
    rel=np.full(len(dn),np.nan);rel[valid]=dn[valid]/tn[valid]
    return {'rows':int(len(dn)),'finite_nonzero_reference_rows':int(valid.sum()),
            'max_abs_l2':float(np.max(dn[np.isfinite(dn)])) if np.isfinite(dn).any() else None,
            'p50_relative':percentile(rel[valid],.50),'p95_relative':percentile(rel[valid],.95),
            'p99_relative':percentile(rel[valid],.99),
            'max_relative':float(np.max(rel[valid])) if valid.any() else None,
            'rows_over_1e-6':int(np.sum(valid&(rel>1e-6)))}
assembly={}
for pw in powers:
    assembly[str(pw)]={
      'mp_secant_vs_center_IFT_action':action_stats(mp_dR[pw],target_dR),
      'fp64_secant_vs_center_IFT_action':action_stats(fp_dR[pw],target_dR),
      'fp64_secant_vs_mp_secant':action_stats(fp_dR[pw],mp_dR[pw]),
      'mp_secant_l2_free':float(np.linalg.norm(mp_dR[pw][free])),
      'fp64_secant_l2_free':float(np.linalg.norm(fp_dR[pw][free])) if np.isfinite(fp_dR[pw]).all() else None,
      'target_linear_residual_derivative_l2_free':float(np.linalg.norm(target_dR[free])),
      'same_state_stress_arithmetic_abs_error_max_paper':float(np.nanmax(FP_eval_noise[pw])) if np.isfinite(FP_eval_noise[pw]).any() else None,
      'same_state_stress_arithmetic_abs_error_p50_paper':percentile(FP_eval_noise[pw],.5),
      'same_state_stress_arithmetic_abs_error_p99_paper':percentile(FP_eval_noise[pw],.99),
      'strict_FP64_endpoint_pass_counts':{
          'plus':int(FP_endpoint_ok[pw][1].sum()),'minus':int(FP_endpoint_ok[pw][0].sum()),
          'both':int((FP_endpoint_ok[pw][1]&FP_endpoint_ok[pw][0]).sum())},
      'strict_FP64_endpoint_failures':int(np.sum(~FP_endpoint_ok[pw][1])+np.sum(~FP_endpoint_ok[pw][0])),
      'paper_local_FP64_secant_vs_MP_IFt':local_stats(FP_secant[pw][paper]-dt*MP_center_tangent[paper],
                                                       dt*MP_center_tangent[paper]),
      'paper_local_MP_secant_vs_MP_IFT':local_stats(MP_secant[pw][paper]-dt*MP_center_tangent[paper],
                                                     dt*MP_center_tangent[paper]),
      'paper_local_FP64_secant_vs_MP_secant':local_stats(FP_secant[pw][paper]-MP_secant[pw][paper],
                                                          MP_secant[pw][paper])}

# Per-tet tangent gate and endpoint branch metrics; preserve unresolved rows.
Crows=np.asarray(a['reference_C'])
CH=np.stack([(Crows[tid]@H[tid].reshape(9)).reshape(3,3) for tid in paper])
local_gate={
 'FP64_reference_C_H_vs_independent_MP_IFT':{
  'max_relative_l2':float(np.max(tangent_relative[np.isfinite(tangent_relative)])) if np.isfinite(tangent_relative).any() else None,
  'p50_relative_l2':percentile(tangent_relative,.5),'p95_relative_l2':percentile(tangent_relative,.95),
  'p99_relative_l2':percentile(tangent_relative,.99),'rows_over_1e-6':int(np.sum(tangent_relative>1e-6)),
  'gate_pass':bool(np.isfinite(tangent_relative).all() and np.all(tangent_relative<=1e-6))},
  'center_MP_state_vs_saved_FP64_state':{'max_abs':float(np.max(np.abs(MP_center_state-a['s0_reference_state'])[np.isfinite(MP_center_state)])) if np.isfinite(MP_center_state).any() else None,
   'max_l2':float(np.max(MP_center_state_delta[np.isfinite(MP_center_state_delta)])) if np.isfinite(MP_center_state_delta).any() else None},
 'center_MP_IFT_rows':int(np.isfinite(MP_center_tangent[paper]).all(axis=(1,2)).sum()),
 'center_MP_projection_iterations_max':int(MP_center_projection_iterations.max()),
 'center_MP_projection_iteration_distribution':{str(i):int(np.sum(MP_center_projection_iterations==i)) for i in np.unique(MP_center_projection_iterations)},
 'branch_summary':{}}
for pw in powers:
    mpb=MP_branch[pw]
    fpb=FP_endpoint_branch[pw]
    plus_same=np.isfinite(mpb[0,:,0])&np.isfinite(mpb[1,:,0])
    msame=(np.sign(mpb[0,:,0])==np.sign(mpb[1,:,0]))&(np.sign(mpb[0,:,1])==np.sign(mpb[1,:,1]))&( (mpb[0,:,2]>1e-14)==(mpb[1,:,2]>1e-14))
    fpsame=(np.sign(fpb[0,:,0])==np.sign(fpb[1,:,0]))&(np.sign(fpb[0,:,1])==np.sign(fpb[1,:,1]))&((fpb[0,:,2]>1e-14)==(fpb[1,:,2]>1e-14))
    local_gate['branch_summary'][str(pw)]={'MP_same_sign_and_FB_den_side_rows':int(np.sum(plus_same&msame)),
       'MP_complete_rows':int(np.sum(plus_same)),'FP64_same_sign_and_FB_den_side_rows':int(np.sum(FP_endpoint_ok[pw][0]&FP_endpoint_ok[pw][1]&fpsame)),
       'FP64_complete_rows':int(np.sum(FP_endpoint_ok[pw][0]&FP_endpoint_ok[pw][1]))}

# Free nodal top residual discrepancies for the MP secant remain descriptive only.
top=[]
pw=26
gap=mp_dR[pw]-target_dR
norms=np.linalg.norm(gap,axis=1); ids=np.argsort(np.where(free,norms,-1))[::-1][:12]
for ni in ids:
    top.append({'node_id':int(ni),'error_xyz':gap[ni].tolist(),'error_l2':float(norms[ni]),
                'target_xyz':target_dR[ni].tolist(),'free':bool(free[ni])})
# Save evidence arrays after all calculations.
np.savez_compressed(OUT/'arrays.npz',
 paper_tet_ids=paper,glue_tet_ids=glue,free_mask=free,
 mp_center_state=MP_center_state,mp_center_tangent_dP_dH=MP_center_tangent[paper],
 mp_center_tangent_relative_to_reference_C_H=tangent_relative,
 mp_center_state_delta_vs_saved_FP64=MP_center_state_delta,
 mp_center_projection_iterations=MP_center_projection_iterations,
 mp_center_branch_dl_y_den=MP_branch_center,
 reference_C_H=CH,
 mp_secant_p26=MP_secant[26],mp_secant_p28=MP_secant[28],
 fp64_secant_p26=FP_secant[26],fp64_secant_p28=FP_secant[28],
 mp_endpoint_states_p26=MP_endpoint_state[26],mp_endpoint_states_p28=MP_endpoint_state[28],
 fp64_endpoint_states_p26=FP_endpoint_state[26],fp64_endpoint_states_p28=FP_endpoint_state[28],
 fp64_endpoint_ok_p26=FP_endpoint_ok[26],fp64_endpoint_ok_p28=FP_endpoint_ok[28],
 fp64_endpoint_residual_p26=FP_endpoint_residual[26],fp64_endpoint_residual_p28=FP_endpoint_residual[28],
 fp64_endpoint_correction_p26=FP_endpoint_correction[26],fp64_endpoint_correction_p28=FP_endpoint_correction[28],
 mp_branch_p26=MP_branch[26],mp_branch_p28=MP_branch[28],
 fp64_branch_p26=FP_endpoint_branch[26],fp64_branch_p28=FP_endpoint_branch[28],
 fp64_same_state_stress_error_p26=FP_eval_noise[26],fp64_same_state_stress_error_p28=FP_eval_noise[28],
 mp_assembled_dR_p26=mp_dR[26],mp_assembled_dR_p28=mp_dR[28],
 fp64_assembled_dR_p26=fp_dR[26],fp64_assembled_dR_p28=fp_dR[28],
 center_linear_residual_derivative=target_dR,
 mp_center_tangent_all_tets=MP_center_tangent)
report={
 'schema':'numi.cardboard.native555-global-ideal-fd-mp-full.v4',
 'classification':'Full environment-0 90-digit independent constitutive projection/IFT and ideal-endpoint directional replay; CPU-only diagnostic. It neither changes native code nor qualifies physical trajectory. MP and strict binary64 outputs are reported separately.',
 'inputs_sha256':{str(p):sha(p) for p in (CAP,ARENA,OLD,OLDREPORT,GLUE/'report.json',GLUE/'corrected_actions.npz',MPDIR/'audit222_corrected.py',PREP/'strict_layers.py',REF/'cardboard_bilinear_fp64_reference.py',REF/'cardboard_compensated_fp64_reference.py')},
 'runner_sha256':sha(Path(__file__)),
 'scope':{'environment':0,'paper_tets':len(paper),'glue_tets':len(glue),'all_tets':len(tets),'nodes':len(nodes),
          'free_nodes':int(free.sum()),'dt_seconds_float32':dt,'epsilon_powers':list(powers),'epsilon_values':epsilons,
          'worker_limit':'one Python process; OMP/BLAS/Accelerate threads set to 1'},
 'input_binding':{'F_center':'capture high plus low float32 words represented in binary64; verified array parity',
  'H':'exact captured float32 F_direction_local_per_s words; deltaF=epsilon*dt*H on all paper and glue tets',
  'accepted_history':'exact captured FP32 state words; all 1728 rows verified against replay arena',
  'material_parameters':'exact per-environment float32 parameter rows and material offsets from capture header',
  'frames':'exact captured float32 material frames/quaternions; row rotations applied to local stresses before force gather',
  'mass_and_velocity':'captured node mass and correction_delta_v words used in dR=dt*dF_internal - M*delta_v',
  'constitutive_rate':'paper and glue laws are rate-independent; captured Fdot is not inserted into this source law'},
 'methods':{
  'MP':'Corrected independent decimal transcription audit222_corrected.py at precision 90; each center/endpoint projected from the immutable accepted history using its elastic predictor and strict MP solver (scaled residual 1e-55, dynamic correction 1e-45; no native seed). Center IFT tangent uses 90-digit Jacobian and directional residual, step 1e-25. Endpoint stress differences are subtracted in Decimal before conversion to binary64.',
  'glue_MP':'90-digit neo-Hookean stress and analytic tangent using source reference equation; all 192 glue endpoint stresses included before nodal assembly.',
  'FP64':'source206 compensated reference using unchanged scaled residual <=2e-12 and dynamic correction <=2e-11; no limits changed. Same-state FP64-vs-MP stress check reevaluates MP stress at the exact FP64 endpoint state to separate stress arithmetic from projection-state differences.',
  'assembly':'full 1920-tet local-to-world stress derivative, captured inverse-rest matrices and volumes, element force assembly, all-tet nodal gather, dt factor, captured mass/correction velocity term, then free-node comparison with the saved center linear residual derivative.'},
 'mass_action_identity':{'max_abs_formula_minus_saved_Ns':float(np.max(np.abs(mass_identity_error))),
  'l2_formula_minus_saved_Ns':float(np.linalg.norm(mass_identity_error))},
 'local_gate_results':local_gate,
 'assembled_action_results':assembly,
 'mp_projection_failures':MP_endpoint_failures,
 'strict_fp64_projection_failures':FP_endpoint_failures,
 'fp64_runtime_warnings':warnings_seen,
 'top_p26_mp_global_free_node_errors':top,
 'limits':['No tolerance or calibration parameter was changed.',
  'The p26/p28 endpoint replay is an ideal binary64/decimal perturbation, not a native FP32 endpoint-noise experiment.',
  'Same-state stress differences are an arithmetic comparison, not a measured native repeatability floor.',
  'A full-mesh tangent/endpoint comparison does not establish full physical coupon convergence, calibration, or assembly qualification.'],
 'elapsed_seconds':time.monotonic()-started}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
# Keep the report and array hashes in a terminal receipt before manifest.
receipt={'schema':'numi.cardboard.native555-global-ideal-fd-mp-full.receipt.v1',
 'exit_status':0,'terminal':True,'elapsed_seconds':report['elapsed_seconds'],
 'mp_center_rows':int(np.isfinite(MP_center_state).all(axis=1).sum()),
 'mp_ift_rows':int(np.isfinite(MP_center_tangent[paper]).all(axis=(1,2)).sum()),
 'mp_paper_endpoint_pairs':{str(p):int(np.isfinite(MP_secant[p][paper]).all(axis=(1,2)).sum()) for p in powers},
 'strict_fp64_endpoint_pairs':{str(p):int((FP_endpoint_ok[p][0]&FP_endpoint_ok[p][1]).sum()) for p in powers}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
files={x.name:sha(x) for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='manifest.json'}
manifest={'schema':'numi.cardboard.native555-global-ideal-fd-mp-full.files.v1',
 'files_sha256':files,'input_sha256':report['inputs_sha256'],
 'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'elapsed_seconds':report['elapsed_seconds'],'mp_ift_rows':receipt['mp_ift_rows'],
 'mp_failures':len(MP_endpoint_failures),'fp64_failures':len(FP_endpoint_failures),
 'local_gate':local_gate['FP64_reference_C_H_vs_independent_MP_IFT'],
 'assembly':assembly,'receipt':receipt},separators=(',',':')))
