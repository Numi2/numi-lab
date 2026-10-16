#!/usr/bin/env python3
"""Bounded MP center-IFT check on paper tets incident to largest corrected 555 nodes."""
import sys,pathlib,json,hashlib,time
from decimal import Decimal, localcontext
import numpy as np
Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
OUT=Q/'analysis/native555-global-ideal-fd-mp-tangent-localization-001'
OUT.mkdir(parents=True,exist_ok=True)
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
ARENA=Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz'
CORR=Q/'analysis/native555-global-ideal-fd-glue-corrected-001'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools'
MPDIR=Q/'analysis/native531-all23-corrected-hill-highprecision-013-001/implementation'
sys.path[:0]=[str(PREP),str(REF),str(MPDIR)]
import strict_layers
import audit222_corrected as mp
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=np.load(ARENA)
header,nall,tall,_=strict_layers.geometry.capture_records(CAP)
tets=sorted((x for x in tall if x['environment']==0),key=lambda x:x['tet_id'])
material=np.asarray(a['material']); paper=np.flatnonzero(material!=2)
histidx={int(t):i for i,t in enumerate(paper)}
params=np.asarray(header['parameters_by_environment'][0],np.float32)
spec={m['index']:m for m in header['materials']}
dt=D=Decimal
dt=Decimal.from_float(float(np.float32(header['dt_seconds'])))
corr=json.loads((CORR/'report.json').read_text())
selected=[1747,1754,1757,828,435,420]
powers=[18,20,22,24,26]
def decmat(x):
    return [[mp.dfloat(v) for v in row] for row in np.asarray(x)]
def decvec(x):
    return [mp.dfloat(v) for v in np.asarray(x).reshape(-1)]
def flmat(x):
    return np.asarray([[float(v) for v in row] for row in x],dtype=np.float64)
def norm(x):
    return float(np.linalg.norm(np.asarray(x,float)))
rows=[]
tstart=time.monotonic()
with localcontext() as ctx:
    ctx.prec=90
    for tid in selected:
        row=histidx[tid]
        mat=int(material[tid]); off=spec[mat]['parameter_offset']
        p=[mp.dfloat(v) for v in params[off:off+28]]
        F0=decmat(a['F_center'][tid])
        old=decvec(a['accepted_paper_history'][row])
        Hf=strict_layers.f32(tets[tid]['center']['F_direction_local_per_s']).astype(np.float64)
        Hd=decmat(Hf)
        z,rootinfo=mp.project(p,F0,old,maxiter=80)
        r0,meta0=mp.residual(p,F0,old,z)
        J=mp.jacobian(p,F0,old,z)
        h=Decimal('1e-25')
        Fp=[[F0[i][j]+h*Hd[i][j] for j in range(3)] for i in range(3)]
        Fm=[[F0[i][j]-h*Hd[i][j] for j in range(3)] for i in range(3)]
        rp,_=mp.residual(p,Fp,old,z); rm,_=mp.residual(p,Fm,old,z)
        rf=[(rp[i]-rm[i])/(2*h) for i in range(15)]
        zd=mp.solve(J,[-x for x in rf])
        Sp=mp.stress(p,Fp,old if False else [z[i]+h*zd[i] for i in range(15)])
        Sm=mp.stress(p,Fm,old if False else [z[i]-h*zd[i] for i in range(15)])
        Tm=[[ (Sp[i][j]-Sm[i][j])/(2*h) for j in range(3)] for i in range(3)]
        Tm_f=flmat(Tm)
        C=np.asarray(a['reference_C'][tid],float)
        Tref=(C@Hf.reshape(9)).reshape(3,3)
        diff=Tm_f-Tref
        centerStress=flmat(mp.stress(p,F0,z))
        old_center=np.asarray(a['s0_reference_stress'][tid],float)
        oldstate=np.asarray(a['s0_reference_state'][row],float)
        center_mp_state=np.asarray([float(x) for x in z])
        endpoint_rows=[]
        for power in powers:
            eps=Decimal(1)/(Decimal(2)**power)
            delta=eps*dt
            pp=[[F0[i][j]+delta*Hd[i][j] for j in range(3)] for i in range(3)]
            mm=[[F0[i][j]-delta*Hd[i][j] for j in range(3)] for i in range(3)]
            zp,ip=mp.project(p,pp,old,maxiter=80)
            zm,im=mp.project(p,mm,old,maxiter=80)
            sp=mp.stress(p,pp,zp); sm=mp.stress(p,mm,zm)
            sec=[[ (sp[i][j]-sm[i][j])/(2*eps) for j in range(3)] for i in range(3)]
            secf=flmat(sec); target=float(dt)*Tm_f
            e=secf-target
            rpmeta=mp.residual(p,pp,old,zp)[1]
            rmmeta=mp.residual(p,mm,old,zm)[1]
            endpoint_rows.append({
             'power':power,'epsilon':float(eps),
             'centered_secant_l2':norm(secf),'ift_target_l2':norm(target),
             'difference_l2':norm(e),'relative_error_to_ift':norm(e)/max(norm(target),1e-300),
             'plus_iterations':int(ip['iterations']),'minus_iterations':int(im['iterations']),
             'plus_residual_mp':ip['residual'],'minus_residual_mp':im['residual'],
             'center_branch':mp.branch(meta0),'plus_branch':mp.branch(rpmeta),'minus_branch':mp.branch(rmmeta),
             'plus_state_shift_l2':norm([float(zp[i]-z[i]) for i in range(15)]),
             'minus_state_shift_l2':norm([float(zm[i]-z[i]) for i in range(15)]),
            })
        rows.append({
         'tet_id':tid,'paper_history_row':row,'material_index':mat,
         'root_mp':rootinfo,'center_mp_scaled_residual':str(mp.norm(r0)),
         'center_state_max_abs_delta_vs_reference':float(np.max(np.abs(center_mp_state-oldstate))),
         'center_stress_max_abs_delta_vs_reference':float(np.max(np.abs(centerStress-old_center))),
         'reference_C_times_H_l2':norm(Tref),'MP_IFT_tangent_l2':norm(Tm_f),
         'MP_vs_reference_C_H_abs_l2':norm(diff),
         'MP_vs_reference_C_H_relative':norm(diff)/max(norm(Tm_f),1e-300),
         'max_abs_tangent_component_error':float(np.max(np.abs(diff))),
         'center_branch':mp.branch(meta0),'epsilon_sweep':endpoint_rows,
        })
elapsed=time.monotonic()-tstart
report={'schema':'numi.cardboard.native555-mp-tangent-localization.v1',
 'classification':'bounded Decimal 90-digit corrected constitutive reference on six source-selected paper tets; not a full mesh gate or native FD qualification',
 'capture_sha256':sha(CAP),'actual_material_replay_arrays_sha256':sha(ARENA),
 'corrected_glue_fd_report_sha256':sha(CORR/'report.json'),
 'corrected_glue_fd_arrays_sha256':sha(CORR/'corrected_actions.npz'),
 'corrected_MP_source_sha256':sha(MPDIR/'audit222_corrected.py'),
 'strict_FP64_reference_sha256':sha(REF/'cardboard_bilinear_fp64_reference.py'),
 'source_preparation_sha256':sha(PREP/'strict_layers.py'),
 'selection_basis':'paper tets incident to largest nodes in prior p20 discrepancy after exact glue endpoint correction: node 859/858/856, plus adjacent 828,435,420; only those selected tets, no post-hoc gate selection.',
 'direction':'exact captured float32 F_direction_local_per_s; finite difference applies deltaF=epsilon*dt*H, accepted history held fixed, same per-material FP32 parameter row.',
 'method':'90-digit Decimal corrected authored equations; independent MP center projection and 15x15 IFT solve; MP directional stress tangent compared with stored source206 reference_C@H. MP endpoints independently projected at p18,20,22,24,26 to compare centered secant with MP IFT.',
 'gates':'No gate relaxed. This is a small diagnostic cohort and cannot qualify full paper/mesh directions.',
 'elapsed_seconds':elapsed,'rows':rows}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
manifest={'schema':'numi.cardboard.native555-mp-tangent-localization-files.v1','files_sha256':{x.name:sha(x) for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='manifest.json'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'elapsed':elapsed,'summary':[{'tet':x['tet_id'],'relC':x['MP_vs_reference_C_H_relative'],'fd':[(y['power'],y['relative_error_to_ift']) for y in x['epsilon_sweep']]} for x in rows]},separators=(',',':')))
