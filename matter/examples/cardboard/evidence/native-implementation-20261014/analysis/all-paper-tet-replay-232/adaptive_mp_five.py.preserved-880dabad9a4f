#!/usr/bin/env python3
"""High-precision adaptive local FD adjudication for the five 216 targets."""
import json, hashlib, math
from pathlib import Path
import numpy as np
import mpmath as mp
BASE=Path('/Users/home/cardboard-implementation-20261014')
RUN=BASE/'solver/graph-schwarz-first-control-final-newton-capture-216'
CAP=RUN/'operator-capture'
REPLAY=BASE/'analysis/smooth-target-pair-audit-225/build/cardboard_solver_pair_replay_realized'
OUT=BASE/'analysis/all-paper-tet-replay-232'
IDS={611,776,1569,1603,1675}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(a): return np.asarray(a,dtype=np.float32)
def mpv(a): return [mp.mpf(float(x)) for x in np.asarray(a,dtype=np.float32).reshape(-1)]
def matrix3(vals):
    vals=list(vals)
    M=mp.matrix(3)
    for i in range(3):
        for j in range(3): M[i,j]=vals[3*i+j]
    return M
def flat(A): return [A[i] for i in range(A.rows*A.cols)]
def mm(A,B): return A*B
def t(A): return A.T

def get_material(params):
    p=mpv(params)
    C=mp.matrix(6,6)
    C[0,0]=p[1]; C[1,1]=p[2]; C[2,2]=p[3]
    C[0,1]=C[1,0]=p[4]; C[0,2]=C[2,0]=p[5]; C[1,2]=C[2,1]=p[6]
    C[3,3]=2*p[9]; C[4,4]=2*p[8]; C[5,5]=2*p[7]
    for i in range(6):
        for j in range(6): C[i,j]/=p[10]
    Fh,G,H,L,M,N=p[11:17]
    B=mp.matrix(6,6)
    B[0,0]=G+H; B[1,1]=Fh+H; B[2,2]=Fh+G
    B[0,1]=B[1,0]=-H; B[0,2]=B[2,0]=-G; B[1,2]=B[2,1]=-Fh
    B[3,3]=2*L; B[4,4]=2*M; B[5,5]=2*N
    return {'C':C,'B':B,'sigma':p[10],'qfloor':p[17],'fb':p[18]}

def strain(F):
    C=t(F)*F
    return mp.matrix([[(C[0,0]-1)/2],[(C[1,1]-1)/2],[(C[2,2]-1)/2],[C[1,2]/2],[C[0,2]/2],[C[0,1]/2]])
def q_and_flow(mat,s):
    q2=(t(s)*mat['B']*s)[0]+mat['qfloor']**2
    q=mp.sqrt(q2)
    bs=mat['B']*s
    w=[1,1,1,mp.mpf('.5'),mp.mpf('.5'),mp.mpf('.5')]
    flow=mp.matrix(6,1)
    for i in range(6):flow[i]=w[i]*bs[i]/q
    return q,bs,flow

def normalized_s(mat,F,ep):
    e=strain(F)-ep
    return mat['C']*e

def stress(mat,F,ep):
    s=normalized_s(mat,F,ep)
    S=mp.matrix([[mat['sigma']*s[0],mat['sigma']*s[5],mat['sigma']*s[4]],
                 [mat['sigma']*s[5],mat['sigma']*s[1],mat['sigma']*s[3]],
                 [mat['sigma']*s[4],mat['sigma']*s[3],mat['sigma']*s[2]]])
    return F*S

def residual(mat,F,old,x):
    ep=mp.matrix([x[i] for i in range(6)])
    s=mp.matrix([x[6+i] for i in range(6)])
    d=x[12]-old[12]
    calc=normalized_s(mat,F,ep)
    q,bs,flow=q_and_flow(mat,s); gap=1-q
    r=mp.matrix(13,1)
    for i in range(6): r[i]=ep[i]-old[i]-d*flow[i]
    for i in range(6): r[6+i]=s[i]-calc[i]
    r[12]=mp.sqrt(d*d+gap*gap+mat['fb']*mat['fb'])-d-gap
    return r

def jacobian(mat,old,x):
    B=mat['B']; C=mat['C']; s=mp.matrix([x[6+i] for i in range(6)])
    q,bs,flow=q_and_flow(mat,s); dl=x[12]-old[12]; gap=1-q; rho=mp.sqrt(dl*dl+gap*gap+mat['fb']**2)
    J=mp.matrix(13,13); w=[1,1,1,mp.mpf('.5'),mp.mpf('.5'),mp.mpf('.5')]
    for i in range(6):
      J[i,i]=1
      J[i,12]=-w[i]*bs[i]/q
      for j in range(6):J[i,6+j]=-dl*w[i]*(B[i,j]/q-bs[i]*bs[j]/q**3)
    for i in range(6):
      for j in range(6):J[6+i,j]=C[i,j]
      J[6+i,6+i]=1
    for i in range(6):J[12,6+i]=(1-gap/rho)*bs[i]/q
    J[12,12]=dl/rho-1
    return J

def project(mat,F,old,maxit=50):
    x=mp.matrix(13,1); trial=normalized_s(mat,F,mp.matrix([old[i] for i in range(6)]))
    for i in range(6):x[i]=old[i]
    for i in range(6):x[6+i]=trial[i]
    x[12]=old[12]
    for it in range(maxit):
      r=residual(mat,F,old,x); norm=max(abs(r[i]) for i in range(13)); scale=max(1,max(abs(x[i]) for i in range(13)))
      if norm < mp.mpf('1e-55')*scale: return x,norm,it
      dx=mp.lu_solve(jacobian(mat,old,x),-r)
      a=mp.mpf(1); accepted=False
      for _ in range(60):
       cand=x+a*dx; rn=residual(mat,F,old,cand); nn=max(abs(rn[i]) for i in range(13))
       if mp.isfinite(nn) and nn < norm:
        x=cand;accepted=True;break
       a/=2
      if not accepted: raise RuntimeError(f'line search fail norm={norm} q={q_and_flow(mat,mp.matrix([x[6+i] for i in range(6)]))[0]}')
    raise RuntimeError('high precision projection exhausted')

def tangent(mat,F,H,old,x):
    de=(t(F)*H+t(H)*F)/2
    ev=mp.matrix([de[0,0],de[1,1],de[2,2],de[1,2],de[0,2],de[0,1]])
    rhs=mp.matrix(13,1); ce=mat['C']*ev
    for i in range(6):rhs[6+i]=ce[i]
    dx=mp.lu_solve(jacobian(mat,old,x),rhs)
    dep=mp.matrix([dx[i] for i in range(6)])
    ds=mat['sigma']*mat['C']*(ev-dep)
    dS=mp.matrix([[ds[0],ds[5],ds[4]],[ds[5],ds[1],ds[3]],[ds[4],ds[3],ds[2]]])
    ep=mp.matrix([x[i] for i in range(6)])
    second=mp.matrix([[mat['sigma']*normalized_s(mat,F,ep)[0],mat['sigma']*normalized_s(mat,F,ep)[5],mat['sigma']*normalized_s(mat,F,ep)[4]],
      [mat['sigma']*normalized_s(mat,F,ep)[5],mat['sigma']*normalized_s(mat,F,ep)[1],mat['sigma']*normalized_s(mat,F,ep)[3]],
      [mat['sigma']*normalized_s(mat,F,ep)[4],mat['sigma']*normalized_s(mat,F,ep)[3],mat['sigma']*normalized_s(mat,F,ep)[2]]])
    return H*second+F*dS

def trial_q(mat,F,old):
    return q_and_flow(mat,normalized_s(mat,F,mp.matrix([old[i] for i in range(6)])))[0]

def as_float(a):
    if isinstance(a,mp.matrix):return [float(x) for x in flat(a)]
    return float(a)

recs=[json.loads(x) for x in CAP.read_text().splitlines() if x.strip()]; header=recs[0]; mats={int(m['index']):m for m in header['materials']}; results=[]
for tid in sorted(IDS):
    tet=next(r for r in recs if r.get('record')=='tetrahedron' and int(r['environment'])==0 and int(r['tet_id'])==tid)
    if int(tet['material_index'])!=1: raise RuntimeError('expected medium material')
    off=int(mats[1]['parameter_offset']); pars=f32(header['parameters_by_environment'][0])[off:off+20]; mat=get_material(pars)
    c=tet['center']; Flo=f32(c.get('F_local_low',[0]*9)).reshape(-1); Fhi=f32(c['F_local']).reshape(-1); F=matrix3([mp.mpf(float(a))+mp.mpf(float(b)) for a,b in zip(Fhi,Flo)])
    Hn=f32(c['F_direction_local_per_s']).reshape(-1); H=matrix3([mp.mpf(float(x)) for x in Hn])
    oldv=mpv(c['accepted_state']); old=mp.matrix(oldv)
    mp.mp.dps=80
    x,norm,it=project(mat,F,old); P0=stress(mat,F,mp.matrix([x[i] for i in range(6)])); T=tangent(mat,F,H,old,x)
    q0=trial_q(mat,F,old)
    hnorm=max(abs(H[i,j]) for i in range(3) for j in range(3))
    # Dimensionless perturbation hF scales max |delta F| directly; epsilon is seconds.
    records=[]
    for hFstr in ['3e-3','1e-3','3e-4','1e-4','3e-5','1e-5','3e-6','1e-6','3e-7','1e-7','3e-8','1e-8','3e-9','1e-9','3e-10','1e-10','3e-11','1e-11','3e-12','1e-12','3e-13','1e-13','3e-14','1e-14','3e-15']:
      hF=mp.mpf(hFstr); dtau=hF/hnorm
      Hp=H/hnorm
      pp_state,rp,ip=project(mat,F+hF*Hp,old); pm_state,rm,im=project(mat,F-hF*Hp,old)
      pp=stress(mat,F+hF*Hp,mp.matrix([pp_state[i] for i in range(6)])); pm=stress(mat,F-hF*Hp,mp.matrix([pm_state[i] for i in range(6)]))
      qplus=trial_q(mat,F+hF*Hp,old); qminus=trial_q(mat,F-hF*Hp,old)
      central=(pp-pm)*(hnorm/(2*hF)); forward=(pp-P0)*(hnorm/hF); backward=(P0-pm)*(hnorm/hF)
      def rel_error(E):return max(abs(E[i,j]-T[i,j]) for i in range(3) for j in range(3))/max(abs(T[i,j]) for i in range(3) for j in range(3))
      records.append({'deltaF_inf':float(hF),'epsilon_s':float(dtau),'q_minus_center_plus':[float(qminus),float(q0),float(qplus)],
        'branches':['transition_unresolved' if abs(float(qminus)-1)<=1e-6 else ('plastic' if qminus>1 else 'elastic'),
                    'transition_unresolved' if abs(float(q0)-1)<=1e-6 else ('plastic' if q0>1 else 'elastic'),
                    'transition_unresolved' if abs(float(qplus)-1)<=1e-6 else ('plastic' if qplus>1 else 'elastic')],
        'same_smooth_branch':len({('transition' if abs(float(q)-1)<=1e-6 else ('p' if q>1 else 'e')) for q in [qminus,q0,qplus]})==1 and abs(float(q0)-1)>1e-6,
        'centered_relative_error':float(rel_error(central)),'forward_relative_error':float(rel_error(forward)),'backward_relative_error':float(rel_error(backward)),
        'centered_abs_error_Pa_per_s':float(max(abs(central[i,j]-T[i,j]) for i in range(3) for j in range(3))), 'central_tangent_matrix':[float(central[i,j]) for i in range(3) for j in range(3)],
        'stress_delta_inf_Pa':float(max(abs(pp[i,j]-pm[i,j]) for i in range(3) for j in range(3))),
        'projection_residuals_mp80':[float(rp),float(rm)],'projection_iterations':[ip,im]})
    # Repeat center/endpoints at 50 and 80 dps for a direct precision-convergence estimate.
    mp.mp.dps=50; x50,n50,_=project(mat,F,old); p50=stress(mat,F,mp.matrix([x50[i] for i in range(6)])); t50=tangent(mat,F,H,old,x50)
    # All source parameters and endpoint words are exactly the same; only mp precision differs.
    center_stress_precision=max(abs(p50[i,j]-P0[i,j]) for i in range(3) for j in range(3))
    tangent_precision=max(abs(t50[i,j]-T[i,j]) for i in range(3) for j in range(3))
    results.append({'tet_id':tid,'grade':'medium','captured_material_index':1,'native_capture_candidate_q_field':c['hill_q_lambda_a_b'][0],
      'independent_center_trial_q':float(q0),'independent_candidate_lambda_increment':float(x[12]-old[12]),'high_precision_projection_residual':float(norm),'iterations':it,
      'precision_convergence_50_vs_80_dps':{'center_stress_abs_Pa':float(center_stress_precision),'directional_tangent_abs_Pa_per_s':float(tangent_precision)},
      'direction_max_abs_per_s':float(hnorm),'tangent_matrix_Pa_per_s':[float(T[i,j]) for i in range(3) for j in range(3)],'center_first_piola_Pa':[float(P0[i,j]) for i in range(3) for j in range(3)],'sweeps':records})
    print('done',tid,'q',float(q0),'lambda',float(x[12]-old[12]),'stress precision',float(center_stress_precision),'T precision',float(tangent_precision))
manifest={'schema':'numi.cardboard.adaptive-high-precision-fd-adjudication.v1','capture':str(CAP),'capture_sha256':sha(CAP),'native_prediction_sha256':sha(RUN/'prediction.json'),'receipt_sha256':sha(RUN/'receipt.json'),
 'source_snapshot':json.loads((RUN/'prediction.json').read_text()).get('source_snapshot'),'methods':'Independent mpmath 80-digit FP64-oracle equation replay; exact FP32 high+low F, captured accepted history, FP32 H and FP32 parameter words; center/one-sided/centered dimensional perturbation sweeps with fixed max|delta F|; branch labels from exact elastic trial q; repeat at 50 vs 80 decimal digits to estimate numerical precision convergence, not empirical stochastic noise.',
 'threshold_policy':'No tolerance relaxed. FD <=1e-6 is reported only where the exact q branch is the same for minus/center/plus and tangent scale is nonzero.',
 'python_mpmath_version':mp.__version__,'results':results}
(OUT/'adaptive-fd-five.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print('wrote',OUT/'adaptive-fd-five.json')
