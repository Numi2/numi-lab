#!/usr/bin/env python3
"""High-precision CPU audit of two captured medium tangent points; no native execution."""
from pathlib import Path
import hashlib, json, struct, sys
import numpy as np
import mpmath as mp

BASE = Path('/Users/home/cardboard-implementation-20261014')
CAP = BASE/'solver/recycle-source-default-first-control-283/operator-capture'
CAP_RUN = BASE/'solver/recycle-source-default-first-control-283'
RUN = BASE/'native/rationalized-all-paper-entry-points-300'
NATIVE_JSONL = BASE/'native/rationalized-all-paper-entry-points-300/output/paper-points-accurate.jsonl'
ALLPOINT_REPORT = BASE/'analysis/retained-paper-allpoint-comparison-300-001/report.json'
REF_NPZ = BASE/'analysis/retained-paper-allpoint-reference-296-001/reference.npz'
REF_PY = BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
SNAPSHOT = BASE/'snapshots/source-scaled-projection-and-coupled-capture-084'
ASSETS = [SNAPSHOT/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter',
          SNAPSHOT/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter']
sys.path.insert(0, str(REF_PY.parent))
import bilinear15_reference as ref


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def f32(x): return float(np.float32(x))
def flat_f32(x): return [f32(v) for v in np.asarray(x).reshape(-1)]
def pair_words(d,k):
    hi=flat_f32(d[k]); lo=flat_f32(d.get(k+'_low',np.zeros_like(d[k])))
    return hi,lo,[a+b for a,b in zip(hi,lo)]
def bits(xs): return [f'0x{struct.unpack("<I",struct.pack("<f",float(v)))[0]:08x}' for v in xs]
def mpf(x):
    if isinstance(x, mp.mpf): return x
    n,d=float(x).as_integer_ratio()
    return mp.mpf(n)/d

def system(pv, fv, oldv):
    p=[mpf(x) for x in pv]; old=[mpf(x) for x in oldv]
    F=mp.matrix(3,3)
    for i,v in enumerate(fv): F[i//3,i%3]=mpf(v)
    C=mp.matrix(6,6)
    C[0,0]=p[1]; C[1,1]=p[2]; C[2,2]=p[3]
    for i,j,k in [(0,1,4),(0,2,5),(1,2,6)]: C[i,j]=C[j,i]=p[k]
    C[3,3]=2*p[9]; C[4,4]=2*p[8]; C[5,5]=2*p[7]
    C=C/p[10]
    E=(F.T*F-mp.eye(3))/2
    e=mp.matrix([E[0,0],E[1,1],E[2,2],E[1,2],E[0,2],E[0,1]])
    def residual(*z):
        z=list(z); s=z[6:12]; dl=z[12]-old[12]
        a,b,c,d,e1,v=s
        q=mp.sqrt(p[11]*(b-c)**2+p[12]*(c-a)**2+p[13]*(a-b)**2+
                  2*p[14]*d*d+2*p[15]*e1*e1+2*p[16]*v*v+p[25]**2)
        flow=[(p[12]*(a-c)+p[13]*(a-b))/q,
              (p[11]*(b-c)+p[13]*(b-a))/q,
              (p[11]*(c-b)+p[12]*(c-a))/q,
              p[14]*d/q,p[15]*e1/q,p[16]*v/q]
        r=[mp.mpf(0)]*15
        for i in range(6): r[i]=z[i]-old[i]-dl*flow[i]
        Ce=C*mp.matrix([e[i]-z[i] for i in range(6)])
        for i in range(6): r[6+i]=z[6+i]-Ce[i]
        pos=[(s[i]+mp.sqrt(s[i]**2+p[23]**2))/2 for i in range(3)]
        den=sum(vv*vv for vv in pos)+p[25]**2
        for i in range(2): r[13+i]=z[13+i]-old[13+i]-dl*pos[i]**2/den
        y=1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q
        rad=mp.sqrt(dl*dl+y*y+p[26]**2)
        denom=rad+dl+y
        # Same real-valued smoothed Fischer-Burmeister equation, rationalized
        # only for stable independent evaluation. This does not alter its root.
        r[12]=(p[26]**2-2*dl*y)/denom if denom else rad-dl-y
        return r
    def stress(*z):
        z=list(z); Ce=C*mp.matrix([e[i]-z[i] for i in range(6)])
        S=mp.matrix([[Ce[0],Ce[5],Ce[4]], [Ce[5],Ce[1],Ce[3]], [Ce[4],Ce[3],Ce[2]]])
        P=F*S*p[10]
        return [P[i,j] for i in range(3) for j in range(3)]
    return p,F,old,e,residual,stress

SCALE=[mp.mpf('1e4')]*6+[mp.mpf(1)]*6+[mp.mpf('1e4')]*3
def norm_scaled(r): return max(abs(r[i])*SCALE[i] for i in range(15))
def jacobian(residual,z):
    J=mp.matrix(15,15)
    for j in range(15):
        for i in range(15):
            J[i,j]=mp.diff(lambda a,ii=i,jj=j: residual(*[a if k==jj else z[k] for k in range(15)])[ii],z[j])
    return J
def project(residual,zstart):
    z=mp.matrix([mpf(v) for v in zstart])
    for it in range(32):
        r=mp.matrix(residual(*list(z))); nr=norm_scaled(r)
        if nr<mp.mpf('1e-60'): return z,it,nr
        J=jacobian(residual,z); d=mp.lu_solve(J,-r); alpha=mp.mpf(1)
        for _ in range(100):
            cand=z+alpha*d; nc=norm_scaled(residual(*list(cand)))
            if nc<nr: z=cand; break
            alpha/=2
        else: raise RuntimeError('MP safeguarded projection line search stalled')
    raise RuntimeError('MP projection iteration limit')
def tangent(fv, pv, oldv, z, h):
    p,F,old,e,residual,stress=system(pv,fv,oldv)
    J=jacobian(residual,z); hh=[mpf(x) for x in h]
    def residual_on_ray(t):
        fp=[F[i//3,i%3]+t*hh[i] for i in range(9)]
        return system(p,fp,old)[4](*list(z))
    dr=mp.matrix([mp.diff(lambda t,ii=i:residual_on_ray(t)[ii],mp.mpf(0)) for i in range(15)])
    dz=mp.lu_solve(J,-dr)
    def stress_on_ray(t):
        fp=[F[i//3,i%3]+t*hh[i] for i in range(9)]
        return system(p,fp,old)[5](*[z[i]+t*dz[i] for i in range(15)])
    return mp.matrix([mp.diff(lambda t,ii=i:stress_on_ray(t)[ii],mp.mpf(0)) for i in range(9)])
def as_np(x): return np.asarray([float(v) for v in x],dtype=np.float64)
def metric(a,b):
    aa=np.asarray(a,dtype=float).reshape(-1); bb=np.asarray(b,dtype=float).reshape(-1); d=aa-bb
    denom=np.linalg.norm(bb)
    return {'relative_l2_to_reference':float(np.linalg.norm(d)/denom) if denom else None,
            'absolute_l2':float(np.linalg.norm(d)), 'maximum_absolute_component':float(np.max(np.abs(d))),
            'componentwise_absolute_error':d.reshape(np.asarray(a).shape).tolist(),
            'candidate_norm_l2':float(np.linalg.norm(aa)), 'reference_norm_l2':float(denom)}
def mp_relative_l2(a,b):
    aa=list(a); bb=list(b)
    num=mp.sqrt(sum((x-y)**2 for x,y in zip(aa,bb)))
    den=mp.sqrt(sum(y*y for y in bb))
    return mp.nstr(num/den if den else mp.mpf('nan'),18)
def matrix9(x): return np.asarray([float(v) for v in x],float).reshape(3,3)
def stable_green_strain(f):
    value=np.asarray(f)
    d=value-np.eye(3,dtype=value.dtype)
    e=(d+d.T+d.T@d)/2
    return e[[0,1,2,1,0,0],[0,1,2,2,2,1]]

def project_fp64_scaled_correction(p,f,old,max_iterations=16):
    """Independent FP64 safeguarded solve using production's scaled-correction gate."""
    original_strain=ref.strain
    ref.strain=lambda value: stable_green_strain(value) if np.asarray(value).shape==(3,3) else np.stack([stable_green_strain(v) for v in value])
    try:
        z=np.asarray(old,dtype=np.float64).copy()
        z[6:12]=ref.stiffness(p)@(stable_green_strain(f)-z[:6])
        scale_floor=float(p[26]); trace=[]
        for iteration in range(max_iterations):
            r=np.asarray(ref.residual(p,f,old,z),dtype=np.float64)
            J=np.asarray(ref.jacobian(p,f,old,z),dtype=np.float64)
            state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),scale_floor)
            row_scale=np.sum(np.abs(J)*state_scale[None,:],axis=1)
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J)) and np.all(row_scale>0)):
                raise RuntimeError('nonfinite/degenerate scaled local system')
            A=J*state_scale[None,:]/row_scale[:,None]
            b=-r/row_scale
            scaled_delta=np.linalg.solve(A,b)
            delta=scaled_delta*state_scale
            correction_norm=float(np.max(np.abs(scaled_delta)))
            residual_norm=float(np.max(np.abs(r)))
            state_norm=float(np.max(np.abs(z)))
            converged_residual=residual_norm<=2.0e-8*(1.0+state_norm)
            record={'iteration':iteration+1,'residual_norm':residual_norm,
                    'scaled_correction_norm':correction_norm,'state_norm':state_norm,
                    'residual_converged':bool(converged_residual),
                    'delta_multiplier':float(delta[12]),
                    'yield_gap':float(1+p[17]*p[19]*p[21]*z[13]/p[10]+
                        p[18]*p[20]*p[22]*z[14]/p[10]-ref.hill(p,z[6:12])[0])}
            if converged_residual and correction_norm<=2.0e-8:
                record['accepted_alpha']=0.0; record['terminal']='converged'
                trace.append(record)
                return z,trace
            accepted=False; alpha=1.0
            for backtrack in range(16):
                trial=z+alpha*delta
                trial_r=np.asarray(ref.residual(p,f,old,trial),dtype=np.float64)
                trial_norm=float(np.max(np.abs(trial_r)))
                if np.isfinite(trial_norm) and trial_norm<=(1.0-1.0e-4*alpha)*residual_norm:
                    record['accepted_alpha']=alpha
                    record['backtracks']=backtrack
                    record['trial_residual_norm']=trial_norm
                    record['terminal']='accepted_newton_step'
                    z=trial; accepted=True; break
                alpha*=0.5
            trace.append(record)
            if not accepted:
                raise RuntimeError(f'FP64 Armijo line search stalled at iteration {iteration+1}')
        raise RuntimeError(f'FP64 projection exceeded {max_iterations} iterations')
    finally:
        ref.strain=original_strain

def tangent_fp64_stable(p,f,old,z,h):
    original_strain=ref.strain
    ref.strain=lambda value: stable_green_strain(value) if np.asarray(value).shape==(3,3) else np.stack([stable_green_strain(v) for v in value])
    try:
        return ref.tangent(np.asarray(p,float),np.asarray(f,float).reshape(3,3),
                           np.asarray(old,float),np.asarray(z,float),
                           np.asarray(h,float).reshape(3,3))
    finally:
        ref.strain=original_strain

def tangent_matrix_fp64_stable(p,f,old,z):
    basis=np.eye(9).reshape(9,3,3)
    # Each action is a 3x3 tensor. Flatten it first, then place that vector
    # in one column; stacking the matrices directly interleaves tensor axes.
    return np.stack([tangent_fp64_stable(p,f,old,z,d).reshape(9) for d in basis],axis=1)

def mp_tangent_matrix(pv,fv,oldv,z):
    """All nine dP/dF basis actions from one high-precision implicit Jacobian."""
    p,F,old,e,residual,stress=system(pv,fv,oldv)
    J=jacobian(residual,z)
    C=mp.matrix(6,6)
    C[0,0]=p[1];C[1,1]=p[2];C[2,2]=p[3]
    for i,j,k in [(0,1,4),(0,2,5),(1,2,6)]:C[i,j]=C[j,i]=p[k]
    C[3,3]=2*p[9];C[4,4]=2*p[8];C[5,5]=2*p[7]
    C=C/p[10]
    Evec=e
    svec=mp.matrix(C*(Evec-mp.matrix([z[i] for i in range(6)])))*p[10]
    def sym_from_vec(v):
        return mp.matrix([[v[0],v[5],v[4]],[v[5],v[1],v[3]],[v[4],v[3],v[2]]])
    S=sym_from_vec(svec)
    out=mp.matrix(9,9)
    for col in range(9):
        H=mp.matrix(3,3);H[col//3,col%3]=1
        dE=(H.T*F+F.T*H)/2
        dev=mp.matrix([dE[0,0],dE[1,1],dE[2,2],dE[1,2],dE[0,2],dE[0,1]])
        dres=mp.matrix(15,1)
        dc=C*dev
        for i in range(6):dres[6+i]=-dc[i]
        dz=mp.lu_solve(J,-dres)
        dsv=(C*(dev-mp.matrix([dz[i] for i in range(6)])))*p[10]
        dP=H*S+F*sym_from_vec(dsv)
        for i in range(3):
            for j in range(3):out[3*i+j,col]=dP[i,j]
    return out

def tangent_matrix_finite_difference_fp64(p,f,old,epsilons):
    original_strain=ref.strain
    ref.strain=lambda value: stable_green_strain(value) if np.asarray(value).shape==(3,3) else np.stack([stable_green_strain(v) for v in value])
    try:
        basis=np.eye(9).reshape(9,3,3); outputs=[]
        for eps in epsilons:
            cols=[]; iterations=[]; residuals=[]; branches=[]
            for H in basis:
                zp,tp=project_fp64_scaled_correction(p,f+eps*H,old)
                zm,tm=project_fp64_scaled_correction(p,f-eps*H,old)
                pp=ref.stress(p,f+eps*H,zp);pm=ref.stress(p,f-eps*H,zm)
                cols.append(((pp-pm)/(2*eps)).reshape(9))
                iterations.append([len(tp),len(tm)])
                residuals.append([tp[-1]['residual_norm'],tm[-1]['residual_norm']])
                def branch_summary(ff,zz):
                    q=float(ref.hill(p,zz[6:12])[0])
                    y=float(1+p[17]*p[19]*p[21]*zz[13]/p[10]+
                            p[18]*p[20]*p[22]*zz[14]/p[10]-q)
                    dl=float(zz[12]-old[12])
                    return {'delta_multiplier':dl,'yield_gap':y,
                            'fb_a_plus_b':dl+y,
                            'side':'elastic_smoothed' if y>dl else 'plastic_smoothed'}
                branches.append({'plus':branch_summary(f+eps*H,zp),
                                 'minus':branch_summary(f-eps*H,zm)})
            outputs.append({'epsilon':eps,'matrix':np.stack(cols,axis=1).tolist(),
                            'iterations_plus_minus':iterations,'terminal_residual_plus_minus':residuals})
            outputs[-1]['branch_plus_minus']=branches
        return outputs
    finally:
        ref.strain=original_strain

def main():
    cap_records=[json.loads(s) for s in CAP.read_text().splitlines() if s.strip()]
    header=cap_records[0]
    assert cap_records[-1]['record']=='integrity' and cap_records[-1]['production_inputs_bitwise_unchanged']
    native_records=[json.loads(s) for s in NATIVE_JSONL.read_text().splitlines() if s.strip()]
    nheader=native_records[0]
    capture_prediction=json.loads((CAP_RUN/'prediction.json').read_text())
    capture_argv=capture_prediction['argv']
    assert '--compensated-source-paper-projection' in capture_argv
    assert '--accurate-source-paper-projection' not in capture_argv
    assert nheader['capture_sha256']==sha(CAP)
    run_receipt=json.loads((RUN/'receipt.json').read_text())
    source_build=json.loads((RUN/'inputs/external/source-build.json').read_text())
    assert run_receipt['execution_started'] and run_receipt['exit_code']==0 and not run_receipt['timed_out']
    assert run_receipt['source_inputs_unchanged'] and source_build['success']
    npz=np.load(REF_NPZ); keymap={tuple(k):i for i,k in enumerate(npz['keys'])}
    native_map={(r['environment'],r['tet_id']):r for r in native_records if r.get('record')=='point'}
    capmap={(r['environment'],r['tet_id']):r for r in cap_records if r.get('record')=='tetrahedron'}
    results=[]
    for env,tet in [(0,1634),(0,1638)]:
        key=(env,tet); row=capmap[key]; c=row['center']; n=native_map[key]; idx=keymap[(env,tet,1)]
        fh,fl,f=pair_words(c,'F_local'); h=flat_f32(c['F_direction_local_per_s'])
        old=flat_f32(c['accepted_state']); nh=flat_f32(n['projected_state_high']); nl=flat_f32(n['projected_state_low'])
        zn=[a+b for a,b in zip(nh,nl)]
        p=[f32(v) for v in header['parameters_by_environment'][env][28:56]]
        # Check this exact source-bound input against the prior FP64 batch input.
        assert np.array_equal(np.asarray(p),np.asarray(npz['parameters'][idx],dtype=np.float32).astype(float))
        assert np.array_equal(np.asarray(f).reshape(3,3),np.asarray(npz['F'][idx],dtype=np.float64))
        assert np.array_equal(np.asarray(old),np.asarray(npz['accepted'][idx],dtype=np.float64))
        assert np.array_equal(np.asarray(h).reshape(3,3),np.asarray(npz['direction'][idx],dtype=np.float64))
        # Construct the exact widened FP32 operands only after setting mpmath
        # precision. mpmath arithmetic at construction time otherwise rounds
        # these values at its default 15 digits.
        mp.mp.dps=90
        pmp,Fmp,oldmp,emp,residual,stress=system(p,f,old)
        # Solve from the exact captured native high+low candidate, not an elastic seed.
        z90,it90,norm90=project(residual,zn)
        t90=tangent(f,p,old,z90,h)
        znmp=mp.matrix([mpf(v) for v in zn])
        tn90=tangent(f,p,old,znmp,h)
        # Repeat root/action at lower precision to establish reference stability.
        mp.mp.dps=70
        p70,F70,o70,e70,r70,s70=system(p,f,old)
        z70,it70,norm70=project(r70,zn)
        t70=tangent(f,p,old,z70,h)
        mp.mp.dps=90
        pmp,Fmp,oldmp,emp,residual,stress=system(p,f,old)
        # Existing NumPy FP64 allpoint reference, preserved for comparison only.
        z_np=np.asarray(npz['projected'][idx],dtype=np.float64)
        t_np=ref.tangent(np.asarray(p,float),np.asarray(f,float).reshape(3,3),np.asarray(old,float),z_np,np.asarray(h,float).reshape(3,3))
        znpmp=mp.matrix([mpf(v) for v in z_np])
        t_mp_at_np_state=tangent(f,p,old,znpmp,h)
        residual_at_np=mp.matrix(residual(*list(znpmp)))
        scaled_residual_at_np=max(abs(residual_at_np[i])*SCALE[i] for i in range(15))
        t_native=np.asarray(n['tangent_pa_per_s'],dtype=np.float64).reshape(3,3)
        cached=np.asarray(row['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'],float).reshape(9,9)
        cache_action=(cached@np.asarray(h,float)).reshape(3,3)
        z_fp64,fp64_projection_trace=project_fp64_scaled_correction(np.asarray(p,float),
            np.asarray(f,float).reshape(3,3),np.asarray(old,float))
        fp64_tangent_matrix=tangent_matrix_fp64_stable(np.asarray(p,float),
            np.asarray(f,float).reshape(3,3),np.asarray(old,float),z_fp64)
        mp_tangent9=mp_tangent_matrix(p,f,old,z90)
        mp_tangent9_np=np.asarray(mp_tangent9.tolist(),dtype=np.float64)
        fd_matrices=tangent_matrix_finite_difference_fp64(np.asarray(p,float),
            np.asarray(f,float).reshape(3,3),np.asarray(old,float),[1.0e-5,1.0e-6,1.0e-7])
        fd_interlevel=[]
        for left,right in zip(fd_matrices,fd_matrices[1:]):
            fd_interlevel.append({'epsilon_coarse':left['epsilon'],'epsilon_fine':right['epsilon'],
                                  'coarse_vs_fine':metric(np.asarray(left['matrix']),np.asarray(right['matrix']))})
        # Branch details at root/candidate, computed using the independent root equations.
        def branch(z):
            ss=list(z[6:12]); a,b,cc,d,e1,v=ss
            q=mp.sqrt(pmp[11]*(b-cc)**2+pmp[12]*(cc-a)**2+pmp[13]*(a-b)**2+
                      2*pmp[14]*d*d+2*pmp[15]*e1*e1+2*pmp[16]*v*v+pmp[25]**2)
            y=1+pmp[17]*pmp[19]*pmp[21]*z[13]/pmp[10]+pmp[18]*pmp[20]*pmp[22]*z[14]/pmp[10]-q
            dl=z[12]-oldmp[12]
            return {'delta_multiplier':mp.nstr(dl,18),'yield_gap':mp.nstr(y,18),
                    'fb_a_plus_b':mp.nstr(dl+y,18),'hill_q':mp.nstr(q,18),
                    'hardening_factor':mp.nstr(1+y+q,18)}
        J90=jacobian(residual,z90)
        root_state=as_np(z90); native_state=np.asarray(zn,float)
        Ffp=np.asarray(f,dtype=np.float64).reshape(3,3)
        Dfp=Ffp-np.eye(3)
        strain_stable_fp64=(Dfp+Dfp.T+Dfp.T@Dfp)/2
        strain_direct_fp64=(Ffp.T@Ffp-np.eye(3))/2
        strain_mp=np.asarray([[float(emp[0]),float(emp[5]),float(emp[4])],
                              [float(emp[5]),float(emp[1]),float(emp[3])],
                              [float(emp[4]),float(emp[3]),float(emp[2])]])
        results.append({
          'point':{'environment':env,'tet_id':tet,'material_index':1,'source_material':'hajali2009_medium_hill_bilinear_fit',
                   'selection':'tet1634 is the worst reported accurate-policy tangent row from allpoint report 296-002; tet1638 is a nearby low-error control selected from that same captured source state.',
                   'native_status':{'stress_valid':n['stress_valid'],'tangent_valid':n['tangent_valid'],'projection_iterations':n['projection_iterations'],
                                    'reported_projection_residual':n['projection_residual'],'reported_projection_state_norm':n['projection_state_norm']},
                   'input_words':{'F_high_f32_bits':bits(fh),'F_low_f32_bits':bits(fl),'F_widened':f,'accepted_state_f32_bits':bits(old),
                                  'direction_f32_bits':bits(h),'native_projected_state_high_f32_bits':bits(nh),'native_projected_state_low_f32_bits':bits(nl)},
                   'source_seed':{'capture_sha256':sha(CAP),'capture_root':header['root'],'capture_newton_iteration':header['newton_iteration'],'source_candidate':row['center']['projected_state'],'source_candidate_low':row['center']['projected_state_low']},
                   'fp64_batch_reference':{'projected_state':z_np.tolist(),'tangent':t_np.tolist(),'iterations':int(npz['iterations'][idx]),'reported_scaled_residual':float(npz['residual'][idx])}},
          'mp_reference':{'precision_digits':90,'equations':'Independent literal 15-state Hill/tensile-hardening implicit equations; exact widened FP32 parameter/input words; high precision Newton root and high precision Jacobian/implicit action. No production bytecode/Metal execution.',
                          'root_iterations_90dps':it90,'root_scaled_residual_90dps':mp.nstr(norm90,12),'root_state':root_state.tolist(),
                          'root_iterations_70dps':it70,'root_scaled_residual_70dps':mp.nstr(norm70,12),
                          '70_to_90_root_state_relative_l2_mp':mp_relative_l2(z70,z90),
                          '70_to_90_tangent_relative_l2_mp':mp_relative_l2(t70,t90),
                          'root_branch':branch(z90),'native_candidate_branch':branch(znmp),'numpy_candidate_branch':branch(znpmp),
                          'numpy_candidate_scaled_residual_mp90':mp.nstr(scaled_residual_at_np,14),
                          'root_to_numpy_candidate_state_relative_l2_mp':mp_relative_l2(z90,znpmp),
                          'root_jacobian_condition_2':float(np.linalg.cond(np.asarray(J90.tolist(),float))),
                          'strain_comparison':{'stable_fp64':strain_stable_fp64.tolist(),
                             'direct_fp64':strain_direct_fp64.tolist(),'mp90':strain_mp.tolist(),
                             'stable_vs_mp_max_abs':float(np.max(np.abs(strain_stable_fp64-strain_mp))),
                             'direct_vs_mp_max_abs':float(np.max(np.abs(strain_direct_fp64-strain_mp))),
                             'stable_vs_direct_max_abs':float(np.max(np.abs(strain_stable_fp64-strain_direct_fp64))),
                             'stable_green_identity':'D=F-I; E=(D+D.T+D.T@D)/2'}},
          'comparisons':{'native_tangent_vs_mp_root':metric(t_native,matrix9(t90)),
                         'native_tangent_vs_mp_at_native_state':metric(t_native,matrix9(tn90)),
                         'mp_root_tangent_vs_mp_at_native_state':metric(matrix9(t90),matrix9(tn90)),
                         'mp_tangent_at_numpy_state_vs_mp_root':metric(matrix9(t_mp_at_np_state),matrix9(t90)),
                         'numpy_tangent_vs_mp_at_numpy_state':metric(t_np,matrix9(t_mp_at_np_state)),
                         'numpy_fp64_tangent_vs_mp_root':metric(t_np,matrix9(t90)),
                         'source283_cache_matrix_action_vs_mp90_at_captured_direction':metric(cache_action,matrix9(t90)),
                         'source283_cached_tangent_matrix_vs_mp90_matrix':metric(cached,mp_tangent9_np),
                         'fp64_scaled_correction_root_vs_mp90_state':metric(z_fp64,root_state),
                         'fp64_scaled_correction_tangent_matrix_vs_mp90_matrix':metric(fp64_tangent_matrix,mp_tangent9_np),
                         'mp90_matrix_action_vs_mp90_directional_derivative':metric(
                            (mp_tangent9*mp.matrix([mpf(v) for v in h])).tolist(),matrix9(t90)),
                         'numpy_fp64_tangent_vs_mp90_componentwise':metric(t_np,matrix9(t90)),
                         'native_projected_state_vs_mp_root':metric(native_state,root_state)},
          'nine_direction_check':{'basis':'Nine row-major unit perturbations of dimensionless F entries; exact captured center F, accepted history, and parameter words are held fixed.',
             'fp64_scaled_correction_iterations':[x['iteration'] for x in fp64_projection_trace],
             'fp64_scaled_correction_trace':fp64_projection_trace,
             'fp64_tangent_matrix':fp64_tangent_matrix.tolist(),'mp90_tangent_matrix':mp_tangent9_np.tolist(),
             'fp64_vs_mp90_full_matrix':metric(fp64_tangent_matrix,mp_tangent9_np),
             'source283_cached_matrix_vs_mp90':metric(cached,mp_tangent9_np),
             'finite_difference_sweep':[{ 'epsilon':x['epsilon'],
                'relative_l2_vs_mp90':metric(np.asarray(x['matrix']),mp_tangent9_np)['relative_l2_to_reference'],
                'absolute_l2_vs_mp90':metric(np.asarray(x['matrix']),mp_tangent9_np)['absolute_l2'],
                'max_component_vs_mp90':metric(np.asarray(x['matrix']),mp_tangent9_np)['maximum_absolute_component'],
                'iterations_plus_minus':x['iterations_plus_minus'],
                'terminal_residual_plus_minus':x['terminal_residual_plus_minus'],
                'branch_plus_minus':x['branch_plus_minus'] } for x in fd_matrices],
             'finite_difference_interlevel_comparison':fd_interlevel,
             'noise_floor_interpretation':'No general noise floor is inferred. Reported are FP64 central-difference epsilon sweeps, root residuals, branch labels, and successive-matrix differences for these two points only.',
             'scope':'CPU FP64 central differences and independent implicit tangent at two captured points; native evidence is one captured directional action and a production cached matrix comparison, not a native finite-difference test.'},
          'native_tangent_matrix':t_native.tolist(),'mp_root_tangent_matrix':matrix9(t90).tolist(),
          'mp_at_native_state_tangent_matrix':matrix9(tn90).tolist(),'captured_cache_action_matrix':cache_action.tolist()})
    comp=json.loads(ALLPOINT_REPORT.read_text())
    a=comp['policies']['accurate']
    prior=json.loads((BASE/'analysis/retained-paper-allpoint-comparison-296-002/report.json').read_text())['policies']['accurate']
    old_failed=[(0,567,0),(0,568,0),(0,569,0),(0,1527,0),(0,1528,0),(0,1529,0)]
    validity=[]
    for env,tet,mat in old_failed:
        rr=native_map[(env,tet)]
        validity.append({'key':[env,tet,mat],'old_policy296_liner_failure':True,
                         'policy300_stress_valid':rr['stress_valid'],'policy300_tangent_valid':rr['tangent_valid'],
                         'projection_failure':rr.get('projection_failure'),'projection_iterations':rr.get('projection_iterations'),
                         'projection_residual':rr.get('projection_residual')})
    source_root=Path(source_build['source_snapshot'])
    artifacts=source_build['artifacts_sha256']
    report={'schema':'numi.cardboard.medium-tangent-state-sensitivity.v4',
            'classification':'Bounded CPU 70/90-digit implicit reference and nine-direction FP64 matrix/central-difference checks at two source-selected medium points, paired with a native300 single-direction accurate-policy result and a separate source283 default-policy cached-matrix capture. This localizes the old discrepancy examples; it does not qualify the entire material or production projection policy.',
            'bindings':{'capture_path':str(CAP),'capture_sha256':sha(CAP),
                        'capture_run_prediction_path':str(CAP_RUN/'prediction.json'),'capture_run_prediction_sha256':sha(CAP_RUN/'prediction.json'),
                        'capture_source_build_path':str(CAP_RUN/'inputs/external/source-build.json'),'capture_source_build_sha256':sha(CAP_RUN/'inputs/external/source-build.json'),
                        'capture_cached_matrix_policy':{'run_id':'recycle-source-default-first-control-283','source_snapshot':capture_prediction['source_snapshot'],
                          'argv_includes_compensated_source_paper_projection':True,'argv_includes_accurate_source_paper_projection':False,
                          'meaning':'The 9x9 cache is a production capture from run283 with the scaled-accuracy switch absent. Its comparison to MP90 is diagnostic only and is not attributed to native300 accurate policy.'},
                        'native_jsonl_path':str(NATIVE_JSONL),'native_jsonl_sha256':sha(NATIVE_JSONL),
                        'native_capture_sha256_in_header':nheader['capture_sha256'],'allpoint_report_path':str(ALLPOINT_REPORT),'allpoint_report_sha256':sha(ALLPOINT_REPORT),
                        'reference_npz_path':str(REF_NPZ),'reference_npz_sha256':sha(REF_NPZ),'reference_python_sha256':sha(REF_PY),
                        'source_snapshot':str(source_root),'frozen_source_manifest_sha256':sha(source_root/'frozen-source.json'),
                        'native_receipt_sha256':sha(RUN/'receipt.json'),'native_source_build_sha256':sha(RUN/'inputs/external/source-build.json'),
                        'source_build_success':source_build['success'],'run_exit_code':run_receipt['exit_code'],'run_timed_out':run_receipt['timed_out'],
                        'source_asset_sha256':{'liner':sha(ASSETS[0]),'medium':sha(ASSETS[1])},
                        'native_artifacts_sha256':{'libmetalrobo.dylib':artifacts[next(k for k in artifacts if k.endswith('/lib/libmetalrobo.dylib'))],
                          'NumiMatter.metallib':artifacts[next(k for k in artifacts if k.endswith('/matter/shaders/NumiMatter.metallib'))],
                          'bilinear15_checker':artifacts[next(k for k in artifacts if k.endswith('/matter/numi-matter-cardboard-bilinear15-capture-diagnostic-check'))],
                          'bilinear15_diagnostic_metallib':artifacts[next(k for k in artifacts if k.endswith('/matter/shaders/cardboard_bilinear15_capture_diagnostic.metallib'))]},
                        'source_snapshot':str(SNAPSHOT),'medium_asset_sha256':sha(ASSETS[1]),
                        'policy296_valid_rows':prior['valid_rows'],'policy296_tangent_rows_above_1e_minus_3':prior['tangent']['rows_above_1e_minus_3'],
                        'allpoint_valid_rows':a['valid_rows'],'allpoint_tangent_rows_above_1e_minus_3':a['tangent']['rows_above_1e_minus_3'],
                        'reported_worst_row':a['tangent']['worst_rows'][0]['key'],'reported_worst_relative_l2':a['tangent']['worst_rows'][0]['metrics']['tangent']['relative_l2']},
            'former_invalid_liner_policy296_rows':validity,
            'results':results,
            'limits':['Two selected medium points only: the prior report worst row and a nearby control. The native single-direction action comes from native300 accurate policy. The separate source283 cached 9x9 matrix was captured with the accurate-source policy switch absent and is diagnostic-only; the nine-direction central differences are independent CPU FP64, not native Metal finite differences.',
                      'The 70/90 digit and stable-strain agreement checks these local replays only; they do not establish a general representability/noise floor.',
                      'Finite-difference errors are reported with three epsilon levels, per-direction terminal residuals and elastic-side branch labels. No general noise floor is inferred from their interlevel differences.',
                      'The native300 3456-row status/one-direction comparison is against the earlier FP64 allpoint reference. It is not a per-row high-precision proof.',
                      'The native report residual field is retained as reported; independent equation residuals are separately computed at candidate and root.',
                      'No GPU, source changes, constitutive calibration, full nonlinear step, or material qualification.'],
            'erratum':{'superseded_preliminary_mp_result':'An early draft created mpmath operands before raising precision from its default 15 digits; those interim values are invalid and are superseded by this report. This script now sets precision before exact binary input construction and binds 70/90 digit comparisons.',
                       'superseded_matrix_assembly_report_sha256':'c4b4bed41dbb5e53e306931af84a2fa7f0273ad3da07a3fda9a63d5d8f238fbb',
                       'matrix_assembly_correction':'The superseded report hash above stacked 3x3 action matrices before reshaping and thereby interleaved matrix axes in the nine-direction FP64 assembly. The corrected code flattens each action to row-major length 9 before assigning it as a matrix column. The old reported full-matrix comparison is invalid; all numbers in this report are regenerated from the corrected assembler.'},
            'interpretation':'For both points, the independent 70- and 90-digit roots/tangent matrices agree to better than 1e-71. The independent FP64 scaled-correction solve takes two iterations, resolves the tiny positive multiplier, and its full 9x9 tangent matrix agrees with MP90 at 2.45e-16 (tet1634) and 2.64e-16 (tet1638); its root state relative error is 2.01e-16 and 4.13e-21. Native300 accurate-policy captured directional actions are 3.66e-8 (tet1634) and 4.42e-9 (tet1638) relative from MP90. Separately, the run283 cached matrix (scaled-accuracy switch absent) differs from MP90 by 6.38e-6 and 1.56e-6; that is not a native300 accurate-policy matrix result. All nine CPU central-difference directions at epsilons 1e-5, 1e-6, and 1e-7 remain on the elastic-smoothed side. Their relative errors to MP90 are 6.35e-6 to 6.38e-6 (tet1634) and 1.54e-6 to 1.56e-6 (tet1638); adjacent epsilon matrices differ by 2.69e-9/2.68e-8 relative. This is a bounded FP64 finite-difference stability check, not a general noise-floor claim. The old NumPy root/action differs from MP90 by 5.79e-5 for tet1634, and MP tangent evaluated at NumPy state matches the NumPy tangent at 3.18e-16, consistent with its iteration-0 stop under a 2e-10 residual threshold. Stable and direct Green strain forms are bit-identical at these two exact captured inputs. The six liner rows invalid under native296 are valid under native300. None of this proves all 3456 derivatives or production qualification.'}
    out=Path(__file__).with_name('report.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(out),'sha256':sha(out),'points':[{'key':x['point']['environment'], 'tet':x['point']['tet_id'],
      'native_vs_mp_root':x['comparisons']['native_tangent_vs_mp_root']['relative_l2_to_reference'],
      'native_vs_mp_native_state':x['comparisons']['native_tangent_vs_mp_at_native_state']['relative_l2_to_reference'],
      'numpy_vs_mp_root':x['comparisons']['numpy_fp64_tangent_vs_mp_root']['relative_l2_to_reference'],
      'mp_at_numpy_vs_mp_root':x['comparisons']['mp_tangent_at_numpy_state_vs_mp_root']['relative_l2_to_reference']} for x in results]},indent=2))

if __name__=='__main__': main()
