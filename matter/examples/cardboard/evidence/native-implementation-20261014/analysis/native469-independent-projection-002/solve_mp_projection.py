#!/usr/bin/env python3
"""Independent high-precision projection replay for the six source-166 misses.

Inputs are the exact binary64 values produced by the sealed source-166 reader
from captured FP32 high/low words. This uses mpmath Newton/line search only; it
never seeds from a native projected state. It preserves the source residual
and the original 2e-12 residual / 2e-11 dynamic-correction gates.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import mpmath as mp
import numpy as np

BASE=Path('/Users/home/cardboard-implementation-20261014')
REPLAY=BASE/'analysis/native469-bilinear-layer-replay-002'
CAPTURE=BASE/'solver/final-newton-fd-capture-469/diagnostics/final-newton'
REFDIR=BASE/'analysis/bilinear-capture-layer-audit-source166-002/candidate/matter/tools'
REF=REFDIR/'cardboard_bilinear_fp64_reference.py'
GEOM=REFDIR/'cardboard_bilinear_capture_layers.py'
AUDIT=BASE/'snapshots/source-right-grip-telemetry-157/matter/tools/cardboard_cached_operator_audit.py'
EXPECTED={
 'arrays.npz':'4361b6aa9f0d269a9733b0bd8f508dbc1204624451611816512dad02c303d27c',
 'report.json':'b474c14872e15b9834af7fbf0e949e3ba7131dd18f11bae15e2725698e3a3eb7',
 'capture':'cf42bdb145317b24a044e8e7561a99dc987df44761b841a08d7f3b28e6cd96d2',
 'reader':'45f8fa26edb1222c61faaf6a2cfb05afb14476b4949b00e1700a34168802d939',
 'reference':'ea5cba69f600f446cc1c129b03bf1fa578d4f26d2d0d62fa1e63b4fa4c8248d5',
 'cached_audit':'2ce80f9b3c989efd058bdbcc29661d10701061cfde459d64cca06c0d6caeee38',
 'material':'76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9',
 'constitutive_source':'f907a680f2ddfd0d1082ad97dc117869f678ea8ff0deb638dea4634f27704260',
}
MATERIAL=BASE/'snapshots/source-angular-and-graph-schwarz-integrated-065/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'
CONSTITUTIVE=BASE/'analysis/bilinear15-fb-value-rationalization-001/constitutive_impl.cpp'
TETS=[750,846,1571,1599,1601,1680]
RESIDUAL_GATE=mp.mpf('2e-12')
CORRECTION_GATE=mp.mpf('2e-11')
SCALES=[mp.mpf('1e4')]*6+[mp.mpf(1)]*6+[mp.mpf('1e4')]*3


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def verify_source166_map(source_report):
    """Recheck every source file hash recorded by the bound source-166 replay."""
    root=Path(source_report['bindings']['source_snapshot'])
    declared=source_report['bindings']['source_hashes_verified_against_prediction']
    if not isinstance(declared,dict) or len(declared)!=787:
        raise ValueError(f'expected 787 source snapshot hashes, got {len(declared) if isinstance(declared,dict) else type(declared).__name__}')
    failures=[]
    for rel,expected in declared.items():
        path=root/rel.removeprefix('source/')
        if not path.is_file():
            failures.append({'path':rel,'actual':'missing','expected':expected})
            continue
        actual=sha(path)
        if actual!=expected: failures.append({'path':rel,'actual':actual,'expected':expected})
    if failures: raise ValueError(f'{len(failures)} source-166 snapshot file hash mismatches: {failures[:3]}')
    canonical=json.dumps(declared,sort_keys=True,separators=(',',':')).encode()
    return {'source_snapshot':str(root),'verified_file_count':len(declared),
            'source_hash_map_sha256':hashlib.sha256(canonical).hexdigest(),
            'frozen_source_metadata_sha256':sha(root/'frozen-source.json')}


def exact_mpf(value):
    """Promote a stored float/integer exactly, with no decimal re-rounding."""
    if isinstance(value,(float,np.floating)):
        n,d=float(value).as_integer_ratio()
    elif isinstance(value,(int,np.integer)):
        n,d=int(value),1
    else:
        raise TypeError(f'unsupported captured scalar {type(value).__name__}')
    return mp.mpf(n)/d


def load_capture_bindings():
    for name in ('arrays.npz','report.json'):
        got=sha(REPLAY/name)
        if got!=EXPECTED[name]: raise ValueError(f'{name} hash mismatch: {got}')
    if sha(CAPTURE)!=EXPECTED['capture']: raise ValueError('capture hash mismatch')
    if sha(REF)!=EXPECTED['reference'] or sha(GEOM)!=EXPECTED['reader'] or sha(AUDIT)!=EXPECTED['cached_audit']:
        raise ValueError('reference/reader/source hash mismatch')
    if sha(MATERIAL)!=EXPECTED['material'] or sha(CONSTITUTIVE)!=EXPECTED['constitutive_source']:
        raise ValueError('material or authored constitutive source hash mismatch')
    source_report=json.loads((REPLAY/'report.json').read_text())
    verified_snapshot=verify_source166_map(source_report)
    sys.path.insert(0,str(REFDIR))
    sys.path.insert(0,str(AUDIT.parent))
    spec=importlib.util.spec_from_file_location('bilinear_fp64',REF)
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    spec=importlib.util.spec_from_file_location('bilinear_geometry',GEOM)
    geom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geom;spec.loader.exec_module(geom)
    archive=np.load(REPLAY/'arrays.npz')
    with CAPTURE.open('r',encoding='utf8') as stream: header=json.loads(next(stream))
    selected={}
    with CAPTURE.open('r',encoding='utf8') as stream:
        next(stream)
        for line in stream:
            row=json.loads(line)
            if row.get('record')=='tetrahedron' and row.get('environment')==0 and row.get('tet_id') in TETS:
                selected[int(row['tet_id'])]=row
    if set(selected)!=set(TETS): raise ValueError('selected tet records incomplete')
    params=np.asarray(header['parameters_by_environment'][0][28:56],dtype=np.float32).astype(np.float64)
    if len(params)!=28: raise ValueError('medium parameter block shape mismatch')
    inputs={}
    paper_ids=np.flatnonzero(archive['material']!=2)
    paper_index={int(tet_id):int(index) for index,tet_id in enumerate(paper_ids)}
    for i in TETS:
        rec=selected[i]; center=rec['center']
        Fhi=np.asarray(center['F_local'],dtype=np.float32).astype(np.float64)
        Flo=np.asarray(center.get('F_local_low',np.zeros((3,3))),dtype=np.float32).astype(np.float64)
        F=Fhi+Flo
        old=np.asarray(center['accepted_state'],dtype=np.float32).astype(np.float64)
        # Ensure replay arrays are the exact outputs of the bound capture reader.
        if not np.array_equal(F,archive['F_center'][i]): raise ValueError(f'tet {i} F mismatch vs reader archive')
        pi=paper_index[i]
        if not np.array_equal(old,archive['accepted_paper_history'][pi]): raise ValueError(f'tet {i} accepted history mismatch vs reader archive row {pi}')
        if int(archive['material'][i])!=1 or rec['material_index']!=1: raise ValueError(f'tet {i} material mismatch')
        inputs[i]={'F':F,'old':old,'p':params.copy(),'paper_row':pi,'native_state':archive['s0_native_state'][pi].copy(),
                   'native_valid':bool(archive['s0_valid'][pi]),'native_stress':np.asarray(center['P_local_Pa'],np.float32).astype(np.float64),
                   'native_tangent':np.asarray(rec['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'],np.float32).astype(np.float64).reshape(9,9),
                   'input_high':Fhi,'input_low':Flo}
    return ref,archive,inputs,header,verified_snapshot,source_report


def asmp_list(values): return [exact_mpf(x) for x in values]


def model(p0,F0,old0):
    p=asmp_list(p0); F=mp.matrix([[exact_mpf(F0[i,j]) for j in range(3)] for i in range(3)])
    old=asmp_list(old0); I=mp.eye(3)
    C=mp.matrix([[p[1],p[4],p[5],0,0,0],[p[4],p[2],p[6],0,0,0],[p[5],p[6],p[3],0,0,0],
                 [0,0,0,2*p[9],0,0],[0,0,0,0,2*p[8],0],[0,0,0,0,0,2*p[7]]])/p[10]
    Eten=(F.T*F-I)/2
    E=mp.matrix([Eten[0,0],Eten[1,1],Eten[2,2],Eten[1,2],Eten[0,2],Eten[0,1]])
    def residual(z):
        ss=z[6:12]; dl=z[12]-old[12]
        a,b,c,d,e,v=ss
        q=mp.sqrt(p[11]*(b-c)**2+p[12]*(c-a)**2+p[13]*(a-b)**2+2*p[14]*d*d+2*p[15]*e*e+2*p[16]*v*v+p[25]**2)
        flow=[p[12]*(a-c)+p[13]*(a-b),p[11]*(b-c)+p[13]*(b-a),p[11]*(c-b)+p[12]*(c-a),p[14]*d,p[15]*e,p[16]*v]
        flow=[x/q for x in flow]
        r=[mp.mpf(0)]*15
        for j in range(6): r[j]=z[j]-old[j]-dl*flow[j]
        elastic=C*(E-mp.matrix(z[:6]))
        for j in range(6): r[6+j]=ss[j]-elastic[j]
        pos=[(ss[j]+mp.sqrt(ss[j]**2+p[23]**2))/2 for j in range(3)]
        wd=sum(x*x for x in pos)+p[25]**2
        r[13]=z[13]-old[13]-dl*pos[0]**2/wd
        r[14]=z[14]-old[14]-dl*pos[1]**2/wd
        y=1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q
        root=mp.sqrt(dl*dl+y*y+p[26]**2);den=root+dl+y
        # Evaluate the authored .nmatter Fischer-Burmeister expression directly.
        # At this precision the unsimplified sqrt expression has no cancellation
        # problem; this avoids reproducing the FP64 helper's 1e-14 rationalization
        # branch threshold in the independent mathematical reference.
        r[12]=root-dl-y
        return r
    return p,F,old,E,C,residual


def scaled_residual(r): return max(abs(r[i])*SCALES[i] for i in range(15))


def jacobian(fun,z):
    J=mp.matrix(15,15)
    for j in range(15):
        def one(x):
            zz=list(z);zz[j]=x
            return fun(zz)
        for i in range(15): J[i,j]=mp.diff(lambda x:one(x)[i],z[j])
    return J


def elastic_trial_seed(p,F,old,E,C):
    stress=C*(E-mp.matrix(old[:6]))
    return list(old[:6])+list(stress)+list(old[12:])


def solve_one(inp,precision=80,max_iterations=32):
    mp.mp.dps=precision
    p,F,old,E,C,fun=model(inp['p'],inp['F'],inp['old'])
    z=elastic_trial_seed(p,F,old,E,C)
    history=[]
    for iteration in range(max_iterations+1):
        r=fun(z);rn=scaled_residual(r)
        J=jacobian(fun,z)
        d=list(mp.lu_solve(J,-mp.matrix(r)))
        cscale=[max(abs(old[k]),abs(z[k]),p[26]) for k in range(15)]
        cn=max(abs(d[k])/cscale[k] for k in range(15))
        history.append({'iteration':iteration,'scaled_residual':mp.nstr(rn,24),
                        'dynamic_scaled_correction':mp.nstr(cn,24),'max_abs_correction':mp.nstr(max(abs(x) for x in d),16)})
        if rn<=RESIDUAL_GATE and cn<=CORRECTION_GATE:
            return {'ok':True,'z':z,'residual':rn,'correction':cn,'iterations':iteration,'history':history,'reason':'both strict gates met','precision_dps':precision}
        alpha=mp.mpf(1);accepted=False
        for backtrack in range(96):
            trial=[z[k]+alpha*d[k] for k in range(15)]
            trial_norm=scaled_residual(fun(trial))
            if mp.isfinite(trial_norm) and trial_norm<rn:
                history[-1]['accepted_alpha']=mp.nstr(alpha,12)
                history[-1]['backtracks']=backtrack
                z=trial;accepted=True;break
            alpha/=2
        if not accepted:
            history[-1]['accepted_alpha']=None;history[-1]['backtracks']=96
            return {'ok':False,'z':z,'residual':rn,'correction':cn,'iterations':iteration,'history':history,'reason':'line-search stalled before both strict gates','precision_dps':precision}
    return {'ok':False,'z':z,'residual':scaled_residual(fun(z)),'correction':cn,'iterations':max_iterations,'history':history,'reason':'iteration ceiling','precision_dps':precision}


def mp_to_float(vals): return np.asarray([float(v) for v in vals],dtype=np.float64)

def mp_stress(p,F,z):
    pmp=[exact_mpf(x) for x in p]
    fm=mp.matrix([[exact_mpf(F[i,j]) for j in range(3)] for i in range(3)])
    zm=list(z);e=(fm.T*fm-mp.eye(3))/2
    ev=[e[0,0],e[1,1],e[2,2],e[1,2],e[0,2],e[0,1]]
    C=mp.matrix([[pmp[1],pmp[4],pmp[5],0,0,0],[pmp[4],pmp[2],pmp[6],0,0,0],[pmp[5],pmp[6],pmp[3],0,0,0],[0,0,0,2*pmp[9],0,0],[0,0,0,0,2*pmp[8],0],[0,0,0,0,0,2*pmp[7]]])/pmp[10]
    sb=pmp[10]*(C*(mp.matrix(ev)-mp.matrix(zm[:6])))
    Sm=mp.matrix([[sb[0],sb[5],sb[4]],[sb[5],sb[1],sb[3]],[sb[4],sb[3],sb[2]]])
    P=fm*Sm
    return np.asarray([[float(P[i,j]) for j in range(3)] for i in range(3)])


def metrics(a,b):
    aa=np.asarray(a,float);bb=np.asarray(b,float);d=aa-bb
    na=float(np.linalg.norm(d));nb=float(np.linalg.norm(bb))
    return {'absolute_l2':na,'reference_l2':nb,'relative_l2':na/nb if nb else None,'max_abs_component':float(np.max(np.abs(d)))}


def evaluate_tangent(ref,inp,z):
    p=inp['p'][None,:];F=inp['F'][None,:,:];old=inp['old'][None,:];zz=np.asarray(z,float)[None,:]
    M=np.zeros((9,9),float)
    for col in range(9):
        H=np.zeros((1,3,3));H.reshape(1,9)[0,col]=1
        M[:,col]=ref.tangent(p,F,old,zz,H).reshape(9)
    native=inp['native_tangent']
    return {'matrix_frobenius':metrics(native,M),'column_relative_l2':[metrics(native[:,j],M[:,j]) for j in range(9)],
            'interpretation':'Same-point cached tangent vs independent complex-step FP64 implicit derivative, after resolved MP center projection. This is a row-local directional matrix check, not a native finite-difference/noise-floor qualification.'}


def elastic_seed_diagnostics(inp,ref):
    """Inspect the independently authored FB residual at the exact elastic seed."""
    p,F,old,E,C,fun=model(inp['p'],inp['F'],inp['old'])
    z=elastic_trial_seed(p,F,old,E,C)
    dl=z[12]-old[12]
    ss=z[6:12]
    a,b,c,d,e,v=ss
    q=mp.sqrt(p[11]*(b-c)**2+p[12]*(c-a)**2+p[13]*(a-b)**2+
              2*p[14]*d*d+2*p[15]*e*e+2*p[16]*v*v+p[25]**2)
    y=1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q
    eps=p[26]
    root=mp.sqrt(dl*dl+y*y+eps*eps)
    den=root+dl+y
    # The existing source-166 Jacobian obtains this row by complex-step through
    # the rationalized value branch. Preserve its observed derivative as an
    # audit datum; the authored unsimplified FB derivative is computed exactly.
    z64=mp_to_float(z)[None,:]
    old64=inp['old'][None,:]
    jac=ref.jacobian(inp['p'][None,:],inp['F'][None,:,:],old64,z64)[0]
    authored_dlambda=dl/root-1
    return {
        'seed':'exact old accepted state plus exact captured-F elastic trial stress; no native projected state used',
        'delta_lambda':mp.nstr(dl,60),'yield_argument_y':mp.nstr(y,60),
        'fb_epsilon':mp.nstr(eps,60),'root':mp.nstr(root,60),
        'rationalized_value_denominator':mp.nstr(den,60),
        'source166_rationalization_branch_cutoff':1e-14,
        'source166_complex_step_d_fb_d_lambda':float(jac[12,12]),
        'authored_unsimplified_d_fb_d_lambda':mp.nstr(authored_dlambda,60),
        'diagnosis':'At this elastic seed, the source-166 rationalized value branch is selected just above its 1e-14 threshold. Complex-step differentiation of that quotient is cancellation-sensitive; direct differentiation of the authored smooth Fischer-Burmeister expression gives dl/root-1. This is an independent-reference Jacobian artifact, not a claim about the compiled native derivative.'
    }


def run(out:Path):
    if out.exists(): raise FileExistsError(f'fresh output required, preserving prior attempts: {out}')
    started=time.time();ref,archive,inputs,header,verified_snapshot,source_report=load_capture_bindings(); out.mkdir(parents=True)
    failures=[]
    for item in json.load((REPLAY/'report.json').open())['states'][0]['projection']['failures']:
        failures.append({'tet':item['tet'],'original_fp64_failure':item['reason']})
    rows=[]
    controls=[]
    # One already-valid nearby reference row verifies the MP/FP64 residual equations agree.
    control_ids=[749,747]
    params=inputs[TETS[0]]['p']
    for i in control_ids:
        pi=inputs.get(i,{}).get('paper_row')
        if pi is None:
            paper_ids=np.flatnonzero(archive['material']!=2); pi=int(np.flatnonzero(paper_ids==i)[0])
        if not bool(archive['s0_valid'][pi]): continue
        inp={'F':archive['F_center'][i],'old':archive['accepted_paper_history'][pi],'p':params,
             'native_state':archive['s0_native_state'][pi],'native_valid':True,
             'native_stress':None,'native_tangent':None}
        p,F,old,E,C,fun=model(inp['p'],inp['F'],inp['old'])
        z=asmp_list(archive['s0_reference_state'][i])
        mp_r=fun(z)
        np_r=ref.residual(inp['p'][None,:],inp['F'][None,:,:],inp['old'][None,:],archive['s0_reference_state'][i][None,:])[0]
        controls.append({'tet':i,'mp_residual_at_source166_state':mp.nstr(scaled_residual(mp_r),16),
                         'source166_fp64_residual_scaled':float(np.max(np.abs(np_r)*np.array([1e4]*6+[1]*6+[1e4]*3))),
                         'meaning':'Equation cross-check at a previously accepted source-166 reference state; does not establish capture-wide validity.'})
    state_outputs={}
    for tet in TETS:
        inp=inputs[tet]
        # Three independently solved precision levels expose stability without
        # first rounding the MP roots to binary64.
        solved={dps:solve_one(inp,precision=dps,max_iterations=32) for dps in (70,80,110)}
        for dps,res in solved.items():
            if not res['ok']:
                raise RuntimeError(f'tet {tet} failed strict {dps}-digit MP solve: {res["reason"]}')
        result=solved[110]
        # Re-evaluate final roots and their differences at 130 digits.
        mp.mp.dps=130
        p,F,old,E,C,fun=model(inp['p'],inp['F'],inp['old'])
        z=result['z']; r=fun(z); J=jacobian(fun,z); corr=list(mp.lu_solve(J,-mp.matrix(r)))
        scales=[max(abs(old[k]),abs(z[k]),p[26]) for k in range(15)]
        rn=scaled_residual(r);cn=max(abs(corr[k])/scales[k] for k in range(15))
        roots={dps:solved[dps]['z'] for dps in (70,80,110)}
        pair_spread={}
        for left,right in ((70,80),(80,110),(70,110)):
            diffs=[abs(roots[left][k]-roots[right][k]) for k in range(15)]
            scaled=[diffs[k]/max(abs(old[k]),abs(roots[left][k]),abs(roots[right][k]),p[26]) for k in range(15)]
            pair_spread[f'{left}_vs_{right}']={
                'max_absolute_component_difference_mp':mp.nstr(max(diffs),45),
                'max_dynamic_scaled_component_difference_mp':mp.nstr(max(scaled),45),
            }
        stability_pass=all(max(abs(roots[a][k]-roots[b][k])/
            max(abs(old[k]),abs(roots[a][k]),abs(roots[b][k]),p[26]) for k in range(15)) < mp.mpf('1e-12')
            for a,b in ((70,80),(80,110),(70,110)))
        if not stability_pass:
            raise RuntimeError(f'tet {tet} 70/80/110-digit roots failed the declared 1e-12 dynamic-scaled stability check')
        zfloat=mp_to_float(z)
        state_outputs[f'tet_{tet}']=zfloat
        # Evaluate the same resolved state in source-166 FP64 arithmetic to expose its roundoff floor.
        fp64r=ref.residual(inp['p'][None,:],inp['F'][None,:,:],inp['old'][None,:],zfloat[None,:])[0]
        fp64rn=float(np.max(np.abs(fp64r)*np.array([1e4]*6+[1]*6+[1e4]*3)))
        fp64j=ref.jacobian(inp['p'][None,:],inp['F'][None,:,:],inp['old'][None,:],zfloat[None,:])[0]
        try:
            fp64corr=np.linalg.solve(fp64j,-fp64r)
            fp64scale=np.maximum(np.maximum(np.abs(inp['old']),np.abs(zfloat)),inp['p'][26])
            fp64cn=float(np.max(np.abs(fp64corr)/fp64scale))
        except Exception as e: fp64cn=None
        stress=mp_stress(inp['p'],inp['F'],z)
        row={'tet':tet,'original_fp64_failure':next(x['original_fp64_failure'] for x in failures if x['tet']==tet),
             'method':'independent 70-, 80-, and 110-digit Newton solves of the authored unsimplified equations, each from the elastic trial seed with strict-decrease backtracking',
             'seed_provenance':'old accepted history plus elastic trial stress; no native projected-state seed used',
             'precision_results':{str(dps):{'passes_both_unchanged_gates':solved[dps]['ok'],
                'iterations':solved[dps]['iterations'],'scaled_residual':mp.nstr(scaled_residual(fun(solved[dps]['z'])),32),
                'dynamic_scaled_correction':mp.nstr(solved[dps]['correction'],32),
                'root_state_mp_decimal':[mp.nstr(x,100) for x in solved[dps]['z']]}
                for dps in (70,80,110)},
             'high_precision_result':{'passes':True,'iterations':result['iterations'],'scaled_residual_130dps':mp.nstr(rn,40),
               'dynamic_scaled_correction_130dps':mp.nstr(cn,40),'residual_gate':'2e-12',
               'dynamic_correction_gate':'2e-11','root_precision_spread':pair_spread,
               'stability_max_dynamic_scaled_difference':max(float(pair_spread[k]['max_dynamic_scaled_component_difference_mp'])
                   for k in pair_spread),
               'stability_gate_dynamic_scaled_difference_threshold':1e-12,'stability_gate_passes':stability_pass,
               'last_newton_steps_110dps':result['history'][-3:]},
             'same_root_in_source166_FP64_arithmetic':{'scaled_residual':fp64rn,'dynamic_scaled_correction':fp64cn,
                'interpretation':'A larger FP64 residual at the independently resolved state is reported as arithmetic floor evidence; it does not replace or relax the high-precision mathematical residual gate.'},
             'native_comparisons':{'projected_state_relative_l2':metrics(inp['native_state'],zfloat),
                                   'stress_P_local_Pa':metrics(inp['native_stress'],stress),
                                   'cached_tangent':evaluate_tangent(ref,inp,zfloat)},
             'branch':{'plastic_multiplier_increment':mp.nstr(z[12]-old[12],20),
                       'yield_residual':mp.nstr(1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10] - mp.sqrt(p[11]*(z[7]-z[8])**2+p[12]*(z[8]-z[6])**2+p[13]*(z[6]-z[7])**2+2*p[14]*z[9]**2+2*p[15]*z[10]**2+2*p[16]*z[11]**2+p[25]**2),20)}}
        if tet==846: row['tet846_fb_quotient_jacobian_diagnosis']=elastic_seed_diagnostics(inp,ref)
        rows.append(row)
    report={'schema':'numi.cardboard.native469-independent-mp-projection.v2',
      'classification':'independent high-precision local constitutive projection at six captured center points; not a solver trajectory, native qualification, or physical validation',
      'scope':{'capture_case':'solver/final-newton-fd-capture-469','environment':0,'root/control':20,'newton':63,'state':'center','tets':TETS,
               'paper_rows':1728,'existing_source166_FP64_valid_center_rows':1722,'input_collection':'exact source-166 reader arrays; F is captured high-float32 plus low-float32, accepted history and material parameters are captured float32 promoted to float64 then exact rational to MP'},
      'bindings':{'capture_sha256':EXPECTED['capture'],'arrays_sha256':EXPECTED['arrays.npz'],'source166_report_sha256':EXPECTED['report.json'],
                  'reader_sha256':EXPECTED['reader'],'numpy_reference_sha256':EXPECTED['reference'],'cached_operator_audit_sha256':EXPECTED['cached_audit'],
                  'material_asset_sha256':EXPECTED['material'],'authored_constitutive_source_sha256':EXPECTED['constitutive_source'],
                  'source166_snapshot_integrity':verified_snapshot,
                  'source166_native_prediction_sha256':source_report['bindings']['prediction_sha256'],
                  'source166_native_receipt_sha256':source_report['bindings']['receipt_sha256'],
                  'source166_native_stderr_sha256':source_report['bindings']['stderr_sha256'],
                  'material_parameter_source':'capture header parameters_by_environment[0], medium offset 28, 28 FP32 parameters',
                  'captured_input_identity':{str(i):{'paper_row':inputs[i]['paper_row'],'material_index':int(archive['material'][i]),'F_center_equals_reader':True,'accepted_history_equals_reader':True} for i in TETS}},
      'strict_gates':{'scaled_residual_max_abs':2e-12,'dynamic_scaled_correction_max_abs':2e-11,'unchanged':True,
                      'residual_scales':[1e4]*6+[1.0]*6+[1e4]*3,'dynamic_state_scale':'max(abs(accepted),abs(candidate),fb_epsilon) per state'},
      'equation_validation_controls':controls,'original_fp64_failures_preserved':failures,'resolved_rows':rows,
      'aggregate':{'all_six_mp_reference_rows_pass_both_unchanged_gates':all(x['high_precision_result']['passes'] for x in rows),
                   'capture_center_reference_valid_rows_after_independent_fallback':1728 if all(x['high_precision_result']['passes'] for x in rows) else 1722,
                   'interpretation':'The MP results resolve these six center projections under the stated equations and exact captured inputs. They do not resolve the 6 misses at the other four deformation states, nor establish the full assembled tangent/derivative or nonlinear solver.'},
      'limitations':['No native projection code or native seed used.','No gate, coefficient, recipe, or constitutive equation was changed.','Native tangent comparisons here use same-state complex-step FP64 reference after MP projection; no measured native finite-difference noise floor or full-derivative qualification.','No full-assembly gate is promoted from six repaired center reference rows.'],
      'wall_seconds':time.time()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'mp_states.npz',**state_outputs)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    args=ap.parse_args();report=run(Path(args.output))
    print(json.dumps({'output':args.output,'resolved_rows':report['aggregate']['all_six_mp_reference_rows_pass_both_unchanged_gates'],'rows':len(report['resolved_rows'])},sort_keys=True))
