#!/usr/bin/env python3
"""Bounded high-precision audit of source173 paper row/tet 172, F[2,0].
No native code, seed, or tolerance changes. Uses the exact FP64 endpoints from
run019 and independent mpmath equations from the retained 005 solver.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, pathlib, sys, time
import numpy as np
ROOT=pathlib.Path('/Users/home/cardboard-implementation-20261014')
OUT=ROOT/'analysis/native174-tet172-mp-iftfd-003'
CAPTURE=ROOT/'analysis/native469-bilinear-layer-replay-002/arrays.npz'
MP_ARRAYS=ROOT/'analysis/native469-mp-fallback-005/run-002/mixed_reference_arrays.npz'
MP_SOLVER=ROOT/'analysis/native469-mp-fallback-005/run_mp_fallback.py'
SOURCE173=ROOT/'snapshots/source-checkpoint-warm-breakdown-173/matter/tools/cardboard_bilinear_fp64_reference.py'
V2_TOOLS=ROOT/'analysis/native174-fp64-referencev2-source173-portable-003/r-ready/matter/tools'
RUN016=ROOT/'analysis/native174-fp64-derivative-qualification-002/run-016-source173-correct-direction-eps8-26'
CENTER=RUN016/'projection-center.npz'
RUN019=ROOT/'analysis/native174-fp64-referencev2-source173-portable-001/run-019-unresolved16-quarter-grid'
RUN019_RECORDS=RUN019/'dense-grid-records.jsonl'
ROW=172; TET=172; FCOMP=(2,0); COL=6
RESID_GATE=2e-12; CORR_GATE=2e-11; FD_GATE=1e-6

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def norm(x): return float(np.linalg.norm(np.asarray(x,dtype=np.float64).reshape(-1)))
def metrics(actual,ref):
    a=np.asarray(actual,dtype=np.float64).reshape(-1); b=np.asarray(ref,dtype=np.float64).reshape(-1)
    d=a-b; den=norm(b)
    return {'relative_l2':float(norm(d)/den) if den else None,'absolute_l2':norm(d),
            'reference_l2':den,'max_abs_component':float(np.max(np.abs(d)))}
def mpfloat(x,mp): return float(mp.mpf(x))
def mparray(mat,mp): return np.array([[float(mat[i,j]) for j in range(mat.cols)] for i in range(mat.rows)],dtype=np.float64)
def mp_metric_numpy(actual,ref_mp,mp):
    d=[]; r=[]
    for i in range(ref_mp.rows):
        for j in range(ref_mp.cols):
            d.append(mp.mpf(float(np.asarray(actual).reshape(ref_mp.rows,ref_mp.cols)[i,j]))-ref_mp[i,j])
            r.append(ref_mp[i,j])
    dn=mp.sqrt(sum(x*x for x in d)); rn=mp.sqrt(sum(x*x for x in r))
    return {'relative_l2':mp.nstr(dn/rn if rn else mp.nan,30),'absolute_l2':mp.nstr(dn,30),
            'reference_l2':mp.nstr(rn,30),'max_abs_component':mp.nstr(max(abs(x) for x in d),30)}
def branch_record(mpmod,p,F,old,z,mp):
    b=mpmod.mp_branch_values(p,old,z)
    return {'dl':mp.nstr(b['lambda_increment'],32),'y':mp.nstr(b['yield_residual'],32),
            'fb_denominator':mp.nstr(b['fb_denominator'],32),
            'dl_sign':int(mp.sign(b['lambda_increment'])),'y_sign':int(mp.sign(b['yield_residual']))}
def main():
    if OUT.exists() and any(OUT.iterdir()) and (OUT/'report.json').exists():
        raise SystemExit('preserving existing report; choose a fresh output namespace')
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    # Bound source arrays and exact mapping before solving.
    cap=np.load(CAPTURE); mpa=np.load(MP_ARRAYS); c=np.load(CENTER)
    p=mpa['paper_parameters_f32'][ROW].astype(np.float64)
    F=c['F'][ROW].astype(np.float64); old=cap['accepted_paper_history'][ROW].astype(np.float64)
    if not np.array_equal(F,cap['F_center'][ROW]): raise SystemExit('run016 center F differs from bound capture')
    if not np.array_equal(p,mpa['paper_parameters_f32'][ROW].astype(np.float64)): raise SystemExit('parameter widening mismatch')
    if int(mpa['paper_material_index'][ROW])!=int(cap['material'][TET]): raise SystemExit('material row/tet join mismatch')
    if int(cap['material'][TET])!=0: raise SystemExit('unexpected material identity; expected liner index 0')
    if int(c['tets'][ROW])!=TET or int(c['rows'][ROW])!=ROW: raise SystemExit('run016 row/tet identity mismatch')
    if not bool(c['valid'][ROW]): raise SystemExit('bound FP64 center is not strict-pass')
    # Strict v2 public API replay on exactly this point.
    sys.path.insert(0,str(V2_TOOLS))
    v1=loadmod('cardboard_compensated_fp64_base','/Users/home/cardboard-implementation-20261014/analysis/native174-fp64-referencev2-source173-portable-003/r-ready/matter/tools/cardboard_compensated_fp64_base.py')
    loadmod('cardboard_dd_fp64',V2_TOOLS/'cardboard_dd_fp64.py')
    v2=loadmod('cardboard_compensated_fp64_reference',V2_TOOLS/'cardboard_compensated_fp64_reference.py')
    base=v1.import_reference(SOURCE173); ref=v2.Reference(base)
    z64,it64,rn64,cn64,ok64,reason64,meta=ref.project_with_metrics(p[None,:],F[None,:,:],old[None,:])
    if not bool(ok64[0]): raise SystemExit(f'v2 strict center failed: {reason64[0]}')
    z64=z64[0]; h=np.zeros((1,3,3),dtype=np.float64); h[0,2,0]=1.0
    tan64=v2.tangent(base,p[None,:],F[None,:,:],old[None,:],z64[None,:],h)[0]
    stress64=base.stress(p[None,:],F[None,:,:],z64[None,:])[0]
    # Import the retained independent 166 MP equation solver. It uses no FP64/native root seed.
    mpmod=loadmod('native469_mp_solver',MP_SOLVER); mp=mpmod.mp
    center_mp={}
    for dps in (70,90,110):
        tic=time.monotonic(); sol=mpmod.solve_one(p,F,old,dps)
        if not sol['ok']: raise SystemExit(f'MP center solve failed at {dps}: {sol["reason"]}')
        mp.mp.dps=dps
        stress=mpmod.mp_stress(p,F,sol['z']); tangent=mpmod.mp_tangent(p,F,old,sol['z'])
        center_mp[dps]={'solution':sol,'stress':stress,'tangent':tangent,
                        'branch':branch_record(mpmod,p,F,old,sol['z'],mp),
                        'elapsed_seconds':time.monotonic()-tic}
    mp.mp.dps=110
    zmp=center_mp[110]['solution']['z']; tmp=center_mp[110]['tangent']
    # Center comparisons isolate same-state stress arithmetic from the root difference.
    zmp64=np.array([float(x) for x in zmp],dtype=np.float64)
    mpstress_at_mp=mpmod.mp_stress(p,F,zmp)
    mpstress_at_z64=mpmod.mp_stress(p,F,[mp.mpf(float(x)) for x in z64])
    stress_fp_at_mp=base.stress(p[None,:],F[None,:,:],zmp64[None,:])[0]
    # Source173 tangent is 3x3 for one F[2,0] perturb; MP matrix column uses row-major F order.
    tan_mp_col=np.array([float(tmp[k,COL]) for k in range(9)],dtype=np.float64).reshape(3,3)
    center_summary={
      'row':ROW,'tet':TET,'material_index':int(cap['material'][TET]),'material_name':'hajali2009_liner_hill_bilinear_fit',
      'direction':{'F_component':list(FCOMP),'row_major_column':COL,'input_direction_norm':1.0},
      'F_center_hex':[[float(x).hex() for x in row] for row in F],
      'F_center':F.tolist(),'accepted_history':old.tolist(),'parameters_f32_promoted':p.tolist(),
      'strict_fp64_v2':{'valid':bool(ok64[0]),'iterations':int(it64[0]),'scaled_residual':float(rn64[0]),
        'dynamic_correction':float(cn64[0]),'residual_gate':RESID_GATE,'correction_gate':CORR_GATE,
        'state_delta_vs_run016_max_abs':float(np.max(np.abs(z64-c['state'][ROW]))),
        'state':z64.tolist()},
      'mp_centers':{},
      'comparisons':{
        'fp64_tangent_vs_mp110_ift':metrics(tan64,tan_mp_col),
        'fp64_stress_at_fp64_root_vs_mp_stress_at_mp_root':metrics(stress64,mparray(mpstress_at_mp,mp)),
        'fp64_stress_at_mp_root_vs_mp_stress_at_same_mp_root':metrics(stress_fp_at_mp,mparray(mpstress_at_mp,mp)),
        'fp64_stress_at_fp64_root_vs_mp_stress_at_same_fp64_root':metrics(stress64,mparray(mpstress_at_z64,mp)),
        'mp_tangent_center_90_vs_110':metrics(mparray(center_mp[90]['tangent'],mp),mparray(tmp,mp)),
        'mp_tangent_center_70_vs_110':metrics(mparray(center_mp[70]['tangent'],mp),mparray(tmp,mp)),
        'mp_center_state_90_vs_110_max_abs':max(abs(center_mp[90]['solution']['z'][k]-zmp[k]) for k in range(15)).__str__(),
        'mp_center_state_70_vs_110_max_abs':max(abs(center_mp[70]['solution']['z'][k]-zmp[k]) for k in range(15)).__str__(),
      }}
    for dps,x in center_mp.items():
        sol=x['solution']; center_summary['mp_centers'][str(dps)]={
            'strict_pass':bool(sol['ok']),'iterations':int(sol['iterations']),
            'scaled_residual':mp.nstr(sol['residual'],40),'dynamic_correction':mp.nstr(sol['correction'],40),
            'reason':sol['reason'],'branch':x['branch'],'state':[mp.nstr(v,35) for v in sol['z']],
            'tangent_action_l2':mp.nstr(mp.sqrt(sum(x['tangent'][k,COL]**2 for k in range(9))),32),
            'elapsed_seconds':x['elapsed_seconds']}
    # Bound and consume the exact unresolved run019 cases for this row+direction.
    records=[]
    for line in RUN019_RECORDS.read_text().splitlines():
        rec=json.loads(line)
        if rec.get('paper_row')==ROW and rec.get('tet')==TET and rec.get('direction_index')==COL and 30.5 <= float(rec.get('epsilon_power', -1)) <= 32.0:
            records.append(rec)
    powers=sorted(float(r['epsilon_power']) for r in records)
    expected=[30.5+i*.25 for i in range(7)]
    if powers!=expected: raise SystemExit(f'run019 row/direction record set mismatch: {powers}')
    # Retain FP64 strict endpoint and MP90 endpoint solutions for each exact FP64-realized perturbation.
    endpoint_rows=[]; stress_mp_by_power={}; stress64_by_power={}; stress_mp_at_fp_state_by_power={}
    for rec in sorted(records,key=lambda x:x['epsilon_power']):
        power=float(rec['epsilon_power']); eps=2.0**(-power)
        fplus=F.copy(); fminus=F.copy(); fplus[FCOMP]+=eps; fminus[FCOMP]-=eps
        dplus=float(fplus[FCOMP]-F[FCOMP]); dminus=float(F[FCOMP]-fminus[FCOMP])
        row={'epsilon_power':power,'nominal_epsilon':eps,'realized_plus_delta':dplus,'realized_minus_delta':dminus,
             'recorded_realized_plus_delta':rec['realized_plus_delta'],'recorded_realized_minus_delta':rec['realized_minus_delta'],
             'F_center_component_hex':float(F[FCOMP]).hex(),'F_plus_component_hex':float(fplus[FCOMP]).hex(),
             'F_minus_component_hex':float(fminus[FCOMP]).hex(),'run019':{k:rec[k] for k in ('plus_strict','minus_strict','plus_residual','minus_residual','plus_correction','minus_correction','plus_iterations','minus_iterations','plus_same_center_branch','minus_same_center_branch','centered_relative_l2','centered_absolute_l2','plus_one_sided_relative_l2','minus_one_sided_relative_l2')},
             'fp64':{},'mp90':{}}
        for side,ff in [('plus',fplus),('minus',fminus)]:
            z,it,rn,cn,ok,why,_=ref.project_with_metrics(p[None,:],ff[None,:,:],old[None,:])
            if not bool(ok[0]): row['fp64'][side]={'strict_pass':False,'iterations':int(it[0]),'scaled_residual':float(rn[0]),'dynamic_correction':float(cn[0]),'reason':str(why[0])}
            else:
                s=base.stress(p[None,:],ff[None,:,:],z)[0]
                row['fp64'][side]={'strict_pass':True,'iterations':int(it[0]),'scaled_residual':float(rn[0]),'dynamic_correction':float(cn[0])}
                stress64_by_power.setdefault(power,{})[side]=s
                mp.mp.dps=90
                mp_same=mpmod.mp_stress(p,ff,[mp.mpf(float(x)) for x in z[0]])
                stress_mp_at_fp_state_by_power.setdefault(power,{})[side]=mp_same
                row['fp64'][side]['stress']=s.tolist(); row['fp64'][side]['state']=z[0].tolist()
                row['fp64'][side]['mp_stress_at_exact_fp64_state']=[[mp.nstr(mp_same[i,j],35) for j in range(3)] for i in range(3)]
                row['fp64'][side]['fp64_stress_vs_mp_same_state']=metrics(s,mparray(mp_same,mp))
            mp.mp.dps=90
            smp=mpmod.solve_one(p,ff,old,90)
            if not smp['ok']:
                row['mp90'][side]={'strict_pass':False,'iterations':int(smp['iterations']),'scaled_residual':mp.nstr(smp['residual'],35),'dynamic_correction':mp.nstr(smp['correction'],35),'reason':smp['reason'],'history':smp['history']}
            else:
                br=branch_record(mpmod,p,ff,old,smp['z'],mp)
                sig=mpmod.mp_stress(p,ff,smp['z']); stress_mp_by_power.setdefault(power,{})[side]=sig
                row['mp90'][side]={'strict_pass':True,'iterations':int(smp['iterations']),'scaled_residual':mp.nstr(smp['residual'],35),'dynamic_correction':mp.nstr(smp['correction'],35),'branch':br,'stress':[[mp.nstr(sig[i,j],35) for j in range(3)] for i in range(3)],'state':[mp.nstr(v,35) for v in smp['z']]}
        row['fp64_endpoint_gate_unchanged']=bool(row['fp64']['plus']['strict_pass'] and row['fp64']['minus']['strict_pass'])
        row['mp_endpoint_gate_unchanged']=bool(row['mp90']['plus']['strict_pass'] and row['mp90']['minus']['strict_pass'])
        endpoint_rows.append(row)
    # Build secants with each side's actual representable spacing; compare both
    # FP64 and MP secants with independently computed center IFTs.
    secants=[]
    for er in endpoint_rows:
        power=er['epsilon_power']; dp=er['realized_plus_delta']; dm=er['realized_minus_delta']
        item={'epsilon_power':power,'branch_preserving_mp':False,'fp64':None,'mp90':None}
        if power in stress64_by_power and set(stress64_by_power[power])=={'plus','minus'}:
            sec=(stress64_by_power[power]['plus']-stress64_by_power[power]['minus'])/(dp+dm)
            sp=(stress64_by_power[power]['plus']-stress64)/dp
            sm=(stress64-sp*dp-stress64_by_power[power]['minus']) if False else (stress64-stress64_by_power[power]['minus'])/dm
            item['fp64']={'centered_vs_fp64_tangent':metrics(sec,tan64),'centered_vs_mp110_ift':metrics(sec,tan_mp_col),
                          'plus_one_sided_vs_mp110_ift':metrics(sp,tan_mp_col),'minus_one_sided_vs_mp110_ift':metrics(sm,tan_mp_col),
                          'centered_stress_action_l2':norm(sec)}
            if power in stress_mp_at_fp_state_by_power and set(stress_mp_at_fp_state_by_power[power])=={'plus','minus'}:
                ep=stress64_by_power[power]['plus']-mparray(stress_mp_at_fp_state_by_power[power]['plus'],mp)
                em=stress64_by_power[power]['minus']-mparray(stress_mp_at_fp_state_by_power[power]['minus'],mp)
                eval_noise=(ep-em)/(dp+dm)
                mp_plus_np=mparray(stress_mp_by_power[power]['plus'],mp); mp_minus_np=mparray(stress_mp_by_power[power]['minus'],mp)
                mp_root_signal=mp_plus_np-mp_minus_np
                root_plus=mparray(stress_mp_at_fp_state_by_power[power]['plus'],mp)-mp_plus_np
                root_minus=mparray(stress_mp_at_fp_state_by_power[power]['minus'],mp)-mp_minus_np
                root_shift_secant=(root_plus-root_minus)/(dp+dm)
                fp64_secant=sec
                mp_secant=mp_root_signal/(dp+dm)
                item['fp64']['same_fp64_state_stress_evaluation_discrepancy']={
                    'plus_endpoint_abs_l2':norm(ep),'minus_endpoint_abs_l2':norm(em),
                    'endpoint_error_difference_l2_pa':norm(ep-em),
                    'secant_error_action_l2_pa_per_F':norm(eval_noise),
                    'secant_error_relative_to_mp_ift':norm(eval_noise)/norm(tan_mp_col),
                    'secant_error_max_abs_component_pa_per_F':float(np.max(np.abs(eval_noise))),
                    'mp_root_endpoint_stress_difference_l2_pa':norm(mp_root_signal),
                    'mp_root_secant_action_l2_pa_per_F':norm(mp_secant),
                    'same_fp64_state_eval_error_to_mp_root_signal_ratio':norm(ep-em)/max(norm(mp_root_signal),1e-300),
                    'mp_projection_root_shift_secant_action_l2_pa_per_F':norm(root_shift_secant),
                    'fp64_secant_vs_mp_root_secant':metrics(fp64_secant,mp_secant),
                    'label':'same-state FP64-minus-MP endpoint stress discrepancies and MP-root versus FP64-root projection shift; finite deterministic point comparisons, not a repeated-evaluation noise floor'}
        if power in stress_mp_by_power and set(stress_mp_by_power[power])=={'plus','minus'}:
            mp.mp.dps=110
            sigp=stress_mp_by_power[power]['plus']; sigm=stress_mp_by_power[power]['minus']; sig0=mpstress_at_mp
            sec=(sigp-sigm)/(mp.mpf(dp)+mp.mpf(dm)); sp=(sigp-sig0)/mp.mpf(dp); sm=(sig0-sigm)/mp.mpf(dm)
            item['branch_preserving_mp']=(endpoint_rows[len(secants)]['mp90']['plus']['branch']['dl_sign']==center_summary['mp_centers']['110']['branch']['dl_sign'] and endpoint_rows[len(secants)]['mp90']['minus']['branch']['dl_sign']==center_summary['mp_centers']['110']['branch']['dl_sign'] and endpoint_rows[len(secants)]['mp90']['plus']['branch']['y_sign']==center_summary['mp_centers']['110']['branch']['y_sign'] and endpoint_rows[len(secants)]['mp90']['minus']['branch']['y_sign']==center_summary['mp_centers']['110']['branch']['y_sign'])
            sec_np=mparray(sec,mp); sp_np=mparray(sp,mp); sm_np=mparray(sm,mp)
            item['mp90']={'centered_vs_mp110_ift':metrics(sec_np,tan_mp_col),'centered_vs_mp90_ift':metrics(sec_np,mparray(center_mp[90]['tangent'],mp)[:,COL].reshape(3,3)),
                          'plus_one_sided_vs_mp110_ift':metrics(sp_np,tan_mp_col),'minus_one_sided_vs_mp110_ift':metrics(sm_np,tan_mp_col),
                          'centered_stress_action_l2':norm(sec_np)}
        secants.append(item)
    # Record input bindings and source lineage. Results remain a selected-point CPU audit.
    bindings={str(path):sha(path) for path in [CAPTURE,MP_ARRAYS,MP_SOLVER,SOURCE173,CENTER,RUN019/'manifest.json',RUN019/'adjudication.json',RUN019_RECORDS,
        V2_TOOLS/'cardboard_compensated_fp64_base.py',V2_TOOLS/'cardboard_compensated_fp64_reference.py',V2_TOOLS/'cardboard_dd_fp64.py',ROOT/'analysis/native174-fp64-referencev2-source173-portable-003/manifest.json']}
    result={'schema':'numi.cardboard.tet172-mp-ift-fd-audit.v1','inputs':bindings,
      'scope':{'paper_row':ROW,'tet':TET,'material_index':int(cap['material'][TET]),'material_name':'hajali2009_liner_hill_bilinear_fit',
        'F_component':list(FCOMP),'row_major_column':COL,'epsilon_powers':powers,'mp_center_precisions_decimal_digits':[70,90,110],
        'mp_endpoint_precision_decimal_digits':90,'native':False,'physical_steps':0},
      'equation_and_gate_contract':{'MP_solver_source':'retained independent MP authored equations in native469-mp-fallback-005; exact binary64 input widening; elastic-history seed only; no native/FP64 candidate seed',
        'FP64':'public source173 v2 Reference.project_with_metrics strict API',
        'strict_scaled_residual_max':RESID_GATE,'strict_dynamic_correction_max':CORR_GATE,'centered_derivative_relative_l2_max':FD_GATE,
        'branch_policy':'Only same center branch comparisons count as smooth; branch transitions reported one-sided and do not pass smooth derivative gate.',
        'noise_policy':'No absolute noise floor inferred from deterministic repeatability or isolated epsilon passes.'},
      'center':center_summary,'endpoints':endpoint_rows,'secants':secants,
      'elapsed_seconds':time.monotonic()-t0,
      'interpretation':'Selected CPU-only independent row/direction audit. It can resolve or refute this one row/direction but cannot close the full nine-basis or native accuracy gates.'}
    (OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT/'report.json'),'elapsed_seconds':result['elapsed_seconds'],'center':center_summary['comparisons'],'endpoint_count':len(endpoint_rows),'strict_fp64_endpoint_pairs':sum(x['fp64_endpoint_gate_unchanged'] for x in endpoint_rows),'strict_mp_endpoint_pairs':sum(x['mp_endpoint_gate_unchanged'] for x in endpoint_rows),'mp_smooth_rows':sum(x['branch_preserving_mp'] for x in secants)},sort_keys=True))
if __name__=='__main__': main()
