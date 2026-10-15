#!/usr/bin/env python3
"""MP90 endpoint-projection and finite-difference audit for captured medium points.

This is a read-only CPU replay. It imports equations from the immutable 1634-001
reference but does not rewrite or execute that package's main/report path.
"""
from pathlib import Path
import hashlib, importlib.util, json, platform, struct, sys, time
import numpy as np
import mpmath as mp

BASE = Path('/Users/home/cardboard-implementation-20261014')
PRIOR = BASE/'analysis/medium-tangent-state-sensitivity-1634-001'
PRIOR_SCRIPT = PRIOR/'replay.py'
PRIOR_REPORT = PRIOR/'report.json'
PRIOR_MANIFEST = PRIOR/'manifest.json'
CAP = BASE/'solver/recycle-source-default-first-control-283/operator-capture'
NATIVE_JSONL = BASE/'native/rationalized-all-paper-entry-points-300/output/paper-points-accurate.jsonl'
OUT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location('medium1634_reference', PRIOR_SCRIPT)
refmod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refmod
spec.loader.exec_module(refmod)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def f32(x): return float(np.float32(x))
def flat_f32(x): return [f32(v) for v in np.asarray(x).reshape(-1)]
def matrix_flat_mp(x): return [mp.mpf(x[i//3,i%3]) for i in range(9)]
def mpf(x):
    if isinstance(x, mp.mpf): return x
    n,d=float(x).as_integer_ratio()
    return mp.mpf(n)/d

def norm_scaled(r):
    scale=[mp.mpf('1e4')]*6+[mp.mpf(1)]*6+[mp.mpf('1e4')]*3
    return max(abs(r[i])*scale[i] for i in range(15))

def state_branch(p, old, z):
    a,b,c,d,e1,v=[z[i] for i in range(6,12)]
    q=mp.sqrt(p[11]*(b-c)**2+p[12]*(c-a)**2+p[13]*(a-b)**2+
              2*p[14]*d*d+2*p[15]*e1*e1+2*p[16]*v*v+p[25]**2)
    y=1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q
    dl=z[12]-old[12]
    return {
        'delta_multiplier':mp.nstr(dl,25),
        'delta_multiplier_abs':mp.nstr(abs(dl),18),
        'delta_multiplier_relative_to_center':None,
        'yield_gap':mp.nstr(y,25),
        'hill_q':mp.nstr(q,25),
        'yield_sign':'nonnegative' if y >= 0 else 'negative',
        'fb_selection':'elastic_smoothed' if y > dl else 'plastic_smoothed',
        'fb_a_plus_b':mp.nstr(dl+y,25),
        'fb_product':mp.nstr(dl*y,25),
        'fb_smoothing_parameter':mp.nstr(p[26],18),
        'fb_product_over_epsilon_squared_quarter':mp.nstr(
            (dl*y)/(p[26]**2/4) if p[26] != 0 else mp.nan,18),
        'hardening_factor':mp.nstr(1+y+q,25),
    }

def project_endpoint(residual, p, old, zstart, tol, max_iter=40):
    """Safeguarded MP Newton; keeps accepted history fixed and returns full status."""
    z=mp.matrix([mpf(v) for v in zstart])
    trace=[]
    terminal='iteration_limit'
    for it in range(max_iter):
        r=mp.matrix(residual(*list(z)))
        nr=norm_scaled(r)
        J=refmod.jacobian(residual,z)
        delta=mp.lu_solve(J,-r)
        correction=max(abs(delta[i]) for i in range(15))
        scaled_correction=max(abs(delta[i])/max(abs(z[i]),abs(old[i]),mp.mpf('1e-8')) for i in range(15))
        record={'iteration':it,'scaled_residual':mp.nstr(nr,12),
                'correction_inf':mp.nstr(correction,12),
                'relative_scaled_correction_inf':mp.nstr(scaled_correction,12),
                'state':state_branch(p,old,z)}
        if nr <= tol and correction <= tol:
            record['accepted_alpha']=0
            record['status']='residual_and_correction_converged'
            trace.append(record); terminal='converged'; break
        accepted=False; alpha=mp.mpf(1)
        for backtrack in range(120):
            candidate=z+alpha*delta
            candidate_norm=norm_scaled(mp.matrix(residual(*list(candidate))))
            if mp.isfinite(candidate_norm) and candidate_norm < nr:
                record['accepted_alpha']=mp.nstr(alpha,18)
                record['backtracks']=backtrack
                record['trial_scaled_residual']=mp.nstr(candidate_norm,12)
                record['trial_state']=state_branch(p,old,candidate)
                record['status']='accepted_newton_step'
                z=candidate; accepted=True; break
            alpha/=2
        if not accepted:
            record['accepted_alpha']=None
            record['backtracks']=120
            record['status']='line_search_stalled'
            trace.append(record); terminal='line_search_stalled'; break
        trace.append(record)
    # A final untouched Newton correction makes the terminal correction explicit.
    final_r=mp.matrix(residual(*list(z)))
    final_J=refmod.jacobian(residual,z)
    final_delta=mp.lu_solve(final_J,-final_r)
    final_correction=max(abs(final_delta[i]) for i in range(15))
    final_scaled_correction=max(abs(final_delta[i])/
        max(abs(z[i]),abs(old[i]),mp.mpf('1e-8')) for i in range(15))
    final_norm=norm_scaled(final_r)
    converged=(terminal=='converged' and final_norm <= tol and final_correction <= tol)
    return z, {
        'status':terminal if converged else ('not_converged_'+terminal),
        'converged':bool(converged),
        'iterations':len([x for x in trace if x['status']=='accepted_newton_step']),
        'trace':trace,
        'terminal_scaled_residual':mp.nstr(final_norm,18),
        'terminal_newton_correction_inf':mp.nstr(final_correction,18),
        'terminal_relative_scaled_correction_inf':mp.nstr(final_scaled_correction,18),
        'terminal_branch':state_branch(p,old,z),
    }

def mp_rel_matrix(a,b):
    num=mp.sqrt(sum((a[i,j]-b[i,j])**2 for i in range(a.rows) for j in range(a.cols)))
    den=mp.sqrt(sum(b[i,j]**2 for i in range(b.rows) for j in range(b.cols)))
    return mp.nstr(num/den if den else mp.nan,18)

def mp_rel_vec(a,b):
    num=mp.sqrt(sum((a[i]-b[i])**2 for i in range(a.rows)))
    den=mp.sqrt(sum(b[i]**2 for i in range(b.rows)))
    return mp.nstr(num/den if den else mp.nan,18)

def mp_to_float_matrix(M):
    return [[float(M[i,j]) for j in range(M.cols)] for i in range(M.rows)]
def mat_from_flat(xs):
    M=mp.matrix(3,3)
    for i,x in enumerate(xs): M[i//3,i%3]=mpf(x)
    return M

def make_equations(pf, ff, oldf):
    return refmod.system(pf,ff,oldf)

def run_point(row, header, native_map, epsilon_strings):
    env,tet=row['environment'],row['tet_id']; c=row['center']
    # Set precision before any of the captured input words are widened to MP.
    mp.mp.dps=90
    p=[f32(v) for v in header['parameters_by_environment'][env][28:56]]
    old=flat_f32(c['accepted_state'])
    fh=flat_f32(c['F_local']); fl=flat_f32(c.get('F_local_low',np.zeros_like(c['F_local'])))
    f=[mpf(a)+mpf(b) for a,b in zip(fh,fl)]
    h=flat_f32(c['F_direction_local_per_s'])
    native=native_map[(env,tet)]
    zstart=[mpf(a)+mpf(b) for a,b in zip(flat_f32(native['projected_state_high']),
                                          flat_f32(native['projected_state_low']))]
    pmp,Fmp,oldmp,emp,residual,stress=make_equations(p,f,old)
    tolerance=mp.mpf('1e-70')
    zcenter,center_projection=project_endpoint(residual,pmp,oldmp,zstart,tolerance)
    center_stress=mp.matrix([stress(*list(zcenter))[i] for i in range(9)])
    analytic=refmod.mp_tangent_matrix(p,f,old,zcenter)
    analytic_action=analytic*mp.matrix([mpf(x) for x in h])
    center_branch=state_branch(pmp,oldmp,zcenter)
    center_lambda=mp.mpf(center_branch['delta_multiplier'])
    results=[]

    for eps_text in epsilon_strings:
        eps=mp.mpf(eps_text)
        columns=[]; plus_actions=[]; minus_actions=[]; endpoints=[]; transitions=[]
        for col in range(9):
            direction=[mp.mpf(0)]*9; direction[col]=mp.mpf(1)
            fplus=[f[i]+eps*direction[i] for i in range(9)]
            fminus=[f[i]-eps*direction[i] for i in range(9)]
            pp,pF,po,pe,rp,sp=make_equations(p,fplus,old)
            pm,mF,mo,me,rm,sm=make_equations(p,fminus,old)
            zp,diagp=project_endpoint(rp,pp,po,zcenter,tolerance)
            zm,diagm=project_endpoint(rm,pm,mo,zcenter,tolerance)
            stressp=mp.matrix([sp(*list(zp))[i] for i in range(9)])
            stressm=mp.matrix([sm(*list(zm))[i] for i in range(9)])
            plus=(stressp-center_stress)/eps
            minus=(center_stress-stressm)/eps
            central=(stressp-stressm)/(2*eps)
            columns.append(central); plus_actions.append(plus); minus_actions.append(minus)
            bp=diagp['terminal_branch']; bm=diagm['terminal_branch']
            same_branch=(bp['yield_sign']==bm['yield_sign'] and bp['fb_selection']==bm['fb_selection'])
            if not same_branch:
                transitions.append(col)
            bp['delta_multiplier_relative_to_center']=mp.nstr(
                (mpf(bp['delta_multiplier'])-center_lambda)/center_lambda,18)
            bm['delta_multiplier_relative_to_center']=mp.nstr(
                (mpf(bm['delta_multiplier'])-center_lambda)/center_lambda,18)
            endpoints.append({'input_direction_column_row_major':col,
                'plus':{'projection':diagp,'state_branch':bp},
                'minus':{'projection':diagm,'state_branch':bm},
                'same_yield_and_FB_branch':same_branch,
                'central_action_vs_analytic_column_rel':mp_rel_vec(central,analytic[:,col]),
                'plus_one_sided_vs_analytic_column_rel':mp_rel_vec(plus,analytic[:,col]),
                'minus_one_sided_vs_analytic_column_rel':mp_rel_vec(minus,analytic[:,col]),
                'plus_minus_one_sided_relative_gap':mp_rel_vec(plus,minus)})
        central_matrix=mp.matrix(9,9)
        plus_matrix=mp.matrix(9,9); minus_matrix=mp.matrix(9,9)
        for col in range(9):
            for rowid in range(9):
                central_matrix[rowid,col]=columns[col][rowid]
                plus_matrix[rowid,col]=plus_actions[col][rowid]
                minus_matrix[rowid,col]=minus_actions[col][rowid]
        results.append({'epsilon_F_entry':eps_text,
            'perturbation_definition':'Each matrix column uses exact MP90 F_plus/minus = captured F +/- epsilon_F_entry * row-major unit basis; accepted history and parameter words are identical to center.',
            'central_matrix_relative_error_vs_MP90_implicit':mp_rel_matrix(central_matrix,analytic),
            'plus_one_sided_matrix_relative_error_vs_MP90_implicit':mp_rel_matrix(plus_matrix,analytic),
            'minus_one_sided_matrix_relative_error_vs_MP90_implicit':mp_rel_matrix(minus_matrix,analytic),
            'successive_one_sided_matrix_gap':mp_rel_matrix(plus_matrix,minus_matrix),
            'branch_transition_columns':transitions,
            'all_endpoint_projections_converged':all(
                e['plus']['projection']['converged'] and e['minus']['projection']['converged'] for e in endpoints),
            'central_matrix':mp_to_float_matrix(central_matrix),
            'endpoint_records':endpoints})

    fdot=[]
    for eps_text in epsilon_strings:
        eps=mp.mpf(eps_text)  # seconds for the captured per-second Fdot
        fplus=[f[i]+eps*mpf(h[i]) for i in range(9)]
        fminus=[f[i]-eps*mpf(h[i]) for i in range(9)]
        pp,pF,po,pe,rp,sp=make_equations(p,fplus,old)
        pm,mF,mo,me,rm,sm=make_equations(p,fminus,old)
        zp,diagp=project_endpoint(rp,pp,po,zcenter,tolerance)
        zm,diagm=project_endpoint(rm,pm,mo,zcenter,tolerance)
        stressp=mp.matrix([sp(*list(zp))[i] for i in range(9)])
        stressm=mp.matrix([sm(*list(zm))[i] for i in range(9)])
        plus=(stressp-center_stress)/eps
        minus=(center_stress-stressm)/eps
        central=(stressp-stressm)/(2*eps)
        fdot.append({'epsilon_seconds':eps_text,
            'perturbation_definition':'F_plus/minus = captured F +/- epsilon_seconds * captured F_direction_local_per_s; same accepted history.',
            'plus_projection':diagp,'minus_projection':diagm,
            'plus_one_sided_relative_error_vs_MP90_analytic_action':mp_rel_vec(plus,analytic_action),
            'minus_one_sided_relative_error_vs_MP90_analytic_action':mp_rel_vec(minus,analytic_action),
            'central_relative_error_vs_MP90_analytic_action':mp_rel_vec(central,analytic_action),
            'one_sided_action_relative_gap':mp_rel_vec(plus,minus),
            'central_action':[[float(central[i*3+j]) for j in range(3)] for i in range(3)],
            'analytic_action':[[float(analytic_action[i*3+j]) for j in range(3)] for i in range(3)]})

    return {'point':{'environment':env,'tet_id':tet,'material_index':row['material_index'],
                     'source_material':'hajali2009_medium_hill_bilinear_fit'},
        'source_inputs':{'parameters_f32':p,'accepted_history_f32':old,
            'F_high_f32':fh,'F_low_f32':fl,'F_widened_exact_words':[[mp.nstr(f[i],18) for i in range(3*j,3*j+3)] for j in range(3)],
            'F_direction_local_per_s_f32':[[h[i*3+j] for j in range(3)] for i in range(3)],
            'epsilon_smoothing_parameter':mp.nstr(pmp[26],18),
            'Fdot_relationship_preserved':True,
            'accepted_history_policy':'same captured accepted_state is passed to every plus/minus endpoint projection; only F varies.'},
        'center_projection':center_projection,
        'center_branch':center_branch,
        'mp90_analytic_full_matrix':mp_to_float_matrix(analytic),
        'mp90_analytic_Fdot_action':[[float(analytic_action[i*3+j]) for j in range(3)] for i in range(3)],
        'nine_direction_endpoint_sweep':results,
        'captured_Fdot_endpoint_sweep':fdot,
        'interpretation_gate':{'all_endpoint_projections_converged':all(
            x['all_endpoint_projections_converged'] for x in results) and all(
            x['plus_projection']['converged'] and x['minus_projection']['converged'] for x in fdot),
            'no_transition_columns':all(not x['branch_transition_columns'] for x in results),
            'not_a_native_or_material_qualification':True}}

def main():
    # Verify the prior package before treating it as an input artifact.
    prior_manifest=json.loads(PRIOR_MANIFEST.read_text())
    assert sha(PRIOR_SCRIPT)==prior_manifest['artifacts']['replay.py']['sha256']
    assert sha(PRIOR_REPORT)==prior_manifest['artifacts']['report.json']['sha256']
    prior_report=json.loads(PRIOR_REPORT.read_text())
    capture_records=[json.loads(s) for s in CAP.read_text().splitlines() if s.strip()]
    assert capture_records[0]['record']=='header'
    assert capture_records[-1]['record']=='integrity'
    assert capture_records[-1]['production_inputs_bitwise_unchanged'] is True
    assert sha(CAP)==prior_report['bindings']['capture_sha256']
    native_records=[json.loads(s) for s in NATIVE_JSONL.read_text().splitlines() if s.strip()]
    assert native_records[0]['capture_sha256']==sha(CAP)
    rows={(r['environment'],r['tet_id']):r for r in capture_records if r.get('record')=='tetrahedron'}
    native={(r['environment'],r['tet_id']):r for r in native_records if r.get('record')=='point'}
    epsilon_strings=['1e-8','1e-9','1e-10','1e-11','1e-12','1e-13','1e-14']
    started=time.time()
    reports=[run_point(rows[(0,t)],capture_records[0],native,epsilon_strings) for t in (1634,1638)]
    report={'schema':'numi.cardboard.medium-tangent-mp90-endpoint-fd.v1',
        'classification':'Independent MP90 endpoint re-projection and full 9x9 central/one-sided finite-difference sweep at two previously captured medium points. Captured F, Fdot, accepted history and native/source bindings are reused. This is numerical diagnostic evidence, not a native policy or material qualification.',
        'bindings':{'prior_package':str(PRIOR),'prior_replay_sha256':sha(PRIOR_SCRIPT),
            'prior_report_sha256':sha(PRIOR_REPORT),'prior_manifest_sha256':sha(PRIOR_MANIFEST),
            'capture_path':str(CAP),'capture_sha256':sha(CAP),
            'native_jsonl_path':str(NATIVE_JSONL),'native_jsonl_sha256':sha(NATIVE_JSONL),
            'source_snapshot':prior_report['bindings']['source_snapshot'],
            'frozen_source_manifest_sha256':prior_report['bindings']['frozen_source_manifest_sha256'],
            'medium_asset_sha256':prior_report['bindings']['medium_asset_sha256']},
        'numerics':{'mpmath_dps':90,'endpoint_scaled_residual_and_newton_correction_tolerance':'1e-70',
            'epsilon_strings':epsilon_strings,
            'matrix_layout':'row-major stress output rows by row-major F-entry perturbation columns; each 3x3 action flattened row-major before matrix-column placement',
            'Fdot_unit':'1/s; captured directional endpoints use epsilon_seconds * captured Fdot',
            'accepted_history':'fixed captured accepted_state on every endpoint, plus and minus',
            'endpoint_projection':'same independent literal 15-state equations as prior replay; fresh safeguarded MP Newton root at each endpoint; no native history mutation'},
        'results':reports,
        'timing':{'elapsed_seconds':time.time()-started},
        'interpretation':{'observed':None,'gate_open':None,
            'claim_limit':'A passing MP90 endpoint sweep would show this finite-difference comparison is stable for these points/directions/epsilon values. It would not make the native finite-difference gate or material qualification pass.'}}
    out=OUT/'report.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(out),'sha256':sha(out),
        'elapsed_seconds':time.time()-started,
        'points':[{'tet_id':x['point']['tet_id'],'center_projection':x['center_projection']['converged'],
            'basis_relative_errors':[e['central_matrix_relative_error_vs_MP90_implicit'] for e in x['nine_direction_endpoint_sweep']],
            'fdot_relative_errors':[e['central_relative_error_vs_MP90_analytic_action'] for e in x['captured_Fdot_endpoint_sweep']],
            'all_roots_converged':x['interpretation_gate']['all_endpoint_projections_converged'],
            'branch_transitions':[e['branch_transition_columns'] for e in x['nine_direction_endpoint_sweep']]} for x in reports]},indent=2))

if __name__=='__main__': main()
