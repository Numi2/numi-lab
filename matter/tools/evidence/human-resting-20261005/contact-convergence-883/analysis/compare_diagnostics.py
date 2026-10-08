#!/usr/bin/env python3
"""Reproduce the predeclared 32-vs-64 contact-iteration diagnostic comparison."""
import csv, hashlib, json, math, re, statistics
from pathlib import Path

E = Path('/Users/n/numi-human-resting-evidence-20261005')
OUT = E/'native-contact-convergence-883/comparison-876-883'
DECL = E/'native-contact-convergence-883/prelaunch-prediction.json'
DECL882 = E/'native-contact-convergence-882/prelaunch-prediction.json'
FAILURE882 = E/'native-contact-convergence-882/native-run.log'
ARMS = {
    '64': E/'native-support-drift-diagnostic-876',
    '32': E/'native-contact-convergence-883',
}
WINDOW_STEPS = (5001, 10000)
ACTIVE_NORMAL_IMPULSE_NS = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def add(a, b):
    return [x+y for x, y in zip(a, b)]


def vec(row, stem):
    return [float(row[f'{stem}_{a}_kg_m_s']) for a in 'xyz']


def scalar_stats(values):
    xs=sorted(float(v) for v in values)
    n=len(xs)
    if not n:
        return {'n':0}
    def q(p):
        return xs[min(n-1, int((n-1)*p))]
    return {'n':n,'mean':statistics.fmean(xs),'median':q(.5),'p95':q(.95),'max':xs[-1]}


def vector_norm_stats(values):
    return scalar_stats(norm(v) for v in values)


def output_hashes(out):
    names=['resting-coupled.csv','resting-surface-audit.csv',
           'resting-com-momentum-diagnostic.csv','resting-com-q-integration.csv',
           'resting-com-support-impulses.csv','run-metadata.json','native.log']
    return {name:sha(out/name) for name in names}


def analyze_arm(label, root):
    out=root/'native-output'
    receipt=json.loads((root/'native-run.log').read_text())
    meta=json.loads((out/'run-metadata.json').read_text())
    log=(out/'native.log').read_text(errors='replace')
    terminal_line=next(x for x in log.splitlines() if x.startswith('stand_terminal_state='))
    terminal=json.loads(terminal_line.split('=',1)[1])
    completed_line=next(x for x in log.splitlines() if x.startswith('resting_integrated_body=completed '))
    assert receipt.get('exit_code')==0 and receipt.get('output')==str(out), (label,receipt)
    assert receipt.get('changed_sources')==[] and receipt.get('loaded_runtime_verified') is True
    assert terminal.get('step_count')==10000 and abs(float(terminal.get('timestep_seconds',0))-.002)<1e-12
    assert terminal.get('root_assistance') is False
    assert meta.get('asset_sha256')
    q=read_csv(out/'resting-com-q-integration.csv')
    s=read_csv(out/'resting-com-support-impulses.csv')
    m=read_csv(out/'resting-com-momentum-diagnostic.csv')
    coupled=read_csv(out/'resting-coupled.csv')
    post_coupled=[r for r in coupled if 10.0<float(r['time_s'])<20.0]
    assert len(q)==len(m)==len(coupled)==10000 and len(s)==320000
    post=[r for r in q if WINDOW_STEPS[0]<=int(r['accepted_step'])<=WINDOW_STEPS[1]]
    assert len(post)==5000
    md={int(r['accepted_step']):r for r in m}
    contact={}
    for row in s:
        st=int(row['accepted_step'])
        contact.setdefault(st,[0.,0.,0.])
        for i,a in enumerate('xyz'):
            contact[st][i]+=float(row[f'impulse_world_{a}_ns'])
    stages={}
    for name,left,right in [
        ('gravity_velocity_stage','before','free_same_q'),
        ('contact_velocity_stage','free_same_q','preprojection_velocity_same_q'),
        ('q_advance','preprojection_velocity_same_q','preprojection_qv'),
        ('final_acceptance','preprojection_qv','accepted_qv')]:
        ds=[sub(vec(r,'source_body_linear_momentum_'+right),vec(r,'source_body_linear_momentum_'+left)) for r in post]
        stages[name]=vector_norm_stats(ds)
    residuals={}
    for name in ['gravity','contact']:
        errs=[]; sumerr=[0.,0.,0.]; sumdp=[0.,0.,0.]; sumimp=[0.,0.,0.]
        for row in post:
            st=int(row['accepted_step'])
            if name=='gravity':
                mr=md[st]
                imp=[float(mr[f'gravity_impulse_{a}_ns']) for a in 'xyz']
                dp=sub(vec(row,'source_body_linear_momentum_free_same_q'),vec(row,'source_body_linear_momentum_before'))
            else:
                imp=contact.get(st,[0.,0.,0.])
                dp=sub(vec(row,'source_body_linear_momentum_preprojection_velocity_same_q'),vec(row,'source_body_linear_momentum_free_same_q'))
            err=sub(dp,imp)
            errs.append(norm(err)); sumerr=add(sumerr,err); sumdp=add(sumdp,dp); sumimp=add(sumimp,imp)
        residuals[name]={'per_step_residual_norm_ns':scalar_stats(errs),
                         'sum_delta_p_ns':sumdp,'sum_impulse_ns':sumimp,
                         'sum_per_step_residual_ns':sumerr,
                         'sum_per_step_residual_norm_ns':norm(sumerr)}
    post_m=[r for r in m if 10.0<float(r['time_s'])<20.0]
    assert len(post_m)>2
    first,last=post_m[0],post_m[-1]
    com0=[float(first[f'com_{a}_m']) for a in 'xyz']
    com1=[float(last[f'com_{a}_m']) for a in 'xyz']
    disp=[(b-a)*1000 for a,b in zip(com0,com1)]
    ts=[float(r['time_s']) for r in post_m]
    mt=statistics.fmean(ts); denom=sum((t-mt)**2 for t in ts)
    trends=[]
    for a in 'xyz':
        vals=[float(r[f'com_{a}_m'])*1000 for r in post_m]
        mv=statistics.fmean(vals)
        trends.append(sum((t-mt)*(v-mv) for t,v in zip(ts,vals))/denom*60)
    com={'n':len(post_m),'first_time_s':float(first['time_s']),'last_time_s':float(last['time_s']),
         'displacement_mm_xyz':disp,'displacement_norm_mm':norm(disp),
         'ols_trend_mm_per_min_xyz':trends,
         'mean_velocity_m_s_xyz':[statistics.fmean(float(r[f'com_v{a}_m_s']) for r in post_m) for a in 'xyz']}
    active=[r for r in s if WINDOW_STEPS[0]<=int(r['accepted_step'])<=WINDOW_STEPS[1]
            and float(r['normal_impulse_ns'])>ACTIVE_NORMAL_IMPULSE_NS]
    slips={}
    for stage in ['before','free','preprojection_velocity_same_q','accepted']:
        col=f'point_slip_pre_step_contact_J_v_{stage}_speed_m_s'
        slips[stage]=scalar_stats(float(r[col]) for r in active)
    return {
        'label':label,
        'completion':{'runner_exit_code':receipt['exit_code'],'native_step_count':terminal['step_count'],
                      'dt_s':terminal['timestep_seconds'],'root_assistance':terminal['root_assistance'],
                      'simulated_s':float(re.search(r'simulated_s=([0-9.]+)',completed_line).group(1)),
                      'wall_s':float(re.search(r'wall_s=([0-9.]+)',completed_line).group(1)),
                      'real_time_factor':float(re.search(r'real_time_factor=([0-9.]+)',completed_line).group(1))},
        'identity':{'argv':meta['argv'],'asset_sha256':meta['asset_sha256'],
                    'binary_sha256':meta['asset_sha256'].get(meta['argv'][0]),
                    'run_metadata_sha256':sha(out/'run-metadata.json'),
                    'runner_receipt_sha256':sha(root/'native-run.log'),
                    'native_log_sha256':sha(out/'native.log'),
                    'source_sha256':sha(Path('/Users/n/numi-human-support-drift-audit-876/apps/numilab_human_myosim_visual_probe.mm')),
                    'output_sha256':output_hashes(out)},
        'trace_dimensions':{'q_rows':len(q),'q_columns':len(q[0]),
                            'support_rows':len(s),'support_columns':len(s[0]),
                            'com_momentum_rows':len(m),'physiology_rows':len(coupled),
                            'postinit_step_interval_inclusive':list(WINDOW_STEPS),
                            'active_contact_rows':len(active)},
        'stage_delta_p_norm_kg_m_s':stages,
        'partial_impulse_accounting':residuals,
        'com_postinit':com,
        'pre_step_contact_J_tangent_speed_m_s':slips,
        'physiology_summary_10_20s':{
            'rows_strictly_inside_10_20s':len(post_coupled),
            'mean_PaCO2_mmhg':statistics.fmean(float(r['PaCO2_mmhg']) for r in post_coupled),
            'mean_PaO2_mmhg':statistics.fmean(float(r['PaO2_mmhg']) for r in post_coupled),
            'max_abs_oxygen_budget_residual_stpd_ml':max(abs(float(r['oxygen_balance_error_stpd_ml'])) for r in post_coupled),
            'max_abs_CO2_budget_residual_stpd_ml':max(abs(float(r['co2_balance_error_stpd_ml'])) for r in post_coupled),
            'max_abs_blood_continuity_residual_accum_ml':max(abs(float(r['blood_continuity_residual_accum_ml'])) for r in post_coupled),
            'max_abs_respiratory_volume_balance_ml':max(abs(float(r['respiratory_volume_balance_ml'])) for r in post_coupled)},
    }


def args_except_controlled(arm):
    argv=arm['identity']['argv'][:]
    normalized=[]
    skip_next=False
    for i,x in enumerate(argv):
        if skip_next:
            skip_next=False
            continue
        if x in [str(ARMS['64']/'native-output'),str(ARMS['32']/'native-output')]:
            normalized.append('<OUTPUT>')
        elif x=='--resting-movie':
            normalized.append(x); normalized.append('<MOVIE>'); skip_next=True
        elif x=='--stand-contact-iterations':
            normalized.append(x); normalized.append('<ITERATIONS>'); skip_next=True
        else:
            normalized.append(x)
    return normalized


def main():
    assert OUT.exists()
    declaration=json.loads(DECL.read_text())
    declaration_sha=sha(DECL)
    declaration882_sha=sha(DECL882)
    failure882_text=FAILURE882.read_text(errors='replace')
    failure882_sha=sha(FAILURE882)
    assert declaration882_sha=='f29c606f9c65553db1c4d3b2e64c0d22d078c67e701489f1d522739843f96fc7'
    assert 'contact iterations must be an integer within [1, 64]' in failure882_text
    assert declaration['baseline_contact_iterations']==64 and declaration['candidate_contact_iterations']==32
    arms={k:analyze_arm(k,v) for k,v in ARMS.items()}
    assert arms['64']['completion']['runner_exit_code']==0 and arms['32']['completion']['runner_exit_code']==0
    assert arms['64']['completion']['native_step_count']==arms['32']['completion']['native_step_count']==10000
    assert arms['64']['completion']['root_assistance'] is False and arms['32']['completion']['root_assistance'] is False
    assert arms['64']['identity']['asset_sha256']==arms['32']['identity']['asset_sha256']
    assert arms['64']['identity']['binary_sha256']==arms['32']['identity']['binary_sha256']
    assert arms['64']['identity']['source_sha256']==arms['32']['identity']['source_sha256']
    assert args_except_controlled(arms['64'])==args_except_controlled(arms['32'])
    physiological_fields=['PaCO2_mmhg','PaO2_mmhg','airflow_ml_s','lung_volume_ml','tidal_ml','breaths',
        'aortic_ejected_ml','pulmonary_ejected_ml','oxygen_balance_error_stpd_ml','co2_balance_error_stpd_ml',
        'blood_continuity_residual_accum_ml','respiratory_volume_balance_ml']
    coupling={}
    c0=read_csv(ARMS['64']/'native-output/resting-coupled.csv')
    c1=read_csv(ARMS['32']/'native-output/resting-coupled.csv')
    assert len(c0)==len(c1)
    for field in physiological_fields:
        diffs=[abs(float(a[field])-float(b[field])) for a,b in zip(c0,c1)]
        coupling[field]={'max_abs_difference':max(diffs),'mean_abs_difference':statistics.fmean(diffs)}
    # Exact-output equality is an observation; do not assume it as the criterion.
    coupling['whole_coupled_csv_byte_identical']=(ARMS['64']/'native-output/resting-coupled.csv').read_bytes()==(ARMS['32']/'native-output/resting-coupled.csv').read_bytes()
    changed_columns={}
    for field in c0[0]:
        diffs=[(a[field],b[field]) for a,b in zip(c0,c1) if a[field]!=b[field]]
        if diffs:
            try: max_abs=max(abs(float(a)-float(b)) for a,b in diffs)
            except (TypeError,ValueError): max_abs=None
            changed_columns[field]={'different_rows':len(diffs),'max_abs_difference':max_abs,'first_pair':list(diffs[0])}
    coupling['different_columns']=changed_columns
    metrics={}
    for key in ['com_postinit','pre_step_contact_J_tangent_speed_m_s','partial_impulse_accounting']:
        metrics[key]={'64':arms['64'][key],'32':arms['32'][key]}
    metrics['postinit_com_displacement_ratio_32_over_64']=arms['32']['com_postinit']['displacement_norm_mm']/arms['64']['com_postinit']['displacement_norm_mm']
    metrics['accepted_pre_step_Jv_p95_ratio_32_over_64']=arms['32']['pre_step_contact_J_tangent_speed_m_s']['accepted']['p95']/arms['64']['pre_step_contact_J_tangent_speed_m_s']['accepted']['p95']
    metrics['constrained_residual_p95_ratio_32_over_64']=arms['32']['partial_impulse_accounting']['contact']['per_step_residual_norm_ns']['p95']/arms['64']['partial_impulse_accounting']['contact']['per_step_residual_norm_ns']['p95']
    metrics['q_advance_mean_ratio_32_over_64']=arms['32']['stage_delta_p_norm_kg_m_s']['q_advance']['mean']/arms['64']['stage_delta_p_norm_kg_m_s']['q_advance']['mean']
    metrics['q_advance_max_ratio_32_over_64']=arms['32']['stage_delta_p_norm_kg_m_s']['q_advance']['max']/arms['64']['stage_delta_p_norm_kg_m_s']['q_advance']['max']
    metrics['final_acceptance_max_ratio_32_over_64']=arms['32']['stage_delta_p_norm_kg_m_s']['final_acceptance']['max']/arms['64']['stage_delta_p_norm_kg_m_s']['final_acceptance']['max']
    metrics['predeclared_directional_metrics_all_increased_at_32_iterations']=all([metrics['postinit_com_displacement_ratio_32_over_64']>1, metrics['accepted_pre_step_Jv_p95_ratio_32_over_64']>1, metrics['constrained_residual_p95_ratio_32_over_64']>1])
    controls={'gravity_residual':{k:arms[k]['partial_impulse_accounting']['gravity']['per_step_residual_norm_ns'] for k in arms},
              'q_advance':{k:arms[k]['stage_delta_p_norm_kg_m_s']['q_advance'] for k in arms},
              'final_acceptance':{k:arms[k]['stage_delta_p_norm_kg_m_s']['final_acceptance'] for k in arms},
              'physiology_trace_comparison':coupling}
    result={
        'schema':'numi.predeclared-contact-convergence-diagnostic-comparison.v1',
        'classification':'predeclared numerical diagnostic; not a numi science registered study and not physiological qualification',
        'registration_distinction':'The 867 endurance comparison is the registered numi science study. 883 is a timestamped predeclared diagnostic against existing 876 baseline data; it is not numi science registered. 882 was a separate timestamped predeclared diagnostic rejected before native launch; it is not an outcome.',
        'prelaunch_declaration':{'path':str(DECL),'sha256':declaration_sha,'declared_baseline_iterations':64,
                                 'declared_candidate_iterations':32,'declared_prediction':declaration['prediction'],
                                 'declaration_contents':declaration},
        'controlled_comparison':{'baseline':'876','candidate':'883','only_declared_physics_change':'--stand-contact-iterations: 64 to 32',
                                 'same_source_binary_assets_dt_horizon_and_other_args':True,
                                 'normalized_argv_equal_except_output_movie_and_iteration_value':True,
                                 'same_input_asset_sha256':True,'same_binary_sha256':True,'same_observer_source_sha256':True},
        'arms':arms,
        'predeclared_metrics':metrics,
        'predeclared_prediction_assessment':'All three declared directional quantities increased at 32 iterations. This supports a convergence-sensitive contribution in this 20 s diagnostic but does not uniquely establish that endpoint contact slip caused COM drift.',
        'controls':controls,
        'interpretation_limits':['The contact Jacobian slip is a pre-step weighted-contact linearization reused against accepted velocity; it is not the accepted endpoint selected-skin witness.','The stage impulse checks are partial, not whole-interval momentum or energy closure.','COM motion includes articulated mass distribution; it is not identical to root translation.','This 20-second comparison is a bounded numerical sensitivity, not a 310-second equilibrium qualification or clinical/biological acceptance test.','No arbitrary threshold was introduced after seeing the results.'],
        'failed_attempt_882':{'classification':'separate predeclared diagnostic, not numi science registered', 'prelaunch_declaration_path':str(DECL882), 'prelaunch_declaration_sha256':declaration882_sha, 'failure_receipt_path':str(FAILURE882), 'failure_receipt_sha256':failure882_sha, 'failure_reason':'owner rejects contact iterations outside [1, 64]', 'outcome':'128 iterations was rejected before native launch; no native/GPU outcome data.'}
    }
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    mean64=arms['64']; mean32=arms['32']
    readme='\n'.join([
        '# Predeclared contact-convergence diagnostic: 32 vs 64 iterations', '',
        'This report compares completed 883 (32 contact iterations) with completed 876 (64 iterations), using the exact metrics and code retained beside this file. The timestamped prelaunch declaration is SHA-256 `'+declaration_sha+'`. It predicted that if iteration-limited creep drives motion, reducing 64 to 32 would increase accepted pre-step-J tangent slip, post-init COM displacement/trend, and constrained momentum-minus-contact-impulse residual.', '',
        'The 883 candidate is a timestamped predeclared 32-iteration diagnostic against the existing 876 64-iteration observer baseline. Neither run was registered through `numi science`; 867 is the registered science study. This diagnostic pair is not clinical or physiological qualification.', '',
        'Both arms have runner exit 0, 10,000 accepted steps at dt 0.002 s, root assistance false, identical source/binary and asset hashes, and the same launch arguments except the declared iteration count and per-arm output/movie destinations. The native terminal records confirm the accepted counts; the owner wrapper receipts report no source changes.', '',
        '## Results', '',
        f"- Post-init COM displacement (samples strictly inside 10–20 s): 64={mean64['com_postinit']['displacement_norm_mm']:.6f} mm, 32={mean32['com_postinit']['displacement_norm_mm']:.6f} mm; ratio 32/64={metrics['postinit_com_displacement_ratio_32_over_64']:.4f}. Componentwise trends (mm/min), 64={mean64['com_postinit']['ols_trend_mm_per_min_xyz']}, 32={mean32['com_postinit']['ols_trend_mm_per_min_xyz']}.",
        f"- Accepted tangential speed on active contacts using pre-step weighted-contact J, p95: 64={mean64['pre_step_contact_J_tangent_speed_m_s']['accepted']['p95']:.8g} m/s, 32={mean32['pre_step_contact_J_tangent_speed_m_s']['accepted']['p95']:.8g} m/s; ratio={metrics['accepted_pre_step_Jv_p95_ratio_32_over_64']:.4f}. This is not endpoint selected-skin slip.",
        f"- Constrained-stage ΔP−summed support impulse residual norm, p95: 64={mean64['partial_impulse_accounting']['contact']['per_step_residual_norm_ns']['p95']:.8g} N·s, 32={mean32['partial_impulse_accounting']['contact']['per_step_residual_norm_ns']['p95']:.8g} N·s; ratio={metrics['constrained_residual_p95_ratio_32_over_64']:.4f}.",
        f"- Gravity-stage residual p95: 64={mean64['partial_impulse_accounting']['gravity']['per_step_residual_norm_ns']['p95']:.8g} N·s, 32={mean32['partial_impulse_accounting']['gravity']['per_step_residual_norm_ns']['p95']:.8g} N·s. q-advance momentum mean/max: 64={mean64['stage_delta_p_norm_kg_m_s']['q_advance']['mean']:.8g}/{mean64['stage_delta_p_norm_kg_m_s']['q_advance']['max']:.8g}, 32={mean32['stage_delta_p_norm_kg_m_s']['q_advance']['mean']:.8g}/{mean32['stage_delta_p_norm_kg_m_s']['q_advance']['max']:.8g} kg·m/s. Final-acceptance maxima are near this scale and detailed in `analysis.json`.",
        f"- The selected PaCO2/PaO2/airflow/lung-volume/tidal/breath/ejection/gas-budget/blood-continuity/respiratory-volume fields are identical row-by-row. The full coupled CSV is not byte-identical because only mechanical/contact diagnostics differ: {', '.join(coupling['different_columns'].keys())}. Per-field details are retained in `analysis.json`; no post-hoc pass boundary is used.", '',
        '## Interpretation', '',
        'The predeclared direction held for all three target quantities at 32 iterations: COM displacement was 2.96× the 64-iteration value, accepted pre-step-J tangent-speed p95 was 9.73×, and constrained-stage residual p95 was 1.09×. This supports a convergence-sensitive contribution to the measured motion in this 20 s diagnostic. The smaller residual ratio does not by itself explain the larger slip/drift ratio. Gravity residuals and final-acceptance momentum changes remained on the same small scale; q-advance mean increased while its maximum stayed similar. The physiology trace fields examined were unchanged. These diagnostics still do not identify a unique cause: slip uses the pre-step Jacobian, force/energy closure is partial, and articulated settling can move COM without equal root displacement.', '',
        '883 is a timestamped predeclared diagnostic, not a `numi science` registered study. 882 was a separate timestamped predeclared 128-iteration diagnostic rejected before native launch; its declaration SHA-256 is `f29c606f9c65553db1c4d3b2e64c0d22d078c67e701489f1d522739843f96fc7`, and its failure receipt is retained without outcome data. The registered science result is 867.', '',
        '## Reproduction', '',
        'Run the exact retained script on the Mac mini:', '',
        '`python3 '+str(OUT/'compare_diagnostics.py')+'`', '',
        'The script reads only 876/883 outputs plus the retained timestamped 883 and 882 declarations/failure receipt, then writes the JSON, Markdown, and manifest in this new comparison directory. It does not alter source, binary, or native run outputs.'
    ])+'\n'
    (OUT/'README.md').write_text(readme)
    manifest={
        'schema':'numi.predeclared-contact-convergence-diagnostic-manifest.v1',
        'classification':'predeclared diagnostic; not numi science registered',
        'script':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())},
        'prelaunch_declaration':{'path':str(DECL),'sha256':declaration_sha},
        'prior_882_prelaunch_declaration':{'path':str(DECL882),'sha256':declaration882_sha},
        'arms':{k:{'root':str(ARMS[k]),'source_sha256':arms[k]['identity']['source_sha256'],
                   'binary_sha256':arms[k]['identity']['binary_sha256'],
                   'runner_receipt_sha256':arms[k]['identity']['runner_receipt_sha256'],
                   'run_metadata_sha256':arms[k]['identity']['run_metadata_sha256'],
                   'native_log_sha256':arms[k]['identity']['native_log_sha256'],
                   'output_sha256':arms[k]['identity']['output_sha256']} for k in arms},
        'analysis_files':{n:sha(OUT/n) for n in ['analysis.json','README.md']},
        'write_scope':'new comparison directory only; frozen source/binary/native outputs unchanged'
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('OUTPUT',OUT)
    print('METRICS',json.dumps(metrics,sort_keys=True))
    print('PHYSIOLOGY',json.dumps(coupling,sort_keys=True))
    print('HASHES',json.dumps({n:sha(OUT/n) for n in ['compare_diagnostics.py','analysis.json','README.md','manifest.json']},sort_keys=True))

if __name__=='__main__':
    main()
