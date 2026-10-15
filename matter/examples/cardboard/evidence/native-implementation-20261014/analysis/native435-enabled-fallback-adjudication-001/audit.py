#!/usr/bin/env python3
"""Source-bound CPU adjudication of native435 fallback run vs native434 control."""
from pathlib import Path
import csv, hashlib, json, math, struct, datetime

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN435 = BASE / 'solver/base-cauchy-first-failure-enabled-435'
RUN434 = BASE / 'solver/base-cauchy-disabled-parity-434'
CMD435 = BASE / 'native-435-command.json'
CMD434 = BASE / 'native-434-command.json'
REPORT433 = BASE / 'analysis/native433-cauchy-kernel-adjudication-001/report.json'
PKG = BASE / 'analysis/native435-enabled-fallback-adjudication-001'


def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def strict_json(path):
    def reject(v): raise ValueError(f'non-standard JSON constant: {v}')
    return json.loads(Path(path).read_text(), parse_constant=reject)
def event_rows(run, prefix):
    rows=[]
    with (run/'stderr.log').open(errors='strict') as f:
        for line_number, line in enumerate(f, 1):
            if line.startswith(prefix):
                rows.append((line_number, json.loads(line.split('=',1)[1])))
    return rows

def json_diffs(a,b,path=''):
    out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b: out.append(path+'/'+k); continue
            out += json_diffs(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): out.append(path+'/length')
        for i,(x,y) in enumerate(zip(a,b)): out += json_diffs(x,y,path+f'/{i}')
    elif a != b: out.append(path or '/')
    return out

def all_files(root):
    return {str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}

def command_parts(path):
    c=strict_json(path); sep=c.index('--')
    return c, c[:sep], c[sep+1:]

def normalize_probe(argv):
    v=list(argv)
    if not v: return v
    v[0]='<source-bound-probe>'
    i=1
    path_opts={'--metallib':'<metallib>','--output':'<output>','--liner-material':'<liner-material>','--medium-material':'<medium-material>'}
    while i < len(v):
        if v[i] in path_opts:
            v[i+1]=path_opts[v[i]]
            i+=2
        else: i+=1
    return v

def parse_driver(prefix):
    # prefix[0] is Python, prefix[1] the source-bound runner script.
    out={'python':prefix[0], 'runner':prefix[1], 'options':{}, 'env':{}, 'positionals':[]}
    i=2
    while i<len(prefix):
        tok=prefix[i]
        if tok=='--env':
            k,v=prefix[i+1].split('=',1); out['env'][k]=v; i+=2
        elif tok.startswith('--') and i+1<len(prefix):
            out['options'][tok]=prefix[i+1]; i+=2
        else:
            out['positionals'].append(tok); i+=1
    return out

def normalize_path_env(value, run_name):
    marker=f'/solver/{run_name}/'
    return '<run>/' + value.split(marker,1)[1] if marker in value else value

def obs_rows(outdir):
    with (outdir/'observations.csv').open(newline='') as f:
        return {(int(r['step']),int(r['environment'])):r for r in csv.DictReader(f)}

def main():
    rec={r:strict_json(r/'receipt.json') for r in (RUN435,RUN434)}
    for run,r in rec.items():
        assert r['schema']=='numi.cardboard.native-run-receipt.v2'
        assert r['execution_started'] and r['source_inputs_unchanged'] and r['changed_inputs']==[]
        assert r['exit_code']==2 and not r['timed_out'] and r['termination_errors']==[]
    c435,p435,probe435=command_parts(CMD435)
    c434,p434,probe434=command_parts(CMD434)
    assert normalize_probe(probe435)==normalize_probe(probe434)
    d435=parse_driver(p435); d434=parse_driver(p434)
    env435=d435['env']; env434=d434['env']
    added_env=sorted(set(env435)-set(env434)); removed_env=sorted(set(env434)-set(env435))
    assert added_env==['NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR'] and removed_env==[]
    assert env435['NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR']=='1:48'
    common_env={}
    for k in sorted(env434):
        a=normalize_path_env(env435[k],RUN435.name); b=normalize_path_env(env434[k],RUN434.name)
        assert a==b,(k,a,b)
        common_env[k]=a
    assert '--input-file' not in d434['options']
    assert d435['options']['--input-file']==str(REPORT433)
    changed_driver_options=sorted(k for k in set(d434['options'])|set(d435['options'])
                                  if d434['options'].get(k)!=d435['options'].get(k))
    assert changed_driver_options==['--build-provenance','--case','--input-file','--prediction',
                                    '--question','--runtime-dir','--source-snapshot']
    assert sha_file(RUN435/'inputs/external/report.json')==sha_file(REPORT433)
    frozen={
        'source133':strict_json(BASE/'snapshots/source-base-state-cauchy-armijo-133/frozen-source.json'),
        'source135':strict_json(BASE/'snapshots/source-cauchy-positive-mass-admission-135/frozen-source.json')}
    source_delta=sorted(k for k in set(frozen['source133']['source_sha256'])|set(frozen['source135']['source_sha256'])
                        if frozen['source133']['source_sha256'].get(k)!=frozen['source135']['source_sha256'].get(k))
    assert source_delta==['matter/src/runtime.mm']
    assert frozen['source133']['parent_revision']==frozen['source135']['parent_revision']
    material_names=['hajali2009_liner_hill_bilinear_fit.nmatter','hajali2009_medium_hill_bilinear_fit.nmatter']
    material_hashes={}
    for name in material_names:
        a=BASE/'snapshots/source-base-state-cauchy-armijo-133/matter/materials'/name
        b=BASE/'snapshots/source-cauchy-positive-mass-admission-135/matter/materials'/name
        assert sha_file(a)==sha_file(b)
        material_hashes[name]=sha_file(a)

    arm435=[x for _,x in event_rows(RUN435,'fem_residual_armijo_iteration=')]
    arm434=[x for _,x in event_rows(RUN434,'fem_residual_armijo_iteration=')]
    assert len(arm435)==len(arm434)==64
    primary=lambda xs,n: [x for x in xs if x['newton_iteration']==n and x['direction_kind']=='newton']
    for n in range(48):
        aa,bb=primary(arm434,n),primary(arm435,n)
        assert len(aa)==len(bb)==1 and aa[0]==bb[0],f'primary Armijo history differs at Newton {n}'
    a48,b48=primary(arm434,48),primary(arm435,48)
    assert len(a48)==len(b48)==1
    diffs48=json_diffs(a48[0],b48[0])
    expected_status48=[
        '/environments/0/solver_status/code',
        '/environments/0/solver_status/diagnostics/0',
        '/environments/0/solver_status/diagnostics/1',
        '/environments/0/solver_status/diagnostics/2',
        '/environments/0/solver_status/diagnostics/3',
        '/environments/0/solver_status/failing_index',
        '/environments/0/solver_status/object_index',
        '/environments/0/solver_status_code'
    ]
    assert diffs48==expected_status48,diffs48
    cauchy=[x for x in arm435 if x['newton_iteration']==48 and x['direction_kind']=='base_cauchy']
    assert len(cauchy)==1 and not [x for x in arm434 if x['direction_kind']=='base_cauchy']
    cauchy_arm=cauchy[0]['environments'][0]
    assert cauchy_arm['accepted'] and cauchy_arm['selected_alpha']==1.0
    assert cauchy_arm['trial_outcome']=='selected_trial_accepted'
    accepted49_52=[]
    for n in range(49,53):
        q=primary(arm435,n); assert len(q)==1 and q[0]['environments'][0]['accepted']
        accepted49_52.append({'newton_iteration':n,'alpha':q[0]['environments'][0]['selected_alpha'],
                              'base_l2':q[0]['environments'][0]['base_l2'],
                              'trial_l2':q[0]['environments'][0]['trial_l2']})
        assert q[0]['environments'][0]['trial_l2'] < q[0]['environments'][0]['base_l2']
    n53=primary(arm435,53); assert len(n53)==1
    e53=n53[0]['environments'][0]
    assert not e53['accepted'] and e53['selected_alpha']==0 and e53['solver_status_code']==10
    assert e53['trial_l2']>e53['base_l2'] and e53['solver_status']['failing_index']==53
    for n in range(54,63):
        q=primary(arm435,n); assert len(q)==1
        assert q[0]['environments'][0]['trial_outcome']=='blocked_by_pretrial_solver_status'
        assert q[0]['environments'][0]['solver_status_code']==10

    # The per-Newton true-residual stream continued to Newton 63. Its entries
    # after the failed trial are explicitly status-gated sentinels, not residuals.
    tr435=[x for _,x in event_rows(RUN435,'fem_true_residual_cycle=')]
    tr434=[x for _,x in event_rows(RUN434,'fem_true_residual_cycle=')]
    assert len(tr435)==len(tr434)==4096
    counts={n:sum(x['newton_iteration']==n for x in tr435) for n in range(64)}
    assert all(counts[n]==64 for n in range(64))
    for n in range(49):
        aa=[x for x in tr434 if x['newton_iteration']==n]
        bb=[x for x in tr435 if x['newton_iteration']==n]
        assert aa==bb,f'true residual cycle differs through primary Newton {n}'
    postfailure=[]
    for x in tr435:
        if x['newton_iteration']>=54:
            env0=x['environments'][0]
            assert env0['solver_status_code']==10 and env0['operator_status_code']==4
            assert env0['true_l2']==-1
            postfailure.append(env0['true_l2'])

    # Compare actual native base-Cauchy vector metrics against the independent
    # report433 kernel/action oracle; this does not extend its certificate scope.
    direction_rows=[x for _,x in event_rows(RUN435,'fem_base_cauchy_direction=')]
    assert len(direction_rows)==1
    direction=direction_rows[0]
    assert direction['newton_iteration']==48 and direction['direction_kind']=='raw_A_transpose_b'
    assert direction['original_newton_certificate_applies_to_cauchy'] is False
    de=direction['environments'][0]
    ref433=strict_json(REPORT433)['cauchy_candidate']
    native_tau=float(de['tau'])
    fp32=lambda x: struct.unpack('<f',struct.pack('<f',float(x)))[0]
    assert fp32(native_tau)==fp32(ref433['tau_metal_float32'])
    g_rel=abs(float(de['gradient_l2'])-float(ref433['metal_compensated_gradient_norm']))/float(ref433['metal_compensated_gradient_norm'])
    ag_rel=abs(float(de['gradient_action_l2'])-float(ref433['metal_compensated_action_norm']))/float(ref433['metal_compensated_action_norm'])
    tau_rel=abs(native_tau-float(ref433['tau_fp64_reference']))/float(ref433['tau_fp64_reference'])
    assert g_rel<1e-8 and ag_rel<1e-8 and tau_rel<2e-7
    base=float(cauchy_arm['base_l2']); trial=float(cauchy_arm['trial_l2'])
    ratio=trial/base; reduction=1-ratio
    assert reduction>0 and cauchy_arm['selected_alpha']==1.0

    # Exact committed geometry and transaction outputs.
    out435=RUN435/'output'; out434=RUN434/'output'
    f435=all_files(out435); f434=all_files(out434)
    assert f435.keys()==f434.keys() and len(f435)==35
    output_diffs=[]
    for rel in sorted(f435):
        a=f435[rel].read_bytes(); b=f434[rel].read_bytes()
        if a!=b: output_diffs.append(rel)
    expected_diffs=['manifest.json','observations.csv','rejected-conditioning-000002/candidate_nodes.bin','rejected-conditioning-000002/manifest.json']
    assert output_diffs==expected_diffs,output_diffs
    accepted_files=['initial.obj','accepted_step_000001_bent.obj','accepted_step_000001_held_reference.obj','accepted_step_000002_held_reference.obj','mesh.json','compiled.nmatterpack','material_state_000002_bent.json']
    accepted_file_equal={x:sha_file(out435/x)==sha_file(out434/x) for x in accepted_files}
    assert all(accepted_file_equal.values())
    assert (out435/'rejected_step_diagnostic_000002_bent.obj').read_bytes()==(out435/'accepted_step_000001_bent.obj').read_bytes()
    assert (out434/'rejected_step_diagnostic_000002_bent.obj').read_bytes()==(out434/'accepted_step_000001_bent.obj').read_bytes()
    obs435=obs_rows(out435); obs434=obs_rows(out434)
    assert obs435.keys()==obs434.keys() and len(obs435)==4
    obs_diffs={}
    for key in sorted(obs435):
        d={k:(obs434[key][k],obs435[key][k]) for k in obs434[key] if obs434[key][k]!=obs435[key][k]}
        if d: obs_diffs[f'{key[0]}:{key[1]}']=d
    assert set(obs_diffs)=={'1:0'}
    assert set(obs_diffs['1:0'])=={'failing_index','fgmres_iterations','status_diagnostic_x','status_diagnostic_y'}
    for key in [(0,0),(0,1),(1,1)]: assert obs435[key]==obs434[key]
    failed_obs=obs435[(1,0)]
    assert failed_obs['step_accepted']=='0' and failed_obs['certificate_raw_accepted_flag']=='0'
    result435=strict_json(out435/'result.json'); result434=strict_json(out434/'result.json')
    for r in (result435,result434):
        assert r['status']=='failed' and r['accepted_steps']==1 and r['physical_validation'] is False
    assert result435['compiled_world_fingerprint']==result434['compiled_world_fingerprint']
    assert result435['material_map_sha256']==result434['material_map_sha256']
    assert result435['authored_region_map_sha256']==result434['authored_region_map_sha256']
    assert result435['frame_map_sha256']==result434['frame_map_sha256']
    payload_equal=(RUN435/'first-status5.bin').read_bytes()==(RUN434/'first-status5.bin').read_bytes()
    assert payload_equal

    def artifact(path):
        p=Path(path); return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha_file(p)}
    source_snapshots={}
    for label,path in [('source133',BASE/'snapshots/source-base-state-cauchy-armijo-133'),
                       ('source135',BASE/'snapshots/source-cauchy-positive-mass-admission-135')]:
        source_snapshots[label]={n:artifact(path/n) for n in ['frozen-source.json','CMakeLists.txt','configure-support-source.json']}
    runs={}
    for label,run,cmd in [('native435',RUN435,CMD435),('native434',RUN434,CMD434)]:
        inp=run/'inputs'; out=run/'output'
        runs[label]={
          'command':artifact(cmd),'receipt':artifact(run/'receipt.json'),'prediction':artifact(run/'prediction.json'),
          'stdout':artifact(run/'stdout.log'),'stderr':artifact(run/'stderr.log'),
          'first_status5_binary':artifact(run/'first-status5.bin'),'first_status5_json':artifact(run/'first-status5.json'),
          'external_source_build':artifact(inp/'external/source-build.json'),
          'runtime_probe':artifact(inp/'runtime/matter/numi-matter-cardboard-probe'),
          'runtime_metallib':artifact(inp/'runtime/matter/shaders/NumiMatter.metallib'),
          'external_cauchy_report':artifact(inp/'external/report.json') if (inp/'external/report.json').exists() else None,
          'output_files':{name:artifact(path) for name,path in sorted(all_files(out).items())}
        }
    report={
      'schema':'numi.cardboard.native435-enabled-fallback-adjudication.v1',
      'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'fallback_local_acceptance_observed_following_newton_rejection; coupon_still_failed',
      'scope':{'native_runs':['434 fallback disabled','435 fallback enabled at environment 0 Newton 48'],
               'cpu_only_adjudication':True,'gpu_runs_by_auditor':0,'source_edits':0},
      'source_and_run_bindings':{'source_snapshots':source_snapshots,'runs':runs,
        'native433_cauchy_report':artifact(REPORT433),
        'command_comparison':{'same_normalized_physical_probe_argv':True,
          'added_environment':{'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR':'1:48'},
          'common_environment_exact_after_runpath_normalization':common_env,
          'changed_driver_options':changed_driver_options,
          'source_snapshot_delta_files':source_delta,
          'source135_scope':'Only runtime.mm differs from source133; frozen-source.json describes the change as fail-closed positive-finite immutable-mass admission for the opted-in fallback.',
          'same_liner_medium_material_sha256':material_hashes,
          'added_external_input':str(REPORT433),
          'driver_source_paths':['source133','source135'],
          'interpretation':'The executable, metallib, source-bound runner and run/source paths differ by their frozen build snapshot. The physical probe argv matches after replacing only executable/output/material/metallib paths. Common solver environment controls match; 435 adds only the opt-in fallback selector. Driver provenance also adds the exact report433 input.'}},
      'primary_history':{'armijo_records_each':64,'primary_newton_records_0_through_47_exact_all_fields':True,
        'primary_newton_48_numeric_and_trial_fields_exact':True,
        'newton48_only_difference_paths':diffs48,
        'newton48_allowed_difference':'The status is cleared in 435 after its primary Armijo rejection so the explicitly selected fallback can be tried; 434 retains terminal status 10.',
        'true_residual_cycles_each':len(tr435),'newton_iteration_range':[0,63],
        'cycles_per_newton_each':64,'true_residual_cycles_0_through_48_exact':True,
        'after_newton53_status_gate':{'newton_iterations':[54,63],'environment0_status_code':10,'operator_status_code':4,
          'true_l2_sentinel':-1,'interpretation':'Trace stream continued, but these are status-gated sentinels, not valid residual measurements.'},
        'armijo_trace_tail':{'records_435':len(arm435),'last_recorded_newton_iteration':max(x['newton_iteration'] for x in arm435),
          'direction_counts':{'newton':sum(x['direction_kind']=='newton' for x in arm435),'base_cauchy':len(cauchy)},
          'trace_stop_reason':'record_cap_64','note':'The extra Cauchy record consumes one slot, so the Armijo trace ends at Newton 62; residual-cycle records run through 63.'}},
      'fallback':{'newton_iteration':48,'direction_kind':direction['direction_kind'],
        'certificate_scope':'The original Newton certificate does not certify the Cauchy direction.',
        'original_newton_converged_flag_environment0':de['original_newton_converged'],
        'native_gradient_l2':de['gradient_l2'],'native_A_gradient_l2':de['gradient_action_l2'],'native_tau':native_tau,
        'independent433_gradient_l2':ref433['metal_compensated_gradient_norm'],
        'independent433_A_gradient_l2':ref433['metal_compensated_action_norm'],
        'independent433_tau_fp32':ref433['tau_metal_float32'],'independent433_tau_fp64':ref433['tau_fp64_reference'],
        'relative_differences':{'gradient_l2':g_rel,'A_gradient_l2':ag_rel,'native_tau_vs_fp64':tau_rel},
        'residual_armijo_environment0':{'base_l2':base,'trial_l2':trial,'ratio':ratio,'fractional_reduction':reduction,
          'selected_alpha':cauchy_arm['selected_alpha'],'accepted':cauchy_arm['accepted'],
          'trial_outcome':cauchy_arm['trial_outcome']},
        'environment1':{k:cauchy[0]['environments'][1][k] for k in ['base_l2','trial_l2','selected_alpha','accepted','certified_noop','trial_outcome']}},
      'post_fallback_history':{'accepted_newton_iterations_49_through_52':accepted49_52,
        'newton53':{'accepted':False,'status_code':e53['solver_status_code'],'failing_index':e53['solver_status']['failing_index'],
          'base_l2':e53['base_l2'],'last_trial_l2':e53['trial_l2'],'selected_alpha':e53['selected_alpha'],
          'line_search_trials':24}},
      'physical_history_and_rollback':{'output_file_count_each':35,'byte_identical_output_files':31,
        'differing_output_paths':output_diffs,'accepted_snapshot_files_byte_identical':accepted_file_equal,
        'rejected_diagnostic_geometry_matches_last_accepted_bent_geometry':True,
        'observations_exact_rows':['step0/env0','step0/env1','step1/env1'],
        'observation_differences':obs_diffs,
        'first_status5_binary_byte_identical':payload_equal,
        'candidate_nodes_note':'The rejected-conditioning candidate_nodes buffer differs; it captures a rejected solver candidate, not a committed accepted state.',
        'terminal_result':{'status':result435['status'],'accepted_steps':result435['accepted_steps'],
          'physical_validation':result435['physical_validation'],'failure':result435['failure'],
          'timed_out':rec[RUN435]['timed_out'],'exit_code':rec[RUN435]['exit_code'],
          'step_1_bent_certificate_observation':{
            'step_accepted':False,'certificate_raw_accepted_flag':False,
            'recorded_certificate_residual':float(failed_obs['certificate_residual']),
            'recorded_certificate_correction':float(failed_obs['certificate_correction']),
            'interpretation':'The raw acceptance flag is false; the retained zero-valued certificate fields do not represent a passing certificate after status-10 rejection.'}}},
      'conclusion':'The opt-in base-state Cauchy fallback was attempted at environment 0, Newton 48 after the numerically identical primary Newton history through Newton 47 and identical primary-48 residual/trial values. Its captured g, A*g and tau agree with the independent native-kernel audit433; alpha=1 reduces the measured Armijo residual by about 13.22%. Newton updates 49–52 then accept. Newton53 rejects with status10. The physical transaction rolls back to the same visible accepted geometry as native434. The 27-control coupon is still failed after one accepted bent control; no convergence, calibration, or physical qualification follows.'
    }
    (PKG/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    r=main()
    print(json.dumps({'status':r['status'],'primary47_exact':r['primary_history']['primary_newton_records_0_through_47_exact_all_fields'],
      'cauchy_alpha':r['fallback']['residual_armijo_environment0']['selected_alpha'],
      'cauchy_residual_ratio':r['fallback']['residual_armijo_environment0']['ratio'],
      'accepted49_52':len(r['post_fallback_history']['accepted_newton_iterations_49_through_52']),
      'newton53_status':r['post_fallback_history']['newton53']['status_code']},sort_keys=True))
