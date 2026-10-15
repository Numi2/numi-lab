#!/usr/bin/env python3
"""Strict source-bound production-history parity audit for native 419."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUNS = {n: BASE/'solver'/p for n,p in {
    '409':'two-stage-certified-coupon-409',
    '415':'two-stage-coupon-first-descent-capture-415',
    '419':'two-stage-first-failure-rich-fd-419'}.items()}
PKG = Path(__file__).resolve().parent
DECODER = BASE/'analysis/native415-status5-parity-001/decode_status5'
CAPTURE_FILTERS = (
    'cardboard_step_timing=', 'fem_coupled_capture_', 'fem_status5_capture_',
    'fem_newton_iterate=', 'fem_newton_trace_', 'fem_newton_terminal_certificate=',
    'fem_operator_fd_', 'fem_regional_rap_capture_')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text())

def tree(root: Path):
    return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}

def compare_tree(left: Path, right: Path):
    a,b=tree(left),tree(right); rows=[]
    for name in sorted(a.keys()|b.keys()):
        if name not in a or name not in b:
            rows.append({'path':name,'equal':False,'reason':'path_missing'}); continue
        x,y=a[name].read_bytes(),b[name].read_bytes()
        row={'path':name,'left_bytes':len(x),'right_bytes':len(y),
             'left_sha256':hashlib.sha256(x).hexdigest(),
             'right_sha256':hashlib.sha256(y).hexdigest(),'equal':x==y}
        if x!=y and name.endswith('.json'):
            jx,jy=json.loads(x),json.loads(y); diffs=[]
            def walk(u,v,p='$'):
                if type(u) is not type(v): diffs.append({'field':p,'left':u,'right':v}); return
                if isinstance(u,dict):
                    for k in sorted(u.keys()|v.keys()):
                        if k not in u or k not in v: diffs.append({'field':p+'/'+k,'missing_side':'left' if k not in u else 'right'})
                        elif u[k]!=v[k]: walk(u[k],v[k],p+'/'+k)
                elif isinstance(u,list):
                    if len(u)!=len(v): diffs.append({'field':p,'left_count':len(u),'right_count':len(v)})
                    for i,(x0,y0) in enumerate(zip(u,v)):
                        if x0!=y0: walk(x0,y0,p+f'/{i}')
                else: diffs.append({'field':p,'left':u,'right':v})
            walk(jx,jy); row['json_differences']=diffs
        rows.append(row)
    return rows

def events(path: Path, prefix: str):
    return [line for line in path.read_text().splitlines() if line.startswith(prefix)]

def command_parts(path: Path):
    cmd=read_json(path); marker=cmd.index('--'); probe=cmd[marker+1:]
    env={cmd[i+1].split('=',1)[0]:cmd[i+1].split('=',1)[1] for i,x in enumerate(cmd[:-1]) if x=='--env'}
    return cmd,probe,env

def normalized_probe(probe):
    out=[]; i=0
    path_options={'--output','--metallib','--liner-material','--medium-material','--glue-material'}
    while i<len(probe):
        x=probe[i]; out.append(x); i+=1
        if x in path_options and i<len(probe): out.append('<'+x[2:]+'-path>'); i+=1
        elif x.startswith('/'):
            out[-1]='<absolute-path>'
    return out

def summarize_capture(path: Path):
    raw=path.read_text().splitlines(); rows=[json.loads(x) for x in raw if x.strip()]
    assert len(rows)==5606, len(rows)
    header, integrity=rows[0],rows[-1]
    assert header['record']=='header' and header['schema']=='numi.cardboard.fem-newton-capture.v1'
    assert (header['root'],header['newton_iteration'],header['environment_count'],header['node_count_per_environment'],header['tetrahedron_count_per_environment'])==(1,48,2,882,1920)
    assert header['production_inputs_bitwise_unchanged'] is True
    assert header['cached_algorithmic_tangent_capture']['available'] is True
    assert header['cached_algorithmic_tangent_capture']['snapshot_bitwise_guard_included'] is True
    assert integrity['record']=='integrity' and integrity['production_inputs_bitwise_unchanged'] is True
    nodes=[r for r in rows if r['record']=='node']; tets=[r for r in rows if r['record']=='tetrahedron']
    assert len(nodes)==2*882 and len(tets)==2*1920
    assert {r['environment'] for r in nodes}=={0,1}
    assert all(len({r['node_id'] for r in nodes if r['environment']==e})==882 for e in (0,1))
    assert all(len({r['tet_id'] for r in tets if r['environment']==e})==1920 for e in (0,1))
    tangent_valid=sum(bool(r['cached_algorithmic_tangent_local']['valid']) for r in tets)
    material_counts={str(m):sum(r['material_index']==m for r in tets) for m in sorted({r['material_index'] for r in tets})}
    return {'sha256':sha(path),'bytes':path.stat().st_size,'jsonl_records':len(rows),
        'node_records':len(nodes),'tetrahedron_records':len(tets),'cached_tangent_valid_tetrahedron_records':tangent_valid,
        'tetrahedra_by_material_index':material_counts,'root':header['root'],'newton_iteration':header['newton_iteration'],
        'environment_count':header['environment_count'],'node_count_per_environment':header['node_count_per_environment'],
        'tetrahedron_count_per_environment':header['tetrahedron_count_per_environment'],
        'epsilon_values':header['epsilon_values'],'dt_seconds':header['dt_seconds'],
        'production_inputs_bitwise_unchanged':integrity['production_inputs_bitwise_unchanged'],
        'cached_tangent_snapshot_bitwise_guard_included':integrity['cached_tangent_snapshot_bitwise_guard_included'],
        'capture_scope':'selected root 1, Newton 48; read-only per-node/tetrahedron operator-FD snapshot'}

def main():
    r={n:read_json(p/'receipt.json') for n,p in RUNS.items()}
    res={n:read_json(p/'output/result.json') for n,p in RUNS.items()}
    for n in RUNS:
        assert r[n]['execution_started'] and r[n]['source_inputs_unchanged']
        assert r[n]['exit_code']==2 and r[n]['timed_out'] is False
        assert res[n]['status']=='failed' and res[n]['accepted_steps']==1 and res[n]['physical_validation'] is False
    # 415 and 419 share source 125 and all physical probe arguments; only observers differ.
    _,p415,e415=command_parts(BASE/'native-415-command.json')
    _,p419,e419=command_parts(BASE/'native-419-command.json')
    assert normalized_probe(p415)==normalized_probe(p419)
    observer_keys={'NM_FEM_COUPLED_CAPTURE_SELECTOR','NM_FEM_COUPLED_CAPTURE_PATH',
                   'NM_FEM_OPERATOR_FD_SELECTOR','NM_FEM_OPERATOR_FD_CAPTURE_PATH',
                   'NM_FEM_REGIONAL_RAP_CAPTURE_ONLY','NM_FEM_NEWTON_TRACE_ROOT',
                   'NM_FEM_STATUS5_CAPTURE_PATH'}
    common_env={k:v for k,v in e415.items() if k not in observer_keys}
    assert common_env=={k:v for k,v in e419.items() if k not in observer_keys}
    out_415_419=compare_tree(RUNS['415']/'output',RUNS['419']/'output')
    assert len(out_415_419)==35 and all(x['equal'] for x in out_415_419)
    out_409_419=compare_tree(RUNS['409']/'output',RUNS['419']/'output')
    diffs=[x for x in out_409_419 if not x['equal']]
    assert len(out_409_419)==35 and [x['path'] for x in diffs]==['manifest.json','rejected-conditioning-000002/manifest.json']
    assert {d['field'] for d in diffs[0]['json_differences']}=={'$/materials/glue_path','$/materials/liner_path','$/materials/medium_path'}
    assert len(diffs[1]['json_differences'])==1 and diffs[1]['json_differences'][0]['field']=='$/device_program_fingerprint'
    assert all(res[n]==res['409'] for n in ('415','419'))
    assert all((RUNS[n]/'stdout.log').read_bytes()==(RUNS['409']/'stdout.log').read_bytes() for n in ('415','419'))
    arm={n:events(RUNS[n]/'stderr.log','fem_residual_armijo_iteration=') for n in RUNS}
    true={n:events(RUNS[n]/'stderr.log','fem_true_residual_cycle=') for n in RUNS}
    assert all(len(arm[n])==64 and arm[n]==arm['409'] for n in ('415','419'))
    assert all(len(true[n])==4096 and true[n]==true['409'] for n in ('415','419'))
    clean={n:[x for x in (RUNS[n]/'stderr.log').read_text().splitlines() if not x.startswith(CAPTURE_FILTERS)] for n in RUNS}
    assert all(len(clean[n])==4164 and clean[n]==clean['409'] for n in ('415','419'))
    sm415=read_json(RUNS['415']/'first-status5.json'); sm419=read_json(RUNS['419']/'first-status5.json')
    assert (RUNS['415']/'first-status5.bin').read_bytes()==(RUNS['419']/'first-status5.bin').read_bytes()
    assert sm415['payload']['sha256']==sm419['payload']['sha256']
    sm415['payload']['path']=sm419['payload']['path']='<normalized-status5-path>'
    assert sm415==sm419
    cap=summarize_capture(RUNS['419']/'operator-capture')
    out={'schema':'numi.cardboard.native419_history_parity.v1',
      'verdict':'production_histories_match; operator_fd_capture_integrity_pass; physical_run_remains_failed',
      'claim_limit':'Diagnostic-on history parity only. Does not qualify the coupon, establish convergence, or adjudicate the captured local derivative.',
      'runs':{n:{'path':str(RUNS[n]),'receipt_sha256':sha(RUNS[n]/'receipt.json'),'stdout_sha256':sha(RUNS[n]/'stdout.log'),
                 'stderr_sha256':sha(RUNS[n]/'stderr.log'),'result_sha256':sha(RUNS[n]/'output/result.json'),
                 'exit_code':r[n]['exit_code'],'timed_out':r[n]['timed_out'],'accepted_steps':res[n]['accepted_steps'],
                 'physical_validation':res[n]['physical_validation']} for n in RUNS},
      'source_bindings':{'409':{'path':str(BASE/'snapshots/source-two-stage-material-merit-123'),'frozen_source_sha256':sha(BASE/'snapshots/source-two-stage-material-merit-123/frozen-source.json')},
                         '415_419':{'path':str(BASE/'snapshots/source-status5-vtccd-diagnostics-125'),'frozen_source_sha256':sha(BASE/'snapshots/source-status5-vtccd-diagnostics-125/frozen-source.json')}},
      'command_comparison':{'415_vs_419_normalized_physical_probe_argv_equal':True,'415_vs_419_nonobserver_environment_equal':True,
          'observer_environment_differences':{'415':{k:v for k,v in e415.items() if k in observer_keys},'419':{k:v for k,v in e419.items() if k in observer_keys}}},
      'production_comparison':{'415_vs_419_output_file_count':len(out_415_419),'415_vs_419_output_byte_identical_count':sum(x['equal'] for x in out_415_419),
          '409_vs_419_output_byte_identical_count':sum(x['equal'] for x in out_409_419),'409_vs_419_expected_metadata_differences':diffs,
          'result_json_equal_across_all_runs':True,'stdout_equal_across_all_runs':True,
          'armijo_records':64,'armijo_equal_across_all_runs':True,'true_residual_records':4096,'true_residual_equal_across_all_runs':True,
          'stderr_remaining_after_declared_capture_and_timing_lines':4164,'stderr_remaining_equal_across_all_runs':True},
      'status5':{'415_payload_sha256':sha(RUNS['415']/'first-status5.bin'),'419_payload_sha256':sha(RUNS['419']/'first-status5.bin'),
          'payload_byte_identical':True,'manifest_equal_after_normalizing_only_payload_path':True,
          'captured_alpha_lane_value':0.0,'alpha_lane_known_bad_in_source125':True,'actual_newton26_selected_alpha':0.0078125,
          'qualification':'Same rejected private alpha-1 witness as 415; stress exceeds the configured maximum, but this is not the final Newton-48 failure.'},
      'operator_fd_capture':cap,
      'file_hashes_415_vs_419':out_415_419,
      'file_hashes_409_vs_419':out_409_419,
      'normalized_trace_hashes':{n:{'armijo_sha256':hashlib.sha256(('\n'.join(arm[n])+'\n').encode()).hexdigest(),'true_residual_sha256':hashlib.sha256(('\n'.join(true[n])+'\n').encode()).hexdigest(),'stderr_clean_sha256':hashlib.sha256(('\n'.join(clean[n])+'\n').encode()).hexdigest()} for n in RUNS},
      'limitations':['419 operator-FD capture is a separate diagnostic artifact; it is not present in 409 or 415 and was not used as physical-history parity evidence.',
          'The capture header/integrity record asserts production inputs were bitwise unchanged; this report validates the record coverage/hash but does not adjudicate derivative accuracy.',
          'The status-5 alpha-lane observer remains unfixed in the source-125 runs; the proposed overlay has syntax/Metal compile checks only and was not used by 419.']}
    (PKG/'audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'verdict':out['verdict'],'output419_vs415':out['production_comparison']['415_vs_419_output_byte_identical_count'],
      'armijo_records':64,'true_residual_records':4096,'clean_stderr_lines':4164,'operator_capture_records':cap['jsonl_records'],
      'operator_capture_bytes':cap['bytes'],'fd_tets':cap['tetrahedron_records'],'fd_nodes':cap['node_records']},indent=2))
if __name__=='__main__': main()
