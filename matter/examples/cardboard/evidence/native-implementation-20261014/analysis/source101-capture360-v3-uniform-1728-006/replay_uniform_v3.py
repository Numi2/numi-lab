#!/usr/bin/env python3
"""Uniform one-epsilon v3-metric FP64 replay of all 1728 source101 rows.

Every active directional row is recomputed from the same frozen capture, its
accepted history, captured direction, independent reference, and dynamic
correction gates. The old 34-row package is used only to validate the historic
selection set; its 32 carried-forward rows are not reused. CPU diagnostic only.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, pathlib, sys, time, warnings

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
HERE=pathlib.Path(__file__).resolve().parent
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
DIRECTIONAL=BASE/'analysis/source101-capture360-directional-fd-003'
DROWS=DIRECTIONAL/'output/all_1728_row_metrics.csv'
PKG005=BASE/'analysis/source101-capture360-small-eps-34-005'
P005_ROWS=PKG005/'output-004/rows.jsonl'
P005_REPORT=PKG005/'output-004/report.json'
P005_MANIFEST=PKG005/'manifest.json'
REPLAY005=PKG005/'replay_continuation.py'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
HELPER=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
OUT=HERE/'output-006'
TARGET=1.0e-7
EXPECTED_CAPTURE='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
EXPECTED_DROWS='226a12e13a5dae5cf17f4189a65d71ba6a5e26c94d14d11407e5712a76979180'
EXPECTED_REFERENCE='c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3'


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader: raise RuntimeError(f'cannot import {path}')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def parse_int(token): return -0.0 if token=='-0' else int(token)

def status_for(row, target):
    c=next((x for x in row['continuation_v3']['epsilon_cases'] if x['target_deltaF_frobenius']==target),None)
    if c is None: raise RuntimeError(f"missing target in tet {row.get('tet_id')}")
    return c

def summarize(rows):
    passed=[];exceeded=[];inconclusive=[];errors=[];unstable=[];failedprojection=[]
    for row in rows:
        c=status_for(row,TARGET)
        status=c.get('strict_1e-6_diagnostic')
        tid=int(row['tet_id'])
        if status=='within_threshold_for_this_direction_and_step': passed.append(tid)
        elif status=='exceeds_threshold_for_this_direction_and_step': exceeded.append(tid)
        else: inconclusive.append((tid,status))
        if c.get('center_and_both_endpoints_v3_converged') is not True:
            failedprojection.append(tid)
        if c.get('branch_signature_stable') is not True: unstable.append(tid)
        e=c.get('central_fd_vs_fp64_tangent',{}).get('relative_l2')
        if e is not None and math.isfinite(float(e)): errors.append((float(e),tid))
    worst=max(errors) if errors else (None,None)
    return {'row_count':len(rows),'within_1e_6_count':len(passed),'exceeds_1e_6_count':len(exceeded),
            'inconclusive_count':len(inconclusive),'v3_projection_gate_failure_count':len(failedprojection),
            'observed_branch_change_count':len(unstable),'max_finite_relative_error':worst[0],
            'max_error_tet_id':worst[1],'exceed_tet_ids':exceeded,
            'inconclusive':inconclusive,'v3_projection_gate_failure_tet_ids':failedprojection,
            'branch_change_tet_ids':unstable}

def main():
    if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    bindings={p.name:sha(p) for p in (CAP,DROWS,REPLAY005,P005_ROWS,P005_REPORT,P005_MANIFEST,REF,HELPER)}
    if bindings[CAP.name]!=EXPECTED_CAPTURE or bindings[DROWS.name]!=EXPECTED_DROWS or bindings[REF.name]!=EXPECTED_REFERENCE:
        raise RuntimeError('immutable source hash mismatch')
    report005=json.loads(P005_REPORT.read_text())
    if report005.get('status')!='completed_bounded_subset': raise RuntimeError('package005 did not complete')
    package005=[json.loads(line,parse_int=parse_int) for line in P005_ROWS.read_text().splitlines() if line.strip()]
    if len(package005)!=34: raise RuntimeError('package005 row count mismatch')
    selected005={int(x['tet_id']):x for x in package005}
    if len(selected005)!=34: raise RuntimeError('duplicate package005 tet id')
    cont=load_module(REPLAY005,'source101_v3_replay_module')
    cont.TARGETS=(TARGET,)
    fd=load_module(HELPER,'source101_directional_helper')
    ref=load_module(REF,'source101_bilinear_reference')
    with CAP.open() as f:
        caprecords=[json.loads(line,parse_int=parse_int) for line in f if line.strip()]
    header=caprecords[0]
    if (header.get('root'),header.get('newton_iteration'),header.get('world_fingerprint_hex'))!=(2,23,'6de3475316b7e8b9'):
        raise RuntimeError('capture selection mismatch')
    if caprecords[-1].get('record')!='integrity' or caprecords[-1].get('production_inputs_bitwise_unchanged') is not True:
        raise RuntimeError('capture integrity binding absent')
    native={int(r['tet_id']):r for r in caprecords if r.get('record')=='tetrahedron' and r.get('environment')==0}
    if len(native)!=1920 or set(native)!=set(range(1920)):
        raise RuntimeError('environment-0 capture tet domain mismatch')
    with DROWS.open(newline='') as f: directional=list(csv.DictReader(f))
    if len(directional)!=1728: raise RuntimeError('directional domain mismatch')
    direction_by_id={int(r['tet_id']):r for r in directional}
    if len(direction_by_id)!=1728: raise RuntimeError('duplicate directional row')
    selected_baseline={tid for tid,r in direction_by_id.items() if float(r['central_fd_relative_error'])>1e-6}
    if selected_baseline!=set(selected005): raise RuntimeError('package005 is not exact baseline>1e-6 set')
    rest_ids=sorted(set(direction_by_id))
    if len(rest_ids)!=1728: raise RuntimeError('expected all 1728 directional rows')
    ptab=fd.f32(header['parameters_by_environment'][0])
    replayed=[]; warning_messages=[]
    # A hard wall is included so the one bounded run cannot silently exceed five minutes.
    deadline=time.monotonic()+270.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i,tid in enumerate(rest_ids,1):
            if time.monotonic()>deadline: raise TimeoutError(f'CPU bound reached after {i-1}/{len(rest_ids)} rows')
            prior=dict(direction_by_id[tid]);prior['legacy_directional_metrics']=dict(direction_by_id[tid])
            replayed.append(cont.replay_target_row(fd,ref,prior,direction_by_id[tid],native[tid],ptab))
        warning_messages=[f'{w.category.__name__}: {w.message} ({w.filename}:{w.lineno})' for w in caught]
    # Uniform v3 replay for all rows; package005 is used only to validate the historic selected-ID set.
    combined=sorted(replayed,key=lambda x:int(x['tet_id']))
    if len(combined)!=1728 or len({int(x['tet_id']) for x in combined})!=1728:
        raise RuntimeError('replay did not produce exactly 1728 unique rows')
    if {int(x['tet_id']) for x in combined}!=set(direction_by_id):
        raise RuntimeError('replayed tet domain differs from source directional rows')
    selected_summary=summarize([x for x in combined if int(x['tet_id']) in selected_baseline])
    nonselected_summary=summarize([x for x in combined if int(x['tet_id']) not in selected_baseline])
    joined_summary=summarize(combined)
    (OUT/'all1728_v3_rows.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in combined))
    elapsed=time.monotonic()-started
    report={'schema':'numi.cardboard.source101-directional-fd-v3-one-epsilon-combination.v1',
      'status':'completed_one_epsilon_cpu_diagnostic',
      'classification':'Bounded CPU-only FP64 directional comparison, same rejected Newton-23 capture, accepted histories and captured local directions; not native-noise-floor, accepted coupled-solve, all-direction, or physical qualification.',
      'bindings':{name:{'path':str(path),'sha256':bindings[path.name]} for name,path in [('capture',CAP),('directional_rows',DROWS),('package005_replay',REPLAY005),('package005_rows',P005_ROWS),('package005_report',P005_REPORT),('package005_manifest',P005_MANIFEST),('reference',REF),('direction_helper',HELPER)]},
      'source_contract':{'root':2,'newton_iteration':23,'world_fingerprint_hex':header['world_fingerprint_hex'],
        'environment':0,'rows':1728,'tet_ids':'0..863 and 960..1823, exactly matching directional row IDs','epsilon_target_deltaF_frobenius':TARGET,
        'accepted_history':'Each row uses its captured environment-0 accepted 15-state history at center and both perturbed F states; center candidate is only a Newton seed when stock projection fails.',
        'tangent':'Independent reference tangent at the v3-converged center candidate, same captured accepted history and F direction.'},
      'gate_contract':{'state_projection':'v3 residual <=2e-10 AND dynamic state-scaled Newton correction <=2e-8 at center and both endpoints; max 4 Newton updates and 24 residual-descent backtracks.',
        'dynamic_correction_scale':'max(abs(old), abs(candidate), parameter[26]) componentwise.',
        'fd_gate':'central stress difference at target ||deltaF||_F=1e-7 compared against matched center tangent; rowwise relative L2 <=1e-6.',
        'branch_predicates':'multiplier-increment sign, yield-gap sign, and elastic_smoothed/plastic_smoothed FB selector only; continuous Hill q and hardening factor retained, no other discrete predicate is exposed.',
        'noise_floor':'No universal noise-floor estimate. No claim for other directions, perturbation sizes, states, or native FP32.'},
      'selection':{'baseline_directional_gt_1e-6_count':len(selected005),'baseline_directional_gt_1e-6_rows_recomputed':len(selected_baseline),
        'baseline_directional_le_1e-6_rows_also_recomputed':len(combined)-len(selected_baseline),'epsilon_common_to_both_sets':TARGET,'join_tet_ids_unique':True},
      'summary':{'recomputed_baseline_selected34':selected_summary,'recomputed_baseline_nonselected1694':nonselected_summary,'uniform_v3_all1728':joined_summary},
      'warnings':warning_messages,'elapsed_seconds':elapsed,
      'interpretation':'This is a fresh uniform v3 run of every one of the 1728 directional rows; package005 results were not carried forward because 32 rows there lacked the v3 dynamic correction gate. A one-epsilon pass exists only if all 1728 are resolved, branch stable, and within threshold; otherwise report explicit failure/inconclusive counts.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'report':str(OUT/'report.json'),'elapsed_seconds':elapsed,
                      'selected34':selected_summary,'baseline_nonselected1694':nonselected_summary,'combined1728':joined_summary,
                      'warning_count':len(warning_messages)},indent=2,allow_nan=False))
if __name__=='__main__': main()
