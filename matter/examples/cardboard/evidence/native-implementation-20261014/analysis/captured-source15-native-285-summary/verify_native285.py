#!/usr/bin/env python3
"""Verify and summarize the immutable native source15 point run 285."""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, statistics, sys

RUN = pathlib.Path('/Users/home/cardboard-implementation-20261014/native/captured-source15-local-accuracy-285')
INPUTS = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/bilinear15-native-capture-overlay-086-v2/evidence/captured-point-inputs.json')
EXPECTED_SCHEMA = 'numi.cardboard.bilinear15-native-captured-point-result.v2'
EXPECTED_SOURCE_HASH = '769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752'
BASE_LINEAGE_TOKEN = '98fe026ee2007982e48f89547a435a2957db3978fb3730e008fdbb99d08f5704'


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit('FAIL: ' + message)

def load_jsonl(path: pathlib.Path):
    values=[]
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            values.append(json.loads(line))
        except Exception as exc:
            raise SystemExit(f'FAIL: {path}:{line_number}: invalid JSON: {exc}')
    return values

def summary_group(rows, key):
    values=[float(r[key]) for r in rows]
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', type=pathlib.Path, default=RUN)
    parser.add_argument('--inputs', type=pathlib.Path, default=INPUTS)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    args=parser.parse_args()
    run=args.run.resolve(); pred=json.loads((run/'prediction.json').read_text())
    receipt=json.loads((run/'receipt.json').read_text())
    records=load_jsonl(run/'stdout.log')
    headers=[x for x in records if x.get('record')=='header']
    cases=[x for x in records if x.get('record')=='case']
    endings=[x for x in records if x.get('record')=='end']
    need(len(headers)==1 and len(endings)==1, 'expected exactly one header and one end record')
    header=headers[0]; ending=endings[0]
    need(header.get('schema')==EXPECTED_SCHEMA, 'unexpected result schema')
    need(receipt.get('exit_code')==0 and not receipt.get('timed_out') and receipt.get('execution_started'), 'native receipt did not complete successfully')
    need(receipt.get('source_inputs_unchanged') is True and not receipt.get('changed_inputs'), 'native source inputs changed during execution')
    need(ending.get('status')=='pass' and ending.get('cases')==16 and ending.get('passed_count')==16 and ending.get('failed_count')==0 and ending.get('inconclusive_count')==0, 'end record is not a complete 16/16 pass')
    need(len(cases)==16 and len({c['id'] for c in cases})==16, 'case rows are incomplete or duplicated')
    need(header.get('physical_steps')==0 and header.get('physical_validation') is False, 'unexpected physical-validation claim')
    need(float(header['relative_gate'])==0.001, 'expected pointwise relative gate is not 1e-3')
    need(header['base_snapshot_manifest_sha256']==BASE_LINEAGE_TOKEN, 'unexpected compiled base-lineage token')

    snapshot=pathlib.Path(pred['source_snapshot']).resolve()
    runtime=pathlib.Path(pred['runtime_dir']).resolve()
    frozen_path=snapshot/'frozen-source.json'
    build_path=runtime/'source-build.json'
    frozen=json.loads(frozen_path.read_text()); build=json.loads(build_path.read_text())
    need(sha(frozen_path)==EXPECTED_SOURCE_HASH, 'frozen source manifest SHA mismatch')
    need(build.get('success') is True and build.get('changed_sources')==[], 'source build failed or changed source files')
    need(build.get('source_sha256')==frozen.get('source_sha256'), 'build source hashes differ from frozen snapshot')
    need(pathlib.Path(build['source_snapshot']).resolve()==snapshot, 'build points at a different source snapshot')
    material=snapshot/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'
    need(sha(material)==header['source_material_sha256'], 'source material SHA differs from result header')
    input_path=args.inputs.resolve()
    need(input_path.is_file() and sha(input_path)==header['input_sha256'], 'captured point input SHA differs from result header')

    # Verify each prediction-bound source, runtime, and external input in place.
    checks=[]
    for key, expected in pred['inputs_sha256'].items():
        prefix, rel=key.split('/',1)
        if prefix=='source': target=snapshot/rel
        elif prefix=='runtime': target=runtime/rel
        elif prefix=='external':
            target=pathlib.Path(pred['external_inputs'][0]).resolve()
            need(rel==target.name, f'unexpected external input mapping: {key}')
        else: raise SystemExit('FAIL: unsupported input prefix '+prefix)
        need(target.is_file(), 'missing bound input '+str(target))
        actual=sha(target)
        need(actual==expected, 'bound input hash mismatch '+key)
        checks.append(prefix)
    counts={p:checks.count(p) for p in ('source','runtime','external')}
    need(len(checks)==782 and counts=={'source':772,'runtime':9,'external':1}, 'unexpected prediction input coverage')
    need(sha(build_path)==pred['inputs_sha256']['external/source-build.json'], 'source-build input hash mismatch')

    # Verify all build artifact hashes, then retain explicit hashes for the run's executable stack.
    artifact_results=[]
    for artifact_path, expected in build['artifacts_sha256'].items():
        path=pathlib.Path(artifact_path)
        need(path.is_file(), 'missing build artifact '+artifact_path)
        actual=sha(path)
        need(actual==expected, 'build artifact hash mismatch '+artifact_path)
        if any(name in path.name for name in ('bilinear15-captured-point-check','NumiMatter.metallib','cardboard_paper_check.metallib','MetalRobo.metallib','libmetalrobo.dylib')):
            artifact_results.append({'path':str(path),'sha256':actual})
    need(len(build['artifacts_sha256'])==73, 'unexpected build artifact inventory')
    need(len(artifact_results)==5, 'expected five direct native runtime artifacts')

    groups={}
    for label in ('N0','N63'):
        group=[c for c in cases if c['capture']==label]
        need(len(group)==8, f'{label} does not have eight material-point cases')
        need(all(c['default']['status']=='pass' and c['optin']['status']=='pass' for c in group), f'{label} has a failed/inconclusive policy row')
        need(all(c['default']['evaluated'] and c['optin']['evaluated'] for c in group), f'{label} has unevaluated policy output')
        need(all(c['default']['comparison_target']=='captured_production_point' and c['optin']['comparison_target']=='independent_fp64_reference' for c in group), f'{label} uses unexpected comparison targets')
        need(all(c['optin']['stress_target_norm_l2']>0 and c['optin']['tangent_target_norm_l2']>0 and c['optin']['state_target_norm_l2']>0 for c in group), f'{label} contains an exactly-zero target')
        need(all(max(c['optin'][metric] for metric in ('stress_relative_l2','tangent_relative_l2','state_relative_l2')) <= float(header['relative_gate']) for c in group), f'{label} exceeds relative gate')
        need(all(c['optin']['repeat_difference_interpretation']=='identical-run variation only; not a representability or accuracy bound' for c in group), f'{label} repeat interpretation mismatch')
        groups[label]={
            'count':len(group),
            'tet_ids':[c['tet_id'] for c in group],
            'branches':sorted(set(c['branch_indicator'] for c in group)),
            'multiplier_increment_min':min(float(c['multiplier_increment']) for c in group),
            'multiplier_increment_max':max(float(c['multiplier_increment']) for c in group),
            'default':{
                'all_status':'pass',
                'iterations':summary_group([c['default'] for c in group],'iterations'),
                'stress_relative_l2_max':max(float(c['default']['stress_relative_l2']) for c in group),
                'tangent_relative_l2_max':max(float(c['default']['tangent_relative_l2']) for c in group),
                'state_relative_l2_max':max(float(c['default']['state_relative_l2']) for c in group),
            },
            'optin':{
                'all_status':'pass',
                'iterations':summary_group([c['optin'] for c in group],'iterations'),
                'stress_relative_l2':summary_group([c['optin'] for c in group],'stress_relative_l2'),
                'tangent_relative_l2':summary_group([c['optin'] for c in group],'tangent_relative_l2'),
                'state_relative_l2':summary_group([c['optin'] for c in group],'state_relative_l2'),
                'projection_residual':summary_group([c['optin'] for c in group],'projection_residual'),
            },
            'repeat_difference_max_abs':{
                field:max(float(c['optin']['repeat_difference_max_abs'][field]) for c in group)
                for field in ('stress','tangent','state')
            },
        }

    rows=[]
    for c in cases:
        rows.append({
            'id':c['id'],'capture':c['capture'],'tet_id':c['tet_id'],
            'branch_indicator':c['branch_indicator'],
            'multiplier_increment':c['multiplier_increment'],
            'default_iterations':c['default']['iterations'],
            'default_relative_errors':{k:c['default'][k] for k in ('stress_relative_l2','tangent_relative_l2','state_relative_l2')},
            'optin_iterations':c['optin']['iterations'],
            'optin_relative_errors':{k:c['optin'][k] for k in ('stress_relative_l2','tangent_relative_l2','state_relative_l2')},
            'optin_projection_residual':c['optin']['projection_residual'],
        })
    summary={
        'schema':'numi.cardboard.captured-source15-native-summary.v1',
        'run_id':'captured-source15-local-accuracy-285',
        'result_schema':header['schema'],
        'verification':'pass',
        'receipt':{'exit_code':receipt['exit_code'],'timed_out':receipt['timed_out'],'source_inputs_unchanged':receipt['source_inputs_unchanged'],'wall_seconds':receipt['wall_seconds']},
        'result_scope':{'cases':16,'passed':16,'failed':0,'inconclusive':0,'physical_steps':0,'physical_validation':False,'relative_gate':header['relative_gate'],'policy_scope':'single captured material-point direction per case'},
        'lineage':{
            'source_snapshot':str(snapshot),'frozen_source_manifest_sha256':sha(frozen_path),
            'compile_time_base_lineage_token':header['base_snapshot_manifest_sha256'],
            'lineage_note':'The header token is compiled into CMake and is distinct from the SHA-256 of this run snapshot frozen-source.json.',
            'prediction_sha256':sha(run/'prediction.json'),'stdout_sha256':sha(run/'stdout.log'),'stderr_sha256':sha(run/'stderr.log'),'receipt_sha256':sha(run/'receipt.json'),
            'input_json_path':str(input_path),'input_json_sha256':sha(input_path),'source_material_sha256':sha(material),
            'source_build_path':str(build_path),'source_build_sha256':sha(build_path),'source_build_success':build['success'],'source_build_changed_sources':build['changed_sources'],
            'prediction_bound_input_counts':counts,'prediction_bound_inputs_verified':len(checks),
            'build_artifacts_verified':len(build['artifacts_sha256']),'direct_native_artifacts':artifact_results,
        },
        'policy_details':{
            'default':header['default_policy'],'optin':header['optin_policy'],
            'optin_compiled_local_iteration_budget':16,'relative_gate':header['relative_gate'],
            'repeat_interpretation':header['repeat_difference_interpretation'],
        },
        'captures':groups,
        'case_rows':rows,
        'all_local_direction_assessment':{
            'status':'not tested by this run',
            'reason':'Each captured point contains one deformationDirection/rateDirection and one 9-component tangent action, not a full 9x9 local map.',
            'feasible_extension':'At each exact point, evaluate nine matrix basis directions H_j and set rateDirection_j=H_j/dt using the same float32 division convention as fem.metalinc. Since captured Fdot is zero and the runtime forms rateDirection=deformationDirection/dt, this reconstructs the local algorithmic tangent map without new physical states.',
            'required_reference_work':'Generate nine independent reference tangent actions per point from the same parameters, accepted state, F_high/F_low, Fdot, dt, and branch; retain basis-by-basis validity and branch diagnostics. Do not infer the nine actions by applying a single directional result.',
        },
    }
    args.output.mkdir(parents=True,exist_ok=True)
    json_path=args.output/'summary.json'
    json_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    def fmt(x): return f'{x:.9g}'
    md=['# Native source15 local-accuracy point summary', '',
        'Run 285 completed with 16/16 pointwise checks passing. This is a bounded material-point replay; it is not a full solver, mesh, or physical validation.', '',
        '## Provenance', '',
        f'- Receipt: exit `{receipt["exit_code"]}`, timed out `{receipt["timed_out"]}`, unchanged inputs `{receipt["source_inputs_unchanged"]}`, wall {receipt["wall_seconds"]:.3f} s.',
        f'- Frozen source snapshot manifest SHA-256: `{sha(frozen_path)}`.',
        f'- Checker header compile-time base-lineage token: `{header["base_snapshot_manifest_sha256"]}`. This is distinct from the frozen-source manifest hash above.',
        f'- Source material SHA-256: `{sha(material)}`; captured input JSON SHA-256: `{sha(input_path)}`.',
        f'- Prediction binds 782 inputs (772 source, 9 runtime, 1 external); all match. Build record SHA-256 `{sha(build_path)}`; build success, no changed sources, and all 73 listed artifact hashes verified.',
        '- Direct native artifacts:',
    ]
    for a in artifact_results: md.append(f'  - `{pathlib.Path(a["path"]).name}`: `{a["sha256"]}`')
    md += ['', '## Results', '', '| Capture | Cases | Branches | Default max stress/tangent/state relative error | Opt-in max stress/tangent/state relative error | Opt-in iterations |', '|---|---:|---|---|---|---|']
    for label in ('N0','N63'):
        g=groups[label]
        d=g['default']; o=g['optin']
        md.append(f'| {label} | 8/8 pass | {", ".join(g["branches"])} | {fmt(d["stress_relative_l2_max"])}/{fmt(d["tangent_relative_l2_max"])}/{fmt(d["state_relative_l2_max"])} | {fmt(o["stress_relative_l2"]["max"])}/{fmt(o["tangent_relative_l2"]["max"])}/{fmt(o["state_relative_l2"]["max"])} | {fmt(o["iterations"]["min"])}–{fmt(o["iterations"]["max"])} |')
    md += ['', 'The opt-in policy stayed below the 1e-3 pointwise relative gate in each nonzero response. Its largest directional tangent error was '
           f'{fmt(groups["N0"]["optin"]["tangent_relative_l2"]["max"])} on N0 and '
           f'{fmt(groups["N63"]["optin"]["tangent_relative_l2"]["max"])} on N63. Default stress and tangent matched the captured production action exactly in these rows; state differences were small but nonzero.',
           '', 'Repeated identical evaluations differed by zero in the recorded rows. This is only a determinism observation; it is not an FP32 accuracy or representability bound. The N0 rows are classified `no_resolved_multiplier_increase`, while N63 rows are `multiplier_increased`; these are pointwise reference branch indicators, not evidence of branch coverage across all perturbation directions.',
           '', '## Directional coverage and limits', '',
           'Each of the 16 points tests one captured tangent direction, not the complete 9x9 local derivative. The exact same point inputs are sufficient to extend the test: use the nine matrix basis directions and the runtime relation `rateDirection = deformationDirection / dt`, producing nine independent FP64 reference actions per point. This would be a new stronger test and should preserve each direction’s projection validity/branch diagnostics. No such basis sweep was executed here.',
           '', 'The native run used zero physical solver steps and reports `physical_validation=false`. It does not qualify the full coupon solve, mesh convergence, or a calibrated production default.', '',
           '## Files', '',
           f'- Original immutable run: `{run}`',
           f'- Verification script: `{pathlib.Path(__file__).resolve()}`',
           f'- Machine-readable summary: `{json_path}`', '']
    md_path=args.output/'README.md'; md_path.write_text('\n'.join(md))
    print(json.dumps({'verification':'pass','summary':str(json_path),'readme':str(md_path),'source_inputs_verified':counts,'build_artifacts_verified':len(build['artifacts_sha256']),'cases':len(cases)},indent=2))

if __name__=='__main__': main()
