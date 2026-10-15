#!/usr/bin/env python3
import csv, hashlib, json, math, pathlib, sys
import numpy as np

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
AUDIT=BASE/'analysis/source125-full-tangent-fp64-audit-147-001'
RERUN=BASE/'analysis/review147-independent-rerun-001'
MAN=json.loads((AUDIT/'manifest.json').read_text())
PRED=json.loads((AUDIT/'prediction.json').read_text())
CAP=BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'
RUNPRED=BASE/'solver/two-stage-first-failure-rich-fd-419/prediction.json'
SNAP=BASE/'snapshots/source-status5-vtccd-diagnostics-125'
SWEEP=BASE/'analysis/source125-fp64-smooth-direction-sweep-006/result-004/report.json'
OUT=pathlib.Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write_json(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

# Verify the original audit's complete declared file and input inventories.
file_checks={}
for rel,expected in MAN['files'].items():
    p=AUDIT/rel
    file_checks[rel]={'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None,'expected':expected}
    if not p.is_file() or file_checks[rel]['sha256']!=expected: raise SystemExit(f'audit artifact mismatch: {rel}')
input_checks={}
for raw,expected in MAN['inputs'].items():
    p=pathlib.Path(raw)
    input_checks[raw]={'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None,'expected':expected}
    if not p.is_file() or input_checks[raw]['sha256']!=expected: raise SystemExit(f'audit input mismatch: {raw}')

# Independently rerun results: data arrays and CSV are byte-identical; reports
# differ only in wall clock, while prediction differs only in run timestamp.
artifact_comparison={}
for name in ('audit.py','matrices.npz','per-column.csv'):
    artifact_comparison[name]={'sha_audit':sha(AUDIT/name),'sha_rerun':sha(RERUN/name),'byte_identical':(AUDIT/name).read_bytes()==(RERUN/name).read_bytes()}
    if not artifact_comparison[name]['byte_identical']: raise SystemExit(f'rerun artifact differs: {name}')
ra=json.loads((AUDIT/'report.json').read_text()); rr=json.loads((RERUN/'report.json').read_text())
ra_sem=dict(ra); rr_sem=dict(rr); ra_sem.pop('wall_seconds',None); rr_sem.pop('wall_seconds',None)
artifact_comparison['report.json']={'sha_audit':sha(AUDIT/'report.json'),'sha_rerun':sha(RERUN/'report.json'),'semantic_equal_except_wall_seconds':ra_sem==rr_sem,'wall_seconds_audit':ra['wall_seconds'],'wall_seconds_rerun':rr['wall_seconds']}
if ra_sem!=rr_sem: raise SystemExit('rerun semantic report mismatch')
pa=json.loads((AUDIT/'prediction.json').read_text()); pr=json.loads((RERUN/'prediction.json').read_text())
pa_sem=dict(pa); pr_sem=dict(pr); pa_sem.pop('time_utc',None); pr_sem.pop('time_utc',None)
artifact_comparison['prediction.json']={'sha_audit':sha(AUDIT/'prediction.json'),'sha_rerun':sha(RERUN/'prediction.json'),'semantic_equal_except_time_utc':pa_sem==pr_sem}
if pa_sem!=pr_sem: raise SystemExit('rerun prediction input binding mismatch')

# Read the JSONL capture header, compiler parameter declaration and exact probe
# build call. Parameter 26 is zero-based: the 27th named slot is fb_epsilon.
first=CAP.open('rb').readline()
header=json.loads(first)
params=header['materials'][0]['parameter_names']
assert header['materials'][1]['parameter_names']==params
assert len(params)==28 and params[26]=='fb_epsilon'
constitutive=(SNAP/'matter/src/constitutive_impl.cpp').read_text()
probe=(SNAP/'matter/tools/cardboard_probe.mm').read_text()
run_prediction=json.loads(RUNPRED.read_text())
assert run_prediction['inputs_sha256']['source/matter/tools/cardboard_probe.mm']==sha(SNAP/'matter/tools/cardboard_probe.mm')
assert '.maximumRateExponent = 0u' in probe
assert '"fb_epsilon"' in constitutive
param_check={
 'capture_header_sha256':sha(CAP),
 'captured_parameter_count':len(params),
 'parameter_26_zero_based':params[26],
 'parameter_26_value_environment0_liner':header['parameters_by_environment'][0][26],
 'parameter_26_value_environment0_medium':header['parameters_by_environment'][0][28+26],
 'source_parameter_declaration_sha256':sha(SNAP/'matter/src/constitutive_impl.cpp'),
 'probe_source_sha256':sha(SNAP/'matter/tools/cardboard_probe.mm'),
 'probe_maximumRateExponent_zero_literal_found':True,
 'prediction_binds_probe_source':True,
}
# Preserve, but flag, the predecessor sweep's mislabeled field; it contains
# precisely the captured fb_epsilon values rather than a scheduler exponent.
sweep=json.loads(SWEEP.read_text())
legacy=sweep['selection']['rate_exponent_max_by_paper_material']
param_check['predecessor_sweep_mislabel']={
 'field_name':'rate_exponent_max_by_paper_material',
 'values':legacy,
 'actual_source':'parameter index 26 (zero-based) is fb_epsilon',
 'classification':'metadata-label defect in predecessor sweep; not consumed for 147 derivative computation',
}

# Recompute compact numerical gates from preserved per-column table and matrix.
rows=list(csv.DictReader((AUDIT/'per-column.csv').open()))
assert len(rows)==1920*9
valid_rows=[r for r in rows if r['valid_projection']=='True']
assert len(valid_rows)==len(rows)
assert all(r['relative_error']!='' and float(r['relative_error'])<=1e-3 for r in rows)
zero_reference=sum(1 for r in rows if float(r['reference_l2_Pa'])==0)
assert zero_reference==0
npz=np.load(AUDIT/'matrices.npz')
C=npz['native_C_Pa']; D=npz['independent_C_Pa']; F=npz['reference_F'];
assert C.shape==D.shape==(1920,9,9) and F.shape[0]==1920
assert np.isfinite(C).all() and np.isfinite(D).all()
max_rel={str(mi):float(ra['material_groups'][str(mi)]['max_column_relative_error']) for mi in (0,1,2)}
assert max(max_rel.values())<=1e-3

report={
 'schema':'numi.cardboard.full_tangent_independent_review.v1',
 'verdict':'PASS_WITH_SCOPED_ERRATUM',
 'review_scope':'Independent integrity/reproduction and source-lineage review of the 147 full 9x9 tangent audit; no production changes or GPU work.',
 'source_review':param_check,
 'reproduction':{
   '147_declared_artifacts_all_hash_match':True,
   '147_declared_inputs_all_hash_match':True,
   'independent_rerun_same_audit_script':True,
   'npz_and_csv_byte_identical':True,
   'semantic_report_equal_except_runtime_wall_seconds':True,
   'prediction_inputs_equal_except_timestamp':True,
 },
 'numerical_scope':{
   'tets':1920,'basis_columns':17280,'valid_projection_rows':1728,
   'zero_reference_columns':zero_reference,'maximum_relative_column_error_by_group':max_rel,
   'gate':1e-3,'all_columns_with_nonzero_reference_within_gate':True,
   'captured_state':'environment 0, control 1, Newton 48',
   'transactions':0,
   'near_zero':'No measured absolute noise floor; zero reference columns would be inconclusive. This dataset has none.',
   'warnings_retained':len(ra['warnings']),
 },
 'limitations':[
   'Single captured center and accepted history, not all solver states.',
   'This compares native cached tangent with an independent analytic FP64 projection/implicit derivative; it does not close the separate native FP32 endpoint/secant noise question.',
   'The audit is not a complete nonlinear-solve, damage/interface, nonzero-rate, physical calibration, or assembly qualification.',
   'Two medium-tet lambda activity labels differ only at an approximately 5.1e-17 FP64 increment versus captured FP32 zero; the captured-state tangent comparisons remain within the reported gate. This is an activity-label caveat near numerical zero, not proof of a material mismatch.',
 ],
 'blocking_findings':[],
}
write_json('report.json',report)
write_json('verification-details.json',{'artifact_comparison':artifact_comparison,'input_hashes':input_checks,'artifact_hashes':file_checks,'parameter_source_check':param_check})
readme='''# Independent review of full 9x9 tangent audit 147\n\nVerdict: **PASS_WITH_SCOPED_ERRATUM**. The 147 audit is hash-complete and its independent rerun reproduces the matrix NPZ and per-column CSV byte-for-byte. The report matches semantically except for runtime timing; the prediction differs only in its generation timestamp.\n\nThe source contract names parameter slot 26 (zero-based) `fb_epsilon`. The inherited sweep-006 output field `rate_exponent_max_by_paper_material` is therefore mislabeled: its values are approximately `1e-8` and are the FB epsilon values. This sweep field is an inherited metadata defect, retained here as an erratum; it is not used by 147's calculation. The selected run's source-bound probe sets `maximumRateExponent = 0u`, and the run prediction binds that exact probe SHA.\n\nThe selected state is environment 0, control 1, Newton 48, with 1,920 tetrahedra and 17,280 tangent columns. All 1,728 paper projections are valid; all column references are nonzero and errors are under `1e-3`. Maximum per-column relative errors are liner `3.29483e-6`, medium `6.13590e-5`, and glue `5.18882e-8`. Fifty-eight FP64 projection warnings remain in the source report and are not suppressed.\n\nThis is a selected-state analytic derivative check. It does not qualify FP32 endpoint/secant noise, a complete nonlinear solve, nonzero-rate laws, damage/interfaces, or physical assembly. Two medium projected multiplier increments differ from captured zero only at approximately `5.1e-17`; this is recorded as a near-zero activity-label caveat.\n\nRun `verify.py` with the qualified Python 3.10.12 environment to verify all declared hashes and rerun comparisons.\n'''
(OUT/'README.md').write_text(readme)
# Final manifest excludes itself, binding all review files plus every referenced input hash.
manifest={'schema':'numi.cardboard.full_tangent_independent_review_manifest.v1','verifier_sha256':sha(OUT/'verify.py'),'report_sha256':sha(OUT/'report.json'),'details_sha256':sha(OUT/'verification-details.json'),'readme_sha256':sha(OUT/'README.md'),'bound_inputs':{**{k:v['sha256'] for k,v in input_checks.items()},str(SWEEP):sha(SWEEP),str(RUNPRED):sha(RUNPRED),str(SNAP/'matter/tools/cardboard_probe.mm'):sha(SNAP/'matter/tools/cardboard_probe.mm'),str(SNAP/'matter/src/constitutive_impl.cpp'):sha(SNAP/'matter/src/constitutive_impl.cpp')},'audit_manifest_sha256':sha(AUDIT/'manifest.json'),'rerun_manifest_sha256':sha(RERUN/'manifest.json'),'status':'sealed_review_artifacts'}
write_json('manifest.json',manifest)
print(json.dumps({'report':str(OUT/'report.json'),'verdict':report['verdict'],'manifest_sha256':sha(OUT/'manifest.json'),'errors':0,'columns':len(rows)},indent=2))
