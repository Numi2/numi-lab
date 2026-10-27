from pathlib import Path
import importlib.util, json, sys
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
owner=N/'connected-batch-619/run_connected_batch619_v8.py'
spec=importlib.util.spec_from_file_location('batch619v8_for772',owner)
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
current=N/'component-merit-retention-631/from-run006-002/final/state.json'
prior=N/'connected-batch-619/run-006-from782-v8/round-001/prepared/constraints/constraint-context.json'
ancestry=N/'connected-batch-619/run-005-from878/round-002/final/state.json'
r=N/'continued-self-cuts-630/from-run006-cut1-002'
failed=Path(json.loads((r/'selection-002.json').read_text())['candidate'])
audit=r/'check-fit-002/audit-130/summary.json'
pairs=r/'check-fit-002/audit-130/surface-self-pairs.npy'
seed=r/'cut-003-hardface-prepared/self-seeds/seed-preflight.json'
out=N/'hard-face-feasibility-634/prepared-from772-001'
owners=h.owners(); pins={str(p.resolve()):h.sha(p) for p in owners}
s,roles,empty,query=h.strict_baseline(current)
h.paths_exist((prior,ancestry,failed,audit,pairs,seed))
out.mkdir(parents=True,exist_ok=False)
depth=out/'depth-413'
argv=[h.PY,h.DEPTH,'--candidate',roles['candidate'],'--point-summary',roles['point_summary'],'--surface-report',roles['surface_report'],'--query-dir',query.parent,'--output-dir',depth,'--expected-candidate-sha256',s['candidate_sha256'],'--expected-point-summary-sha256',s['point_summary_sha256'],'--expected-surface-report-sha256',s['surface_report_sha256'],'--expected-query-receipt-sha256',h.sha(query),'--expected-owner-sha256',h.DEPTH_OWNER_SHA]
print('refreshing depth constraints from strict 772 baseline',flush=True)
h.stage(argv,out/'depth-stage',slot=True,sources=owners+(current,roles['candidate'],roles['point_summary'],roles['surface_report'],query,prior,ancestry,failed,audit,pairs,seed),timeout=1800)
dnpz=depth/'point-depth-rows.npz'; drep=depth/'365-compatible-report.json'
h.paths_exist((dnpz,drep))
ctx=out/'current-inputs.json'
manifest={'state':str(current),'state_sha256':h.sha(current),'candidate':str(roles['candidate']),'candidate_sha256':h.sha(roles['candidate']),'point_summary':str(roles['point_summary']),'point_summary_sha256':h.sha(roles['point_summary']),'surface_report':str(roles['surface_report']),'surface_report_sha256':h.sha(roles['surface_report']),'query_receipt':str(query),'guard_context':str(prior),'source_parent_context':str(ancestry),'failed_candidate':str(failed),'failed_audit130':str(audit),'failed_pairs':str(pairs),'depth_rows_npz':str(dnpz),'depth_rows_report':str(drep),'prior_seed_report':str(seed),'native_admission':False,'accepted_physical_steps':0}
h.writej(ctx,manifest)
prep=out/'prepared'
cmd=[h.PY,h.PREP613,'--current-inputs',ctx,'--depth-rows-npz',dnpz,'--depth-rows-report',drep,'--fit-template',h.FIT_TEMPLATE,'--matched-template',h.MATCHED_TEMPLATE,'--fit-entry',h.FIT603,'--out',prep,'--prior-seed-report',seed]
print('preparing matched guards, targets, ancestry and self seeds',flush=True)
h.stage(cmd,out/'prepare-stage',slot=False,sources=owners+(ctx,current,roles['candidate'],roles['point_summary'],roles['surface_report'],query,dnpz,drep,prior,ancestry,failed,audit,pairs,seed),timeout=3600)
p=h.readj(prep/'prepared.json')
if p.get('fit_not_run') is not True: raise RuntimeError('unexpected fit')
if not isinstance(p.get('source_pins'),dict) or not p['source_pins']: raise RuntimeError('missing source pins')
h.verify_pins(p['source_pins']);h.verify_pins(pins)
print(json.dumps({'status':'prepared-only','output':str(prep),'candidate_sha256':s['candidate_sha256'],'fit_not_run':True,'native_admission':False,'accepted_physical_steps':0},sort_keys=True),flush=True)
