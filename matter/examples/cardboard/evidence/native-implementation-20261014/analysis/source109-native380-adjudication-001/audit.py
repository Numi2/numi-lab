from pathlib import Path
import json,hashlib,shutil
B=Path('/Users/home/cardboard-implementation-20261014');C=B/'solver/explicit32-certified-coupon-380';P=B/'solver/cached-true-residual-certification-coupon-370';O=B/'analysis/source109-native380-adjudication-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((C/'receipt.json').read_text());assert r['execution_started'] and r['exit_code']==2 and not r['timed_out'] and not r['termination_errors'] and r['source_inputs_unchanged']
m=json.loads((C/'output/manifest.json').read_text());assert m['solver']['compiled_local_material_newton_iterations']==[32]*3
rows=[]
for line in (C/'stderr.log').read_text().splitlines():
 if line.startswith('fem_residual_armijo_iteration='):
  row=json.loads(line.split('=',1)[1]);e=row['environments'][0]
  rows.append(dict(iteration=row['newton_iteration'],base_health=e['base_health'],status=e['solver_status'],pretrial=e['pretrial_status'],trial_outcome=e['trial_outcome']))
first=next(x for x in rows if x['status']['code']!=0)
comparisons={}
for name in ['initial.obj','mesh.json','accepted_step_000001_bent.obj','accepted_step_000001_held_reference.obj']:
 comparisons[name]=dict(equal=sha(C/'output'/name)==sha(P/'output'/name),native380_sha256=sha(C/'output'/name),native370_sha256=sha(P/'output'/name))
obs=(C/'output/observations.csv').read_text().splitlines();old=(P/'output/observations.csv').read_text().splitlines();assert obs[:3]==old[:3]
out=dict(schema='numi.cardboard.native380-adjudication.v1',verdict='FAIL_MATERIAL_PROJECTION_AFTER_EFFECTIVE_BUDGET_REPAIR',receipt=r,actual_compiled_material_budgets=[32,32,32],accepted_control_count=1,required_control_count=27,first_observed_failed_iteration=first,root1_trace_rows=len(rows),control0_csv_rows_equal=True,physical_geometry_comparisons=comparisons,inputs={str(p):sha(p) for p in [C/'receipt.json',C/'output/manifest.json',C/'output/result.json',C/'stderr.log',C/'output/compiled.nmatterpack',P/'output/compiled.nmatterpack']},limits='Failure674 must be replayed independently. Source379 local repair remains valid; full coupon and numerical convergence remain unqualified. No unaccepted geometry is an animation frame.')
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2(__file__,O/'audit.py');(O/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in O.iterdir() if p.is_file()}),indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['inputs','receipt']},indent=2))
