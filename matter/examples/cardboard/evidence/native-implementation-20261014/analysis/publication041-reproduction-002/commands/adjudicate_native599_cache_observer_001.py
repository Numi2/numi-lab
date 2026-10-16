from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native599-corrected-cache-observer-parity-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
new=q/'solver/primary-cache-action-diagnostic-regression-599';old=q/'solver/composed241-seven-phase-regression-588'
r=json.loads((new/'receipt.json').read_text());assert r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out']
names=[p.name for p in (old/'output').iterdir() if p.is_file() and (p.name.startswith(('accepted_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json','observations.csv'])]
rows=[];inputs=[new/'receipt.json',old/'receipt.json',Path(__file__)]
for n in sorted(names):
 p=new/'output'/n;o=old/'output'/n;assert sha(p)==sha(o),n
 rows.append({'path':n,'sha256':sha(p),'bytes':p.stat().st_size});inputs.extend([p,o])
assert len(rows)==32,len(rows)
cap=new/'diagnostics/capture-operator-fd-capture/capture'
with cap.open() as f:head=json.loads(next(f))
print('header_kind',list(head)[:5])
policy=head['operator_action_policy'];assert policy['cache_mode']==1 and policy['selection']=='matches_primary_FGMRES_cache_mode'
report={'schema':'numi.cardboard.native599-corrected-cache-observer-parity.v1','verdict':'PASS_COMPLETE_TINY_SEVEN_PHASE_PRODUCTION_HISTORY_BYTE_PARITY','comparison_case':str(old.relative_to(q)),'case':str(new.relative_to(q)),'production_files':rows,'operator_action_policy':policy,'capture_sha256':sha(cap),'limits':['Tiny .01degree7control regression only; not the requested.375degree60control coupon or numerical convergence.','This proves corrected cached observer production history parity; quantitative cached action comparison is separate and no noise floor inferred here.'],'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');inputs.append(cap)
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'production_files_exact':len(rows),'cache_mode':policy['cache_mode']}))
