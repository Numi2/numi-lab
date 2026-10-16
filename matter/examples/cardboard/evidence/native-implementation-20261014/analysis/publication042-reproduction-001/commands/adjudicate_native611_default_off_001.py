from pathlib import Path
import json,hashlib,datetime,struct
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=q/'analysis/native611-integrated259-default-off-history-adjudication-001';d.mkdir(exist_ok=False);new=q/'solver/integrated259-default-off-seven-phase-611';old=q/'solver/best-of255-default-off-seven-phase-607'
r=json.loads((new/'receipt.json').read_text());assert r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out']
names=[p.name for p in (old/'output').iterdir() if p.is_file() and (p.name.startswith(('accepted_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json','observations.csv','initial.obj','result.json'])]
assert len(names)==32
rows=[];inputs=[Path(__file__),new/'receipt.json',old/'receipt.json',q/'analysis/source259-build-003/report.json']
for n in sorted(names):
 p=new/'output'/n;o=old/'output'/n;assert sha(p)==sha(o),n;rows.append({'path':n,'sha256':sha(p),'bytes':p.stat().st_size});inputs.extend([p,o])
report={'schema':'numi.cardboard.native611-integrated259-default-off-history-adjudication.v1','verdict':'PASS_COMPLETE_TINY_SEVEN_PHASE_HISTORY_BYTE_PARITY','case':str(new.relative_to(q)),'comparison_case':str(old.relative_to(q)),'production_files':rows,'wall_seconds':r['wall_seconds'],'full_target_coupon_or_convergence_qualified':False,'shipping_only_observable_changes_exercised':False,'scope':'Integrated259-003 completes the unchanged .01degree seven-control coupon regression with best-of disabled. All32 production files are byte-identical to607/603. The263 pipeline-registration repair and current420b4f86 Human source are included. Enabled best-of requires its own fresh native history.','all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n');print('611',sha(d/'report.json'))
