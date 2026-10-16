from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=q/'analysis/native606-607-best-of-root-adjudication-001';d.mkdir(exist_ok=False)
component=q/'native/best-of-cgls-production-selection-606';new=q/'solver/best-of255-default-off-seven-phase-607';old=q/'solver/composed254-seven-phase-regression-603'
inputs=[Path(__file__),component/'receipt.json',component/'stdout.log',component/'stderr.log',new/'receipt.json',old/'receipt.json',q/'snapshots/source-best-of-cgls-255-001/frozen-source.json',q/'build-source255-best-of-005/source-build.json']
for c in [component,new]:
 r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out']
assert (component/'stdout.log').read_text().strip()=='pure_fem_cgls_best_of_production_kernel=PASS checks=414 physical_transactions=0'
names=[p.name for p in (old/'output').iterdir() if p.is_file() and (p.name.startswith(('accepted_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json','observations.csv','initial.obj','result.json'])]
assert len(names)==32
rows=[]
for n in sorted(names):
 p=new/'output'/n;o=old/'output'/n;assert sha(p)==sha(o),n;rows.append({'path':n,'sha256':sha(p),'bytes':p.stat().st_size});inputs.extend([p,o])
report={'schema':'numi.cardboard.native606-607-best-of-root-adjudication.v1','component_verdict':'PASS_ACTUAL_METAL_TRANSACTION_KERNELS_TEN_ADVERSARIAL_CASES','component_checks':414,'component_physical_transactions':0,'production_verdict':'PASS_DEFAULT_OFF_COMPLETE_TINY_SEVEN_PHASE_HISTORY_BYTE_PARITY','production_files':rows,'comparison_case':str(old.relative_to(q)),'case':str(new.relative_to(q)),'scope':'Selection kernels restore all2EN rows and complete saved metadata for invalid/losing/unselected alternatives; default-off runtime completes the unchanged .01 degree seven-control history. Enabled best-of behavior requires separate physical run.','full_target_coupon_or_convergence_qualified':False,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n');print(json.dumps({'report_sha256':sha(d/'report.json'),'production_files_exact':len(rows)}))
