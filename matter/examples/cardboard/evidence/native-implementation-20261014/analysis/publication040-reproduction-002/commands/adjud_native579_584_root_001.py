from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native579-584-root-adjudication-001';d.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={};rows=[]
for n,needle in [('refined-coupon-cgls-579','arena_20682_native_check=PASS checks=165559'),('maximum-coupon-cgls-580','arena_32768_native_check=PASS checks=262247'),('refined-coupon-true-residual-581','kernel contract 131783 checks'),('refined-regional-native-regression-582','PASS regional RAP/block-Jacobi')]:
 p=q/'native'/n;r=json.loads((p/'receipt.json').read_text());out=(p/'stdout.log').read_text()
 assert r['execution_started'] and r['source_inputs_unchanged'] and r['exit_code']==0 and not r['timed_out'] and needle in out and (p/'stderr.log').stat().st_size==0
 for f in [p/'receipt.json',p/'stdout.log',p/'stderr.log']:inputs[str(f)]=sha(f)
 rows.append({'case':n,'component_gate_pass':True,'physical_mesh_time_convergence':False,'full_shipping_operator_qualified':False,'stdout_sha256':sha(p/'stdout.log')})
p=q/'snapshots/source-refined-regional-graph-240-003/matter/tools/fgmres_true_residual_native_check.mm';t=p.read_text()
assert 'if(argc==2){checkLargeNodeResidualMeasurement(device,queue,measure,20682u);checkLargeNodeResidualMeasurement(device,queue,measure,32768u);}' in t
inputs[str(p)]=sha(p)
off=q/'native/contact-trial-observer-off-583';on=q/'native/contact-trial-observer-on-584'
parity={}
for name in ['transaction-000000-before.bin','transaction-000000-after.bin','transaction-000000.json','transaction-000000-first-failure.bin']:
 a=off/'accepted-state-output/dense'/name;b=on/'accepted-state-output/dense'/name
 parity[name]={'off_sha256':sha(a),'on_sha256':sha(b),'byte_equal':a.read_bytes()==b.read_bytes()}
 for f in [a,b]:inputs[str(f)]=sha(f)
assert all(x['byte_equal'] for x in parity.values())
rollback={n:(p/'accepted-state-output/dense/transaction-000000-before.bin').read_bytes()==(p/'accepted-state-output/dense/transaction-000000-after.bin').read_bytes() for n,p in [('off',off),('on',on)]}
assert all(rollback.values())
parsed=[];bad=[]
for i,line in enumerate((on/'stderr.log').read_text().splitlines(),1):
 if line.startswith('fem_deformable_limit_trace='):
  try:parsed.append(json.loads(line.split('=',1)[1]))
  except json.JSONDecodeError as e:bad.append({'line':i,'error':str(e)})
assert len(parsed)==37 and len(bad)==27
for p in [off,on]:
 for n in ['receipt.json','command.json','stderr.log','stdout.log']:inputs[str(p/n)]=sha(p/n)
r={'schema':'numi.cardboard.native579-584-root-adjudication.v1','component_results':rows,'native581_metadata_erratum':{'executed_large_node_measurements':[4096,8192,20682,32768],'proof':'Source-bound --refined-coupon branch runs 20682 and32768 before the existing4096/8192 checks; process exits0 after all assertions.','stdout_defect':'Hardcoded label lists only4096,8192 and contains a literal backslash-n; preserve raw581 stdout and repair emitter in a new snapshot.','full_FEM_operator_qualified':False},'native583_584':{'native_failure_preserved':True,'off_on_production_parity':parity,'transactional_rollback_exact':rollback,'trace_lines_total':64,'valid_empty_contact_records':len(parsed),'malformed_nonempty_records':bad,'trace_interpretation':'INCONCLUSIVE: missing barrier-array closing bracket prevents JSON decode. Preserve raw outputs; a corrected source and fresh pair are required.','nonlinear_contact_qualified':False},'all_inputs_unchanged':all(sha(p)==h for p,h in inputs.items()),'all_six_full_sequence_axes_open':True,'inputs_sha256':inputs,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'report_sha256':sha(d/'report.json'),'script_sha256':sha(__file__),'inputs_sha256':inputs},indent=2)+'\n')
print(json.dumps({'report':str(d/'report.json'),'sha256':sha(d/'report.json'),'all_inputs_unchanged':r['all_inputs_unchanged'],'off_on_parity':True,'rollback':rollback,'components':len(rows)}))
