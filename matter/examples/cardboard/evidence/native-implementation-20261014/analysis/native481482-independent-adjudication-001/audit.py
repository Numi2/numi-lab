from pathlib import Path
import csv,hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/native481482-independent-adjudication-001'
assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
I=B/'analysis/closed-native479480481482483-transfer-inventory-001/inventory.json';iv=json.loads(I.read_text())
roots={'481':B/'solver/macmini-warm-cgls-control20-capture-481','482':B/'solver/macmini-warm-cgls-control20-capture-off-482'}
P=B/'analysis/native481482-warm-control20-capture-command-001';verified={};receipts={}
for k,root in roots.items():
 files=[f for f in iv[k]['files'] if not f['path'].startswith('inputs/')]
 for f in files:
  p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],(k,f['path'])
 r=P/('execution-'+k)/'receipt.json';assert sha(r)==iv[k]['terminal_receipt_sha256'];receipts[k]=json.loads(r.read_text())
 assert not receipts[k]['timed_out'] and not receipts[k]['space_abort'] and not receipts[k]['errors']
 verified[k]={'files':len(files),'bytes':sum(f['bytes'] for f in files),'duplicate_inputs_preserved_remote':len(iv[k]['files'])-len(files),'terminal_receipt_sha256':sha(r)}
assert receipts['481']['wrapper_exit_code']==3
n481=json.loads((roots['481']/'receipt.json').read_text());assert n481['exit_code']==1 and n481['source_inputs_unchanged']
assert 'coupon checkpoint non-path NM_/NUMI_MATTER_ settings differ' in (roots['481']/'stderr.log').read_text()
assert receipts['482']['wrapper_exit_code']==2
n482=json.loads((roots['482']/'receipt.json').read_text());assert n482['exit_code']==2 and n482['source_inputs_unchanged'] and not n482['timed_out']
out=roots['482']/'output';prior=B/'solver/macmini-warm-cgls-control0-resume-477/output'
rows=lambda p:list(csv.DictReader(p.open()))
x=rows(out/'observations.csv');y=[r for r in rows(prior/'observations.csv') if r['step']=='20'];assert len(x)==2 and x==y
res=json.loads((out/'result.json').read_text());assert res['status']=='resumed_diagnostic_segment_failed' and res['accepted_steps']==0
assert x[0]['step_accepted']=='0' and x[0]['status_code']=='10' and float(x[0]['certificate_residual'])>5e-7
assert x[1]['step_accepted']=='1' and x[1]['status_code']=='0'
common=[];excluded={'result.json','coupon-resume-segment.json','observations.csv'}
for p in sorted(out.rglob('*')):
 if not p.is_file() or p.name in excluded:continue
 rel=p.relative_to(out);q=prior/rel
 if not q.is_file():continue
 assert sha(p)==sha(q),str(rel);common.append({'path':str(rel),'sha256':sha(p),'bytes':p.stat().st_size})
assert len(common)==35
def traces(p):
 ans={}
 for line in p.read_text().splitlines():
  key,sep,s=line.partition('=')
  if not sep or key not in ('fem_true_residual_cycle','fem_base_cgls_direction'):continue
  v=json.loads(s)
  if v.get('root')==20:ans.setdefault(key,[]).append(v)
 return ans
a=traces(roots['482']/'stderr.log');b=traces(prior.parent/'stderr.log')
assert a==b and len(a['fem_true_residual_cycle'])==4096 and len(a['fem_base_cgls_direction'])==64
report={'schema':'numi.cardboard.native481482.adjudication.v1','inventory_sha256':sha(I),'transfers':verified,
 'native481':{'verdict':'INVALID_DIAGNOSTIC_RESUME_ADMISSION','native_exit_code':1,'wrapper_exit_code':3,'physical_step_executed':False,'reason':'Original strict checkpoint guard rejects changed diagnostic selectors before stepping; capture is missing. Source182 must preserve physical bindings and separately admit authenticated observer changes.'},
 'native482':{'verdict':'PASS_FAILED_STEP_REPRODUCTION_ONLY','control20_observation_rows_identical':True,'common_physical_output_files_identical':len(common),'true_residual_cycle_records_identical':4096,'cgls_direction_records_identical':64,'timing_excluded':True,'complete_protocol':False,'accepted_bent_steps':0,'retained_control19_rollback_angle_deg':float(x[0]['measured_right_grip_angle_deg']),'final_nonlinear_residual':float(x[0]['certificate_residual'])},
 'qualification_axes':{k:False for k in ['solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly','crush_impact_prediction']}}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'matched-physical-output-files.json').write_text(json.dumps(common,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'files_sha256':{p.name:sha(p) for p in sorted(O.iterdir())}},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(O/'manifest.json'),'transfers':verified,'physical_outputs_identical':len(common),'true_residual_records':4096,'cgls_records':64}))
