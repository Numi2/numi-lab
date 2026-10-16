from pathlib import Path
import csv,hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014'); O=B/'analysis/native479480483-independent-adjudication-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
I=B/'analysis/closed-native479480483-transfer-inventory-001/inventory.json';iv=json.loads(I.read_text())
assert not O.exists();O.mkdir()
roots={'479':B/'solver/macmini-warm-cgls-control19-checkpoint-479','480':B/'solver/macmini-warm-cgls-control19-checkpoint-480','483':B/'solver/macmini-rejected-encoder-cleanup-483'}
packages={'479':'native479-control19-checkpoint-command-001','480':'native480-control19-checkpoint-command-001','483':'native483-rejected-encoder-cleanup-command-001'}
verified={}; receipts={}
for k,root in roots.items():
 files=[f for f in iv[k]['files'] if not f['path'].startswith('inputs/')]
 for f in files:
  p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],(k,f['path'])
 receipt=B/'analysis'/packages[k]/('execution-'+k)/'receipt.json'
 assert sha(receipt)==iv[k]['terminal_receipt_sha256'];r=json.loads(receipt.read_text())
 assert not r['timed_out'] and not r['space_abort'] and not r['errors']
 receipts[k]=r
 verified[k]={'noninput_files':len(files),'noninput_bytes':sum(f['bytes'] for f in files),'duplicate_inputs_preserved_remote':len(iv[k]['files'])-len(files),'terminal_receipt_sha256':sha(receipt)}
assert receipts['479']['wrapper_exit_code']==1
e479=(roots['479']/'stderr.log').read_text();assert 'resume' in e479.lower() and 'checkpoint' in e479.lower()
assert receipts['480']['wrapper_exit_code']==0
native=json.loads((roots['480']/'receipt.json').read_text());assert native['exit_code']==0 and native['source_inputs_unchanged'] and not native['timed_out']
out=roots['480']/'output';res=json.loads((out/'result.json').read_text());assert res['accepted_steps']==20 and res['status']=='accepted_checkpoint_prefix_saved'
cp=out/'checkpoints/coupon-control-000019.nmcb';meta={}
for line in cp.with_suffix('.resume').read_text().splitlines():
 k,sep,v=line.partition('=');assert sep and k not in meta;meta[k]=v
assert meta['checkpoint_sha256']==sha(cp) and int(meta['checkpoint_bytes'])==cp.stat().st_size
assert meta['accepted_control_index']=='19' and meta['next_control_index']=='20' and meta['both_environments_accepted']=='true'
assert meta['prefix_manifest_sha256']==sha(cp.with_suffix('.prefix'))
prefix=[]
for line in cp.with_suffix('.prefix').read_text().splitlines():
 if not line.startswith('file='):continue
 name,n,digest,obs=line[5:].split('|');p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 target=out/p;assert target.stat().st_size==int(n) and sha(target)==digest;prefix.append(name)
def rows(p):return list(csv.DictReader(p.open()))
actual=rows(out/'observations.csv')
p476=B/'solver/macmini-warm-cgls-control0-prefix-476/output';p477=B/'solver/macmini-warm-cgls-control0-resume-477/output'
expected=rows(p476/'observations.csv')+[r for r in rows(p477/'observations.csv') if int(r['step'])<20]
assert len(actual)==len(expected)==40 and actual==expected
for r in actual:assert r['step_accepted']=='1' and r['status_code']=='0' and float(r['certificate_residual'])<=5e-7 and float(r['min_J'])>0
geometries=[]
for p in sorted(out.glob('accepted_step_*.obj')):
 choices=[q for q in (p476/p.name,p477/p.name) if q.exists()];assert choices
 assert all(sha(q)==sha(p) for q in choices),p.name
 geometries.append({'name':p.name,'sha256':sha(p)})
assert len(geometries)==40
assert receipts['483']['wrapper_exit_code']==2
n483=json.loads((roots['483']/'receipt.json').read_text());assert n483['exit_code']==2 and n483['source_inputs_unchanged']
r483=json.loads((roots['483']/'output/result.json').read_text());assert r483['status']=='failed' and r483['accepted_steps']==0 and 'requested base-state Cauchy fallback is unsupported' in r483['failure']
e483=(roots['483']/'stderr.log').read_text();assert 'requires_true_residual_certification_for_the_same_selected_newton_state' in e483 and 'Command encoder released without endEncoding' not in e483
report={'schema':'numi.cardboard.native479480483.adjudication.v1','inventory_sha256':sha(I),'transfers':verified,
 'native479':{'verdict':'INVALID_ARGUMENTS_resume_and_checkpoint_save','stderr':e479,'gpu_physical_transaction':False},
 'native480':{'verdict':'PASS_accepted_control19_checkpoint_and_observed_resume_parity','checkpoint_sha256':sha(cp),'checkpoint_bytes':cp.stat().st_size,'metadata_sha256':sha(cp.with_suffix('.resume')),'prefix_sha256':sha(cp.with_suffix('.prefix')),'prefix_files_verified':len(prefix),'all_40_observation_rows_bitwise_strings_identical':True,'all_40_accepted_OBJ_bytes_identical':True,'checkpoint_internal_state_second_independent_capture_available':False,'full_protocol_complete':False,'control19':actual[-2:]},
 'native483':{'verdict':'PASS_expected_configuration_rejection_without_encoder_abort','source181':True,'expected_native_exit_code':2,'accepted_steps':0,'failure':r483['failure'],'physics_solver_qualification':False},
 'qualification_axes':{k:False for k in ['solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly','crush_impact_prediction']}}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'matched-accepted-geometry.json').write_text(json.dumps(geometries,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'files_sha256':{p.name:sha(p) for p in sorted(O.iterdir())}},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(O/'manifest.json'),'checkpoint_sha256':sha(cp),'prefix_files':len(prefix),'observation_rows':len(actual),'geometry_files':len(geometries),'transfers':verified}))
