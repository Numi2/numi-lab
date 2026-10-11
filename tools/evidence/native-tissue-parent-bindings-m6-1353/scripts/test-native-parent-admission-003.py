from pathlib import Path
import json,subprocess,copy,time,hashlib
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'parent-bindings-native-identity-1353-003';D=C/'negative-admission';D.mkdir(exist_ok=False)
decl=json.loads((C/'baseline/run-declaration.json').read_text());base=json.loads((C/'resting-anatomy-receipt.json').read_text())
idx=decl['argv'].index('numilab_human.resting_run');env=dict(v.split('=',1) for v in decl['argv'][2:idx-2])
cases=[
 ('child_hash','child payload identity mismatch',lambda x:x.update(child_payload_sha256='0'*64)),
 ('source_hash','source SHA-256 mismatch',lambda x:x['source_payloads'][0].update(sha256='0'*64)),
 ('negative_coefficient','coefficient must be positive',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(3,-1)),
 ('nonunit','partition of unity',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(3,.5)),
 ('vertex_oob','raw vertex is absent',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(2,1000000000)),
 ('duplicate_term','repeated raw vertex term',lambda x:x['surfaces'][0]['vertices'][0].append([0,163,0,1])),
 ('source_oob','source index invalid',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(0,8)),
 ('boolean_coefficient','coefficient must be numeric',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(3,True)),
 ('absent_child','child row absent or duplicated',lambda x:x['surfaces'][0].update(stable_id=999)),
 ('duplicate_child','child row absent or duplicated',lambda x:x['surfaces'].append(copy.deepcopy(x['surfaces'][0]))),
 ('rest_geometry_shift','changes registered rest geometry',lambda x:x['surfaces'][0]['vertices'][0][0].__setitem__(2,100)),
]
rows=[]
for name,expected,mutate in cases:
 r=copy.deepcopy(base);mutate(r['passive_tissue_parent_bindings'])
 # Keep unchanged cardiac sidecars resolved to the validated parent receipt.
 for descriptor in r['provenance']['cardiac_geometry_binding']['common_field'].values():
  if isinstance(descriptor,dict) and 'path' in descriptor:
   path=Path(descriptor['path'])
   if not path.is_absolute():descriptor['path']=str(C/path)
 path=D/(name+'.json');path.write_text(json.dumps(r))
 argv=list(decl['owner_cli_preview']['native_argv']);argv[argv.index('--resting-anatomy-receipt')+1]=str(path)
 argv[4]=str(D/(name+'-native'));argv[argv.index('--resting-movie')+1]=str(D/(name+'-native')/'native-viewer.mov')
 log=D/(name+'.log');at=time.monotonic()
 with log.open('w') as f:result=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
 text=log.read_text();passed=result.returncode!=0 and expected in text
 row=dict(case=name,returncode=result.returncode,expected=expected,passed=passed,wall_seconds=time.monotonic()-at,receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
 rows.append(row);(D/'report.json').write_text(json.dumps(dict(complete=False,checks=rows),indent=2));print(json.dumps(row),flush=True)
 if not passed:raise RuntimeError(name)
(D/'report.json').write_text(json.dumps(dict(complete=True,passed=all(r['passed'] for r in rows),checks=rows),indent=2))
