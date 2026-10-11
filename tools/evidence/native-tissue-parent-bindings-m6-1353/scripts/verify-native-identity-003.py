from pathlib import Path
import json,hashlib,csv,sys,numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'parent-bindings-native-identity-1353-003';P=A/'temporalis163-native-m6-001';O=C/'verification.json'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=C/'baseline/native-run';pp=P/'baseline/native-run'
a=list(csv.DictReader((cp/'resting-coupled.csv').open()));b=list(csv.DictReader((pp/'resting-coupled.csv').open()))
execution=json.loads((C/'baseline/execution.json').read_text())
assert execution['returncode']==0 and not execution['changed_inputs'] and execution['native_argv_matches_prepared_cli_preview'] and execution['native_environment_matches_prepared_cli_preview']
prior=json.loads((P/'native-verification-001/report.json').read_text());assert prior['all_native_geometry_checks_pass']
inventory=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
keys,*_=ca._load_target_inventory(inventory);keys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163)}
checks=[];pins={str(p):sha(p) for p in (Path(__file__),cp/'resting-coupled.csv',pp/'resting-coupled.csv',C/'baseline/execution.json',C/'baseline/run-declaration.json',C/'resting-anatomy-receipt.json',P/'native-verification-001/report.json')}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 c=cp/'accepted-geometry'/f'step-{step}.mrvpack';p=pp/'accepted-geometry'/f'step-{step}.mrvpack'
 x,xs,_=ca._pack_surfaces(c,keys);y,ys,_=ca._pack_surfaces(p,keys)
 cr=json.loads(c.with_suffix('.receipt.json').read_text());pr=json.loads(p.with_suffix('.receipt.json').read_text())
 assert set(xs)==set(ys) and len(xs)==867
 different=[]
 for key in xs:
  f=np.array(xs[key]['faces'],int);g=np.array(ys[key]['faces'],int)
  if not np.array_equal(x[f],y[g]):different.append(dict(key=key,max_m=float(np.linalg.norm(x[f].astype(float)-y[g].astype(float),axis=2).max())))
 checks.append(dict(step=step,body_poses_exact=cr['accepted_body_poses']==pr['accepted_body_poses'],surfaces=867,different_surfaces=different))
 for f in (c,p,c.with_suffix('.receipt.json'),p.with_suffix('.receipt.json')):pins[str(f)]=sha(f)
report=dict(complete=True,trace_rows=len(a),physiology_exact=a==b,checks=checks,input_sha256=pins,wall_seconds=execution['wall_seconds'],real_time_factor=float(a[-1]['time_s'])/execution['wall_seconds'],boundary='Identity source-parent reference regression in integrated native scene. No new anatomy or nontrivial parent interpolation qualification.')
report['passed']=len(a)==1000 and a==b and all(r['body_poses_exact'] and not r['different_surfaces'] for r in checks)
O.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)
assert report['passed']
