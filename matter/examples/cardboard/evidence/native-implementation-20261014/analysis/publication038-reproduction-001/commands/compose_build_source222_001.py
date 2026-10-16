from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,os,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-composed-strict-periodic-freebox-210-buildfix-003';s=q/'snapshots/source-capacity-gates-checkpoint-proof-222-001';d=q/'analysis/source222-composition-build-001';b=q/'build-source222-gates-proof-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());assert sha(base/'frozen-source.json')=='477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931'
s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in f['source_sha256'].items():
 assert sha(base/n)==h,n
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
layers=[('source-capacity-damage-debond-gates-217-001','baa52c11437d63844a5837657eb16d821a910aa9a60c8dd83c646488296b7c52',['matter/tools/cardboard_stage_resolution_study.py','matter/tools/test_cardboard_stage_resolution_study.py','matter/tools/cardboard_stage_resolution_study.md']),('source-qualification-checkpoint-proof-221-003','fb885540a2f096ee70df5a03701a8d2710832c6df402e8e097ff95c736e7c1a5',['matter/tools/cardboard_probe.mm'])]
records=[]
for name,h,expected in layers:
 donor=q/'snapshots'/name;df=json.loads((donor/'frozen-source.json').read_text());assert sha(donor/'frozen-source.json')==h
 actual=[n for n,x in df['source_sha256'].items() if x!=f['source_sha256'].get(n)]
 # Both overlays directly descend210 and touch disjoint files.
 assert sorted(actual)==sorted(expected),(name,actual)
 assert all(sha(donor/n)==v for n,v in df['source_sha256'].items())
 for n in expected:shutil.copy2(donor/n,s/n)
 records.append({'snapshot':str(donor),'frozen_sha256':h,'changed':expected})
parentmap=dict(f['source_sha256']);f['source_sha256']={n:sha(s/n) for n in sorted(parentmap)}
changed=[n for n,h in parentmap.items() if h!=f['source_sha256'][n]];assert len(changed)==4
f['source222_composition']={'parent_snapshot':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'layers':records,'changed':sorted(changed),'physical_operator_or_material_law_changed':False}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','4','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-coupon-resume-check']
(d/'command.json').write_text(json.dumps({'argv':argv,'source_frozen_sha256':sha(s/'frozen-source.json'),'layers':records,'changed':sorted(changed),'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:r=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'},cwd=s)
test=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/test_cardboard_stage_resolution_study.py')]
with (d/'tests.stdout').open('xb') as out,(d/'tests.stderr').open('xb') as err:rt=subprocess.run(test,stdout=out,stderr=err,cwd=s,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
report={'schema':'numi.cardboard.source222-composition-build.v1','build_exit_code':r.returncode,'capacity_tests_exit_code':rt.returncode,'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'changed':sorted(changed),'layers':records,'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'physical_operator_or_material_law_changed':False,'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(r.returncode or rt.returncode or int(not report['all_inputs_unchanged']))
