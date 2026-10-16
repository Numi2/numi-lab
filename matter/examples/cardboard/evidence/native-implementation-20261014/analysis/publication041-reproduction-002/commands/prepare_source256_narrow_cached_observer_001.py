from pathlib import Path
import json,hashlib,shutil,subprocess,datetime,difflib,os
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-refined-regional-graph-240-003';patchbase=q/'snapshots/source-contact-compaction-fixture-249-001';incoming=q/'snapshots/source-primary-cache-fd-action-253-003';s=q/'snapshots/source-legacy-program-primary-cache-observer-256-001';d=q/'analysis/source256-narrow-primary-cache-observer-001';d.mkdir(exist_ok=False);s.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text())
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 p=base/n;assert sha(p)==h,n;z=s/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z)
shutil.copy2(base/'configure-support-source.json',s/'configure-support-source.json')
changed=json.loads((q/'analysis/source253-final-primary-cache-observer-package-005/report.json').read_text())['changes'];merges=[]
for row in changed:
 n=row['path'];cur=s/n
 if (patchbase/n).exists():
  assert cur.exists(),n
  p=subprocess.run(['git','merge-file','-p',str(cur),str(patchbase/n),str(incoming/n)],capture_output=True)
  (d/(Path(n).name+'.merge.stderr')).write_bytes(p.stderr)
  if p.returncode:
   (d/(Path(n).name+'.conflict')).write_bytes(p.stdout)
   raise RuntimeError('three-way conflict '+n+' exit'+str(p.returncode))
  cur.write_bytes(p.stdout);merges.append(n)
 else:
  assert not cur.exists();cur.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(incoming/n,cur)
names=set(f['source_sha256'])|{x['path'] for x in changed}
metal=[n for n in names if n.endswith(('.metal','.metalinc'))];assert all(sha(s/n)==sha(base/n) for n in metal)
g={**f,'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'source256_observer_only':{'incoming':str(incoming),'purpose':'Correct cached FD action while retaining exact240 Metal sources and checkpoint program identity; source600 correctly rejected incompatible253 program.','unchanged_metal_input_files':len(metal)},'source_sha256':{n:sha(s/n) for n in sorted(names)}}
(s/'frozen-source.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n')
patch=[]
for row in changed:
 n=row['path'];patch.extend(difflib.unified_diff((base/n).read_text().splitlines(keepends=True) if (base/n).exists() else [],(s/n).read_text().splitlines(keepends=True),fromfile='a/'+n,tofile='b/'+n))
(d/'source256.patch').write_text(''.join(patch))
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'};py=str(q/'venvs/cardboard-reference-185/bin/python3');tests=[]
for i,n in enumerate(['test_cardboard_primary_cache_action.py','test_cardboard_cached_operator_audit.py','test_cardboard_solver_environment.py']):
 p=subprocess.run([py,'-B',str(s/'matter/tools'/n)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/f'cpu-{i}.stdout').write_bytes(p.stdout);(d/f'cpu-{i}.stderr').write_bytes(p.stderr);tests.append({'name':n,'exit_code':p.returncode});assert p.returncode==0,n
r={'schema':'numi.cardboard.source256-narrow-observer-composition.v1','source':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'changes':[r['path'] for r in changed],'three_way_merges':merges,'all_metal_inputs_byte_identical_to240':len(metal),'all_production_settings_and_operator_unchanged':True,'CPU_tests':tests,'GPU_execution':False,'metallib_identity_and_checkpoint_compatibility_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in [base/'frozen-source.json',patchbase/'frozen-source.json',incoming/'frozen-source.json',Path(__file__)]},'artifacts':{p.name:sha(p) for p in d.iterdir() if p.is_file()},'frozen_source_sha256':sha(s/'frozen-source.json')},indent=2)+'\n');print(json.dumps(r))
