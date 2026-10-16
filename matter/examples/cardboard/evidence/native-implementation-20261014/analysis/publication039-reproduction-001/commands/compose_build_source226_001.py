from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,os,sys,difflib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=q/'snapshots/source-capacity-gates-checkpoint-proof-222-001';ancestor=q/'snapshots/source207-buildfix-002';donor=q/'snapshots/source-pure-fem-armijo-cgls-arena-215-002';s=q/'snapshots/source-composed-pure-8192-on222-226-001';d=q/'analysis/source226-composition-build-001';b=q/'build-source226-pure-arenas-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());af=json.loads((ancestor/'frozen-source.json').read_text());df=json.loads((donor/'frozen-source.json').read_text())
assert sha(base/'frozen-source.json')=='698a3b1106e6997aa251a8090894adcb8b2850085b70a83f681ccd6fd6f383b6'
assert sha(ancestor/'frozen-source.json')=='5f7d27b1c56784559b444434c90e96668dd5e04caabf87688590420c7fc954fd'
assert sha(donor/'frozen-source.json')=='9c94ee26141a1dec60bb133bc8588a8934013786e09ee2a60551f17610e08658'
for root,m in [(base,f),(ancestor,af),(donor,df)]:assert all(sha(root/n)==h for n,h in m['source_sha256'].items())
assert not set(af['source_sha256'])-set(df['source_sha256'])
s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n in f['source_sha256']:
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
delta=[n for n,h in df['source_sha256'].items() if af['source_sha256'].get(n)!=h];records=[]
for n in delta:
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);new=n not in af['source_sha256'];current=f['source_sha256'].get(n);old=af['source_sha256'].get(n)
 if current==old or current is None:
  shutil.copy2(donor/n,p);mode='new' if new else 'direct'
 elif current==df['source_sha256'][n]:mode='already_identical'
 else:
  cp=subprocess.run(['git','merge-file','-p',str(p),str(ancestor/n),str(donor/n)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  (d/(n.replace('/','__')+'.merge.stderr')).write_bytes(cp.stderr)
  (d/(n.replace('/','__')+'.merge.output')).write_bytes(cp.stdout)
  assert cp.returncode==0,{'file':n,'merge_exit':cp.returncode}
  p.write_bytes(cp.stdout);mode='three_way_clean'
 records.append({'file':n,'mode':mode,'ancestor_sha256':old,'base222_sha256':current,'donor215_sha256':df['source_sha256'][n],'merged_sha256':sha(p)})
keys=set(f['source_sha256'])|set(df['source_sha256']);f['source_sha256']={n:sha(s/n) for n in sorted(keys)}
f['source226_composition']={'parent_snapshot':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'ancestor':str(ancestor),'ancestor_frozen_sha256':sha(ancestor/'frozen-source.json'),'donor':str(donor),'donor_frozen_sha256':sha(donor/'frozen-source.json'),'files':records,'scope':'Checked optional true-residual and pure-FEM Armijo/CGLS arenas at8192 nodes; physical equations and outer acceptance criteria unchanged; combined execution unqualified.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
patch=''.join(''.join(difflib.unified_diff((base/n).read_text().splitlines(True) if (base/n).exists() else [],(s/n).read_text().splitlines(True),fromfile='a/'+n if (base/n).exists() else '/dev/null',tofile='b/'+n)) for n in sorted(delta))
(d/'source.patch').write_text(patch)
targets=['numi-matter-cardboard-probe','numi-matter-cardboard-coupon-resume-check','numi-matter-true-residual-arena-plan-check','numi-matter-pure-fem-armijo-cgls-arena-plan-check','numi-matter-fem-cgls-arena-8192-check','numi-matter-fgmres-true-residual-native-check']
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','2']
for t in targets:argv+=['--target',t]
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
(d/'command.json').write_text(json.dumps({'argv':argv,'source_frozen_sha256':sha(s/'frozen-source.json'),'files':records,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:r=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=s)
tests=[]
if r.returncode==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 cmds += [[str(b/n)] for n in ['numi-matter-true-residual-arena-plan-check','numi-matter-pure-fem-armijo-cgls-arena-plan-check','numi-matter-cardboard-coupon-resume-check']]
 for i,a in enumerate(cmds):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:rc=subprocess.run(a,stdout=out,stderr=err,cwd=s,env=env).returncode
  tests.append({'argv':a,'exit_code':rc,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
report={'schema':'numi.cardboard.source226-composition-build.v1','build_exit_code':r.returncode,'tests':tests,'all_pass':r.returncode==0 and all(t['exit_code']==0 for t in tests),'source_snapshot':str(s),'source_count':len(keys),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'source_patch_sha256':sha(d/'source.patch'),'files':records,'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'gpu_execution':False,'physical_transactions':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
