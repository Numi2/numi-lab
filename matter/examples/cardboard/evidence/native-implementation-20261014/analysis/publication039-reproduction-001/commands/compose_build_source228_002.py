from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,os,sys,difflib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=q/'snapshots/source-composed-pure-8192-on222-226-001';ancestor=q/'snapshots/source207-buildfix-002';donor=q/'snapshots/source216-integrated-configure-support-006';s=q/'snapshots/source-composed-pure-proxy-228-002';d=q/'analysis/source228-composition-build-002';b=q/'build-source228-integrated-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());af=json.loads((ancestor/'frozen-source.json').read_text());df=json.loads((donor/'frozen-source.json').read_text())
assert sha(base/'frozen-source.json')=='313f5a81883162bb5917773b3b3f576cf8991932ff4eed2e8af11e36cb13cba8'
assert sha(ancestor/'frozen-source.json')=='5f7d27b1c56784559b444434c90e96668dd5e04caabf87688590420c7fc954fd'
assert sha(donor/'frozen-source.json')=='26636e0efcd70b8aef95afb82475c1d5f69eec7a1f26c7191d355d4fe55de67e'
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
  if cp.returncode==1 and n=='matter/src/runtime.mm':
   merged=cp.stdout.decode();lines=merged.splitlines(True)
   markers=[l for l in lines if l.startswith(('<<<<<<< ','=======','>>>>>>> '))]
   assert len(markers)==3,markers
   start=next(i for i,l in enumerate(lines) if l.startswith('<<<<<<< '))
   middle=next(i for i,l in enumerate(lines) if l.startswith('======='))
   end=next(i for i,l in enumerate(lines) if l.startswith('>>>>>>> '))
   assert ''.join(lines[start+1:middle])=='#include "numi/matter/detail/true_residual_arena_plan.hpp"\\n#include "numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp"\\n'
   assert ''.join(lines[middle+1:end])=='#include "numi/matter/detail/free_box_true_residual_proxy_admission.hpp"\\n'
   p.write_text(''.join(l for i,l in enumerate(lines) if i not in [start,middle,end]));mode='manual_union_three_disjoint_include_lines'
  else:
   assert cp.returncode==0,{'file':n,'merge_exit':cp.returncode}
   p.write_bytes(cp.stdout);mode='three_way_clean'
 records.append({'file':n,'mode':mode,'ancestor_sha256':old,'base226_sha256':current,'donor216_sha256':df['source_sha256'][n],'merged_sha256':sha(p)})
keys=set(f['source_sha256'])|set(df['source_sha256']);f['source_sha256']={n:sha(s/n) for n in sorted(keys)}
f['source228_composition']={'parent_snapshot':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'ancestor':str(ancestor),'ancestor_frozen_sha256':sha(ancestor/'frozen-source.json'),'donor':str(donor),'donor_frozen_sha256':sha(donor/'frozen-source.json'),'files':records,'scope':'Compose the checked free-BOX/prescribed-BOX/capsule proxy admission and manufactured operator checker onto strict-inner pure8192 source226; exact production equations/acceptance unchanged, native combined trajectory unqualified.'}
cfg=json.loads((donor/'configure-support-source.json').read_text())
for n,h in cfg['files'].items():
 assert sha(donor/n)==h,n
 p=s/n
 if p.exists():assert sha(p)==h,n
 else:p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,p)
shutil.copy2(donor/'configure-support-source.json',s/'configure-support-source.json')
f['source228_composition']['configure_support_manifest_sha256']=sha(s/'configure-support-source.json')
f['source228_composition']['configure_support_files']=len(cfg['files'])
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
patch=''.join(''.join(difflib.unified_diff((base/n).read_text().splitlines(True) if (base/n).exists() else [],(s/n).read_text().splitlines(True),fromfile='a/'+n if (base/n).exists() else '/dev/null',tofile='b/'+n)) for n in sorted(delta))
(d/'source.patch').write_text(patch)
targets=['numi-matter-cardboard-probe','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-coupled-free-box-operator-check','numi-matter-cardboard-free-box-proxy-admission-check','numi-matter-cardboard-full-contact-armijo-check','numi-matter-fgmres-true-residual-native-check','numi-matter-fem-cgls-arena-8192-check']
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for t in targets:argv+=['--target',t]
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
(d/'command.json').write_text(json.dumps({'argv':argv,'source_frozen_sha256':sha(s/'frozen-source.json'),'files':records,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:r=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=s)
tests=[]
if r.returncode==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 cmds += [[str(b/'matter'/n)] for n in ['numi-matter-cardboard-free-box-proxy-admission-check','numi-matter-cardboard-coupon-resume-check']]
 for i,a in enumerate(cmds):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:rc=subprocess.run(a,stdout=out,stderr=err,cwd=s,env=env).returncode
  tests.append({'argv':a,'exit_code':rc,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
report={'schema':'numi.cardboard.source228-composition-build.v1','build_exit_code':r.returncode,'tests':tests,'all_pass':r.returncode==0 and all(t['exit_code']==0 for t in tests),'source_snapshot':str(s),'source_count':len(keys),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'source_patch_sha256':sha(d/'source.patch'),'files':records,'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'gpu_execution':False,'physical_transactions':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
