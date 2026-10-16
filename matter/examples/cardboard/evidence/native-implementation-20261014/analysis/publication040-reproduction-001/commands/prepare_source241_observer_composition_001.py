from pathlib import Path
import json,hashlib,shutil,difflib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-true-residual-native-reporting-243-001';s=q/'snapshots/source-strict-refined-contact-observers-241-001';d=q/'analysis/source241-observer-composition-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();s.mkdir();d.mkdir()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text())
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
bind=lambda p:json.loads((p/'frozen-source.json').read_text())['source_sha256']
a=q/'snapshots/source-integrated-configure-support-219-001';b=q/'snapshots/source-first-failure-dispatch-224-001';c=q/'snapshots/source-first-failure-dispatch-233-001';e=q/'snapshots/source-deformable-contact-trial-trace-236-001';z=q/'snapshots/source-deformable-contact-trial-trace-236-002'
eb=json.loads((q/'analysis/deformable-contact-trial-alpha-trace-236-001/source236-overlay.json').read_text())['candidate_source_bindings']
zb=json.loads((q/'analysis/deformable-contact-trial-alpha-trace-236-002/build-source236-001/source-build.json').read_text())['source_sha256']
stages=[('source224',a,b,bind(a),bind(b)),('source233',b,c,bind(b),bind(c)),('source236001',c,e,bind(c),eb),('source236002',e,z,eb,zb)]
results=[];added=set()
for label,parent,child,pm,cm in stages:
 names=sorted(n for n,h in cm.items() if pm.get(n)!=h)
 patch=''
 for n in names:
  assert sha(child/n)==cm[n],n
  before=(parent/n).read_text().splitlines(keepends=True) if n in pm else []
  if n in pm:assert sha(parent/n)==pm[n],n
  after=(child/n).read_text().splitlines(keepends=True)
  patch+=''.join(difflib.unified_diff(before,after,fromfile='a/'+n if n in pm else '/dev/null',tofile='b/'+n))
  if n not in pm:added.add(n)
 pp=d/(label+'.patch');pp.write_text(patch)
 check=subprocess.run(['git','apply','--check',str(pp)],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (d/(label+'.check.stdout')).write_bytes(check.stdout);(d/(label+'.check.stderr')).write_bytes(check.stderr)
 results.append({'stage':label,'changed':names,'patch_sha256':sha(pp),'check_exit_code':check.returncode})
 (d/'stage-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 assert check.returncode==0,check.stderr.decode()
 p=subprocess.run(['git','apply',str(pp)],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 assert p.returncode==0,p.stderr.decode()
for n in list(f['source_sha256'])+list(added):
 if (s/n).is_file():f['source_sha256'][n]=sha(s/n)
for n in cfg['files']:cfg['files'][n]=sha(s/n)
f['source241_composition']={'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'stages':results,'physical_operator_changed':False,'diagnostics_default_off':True,'native_composition_qualification':False}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
(d/'report.json').write_text(json.dumps({'schema':'numi.cardboard.source241-observer-composition.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_count':len(f['source_sha256']),'stages':results,'patches_applied':True,'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print((d/'report.json').read_text(),flush=True)
