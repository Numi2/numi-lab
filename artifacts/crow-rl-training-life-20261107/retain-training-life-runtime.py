from pathlib import Path
import hashlib,json,shutil,sys
root=Path('/Users/n/Developer/numi-crow-rl-20261107');version=sys.argv[1]
plan=json.loads((root/f'.numi/runs/american-crow/20261107-life-training-probe-{version}/plan.json').read_text())
out=root/f'.numi/retained-runtime/life-training-probe-{version}';out.mkdir(parents=True,exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rows=[]
for key,relative in [('train','bin/metalrobo_task_train'),('rollout','bin/metalrobo_task_rollout'),('library','lib/libmetalrobo.dylib'),('metallib','shaders/MetalRobo.metallib'),('retinal_metallib','shaders/RetinalStream.metallib')]:
 item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
 p=out/relative;p.parent.mkdir(exist_ok=True);shutil.copy2(source,p);assert sha(p)==item['sha256']
 rows.append({'kind':key,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
for source in sorted((root/'python/metalrobo').rglob('*.py')):
 p=out/'python/metalrobo'/source.relative_to(root/'python/metalrobo');p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,p);assert sha(p)==sha(source)
 rows.append({'kind':'python-source','path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
(out/'receipt.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'runtime':str(out),'files':len(rows)}))
