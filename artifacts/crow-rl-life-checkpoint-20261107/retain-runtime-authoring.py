from pathlib import Path
import hashlib,json,shutil
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
plan=json.loads((root/'.numi/runs/american-crow/20261107-life-admission-06/plan.json').read_text())
out=root/'.numi/retained-runtime/life-admission-06-final';out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
rows=[]
for key,relative in [('life_check','bin/metalrobo_crow_life_checkpoint_check'),('retinal_api_check','bin/metalrobo_crow_retinal_api_check'),('train','bin/metalrobo_task_train'),('rollout','bin/metalrobo_task_rollout'),('library','lib/libmetalrobo.dylib'),('metallib','shaders/MetalRobo.metallib'),('retinal_metallib','shaders/RetinalStream.metallib')]:
 item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
 p=out/relative;p.parent.mkdir(exist_ok=True);shutil.copy2(source,p);assert sha(p)==item['sha256']
 rows.append({'kind':key,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
(out/'receipt.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'runtime':str(out),'files':len(rows)}))
