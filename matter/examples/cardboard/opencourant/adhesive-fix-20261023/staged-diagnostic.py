#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,hashlib,subprocess,time
R=Path('/Users/n/cardboard-opencourant-20261023/adhesive-fix')
T=R.parent
stage=int(sys.argv[1]); parent=R/'runs/staged-bond-birth-03'
root=R/f'runs/valgrind-staged-birth-03-stage{stage}'
root.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
engine=R/'candidate03/runtime/OpenCourant/exec/engine_type43'
deck=next(parent.glob(f'*_{stage:04d}.rad'))
files=[deck]+list(parent.glob(f'*_{stage-1:04d}_*.rst'))
inputs={}
for p in files:
 q=root/p.name;q.write_bytes(p.read_bytes());inputs[q.name]=sha(q)
env=os.environ.copy();env.update(RAD_CFG_PATH=str(T/'runtime/OpenCourant/hm_cfg_files'),LD_LIBRARY_PATH=str(T/'runtime/OpenCourant/extlib/hm_reader/linuxa64')+':'+str(T/'runtime/OpenCourant/extlib/h3d/lib/linuxa64'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
argv=['/usr/bin/time','-v','timeout','--kill-after=10','120','valgrind','--track-origins=yes','--leak-check=no','--error-limit=no','--log-file=valgrind.log',str(engine),'-i',deck.name,'-nt','1']
record={'schema':'cardboard.diagnostic.replay.v1','parent':str(parent),'parent_receipt_sha256':sha(parent/'run.json'),'inputs_sha256':inputs,'engine_sha256':sha(engine),'argv':argv,'diagnostic_only':True}
(root/'replay.json').write_text(json.dumps(record,indent=2))
start=time.monotonic()
with (root/'stdout').open('w') as out,(root/'stderr').open('w') as err: p=subprocess.run(argv,cwd=root,env=env,stdout=out,stderr=err)
record.update(exit_code=p.returncode,elapsed_s=time.monotonic()-start,outputs_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='replay.json'})
(root/'replay.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'stage':stage,'exit_code':p.returncode,'elapsed_s':record['elapsed_s']}))

