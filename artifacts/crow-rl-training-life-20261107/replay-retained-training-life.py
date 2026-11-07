from pathlib import Path
import hashlib,json,os,struct,sys
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
sys.path.insert(0,str(root/'python'))
from metalrobo.crow_environment import execute,write,sha
import mlx.core as mx
import numpy as np
for version in ('04','05'):
 trial=root/f'.numi/runs/american-crow/20261107-life-training-probe-{version}'
 runtime=root/f'.numi/retained-runtime/life-training-probe-{version}'
 output=root/f'.numi/runs/american-crow/20261107-training-life-retained-replay-{version}';output.mkdir(exist_ok=False)
 receipt=json.loads((trial/'resume-chunk8.receipt.json').read_text())
 command=receipt['command'];command[0]=str(runtime/'bin/metalrobo_task_train')
 for option,path in [('--metallib',runtime/'shaders/MetalRobo.metallib'),('--native-library',runtime/'lib/libmetalrobo.dylib'),('--python-root',runtime/'python')]:command[command.index(option)+1]=str(path)
 for option,name in [('--updated-policy-pack','updated.policypack'),('--deployment-policy-pack','candidate.policypack'),('--incumbent-policy-pack','incumbent.policypack'),('--rollout-pack','rollout.rolloutpack'),('--output-learner-state','learner.safetensors'),('--checkpoint-directory','checkpoints')]:command[command.index(option)+1]=str(output/name)
 rows=json.loads((runtime/'receipt.json').read_text())
 for row in rows:assert sha(Path(row['path']))==row['sha256']
 write(output/'plan.json',{'schema':'numi.training-life-retained-runtime-replay.v1','version':version,'command':command,'runtime':rows,'source_life':str(trial/'split/checkpoints/revision-00000003.trainingcheckpoint'),'prediction':'all future rollout/policy/optimizer/native components equal original reference at both next boundaries','scope':'retained exact runtime and Python source; same installed MLX/NumPy/Python dependencies; no qualification seeds'})
 os.environ['DYLD_LIBRARY_PATH']=str(runtime/'lib')
 value=execute(output,'replay',command,root,180)
 assert value['life_state_restored'] and value['learner_state_restored'] and value['final_policy_revision']==5 and value['failed_environment_steps']==0
 def components(path):
  data=path.read_bytes();assert data[:8]==b'NUMILF01';offset=104;result=[]
  for _ in range(3):
   count,=struct.unpack_from('<Q',data,offset);offset+=8;result.append(data[offset:offset+count]);offset+=count
  assert offset==len(data);return result
 for revision in (4,5):
  a=trial/'reference/checkpoints'/f'revision-{revision:08}.trainingcheckpoint'
  b=output/'checkpoints'/f'revision-{revision:08}.trainingcheckpoint'
  for name in ('policy.policypack','candidate.policypack','rollout.rolloutpack'):assert sha(a/name)==sha(b/name)
  aa,ma=mx.load(a/'learner.safetensors',return_metadata=True);ab,mb=mx.load(b/'learner.safetensors',return_metadata=True)
  assert ma==mb and set(aa)==set(ab)
  for key in aa:
   x,y=np.asarray(aa[key]),np.asarray(ab[key]);assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
  assert components(a/'life.native')==components(b/'life.native')
 source=trial/'split/checkpoints/revision-00000003.trainingcheckpoint'
 manifest=json.loads((source/'manifest.json').read_text())
 for name,row in manifest['files'].items():assert sha(source/name)==row['sha256']
 write(output/'summary.json',{'status':'passed','version':version,'both_future_boundaries_bit_identical':True,'original_checkpoint_unchanged':True,'dependency_scope':'matching retained binaries and package sources; installed interpreter and MLX/NumPy distributions identity-checked by worker'})
 print(json.dumps({'version':version,'status':'passed'}),flush=True)
