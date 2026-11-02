from pathlib import Path
import subprocess,json,time,os,hashlib
R=Path('/Users/numi/Work/skin-fat-native-integration-209');S=Path(__file__).resolve().parent
rows=[]
def run(name,args,env=None):
 start=time.monotonic()
 with (S/(name+'.stdout.log')).open('w') as o,(S/(name+'.stderr.log')).open('w') as e:
  p=subprocess.run(list(map(str,args)),cwd=R,stdout=o,stderr=e,env=env,timeout=300)
 row={'name':name,'argv':list(map(str,args)),'exit_code':p.returncode,'elapsed_s':time.monotonic()-start};rows.append(row);(S/'progress-controls.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);assert p.returncode==0,(S/(name+'.stderr.log')).read_text()[-4000:]
env=os.environ.copy();env['PATH']=str(S/'build-tools')+':'+env['PATH']
run('progress-production-build',['/opt/homebrew/bin/cmake','--build',R/'build-native','--target','numi-human-native','--parallel','1'],env)
run('post-progress-reduction-control',[S/'combined-reduction-test',R/'build-native/matter/shaders/NumiMatter.metallib'])
env=os.environ.copy();env['NUMI_HUMAN_PARALLEL_CONTACT_AUDIT']='1';env['NUMI_HUMAN_PARALLEL_CONTACT_WRENCH']='1';env['NUMI_GPU_PROGRESS']='1';env['NUMI_GPU_PROGRESS_ROOT']='0'
lib=R/'build-native/matter/shaders/NumiMatter.metallib'
run('native-progress-inactive',[S/'native-test',R/'matter/materials/silicone.nmatter',lib],env)
run('native-progress-active',[S/'native-test',R/'matter/materials/silicone.nmatter',lib,'active-contact'],env)
assert 'gpu_progress=' in (S/'native-progress-active.stderr.log').read_text()
print('all controls PASS; optional shared-event instrument executed',flush=True)
