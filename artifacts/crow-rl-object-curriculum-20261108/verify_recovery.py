"""Exercise fixed-policy and coupled training continuation from actual Git blobs."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import sys
import subprocess
import time

root=Path.cwd()
base=root/'artifacts/crow-rl-object-curriculum-20261108'
tmp=Path('/private/tmp/crow-object-curriculum-git-recovery-20261108')
metadata=json.loads((base/'runtime-recovery.json').read_text())
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
def write(name,value):
    with (base/name).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')
plan={
    'budget':'One186-tick two-lane fixed-policy future and two97-tick two-lane coupled training updates; timeout180s each. No retries or tuning.',
    'prediction':'Git-restored prefix life/exact policy and coupled optimizer checkpoint reproduce original future rollout bytes, all native physical/GRU/retinal components, all model/Adam arrays/metadata and scientific learner metrics.',
    'runtime':'All executables, native library, three shaders, canonical sensor assets and Python sources read from committed Git objects; no raw evidence/runtime input reused.',
    'measurement_scope':'Record elapsed time and allocator measurements; exclude those process-dependent quantities from causal byte/tensor/scientific equality.',
    'scope':'Same device/OS and external Python/MLX environment; scripted engineering continuation, not learned manipulation or general ecological qualification.',
}
parser=argparse.ArgumentParser()
parser.add_argument('--verify-existing',action='store_true')
parser.add_argument('--analysis-name',default='verification-reanalysis')
args=parser.parse_args()
reuse=args.verify_existing
analysis_name=args.analysis_name
if not analysis_name.replace('-', '').isalnum():raise ValueError('Invalid analysis name')
plan['execution_mode']='Verify command/hash-validated original successful executions after correcting optional allocator-history fields' if reuse else 'Fresh execution'
write(analysis_name+'-plan.json' if reuse else 'verification-plan.json',plan)
if reuse:
    restoration=json.loads((tmp/'runtime-restoration.json').read_text())
    for record in restoration['files']:
        if a.file_hash(tmp/record['path'])!=record['sha256']:
            raise RuntimeError('Restored Git input changed before reanalysis')
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')}
env.update(DYLD_PRINT_LIBRARIES='1',METALROBO_LIBRARY=str(tmp/'lib/libmetalrobo.dylib'),
           METALROBO_METALLIB=str(tmp/'shaders/MetalRobo.metallib'),PYTHONPATH=str(tmp/'python'))
def execute(name,command):
    if reuse:
        if json.loads((base/(name+'.command.json')).read_text())!=command:
            raise RuntimeError('Retained recovery command differs: '+name)
        receipt=json.loads((base/(name+'.receipt.json')).read_text())
        stdout=(base/(name+'.stdout')).read_text()
        stderr=(base/(name+'.stderr')).read_text()
        if (receipt['returncode']!=0 or receipt['stdout_sha256']!=hashlib.sha256(stdout.encode()).hexdigest() or
            receipt['stderr_sha256']!=hashlib.sha256(stderr.encode()).hexdigest()):
            raise RuntimeError('Retained recovery execution receipt differs: '+name)
    else:
        write(name+'.command.json',command)
        started=time.monotonic()
        p=subprocess.run(command,cwd=tmp,env=env,capture_output=True,text=True,timeout=180)
        stdout,stderr=p.stdout,p.stderr
        (base/(name+'.stdout')).write_text(stdout)
        (base/(name+'.stderr')).write_text(stderr)
        write(name+'.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-started,
            'stdout_sha256':hashlib.sha256(stdout.encode()).hexdigest(),
            'stderr_sha256':hashlib.sha256(stderr.encode()).hexdigest()})
        if p.returncode:raise RuntimeError('Git-restored execution failed: '+name)
    loaded=[s for s in stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    if not loaded or any(not s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded):
        raise RuntimeError('Wrong recovery library: '+name)
    v=json.loads(stdout)
    if v['failed_environment_steps']!=0:raise RuntimeError('Native recovery rejected a transition')
    return v
def replace(command,flag,value):
    command[command.index(flag)+1]=str(value)
def parts(path):
    data=path.read_bytes()
    tag,abi,reserved,identity,size,digest=struct.unpack_from('<8sIIQQ32s',data)
    if tag!=b'NUMILF01' or abi!=1 or reserved or size!=len(data)-64 or hashlib.sha256(data[64:]).digest()!=digest:
        raise RuntimeError('Invalid native life envelope')
    offset=104
    values=[]
    for _ in range(3):
        length,=struct.unpack_from('<Q',data,offset)
        offset+=8
        values.append(data[offset:offset+length])
        offset+=length
    if offset!=len(data) or values[2]:raise RuntimeError('Wrong object-curriculum component contract')
    return values
try:
    c=list(metadata['fixed_reference']['command'])
    c[2]=str(tmp/'bin/metalrobo_task_rollout')
    for flag,value in {
        '--policy-pack':tmp/'input/push.policypack',
        '--metallib':tmp/'shaders/MetalRobo.metallib',
        '--visual-observation-config':tmp/'environments/american-crow/object-push-v1/retinal-observation.json',
        '--object-push-curriculum':tmp/'environments/american-crow/object-push-v1/curriculum.json',
        '--resume-life-checkpoint':tmp/'input/prefix.life',
        '--save-life-checkpoint':tmp/'fixed-future.life',
        '--rollout-pack':tmp/'fixed-future.rolloutpack',
    }.items():replace(c,flag,value)
    fixed=execute('fixed-policy-recovery',c)
    if fixed['accepted_lifetime_control_ticks_by_environment']!=[345,345] or not fixed['life_checkpoint_restored']:
        raise RuntimeError('Recovered fixed-policy clock/restore differs')
    if a.file_hash(tmp/'fixed-future.rolloutpack')!=metadata['fixed_reference']['rollout_sha256']:
        raise RuntimeError('Recovered fixed-policy rollout differs')
    if parts(tmp/'fixed-future.life')!=parts(tmp/'reference/future.life'):
        raise RuntimeError('Recovered fixed-policy native hidden state differs')
    target=tmp/'training'
    target.mkdir(exist_ok=reuse)
    c=list(metadata['training_reference']['command'])
    c[2]=str(tmp/'bin/metalrobo_task_train')
    for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
                 '--rollout-pack','--output-learner-state','--checkpoint-directory'):
        replace(c,flag,target/Path(c[c.index(flag)+1]).name)
    for flag,value in {
        '--resume-training-checkpoint':tmp/'input/source.trainingcheckpoint',
        '--python-root':tmp/'python','--native-library':tmp/'lib/libmetalrobo.dylib',
        '--metallib':tmp/'shaders/MetalRobo.metallib',
        '--visual-observation-config':tmp/'environments/american-crow/object-push-v1/retinal-observation.json',
        '--object-push-curriculum':tmp/'environments/american-crow/object-push-v1/curriculum.json',
    }.items():replace(c,flag,value)
    learning=execute('training-recovery',c)
    if learning['training_samples']!=388 or learning['accepted_lifetime_control_ticks_by_environment']!=[291,291] or not learning['life_state_restored']:
        raise RuntimeError('Recovered coupled training clock/sample/restore differs')
    measurements=[]
    ignored={'update_seconds','mlx_peak_memory_bytes','mlx_active_memory_bytes','mlx_cache_memory_bytes'}
    for got,want in zip(learning['learning_update_history'],metadata['training_reference']['summary']['learning_update_history'],strict=True):
        got=got.copy();want=want.copy()
        present=ignored & got.keys()
        if present!=ignored & want.keys() or not {'update_seconds','mlx_peak_memory_bytes'}<=present:
            raise RuntimeError('Process measurement schema differs')
        measurements.append({'original':{k:want[k] for k in present},'recovered':{k:got[k] for k in present}})
        for k in present:
            if not math.isfinite(got[k]) or got[k]<0:raise RuntimeError('Invalid recovered process measurement')
            got.pop(k);want.pop(k)
        if got!=want:raise RuntimeError('Recovered scientific learner metrics differ')
    import mlx.core as mx
    import numpy as np
    exact=[]
    tensors=0
    native=0
    for name,record in metadata['training_reference']['files'].items():
        p=target/name
        q=tmp/'reference/training'/name
        if p.name=='learner.safetensors':
            x,xm=mx.load(p,return_metadata=True);y,ym=mx.load(q,return_metadata=True)
            if xm!=ym or set(x)!=set(y):raise RuntimeError('Recovered model/Adam metadata differs')
            for key in x:
                if np.asarray(x[key]).tobytes()!=np.asarray(y[key]).tobytes():
                    raise RuntimeError('Recovered model/Adam tensor differs: '+key)
                tensors+=1
        elif p.name=='life.native':
            if parts(p)!=parts(q):raise RuntimeError('Recovered coupled native hidden state differs')
            native+=1
        elif p.suffix in ('.policypack','.rolloutpack'):
            if a.file_hash(p)!=record['sha256']:raise RuntimeError('Recovered canonical bytes differ: '+name)
            exact.append(name)
    result={'fixed_policy_future_rollout_byte_exact':True,'fixed_policy_world_retina_GRU_exact':True,
        'training_policy_rollout_files_byte_exact':exact,'model_Adam_arrays_exact':tensors,
        'model_Adam_metadata_exact':True,'coupled_native_boundaries_exact':native,
        'scientific_learner_metrics_exact':True,'process_measurements':measurements,
        'fixed_accepted_lifetime_ticks':fixed['accepted_lifetime_control_ticks_by_environment'],
        'training_accepted_lifetime_ticks':learning['accepted_lifetime_control_ticks_by_environment']}
    files=[]
    directories=[]
    for directory,subdirs,names in os.walk(tmp,followlinks=False):
        for name in subdirs:
            p=Path(directory)/name
            if p.is_symlink():raise RuntimeError('Unexpected recovery directory symlink')
            directories.append(p)
        for name in names:
            p=Path(directory)/name;s=p.lstat()
            if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
            files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),
                          'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
    handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
    if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovery handles')
    for f in files:
        p=Path(f['path']);s=p.lstat()
        if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or stat.S_IMODE(s.st_mode)!=f['mode'] or a.file_hash(p)!=f['sha256']:
            raise RuntimeError('Recovery file changed before cleanup')
    report={'status':'passed','verification_reuses_completed_executions':reuse,'archive_commit':a.git('rev-parse','HEAD').decode().strip(),
        'results':result,'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),
        'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,
            'removed_logical_bytes':sum(f['bytes'] for f in files)}}
    write('verification-before-cleanup.json',report)
    for f in files:Path(f['path']).unlink()
    for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):p.rmdir()
    tmp.rmdir()
    report['temporary_cleanup']['status']='removed'
    write('verification.json',report)
    print(json.dumps({'status':'passed','model_Adam_arrays_exact':tensors,
        'training_files_byte_exact':len(exact),'temporary_bytes_removed':sum(f['bytes'] for f in files)}))
except Exception as exc:
    write(analysis_name+'-failure.json' if reuse else 'verification-failure.json',{'error':str(exc),'type':type(exc).__name__})
    raise
