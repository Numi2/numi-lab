#!/usr/bin/env python3
import hashlib, json, os, pathlib, subprocess, sys, time
T=pathlib.Path('/Users/n/cardboard-opencourant-20261023')
case=sys.argv[1]
label=sys.argv[2]
parent=T/'runs'/case
root=T/'adhesive-fix'/'runs'/label
root.mkdir(parents=True,exist_ok=False)
files=list(parent.glob('*_0001.rad'))+list(parent.glob('*_0000_0001.rst'))
assert len(files)==2, files
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
for p in files:
    q=root/p.name;q.write_bytes(p.read_bytes());inputs[q.name]=sha(q)
engine=T/'runtime/OpenCourant/exec/engine_linuxa64_gf_ompi'
if len(sys.argv)>3: engine=pathlib.Path(sys.argv[3])
env=os.environ.copy()
env.update(RAD_CFG_PATH=str(T/'runtime/OpenCourant/hm_cfg_files'),LD_LIBRARY_PATH=str(T/'runtime/OpenCourant/extlib/hm_reader/linuxa64')+':'+str(T/'runtime/OpenCourant/extlib/h3d/lib/linuxa64'),OMP_NUM_THREADS='1')
command=['/usr/bin/time','-v','timeout','--signal=TERM','--kill-after=10','120','valgrind','--tool=memcheck','--track-origins=yes','--error-limit=no','--leak-check=no','--num-callers=24','--log-file=valgrind.log',str(engine),'-i',next(root.glob('*.rad')).name,'-nt','1']
receipt={'schema':'cardboard.diagnostic.replay.v1','parent':str(parent),'inputs_sha256':inputs,'engine':str(engine),'engine_sha256':sha(engine),'argv':command,'diagnostic_only':True,'env':{k:env[k] for k in ['RAD_CFG_PATH','LD_LIBRARY_PATH','OMP_NUM_THREADS']}}
(root/'replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
t0=time.monotonic()
with (root/'stdout').open('w') as out,(root/'stderr').open('w') as err:
    p=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=err)
receipt.update(exit_code=p.returncode,elapsed_s=time.monotonic()-t0,outputs_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='replay.json'})
(root/'replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'root':str(root),'exit_code':p.returncode,'elapsed_s':receipt['elapsed_s']}))
