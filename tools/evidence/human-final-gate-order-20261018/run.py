from pathlib import Path
import os,subprocess,time,json,hashlib
c=Path('/Users/n/numi-human-separation-20261018/C');p=c/'current-operator-integration-207';d=c/'body-final-gate-order-215'; ev=d/'evidence';ev.mkdir(exist_ok=True)
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=os.environ.copy();pins={'METALROBO_TEST_ARTICULATED_SOURCE':p/'shaders/operator-modules.tsv','NUMI_HUMAN_STAND_SOURCE':p/'shaders/StandSource.metal'}
for k,v in pins.items():e[k]=str(v);e[k+'_SHA256']=h(v)
argv=['/Users/n/numi-human-prep-venv-20261005/bin/python3','/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py','--',str(d/'build/final-gate-probe')]
for name,lib,expected in [('old-order',c/'blood-transfer-208/build/libmetalrobo.dylib',1),('new-order',d/'build/libmetalrobo.dylib',0)]:
 e['DYLD_LIBRARY_PATH']=str(lib.parent);record={'argv':argv,'environment':{k:e[k] for k in list(pins)+[k+'_SHA256' for k in pins]+['DYLD_LIBRARY_PATH']},'library_sha256':h(lib),'binary_sha256':h(d/'build/final-gate-probe'),'expected_exit':expected}
 t=time.monotonic();r=subprocess.run(argv,env=e,capture_output=True,text=True);record.update(exit=r.returncode,wall_seconds=time.monotonic()-t,library_sha256_after=h(lib));(ev/(name+'.log')).write_text(r.stdout+r.stderr);(ev/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(name,r.returncode,r.stdout+r.stderr,flush=True)
 if r.returncode!=expected:raise SystemExit('unexpected result '+name)
