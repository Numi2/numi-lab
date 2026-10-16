from pathlib import Path
import hashlib,json,importlib.util,sys,os,subprocess,datetime,time,shutil
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');S=B/'snapshots/source-scaled-contact-geometry-188-buildfix002';D=B/'build-source188-002';P=B/'analysis/native510-contact-free-coupon-regression-001';P.mkdir();(P/'runner.py').write_bytes(Path(__file__).read_bytes());Old=B/'analysis/native497498-diagnostic-replay-command-002';old=json.loads((Old/'command-spec.json').read_text());C=B/'solver/warm-cgls-source188-control20-regression-510';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());build=json.loads((D/'source-build.json').read_text());assert build['success'] and f['source_sha256']==build['source_sha256']
for n,h in f['source_sha256'].items():assert sha(S/n)==h
for n,h in build['artifacts_sha256'].items():assert sha(Path(n))==h
spec=importlib.util.spec_from_file_location('prior',Old/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);prefix=mod.prefix()
for row in old['material_path_bindings']:assert sha(Path(row['original']))==row['sha256']
assert not [x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if 'numi-matter-' in x and not x.split()[1].startswith('Z')]
native=[x.replace(str(B/'build-source182-003'),str(D)).replace(old['cases']['498']['case'],str(C)) for x in old['cases']['498']['native_argv']]
command=[sys.executable,'-B',str(S/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(S),'--runtime-dir',str(D),'--build-provenance',str(D/'source-build.json'),'--case',str(C),'--timeout','900','--question','Do the source188 contact geometry and accounting repairs preserve the exact failing contact-free coupon control20 history?','--prediction','Contact-free production histories should match source182 native498 exactly; failure remains a failed nonlinear solve, not convergence qualification.','--diagnostic-run']
for k,v in sorted(old['environment'].items()):command+=['--env',k+'='+v]
shutil.copy2(B/'analysis/source188-native-validation-001/cpu-checks/receipt.json',P/'cpu-validation-receipt.json')
inputs=[Old/'command-spec.json',S/'frozen-source.json',P/'cpu-validation-receipt.json']
CP=B/'solver/warm-cgls-control19-checkpoint-480'
inputs+=[CP/n for n in ['receipt.json','output/manifest.json','output/observations.csv','output/checkpoints/coupon-control-000019.nmcb','output/checkpoints/coupon-control-000019.resume','output/checkpoints/coupon-control-000019.prefix']]
inputs+=[Path(row['original']) for row in old['material_path_bindings']]
for p in inputs:command+=['--input-file',str(p)]
command+=['--']+native
prediction={'argv':command,'physical_environment':old['environment'],'prefix_verified':prefix,'inputs_sha256':{str(p):sha(p) for p in inputs},'source_build_sha256':sha(D/'source-build.json'),'frozen_sha256':sha(S/'frozen-source.json'),'one_gpu_worker':True,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'prediction.json').write_text(json.dumps(prediction,indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_CARDBOARD_','NUMI_MATTER_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';start=time.monotonic()
with (P/'wrapper.log').open('xb') as log:
 try:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=960);code=r.returncode;timed=False
 except subprocess.TimeoutExpired:code=None;timed=True
receipt={'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-start,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_sha256':sha(P/'wrapper.log'),'all_six_qualification_axes_complete':False}
(P/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
