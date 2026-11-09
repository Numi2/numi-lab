from pathlib import Path
import hashlib,json,subprocess,tarfile
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109'
files=sorted(B.glob('*.air'))+sorted((R/'build/shaders').glob('MetalRobo*.air'))
files += [B/(x+'.metallib') for x in ['symmetric-launch-wings-off-control','symmetric-launch-wings-off-push','symmetric-launch-wings-on-control','symmetric-launch-wings-on-push']]
files += [B/'materialized/libmetalrobo.dylib',B/'complete-observer/libmetalrobo.dylib',B/'toe-envelope/libmetalrobo.dylib',R/'build/bin/metalrobo_task_rollout']
files += [Path('/tmp')/n for n in ['crow-v15-grip-screen-20261109','crow-v15-coordinated-screen-20261109','crow-v15-resident-trace-20261109','crow-v15-feedback-probe-20261109','crow-v15-perch-offset-20261109','crow-v15-perch-control-20261109','crow-v15-perch-identity-20261109','crow-v15-perch-perturb-20261109']]
files=[p for p in files if p.exists()];index=[]
with tarfile.open(A/'native-runtime.tar.gz','w:gz',compresslevel=9) as t:
 for p in files:
  name=str(p.relative_to(R)) if p.is_relative_to(R) else 'tmp/'+p.name; t.add(p,arcname=name,recursive=False);index.append({'path':name,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(A/'native-runtime-index.json').write_text(json.dumps({'files':index,'tar_sha256':hashlib.sha256((A/'native-runtime.tar.gz').read_bytes()).hexdigest(),'base_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'flight_source_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest()},indent=2)+'\n')
print('archived',len(index),'files',round((A/'native-runtime.tar.gz').stat().st_size/1e6,1),'MB')
