"""Short native admission of the source-bound bladder and motion-conditioned taenia repairs."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time

E=Path('/Users/n/numi-human-resting-evidence-20261005')
template=E/'cardiac-map-multiregion-affine-repair-011/native-fullchain073-6s-001/invocation.json'
out=E/'bladder-taenia-native-114';out.mkdir(exist_ok=False)
original=json.loads(template.read_text());argv=original['argv'].copy();argv[4]=str(out)
new=E/'bladder-cardiac-input-113'
argv[argv.index('--torso-anatomy-payload')+1]=str(new/'resting-thorax.nhanatomy')
argv[argv.index('--resting-anatomy-receipt')+1]=str(new/'resting-anatomy-receipt.json')
argv[argv.index('--resting-movie')+1]=str(out/'native-viewer.mov')
env={**os.environ,**original['environment'],'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS':'0,639,2783,2999'}
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
replaced=['fullchain073-native-input-003/resting-', 'ventricular-wall-map-refinement-fullchain073-002.json']
assets={p:sha(p) for p in original['asset_sha256'] if not any(x in p for x in replaced)}
assert all(v==original['asset_sha256'][p] for p,v in assets.items())
assets.update({str(new/f):sha(new/f) for f in ['resting-thorax.nhanatomy','resting-anatomy-receipt.json','ventricular-wall-map-refinement.json']})
source=Path('/Users/n/numi-human-resting-cardiac-source-026')
source_hashes={p:sha(source/p) for p in original['source_file_sha256']}
assert source_hashes==original['source_file_sha256']
inv={'scope':'Six-second native admission of passive bladder and motion-conditioned taenia repairs. Known respiratory source, lung-rib, and cardiac neighbor defects remain; not final qualification.',
     'template_invocation':str(template),'template_sha256':sha(template),'driver_sha256':sha(__file__),
     'source_root':str(source),'source_revision':original['source_revision'],'source_diff_sha256':original['source_diff_sha256'],
     'source_file_sha256':source_hashes,'binary_sha256':sha(argv[0]),'asset_sha256':assets,
     'argv':argv,'environment':{k:v for k,v in env.items() if k.startswith('NUMI_')},'device':original['device'],
     'cwd':original['cwd'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
path=out/'invocation.json';path.write_text(json.dumps(inv,indent=2)+'\n')
start=time.monotonic()
with (out/'native.log').open('w') as log:
    result=subprocess.run(argv,cwd=original['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
elapsed=time.monotonic()-start
inv.update(return_code=result.returncode,wall_seconds=elapsed,real_time_factor=(6/elapsed if result.returncode==0 else None),
           finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),asset_sha256_after={p:sha(p) for p in assets})
path.write_text(json.dumps(inv,indent=2)+'\n');assert inv['asset_sha256_after']==assets
print(json.dumps({'return_code':result.returncode,'wall_seconds':elapsed,'real_time_factor':inv['real_time_factor']}),flush=True)
raise SystemExit(result.returncode)
