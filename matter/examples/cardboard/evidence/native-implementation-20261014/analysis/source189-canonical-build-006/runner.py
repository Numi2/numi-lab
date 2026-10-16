from pathlib import Path
import json,hashlib,datetime,subprocess,os
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');S=Q/'snapshots/source-shared-contact-trial-189-buildfix003';Base=Q/'snapshots/source-scaled-contact-geometry-188-buildfix002';O=Q/'analysis/source189-canonical-build-006';O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());bf=json.loads((Base/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
expected={'matter/src/runtime.mm':'9ed2a0011bbecb138ac1bcb9089a07899a34b5d39b43e1cfce3b2e42565f4e01','matter/src/metal/fem.metalinc':'bbd6abd764f723e6f0b266b80a3c99e100384e68e5ca5ae7cdd29cad36fb6a43','matter/src/metal/contact.metalinc':'deb984d07de73477c294446eaacaae4e8a8bf374d895056018b9d8047d491cec'}
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
for n,h in support['files'].items():assert sha(S/n)==h,n
changed={n:h for n,h in f['source_sha256'].items() if bf['source_sha256'].get(n)!=h}
assert changed==expected
assert len(f['source_sha256'])==814 and len(support['files'])==1034
assert sha(S/'matter/tools/cardboard_full_contact_armijo_check.mm')==bf['source_sha256']['matter/tools/cardboard_full_contact_armijo_check.mm']
failure=Q/'analysis/source189-canonical-build-003'
(Q/'analysis/source189-canonical-build-005/failure.json').write_text(json.dumps({'status':'PREFLIGHT_SCRIPT_ASSERTION_FAILURE','before_build':True,'before_native':True,'reason':'Script mistakenly compared full-contact checker source with geometry-checker source hash; source189003 itself passes all 814 source and 1034 support hashes.'},indent=2)+'\n')
pred={'selected':814,'support':1034,'frozen_sha256':sha(S/'frozen-source.json'),'base_frozen_sha256':sha(Base/'frozen-source.json'),'changed':expected,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stale_candidate_manifest_rejected':str(failure)}
(O/'verified-source.json').write_text(json.dumps(pred,indent=2)+'\n')
cmd=[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(S/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(S),'--build-dir',str(Q/'build-source189-003'),'--jobs','4']
for n in ['numi-matter-cardboard-probe','numi-matter-cardboard-shipping-memory-check','numi-matter-cardboard-full-contact-scratch-budget-check','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-contact-geometry-check']:cmd+=['--target',n]
(O/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
with (O/'build.log').open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
(O/'runner.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'exit':r.returncode,'source_verified':pred,'provenance':str(Q/'build-source189-003/source-build.json')}),flush=True)
