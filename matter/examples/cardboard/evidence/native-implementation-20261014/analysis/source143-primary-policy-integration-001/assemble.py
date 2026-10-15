from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-bounded-native-cgls-mask-repair-142';N=B/'snapshots/source-cgls-primary-policy-143';O=B/'analysis/source143-primary-policy-integration-001'
A=B/'analysis/base-cgls-primary-policy-overlay-143-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (A/'manifest.json').is_file(),'sealed overlay required'
review=B/'analysis/source143-primary-policy-independent-review-001/report.json'
assert review.is_file(),'independent policy review required'
review_data=json.loads(review.read_text())
assert review_data.get('blocking_findings')==[],review_data
changes=['matter/src/runtime.mm','matter/src/metal/base_cgls_runtime.metalinc']
frozen=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,d in {**frozen['source_sha256'],**support['files']}.items():assert sha(S/p)==d,p
for p in changes:assert (A/'overlay'/p).is_file() and sha(A/'overlay'/p)!=sha(S/p),p
assert not N.exists() and not O.exists()
v=os.statvfs(B);available=v.f_bavail*v.f_frsize
assert available>2*1024**3,available
O.mkdir()
(O/'disk-preflight.json').write_text(json.dumps({'available_bytes':available,'minimum_bytes':2*1024**3},indent=2)+'\n')
c=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True)
(O/'clone.log').write_text(c.stdout+c.stderr);assert c.returncode==0
for p in changes:shutil.copy2(A/'overlay'/p,N/p);frozen['source_sha256'][p]=sha(N/p)
frozen['base_snapshot']=str(S)
frozen['scope']='Source142 plus default-off explicitly selected CGLS-primary residual-minimizing direction policy after a bounded Newton index on one existing all-Newton selected control. Private eligibility requires healthy base, original linear certificate and residual above no-op bound; CGLS installer clears that original certificate. Ordinary strict nonlinear Armijo, exact tested update fraction, final residual/geometry certificates and transactional rollback are unchanged. No material, mass, physical operator, loading, timestep or tolerance changes. New policy awaits native qualification.'
(N/'frozen-source.json').write_text(json.dumps(frozen,indent=2)+'\n')
for p,d in {**frozen['source_sha256'],**support['files']}.items():assert sha(N/p)==d,p
record={'schema':'numi.cardboard.source143_policy_integration.v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(S),'snapshot':str(N),'changed_files':{p:sha(N/p) for p in changes},'source_count':len(frozen['source_sha256']),'overlay_manifest_sha256':sha(A/'manifest.json'),'independent_review_sha256':sha(review),'physical_steps':0,'status':'AWAITING_NATIVE_QUALIFICATION'}
(O/'integration.json').write_text(json.dumps(record,indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
command=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-143'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe']
(B/'build-command-143.json').write_text(json.dumps(command,indent=2)+'\n')
with (B/'build143-runner.log').open('x') as out:code=subprocess.call(command,stdout=out,stderr=subprocess.STDOUT)
p=R/'build-cardboard-provenance-143/source-build.json'
(O/'post-build.json').write_text(json.dumps({'exit_code':code,'source_build_sha256':sha(p) if p.exists() else None},indent=2)+'\n')
print('build143',code,flush=True)
raise SystemExit(code)
