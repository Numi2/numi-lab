from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-valid-summary-tet858-121';N=B/'snapshots/source-native-tet858-vtccd-traces-122';O=B/'analysis/source122-integration-001'
A=B/'analysis/private-trial-vtccd-call-observer-overlay-001';P=B/'analysis/coupon-392-status5-audit-002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
assert sha(A/'manifest.json')=='8ef93526fdd19f6d98bc278a76c821e3df79702401ede6c0d7726d49363fafe6'
a=json.loads((A/'manifest.json').read_text())
host=P/'cardboard_bilinear15_capture_diagnostic_check.mm';assert sha(host)=='b04bb295db1eba1c10c1001db3b2e90564099e2afb453fd630579bc5a6c76e0d'
assert not N.exists() and not O.exists();O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for row in a['changed_files']:
 p=row['path'];assert sha(S/p)==row['base_sha256'];source=A/'overlay'/p;assert sha(source)==row['candidate_sha256'];shutil.copy2(source,N/p)
shutil.copy2(host,N/'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
selected={p:sha(N/p) for p in old['source_sha256']};extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source121 plus exact point401-bound tet858 native local trace host and opt-in inline private VTCCD call observer. No physical operator or acceptance change.','source_sha256':selected},indent=2)+'\n')
(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),observer_manifest_sha256=sha(A/'manifest.json'),trace_host_sha256=sha(host),changes={p:dict(source121=old['source_sha256'][p],source122=h) for p,h in selected.items() if old['source_sha256'][p]!=h},physical_steps=0),indent=2)+'\n')
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-122'),'--cmake-source-relative','.','--jobs','4']
for target in ['numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-bilinear15-capture-diagnostic-check','numi-matter-cardboard-source392-tet858-point-check']:cmd+=['--target',target]
(B/'build-command-122.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build122-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-122/source-build.json')),indent=2)+'\n')
print('build122',code,flush=True);raise SystemExit(code)
