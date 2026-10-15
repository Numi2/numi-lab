from pathlib import Path
import hashlib, json, shutil, subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-trial-summary-face-guard-119';N=B/'snapshots/source-padded-trial-summary-120'
O=B/'analysis/source120-integration-001';A=B/'analysis/armijo-all-trial-summary-overlay-002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
a=json.loads((A/'manifest.json').read_text())
assert a['schema']=='numi.matter.armijo_all_trial_summary_overlay.v2'
for row in a['overlay_files']:assert sha(A/row['path'])==row['sha256'],row['path']
assert a['base_files'][0]['sha256']==sha(S/'matter/src/runtime.mm')
assert not N.exists() and not O.exists();O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
shutil.copy2(A/'overlay/matter/src/runtime.mm',N/'matter/src/runtime.mm')
shutil.copy2(R/'matter/tools/cardboard_resolution_study.md',N/'matter/tools/cardboard_resolution_study.md')
selected={p:sha(N/p) for p in old['source_sha256']};extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source119 plus diagnostic scratch-layout/preflight accounting correction from overlay002. Existing physical operators and all acceptance thresholds remain unchanged. Temporal-study documentation corrected to actual stricter tolerance/factor policy; equivalent force accuracy remains unqualified.','source_sha256':selected},indent=2)+'\n')
(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
changes={p:{'source119':old['source_sha256'][p],'source120':h} for p,h in selected.items() if old['source_sha256'][p]!=h}
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),overlay_manifest_sha256=sha(A/'manifest.json'),changes=changes,source_sha256=selected,support_sha256=extra,physical_steps=0),indent=2)+'\n')
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=[str(Path('/Users/home/.pyenv/versions/3.10.12/bin/python3')),str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-120'),'--cmake-source-relative','.','--jobs','4']
for target in ['numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-probe','numi-matter-cardboard-bilinear15-capture-diagnostic-check','numi-matter-cardboard-contact-ccd-native-check']:
 cmd+=['--target',target]
(B/'build-command-120.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build120-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-120/source-build.json')),indent=2)+'\n')
print('build120',code,flush=True);raise SystemExit(code)
