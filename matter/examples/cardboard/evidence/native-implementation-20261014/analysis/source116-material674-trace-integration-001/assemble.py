from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');OLD=B/'snapshots/source-exact-material-point-112';NEW=B/'snapshots/source-material674-trace-116';O=B/'analysis/source116-material674-trace-integration-001';P=B/'analysis/source380-generalized-material-point-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text());s=json.loads((OLD/'configure-support-source.json').read_text());m=json.loads((P/'source-overlay.json').read_text())
assert sha(P/'source-overlay.patch')==m['overlay_patch_sha256']
for p,h in {**a['source_sha256'],**s['files']}.items():assert sha(OLD/p)==h,p
assert not NEW.exists();O.mkdir(exist_ok=False)
# APFS clone preserves independent paths and exact bytes without allocating a
# second physical copy of every unchanged source file on this nearly full disk.
q=subprocess.run(['/bin/cp','-cR',str(OLD),str(NEW)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for p,h in {**a['source_sha256'],**s['files']}.items():assert sha(NEW/p)==h,p
for p,h in m['changed_sources'].items():assert sha(NEW/p)==h['base_sha256'],p
q=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'source-overlay.patch')],cwd=NEW,capture_output=True,text=True);(O/'patch.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for p,h in m['changed_sources'].items():assert sha(NEW/p)==h['overlay_sha256'],p
selected={p:sha(NEW/p) for p in a['source_sha256']}
(NEW/'frozen-source.json').write_text(json.dumps({**a,'base_snapshot':str(OLD),'scope':'Diagnostic-only exact compiled-package medium674 projection with32 trace slots and trace-off comparison. Native382 separately confirms all Fhi/lo/Fdot input words bit-exact production deformation readback.','source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']};assert extra==s['files']
(NEW/'configure-support-source.json').write_text(json.dumps({**s,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(NEW),source_overlay_sha256=sha(P/'source-overlay.json'),changes=m['changed_sources'],all_source_bytes_verified_after_APFS_clone=True,selected_source_files=len(selected),support_files=len(extra),production_equations_tolerances_and_iteration_budgets_unchanged=True),indent=2)+'\n');shutil.copy2(__file__,O/'assemble.py')
cmd=[x.replace('source-exact-material-point-112','source-material674-trace-116').replace('provenance-112','provenance-116').replace('numi-matter-cardboard-source370-material-point-check','numi-matter-cardboard-bilinear15-capture-diagnostic-check') for x in json.loads((B/'build-command-112.json').read_text())]
(B/'build-command-116.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build116-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(NEW/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-116/source-build.json')),indent=2)+'\n');print('build116',code,flush=True);raise SystemExit(code)
