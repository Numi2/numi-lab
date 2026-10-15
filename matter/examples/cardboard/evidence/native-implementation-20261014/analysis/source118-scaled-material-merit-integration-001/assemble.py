from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');OLD=B/'snapshots/source-material674-trace-116';NEW=B/'snapshots/source-scaled-material-merit-118';O=B/'analysis/source118-scaled-material-merit-integration-001';P=B/'analysis/source380-dimensionless-armijo-overlay-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text());s=json.loads((OLD/'configure-support-source.json').read_text());m=json.loads((P/'source-overlay.json').read_text())
assert sha(P/'source-overlay.json')=='034882884a999c66df264ccaf7120c5b5b73f217a970ca9da9bb1842d5b9cdc9'
for p,h in m['files'].items():assert sha(P/p)==h,p
assert sha(P/'source-overlay.patch')==m['overlay']['patch_sha256']
for p,h in {**a['source_sha256'],**s['files']}.items():assert sha(OLD/p)==h,p
assert not NEW.exists();O.mkdir(exist_ok=False)
# APFS clone preserves independent paths and exact bytes without allocating a
# second physical copy of every unchanged source file on this nearly full disk.
q=subprocess.run(['/bin/cp','-cR',str(OLD),str(NEW)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for p,h in {**a['source_sha256'],**s['files']}.items():assert sha(NEW/p)==h,p

prod=B/'snapshots/source-private-ccd-fracture-115'
production_changes=json.loads((B/'analysis/source115-private-ccd-fracture-integration-002/integration.json').read_text())['changes']
for p,h in production_changes.items():
 assert sha(prod/p)==h,p
 shutil.copy2(prod/p,NEW/p)
common=NEW/m['base']['path']
raw=common.read_text();assert raw.count('kMaterialProjectionTraceIterationCapacity = 32u;')==1
common.write_text(raw.replace('kMaterialProjectionTraceIterationCapacity = 32u;','kMaterialProjectionTraceIterationCapacity = 16u;'))
assert sha(common)==m['base']['sha256']
q=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'source-overlay.patch')],cwd=NEW,capture_output=True,text=True);(O/'patch.log').write_text(q.stdout+q.stderr);assert q.returncode==0
assert sha(common)==m['overlay']['sha256']
raw=common.read_text();assert raw.count('kMaterialProjectionTraceIterationCapacity = 16u;')==1
common.write_text(raw.replace('kMaterialProjectionTraceIterationCapacity = 16u;','kMaterialProjectionTraceIterationCapacity = 32u;'))
selected={p:sha(NEW/p) for p in a['source_sha256']}
(NEW/'frozen-source.json').write_text(json.dumps({**a,'base_snapshot':str(OLD),'scope':'Candidate local-projection dimensionless Armijo merit under existing accurate bilinear15 policy. Keeps source115 production CCD and damage guards, source116 exact380 point diagnostic and32 trace slots. No residual/correction tolerance, iteration budget, constitutive equation or recipe changes.','source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']};assert extra==s['files']
(NEW/'configure-support-source.json').write_text(json.dumps({**s,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(NEW),source_overlay_sha256=sha(P/'source-overlay.json'),common_base_sha256=m['base']['sha256'],common_candidate_sha256=sha(common),source115_production_changes=production_changes,all_source_bytes_verified_after_APFS_clone=True,selected_source_files=len(selected),support_files=len(extra),production_equations_tolerances_and_iteration_budgets_unchanged=True,line_search_merit_changed_only_for_existing_accurate_policy=True),indent=2)+'\n');shutil.copy2(__file__,O/'assemble.py')
cmd=json.loads((B/'build-command-116.json').read_text())
cmd=[x.replace('source-material674-trace-116','source-scaled-material-merit-118').replace('provenance-116','provenance-118') for x in cmd]
cmd+=['--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-full-contact-armijo-check']
(B/'build-command-118.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build118-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(NEW/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-118/source-build.json')),indent=2)+'\n');print('build118',code,flush=True);raise SystemExit(code)
