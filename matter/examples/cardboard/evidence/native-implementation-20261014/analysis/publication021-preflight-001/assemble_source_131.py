from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-hybrid-status-includes-129';N=B/'snapshots/source-hybrid-outcome-capture-131';O=B/'analysis/source131-hybrid-outcome-integration-001';A=B/'analysis/selector5-6-runtime-outcome-overlay-002';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(S/p)==h,p
j=json.loads((A/'manifest.json').read_text())
for row in j['files']:assert sha(A/row['path'])==row['sha256'],row['path']
assert sha(A/'selector-runtime-complete-outcomes.patch')==j['patch_sha256'];assert not N.exists();O.mkdir();q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for dry in [True,False]:
 cmd=['patch','--batch']+(['--dry-run'] if dry else [])+['-p1','-i',str(A/'selector-runtime-complete-outcomes.patch')];q=subprocess.run(cmd,cwd=N,capture_output=True,text=True);(O/('patch-dry.log' if dry else 'patch-apply.log')).write_text(q.stdout+q.stderr);assert q.returncode==0
selected={p:sha(N/p) for p in f['source_sha256']};support={p:sha(N/p) for p in u['files']}
(N/'frozen-source.json').write_text(json.dumps({**f,'base_snapshot':str(S),'scope':'Source129 plus checker-only capture of both selector5/6 runtime outcomes before unchanged parity gates. No production change.','source_sha256':selected},indent=2)+'\n');(N/'configure-support-source.json').write_text(json.dumps({**u,'files':support},indent=2)+'\n');(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),overlay_manifest_sha256=sha(A/'manifest.json'),changed_files={p:dict(base=f['source_sha256'][p],candidate=h) for p,h in selected.items() if f['source_sha256'][p]!=h},production_changes=False,physical_steps=0),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-131'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-preconditioner-gpu-check'];(B/'build-command-131.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build131-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_build_sha256=sha(R/'build-cardboard-provenance-131/source-build.json')),indent=2)+'\n');print('build131',code,flush=True);raise SystemExit(code)
