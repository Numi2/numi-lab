from pathlib import Path
import hashlib,json,subprocess,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-cache-projection-reuse-capture419-132';N=B/'snapshots/source-cache-projection-reuse-checker-fixed-134';O=B/'analysis/source134-cache-reuse-integration-001';A=B/'analysis/tangent-cache-projection-reuse-parity-125-003-checker-binding-and-nserror';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(S/p)==h,p
assert sha(A/'source-overlay.json')=='540459d84b26708544c296ce835084657dd9976dc5f8522da26b6fded5b762b6';j=json.loads((A/'source-overlay.json').read_text())
for p,h in j['files_sha256'].items():assert sha(A/p)==h,p
assert not N.exists();O.mkdir();q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for dry in [True,False]:
 cmd=['patch','--batch']+(['--dry-run'] if dry else [])+['-p1','-i',str(A/'prototype/checker-source132-003.patch')];q=subprocess.run(cmd,cwd=N,capture_output=True,text=True);(O/('patch-dry.log' if dry else 'patch-apply.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
newfile='matter/tools/cardboard_tangent_cache_projection_reuse_check.mm'
selected={p:sha(N/p) for p in list(f['source_sha256'])};support={p:sha(N/p) for p in u['files']}
(N/'frozen-source.json').write_text(json.dumps({**f,'base_snapshot':str(S),'scope':'Source132 plus checker-only binding digest and lazy NSError handling fixes. No kernel/runtime/physical-policy changes.','source_sha256':selected},indent=2)+'\n');(N/'configure-support-source.json').write_text(json.dumps({**u,'files':support},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),overlay_manifest_sha256=sha(A/'source-overlay.json'),changed_files={p:dict(base=f['source_sha256'].get(p),candidate=h) for p,h in selected.items() if f['source_sha256'].get(p)!=h},physical_steps=0),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-134'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-tangent-cache-projection-reuse-check'];(B/'build-command-134.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build134-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_build_sha256=sha(R/'build-cardboard-provenance-134/source-build.json')),indent=2)+'\n');print('build134',code,flush=True);raise SystemExit(code)
