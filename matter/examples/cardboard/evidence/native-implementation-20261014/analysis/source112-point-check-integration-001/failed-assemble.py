from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
OLD=B/'snapshots/source-explicit-material-budget-109';NEW=B/'snapshots/source-exact-material-point-112';O=B/'analysis/source112-point-check-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text());s=json.loads((OLD/'configure-support-source.json').read_text())
for k,h in {**a['source_sha256'],**s['files']}.items():assert sha(OLD/k)==h,k
assert not NEW.exists();O.mkdir(exist_ok=False);shutil.copytree(OLD,NEW)
changes={}
for n,h in [(111,'7598760eef248a2ef09c095a765238843827f2128b5cd8279ec4f4ef303f3de0'),(112,'d47a3b52fd79aea445f9bb411f58027f1bbc6e60450494d197590323f6942bef')]:
 P=B/f'analysis/exact-source370-point-replay-overlay-{n}-001';assert sha(P/'source-overlay.json')==h;m=json.loads((P/'source-overlay.json').read_text())
 patch=next(p for p in P.glob('*.patch'));assert sha(patch)==m['patch_sha256']
 for row in m['files']:
  t=NEW/row['path'];assert (sha(t) if t.exists() else None)==row['base_sha256'],row['path']
 r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(patch)],cwd=NEW,capture_output=True,text=True);(O/f'patch{n}.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 for row in m['files']:assert sha(NEW/row['path'])==row['candidate_sha256'],row['path'];changes[row['path']]=row['candidate_sha256']
selected={k:sha(NEW/k) for k in set(a['source_sha256'])|set(changes)}
(NEW/'frozen-source.json').write_text(json.dumps({**a,'base_snapshot':str(OLD),'scope':'Diagnostic-only production helper exact370 material point16/32 private descriptor evaluation; no physical trajectory qualification','source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']};assert extra==s['files']
(NEW/'configure-support-source.json').write_text(json.dumps({**s,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(NEW),selected_source_files=len(selected),support_files=len(extra),changes=changes,unmodified_production_physics_from_source109=True),indent=2)+'\n');shutil.copy2(__file__,O/'assemble.py')
cmd=[str(Path('/Users/home/.pyenv/versions/3.10.12/bin/python3')),str(NEW/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(NEW),'--build-dir',str(R/'build-cardboard-provenance-112'),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-source370-material-point-check']
(B/'build-command-112.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build112-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for k,h in {**selected,**extra}.items():assert sha(NEW/k)==h,k
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-112/source-build.json')),indent=2)+'\n');print('build112',code,flush=True);raise SystemExit(code)
