from pathlib import Path
import hashlib,json,subprocess,shutil,difflib
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-status5-vtccd-diagnostics-125';D=B/'analysis/native-base-cauchy-fallback-133/source';N=B/'snapshots/source-base-state-cauchy-armijo-133';O=B/'analysis/source133-cauchy-runtime-integration-001';A=B/'analysis/native-base-cauchy-kernel-133-metal-overlay-001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());u=json.loads((S/'configure-support-source.json').read_text());m=json.loads((A/'manifest.json').read_text());assert sha(A/'manifest.json')=='ad321f2d1aa6ebdca456d6bc94b1a91bbbb55469fa040101a0a618021938614c'
for r in m['inventory']:assert sha(A/r['path'])==r['sha256'],r['path']
changed=['matter/src/metal/fem.metalinc','matter/src/metal/fgmres.metalinc','matter/src/runtime.mm'];new='matter/tools/test_base_cauchy_fallback_reference.py'
for p,h in {**f['source_sha256'],**u['files']}.items():
 assert sha(S/p)==h,p
 if p not in changed:assert sha(D/p)==h,p
for p in changed[:2]:assert sha(D/p)==m['candidate_files'][p]['sha256']
assert sha(D/new)==m['candidate_files'][new]['sha256'];assert not N.exists();O.mkdir();q=subprocess.run(['/bin/cp','-cR',str(D),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
selected={p:sha(N/p) for p in f['source_sha256']};selected[new]=sha(N/new)
(N/'frozen-source.json').write_text(json.dumps({**f,'base_snapshot':str(S),'scope':'Opt-in exact control/Newton pure-FEM base-residual Cauchy fallback after exhausted primary Armijo. Same physical operator, material parameters, 24-trial policy and final nonlinear acceptance. Requires original Newton true-residual certification; the Cauchy direction is separately identified and is never labeled as linearly certified. Full field-half RHS must be exactly finite zero.','source_sha256':selected},indent=2)+'\n')
(N/'configure-support-source.json').write_bytes((S/'configure-support-source.json').read_bytes())
patch=''
for p in changed:patch+=''.join(difflib.unified_diff((S/p).read_text().splitlines(True),(N/p).read_text().splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(O/'source133.patch').write_text(patch)
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),metal_overlay_manifest_sha256=sha(A/'manifest.json'),changed_files={p:dict(base=sha(S/p),candidate=sha(N/p)) for p in changed},new_files={new:sha(N/new)},physical_steps=0),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-133'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe'];(B/'build-command-133.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build133-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_build_sha256=sha(R/'build-cardboard-provenance-133/source-build.json')),indent=2)+'\n');print('build133',code,flush=True);raise SystemExit(code)
