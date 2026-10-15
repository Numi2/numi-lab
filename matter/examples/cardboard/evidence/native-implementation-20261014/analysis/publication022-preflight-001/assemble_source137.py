from pathlib import Path
import hashlib, json, shutil, subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-cache-projection-reuse-runtime-136'
N=B/'snapshots/source-repeated-cauchy-fallback-137'
A=B/'analysis/cauchy-operator-tools-abi-overlay-137-001'
P=B/'analysis/repeated-cauchy-selector-overlay-137-001'
O=B/'analysis/source137-repeated-fallback-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'manifest.json')=='54471f654aef8d524ea71f6e262138c49546214eabd53e9d099af8759e2b9755'
m=json.loads((A/'manifest.json').read_text())
for row in m['base_files']:
    assert sha(S/row['path'])==row['sha256'],row['path']
for row in m['candidate_files']:
    assert sha(A/'candidate'/row['path'])==row['sha256'],row['path']
j=json.loads((P/'integration.json').read_text())
assert sha(S/j['path'])==j['base_sha256']
assert sha(P/'candidate/runtime.mm')==j['candidate_sha256']=='827eeecfa8c85b9c4d955da4b0fced2e9c479d0da0a9f8c419d9126e1049651e'
f=json.loads((S/'frozen-source.json').read_text())
u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(S/p)==h,p
assert not N.exists()
O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True)
(O/'clone.log').write_text(q.stdout+q.stderr)
assert q.returncode==0
changed={j['path']:j['candidate_sha256']}
shutil.copy2(P/'candidate/runtime.mm',N/j['path'])
for row in m['candidate_files']:
    p=row['path'];shutil.copy2(A/'candidate'/p,N/p);changed[p]=row['sha256']
f['source_sha256'].update(changed)
f['base_snapshot']=str(S)
f['scope']='Source136 plus bounded control:all Cauchy globalization selector, certification of every selected original Newton solve, selected diagnostic ceiling512, and explicit zero fallbackMetrics buffer26 in five direct operator tools. Cache reuse remains opt-in and has a retained initialization rejection; execute repeated-fallback experiments with reuse unset. No operator or tolerance changes; no complete physical qualification.'
(N/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(N/p)==h,p
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),
    abi_overlay_manifest_sha256=sha(A/'manifest.json'),
    selector_integration_sha256=sha(P/'integration.json'),changed_files=changed,
    physical_steps=0),indent=2)+'\n')
targets=json.loads((R/'build-cardboard-provenance-104/source-build.json').read_text())['targets']
targets += ['numi-matter-cardboard-vtccd-budget-replay-check']
assert len(set(targets))==len(targets)==69
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-137'),'--cmake-source-relative','.','--jobs','4']
for t in targets:cmd += ['--target',t]
(B/'build-command-137.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build137-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,
    targets=len(targets),source_build_sha256=sha(R/'build-cardboard-provenance-137/source-build.json')),indent=2)+'\n')
print('build137',code,flush=True)
raise SystemExit(code)
