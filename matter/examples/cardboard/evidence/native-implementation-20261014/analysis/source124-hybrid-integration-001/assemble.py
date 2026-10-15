from pathlib import Path
import hashlib,json,subprocess,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-two-stage-material-merit-123';N=B/'snapshots/source-hybrid-regional-preconditioner-124';O=B/'analysis/source124-hybrid-integration-001'
A=B/'analysis/partial-graph-hybrid-selector6-overlay-001';C=B/'analysis/partial-graph-hybrid-selector6-native-check-overlay-002';M=B/'analysis/source392-trace-input-provenance-overlay-001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
for a,h in [(A,'f9641811e5296f9b55ec98acf2a8800e47b216b8be7115c98477518329b8b554'),(C,'f8cb30b6134a13f2763ad7b8b02d21da8ad7dd0a34b2b30f864e5dfcd8c2e96e')]:
 assert sha(a/'source-overlay.json')==h
 j=json.loads((a/'source-overlay.json').read_text());assert sha(a/'source-overlay.patch')==j['source_overlay_patch_sha256']
 for row in j['files']:assert sha(a/'source'/row['path'])==row['overlay_sha256'],row['path']
assert sha(M/'manifest.json')=='21edbdc44d11dae0fa9f238ff1a9d6a36e4ea8b9685f823a75fcacf6dfb9a3d9'
assert not O.exists();O.mkdir()
if not N.exists():
 q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
else:
 for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(N/p)==h,p
 (O/'clone.log').write_text('Reused untouched source123 clone from preserved logging-path setup failure; selected/support bytes all verified before applying patches.\n')
for number,a in [(1,A),(2,C)]:
 for dry in [True,False]:
  cmd=['patch','--batch']+(['--dry-run'] if dry else [])+['-p1','-i',str(a/'source-overlay.patch')];q=subprocess.run(cmd,cwd=N,capture_output=True,text=True);(O/(f'patch{number}-'+('dry' if dry else 'apply')+'.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
shutil.copy2(M/'cardboard_bilinear15_capture_diagnostic_check.mm',N/'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
p=N/'matter/src/metal/common.metalinc';s=p.read_text();before='''        // entire line search. This compares dimensionless residual rows and
        // leaves the independently checked raw residual/correction gates
        // unchanged.
''';after='''        // entire line search. The scaled merit is used after the raw gate
        // passes; both the raw and dimensionless trial norms are recorded.
        // The independently checked stopping gates remain unchanged.
''';assert s.count(before)==1;p.write_text(s.replace(before,after))
selected={p:sha(N/p) for p in old['source_sha256']};extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source123 plus explicit experimental selector6 partial graph/block hybrid preconditioner, bounded native checker, and diagnostic provenance label correction. Physical material/operator/tolerance/acceptance unchanged; selector6 requires separate native qualification.','source_sha256':selected},indent=2)+'\n');(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),overlay1_sha256=sha(A/'source-overlay.json'),overlay2_sha256=sha(C/'source-overlay.json'),changes={p:dict(source123=old['source_sha256'][p],source124=h) for p,h in selected.items() if old['source_sha256'][p]!=h},physical_steps=0,native_scalability_qualified=False),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=json.loads((B/'build-command-119.json').read_text());src=cmd[cmd.index('--source-snapshot')+1];build=cmd[cmd.index('--build-dir')+1];cmd=[x.replace(src,str(N)).replace(build,str(R/'build-cardboard-provenance-124')) for x in cmd];cmd+=['--target','numi-matter-cardboard-source392-tet858-point-check'];(B/'build-command-124.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build124-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-124/source-build.json')),indent=2)+'\n');print('build124',code,flush=True);raise SystemExit(code)
