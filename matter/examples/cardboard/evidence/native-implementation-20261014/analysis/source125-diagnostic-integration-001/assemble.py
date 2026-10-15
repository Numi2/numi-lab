from pathlib import Path
import hashlib,json,subprocess,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-two-stage-material-merit-123';N=B/'snapshots/source-status5-vtccd-diagnostics-125';O=B/'analysis/source125-diagnostic-integration-001'
A=B/'analysis/source392-status5-observer-overlay-122';C=B/'analysis/private-vtccd-budget-replay-overlay-001';M=B/'analysis/source392-trace-input-provenance-overlay-001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
for a,h in [(A,'1fb105ca0cf421c4ec2e97bdb6f3141ef81f49dd2d94aa27065864063d2665b2'),(C,'a7a6a0a361862c0afa409793e35a1091f5eceefedd0aa266378658f85ea112a3')]:
 assert sha(a/'manifest.json')==h
 j=json.loads((a/'manifest.json').read_text())
 assert sha(a/'overlay.patch')==j.get('patch_sha256',j.get('overlay_patch_sha256'))
assert sha(M/'manifest.json')=='21edbdc44d11dae0fa9f238ff1a9d6a36e4ea8b9685f823a75fcacf6dfb9a3d9'
assert not O.exists();O.mkdir()
if not N.exists():
 q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
else:
 for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(N/p)==h,p
 (O/'clone.log').write_text('Reused untouched source123 clone from preserved logging-path setup failure; selected/support bytes all verified before applying patches.\n')
for number,a in [(1,A),(2,C)]:
 for dry in [True,False]:
  cmd=['patch','--batch']+(['--dry-run'] if dry else [])+['-p1','-i',str(a/'overlay.patch')];q=subprocess.run(cmd,cwd=N,capture_output=True,text=True);(O/(f'patch{number}-'+('dry' if dry else 'apply')+'.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
shutil.copy2(M/'cardboard_bilinear15_capture_diagnostic_check.mm',N/'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
p=N/'matter/src/metal/common.metalinc';s=p.read_text();before='''        // entire line search. This compares dimensionless residual rows and
        // leaves the independently checked raw residual/correction gates
        // unchanged.
''';after='''        // entire line search. The scaled merit is used after the raw gate
        // passes; both the raw and dimensionless trial norms are recorded.
        // The independently checked stopping gates remain unchanged.
''';assert s.count(before)==1;p.write_text(s.replace(before,after))
source_paths=set(old['source_sha256']); source_paths.add('matter/tools/cardboard_vtccd_iteration_budget_check.mm'); selected={p:sha(N/p) for p in sorted(source_paths)};extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source123 plus opt-in first status5 observer, exact production-default16 VTCCD budget diagnostic replay, and metadata label correction. Physical operator, thresholds, material and acceptance unchanged.','source_sha256':selected},indent=2)+'\n');(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),overlay1_sha256=sha(A/'manifest.json'),overlay2_sha256=sha(C/'manifest.json'),changes={p:dict(source123=old['source_sha256'].get(p),source125=h) for p,h in selected.items() if old['source_sha256'].get(p)!=h},physical_steps=0,native_scalability_qualified=False),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-125'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-vtccd-budget-replay-check','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-source392-tet858-point-check'];(B/'build-command-125.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build125-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-125/source-build.json')),indent=2)+'\n');print('build125',code,flush=True);raise SystemExit(code)
