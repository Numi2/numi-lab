from pathlib import Path
import difflib, hashlib, json, subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-repeated-cauchy-fallback-137'
N=B/'snapshots/source-coupon-cache-admission-139'
A=B/'analysis/cache-projection-reuse-admission-overlay-138-001'
O=B/'analysis/source139-cache-admission-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'manifest.json')=='fbb3c61cfd7367c4c32bd65a59ff225969e82b8e2451642d91d5fffc81dc5484'
m=json.loads((A/'manifest.json').read_text())
for row in m['inventory']:assert sha(A/row['path'])==row['sha256'],row['path']
f=json.loads((S/'frozen-source.json').read_text())
u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(S/p)==h,p
assert not N.exists();O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True)
(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
patch=(A/'overlay.patch').read_bytes()
for mode in ['dry-run','apply']:
    cmd=['patch','--batch','-p1','-d',str(N)]
    if mode=='dry-run':cmd.insert(1,'--dry-run')
    q=subprocess.run(cmd,input=patch,capture_output=True)
    log=q.stdout+q.stderr;(O/(mode+'.log')).write_bytes(log)
    assert q.returncode==0 and b'fuzz' not in log.lower(),log.decode()
assert sha(N/'matter/src/metal/fem.metalinc')==m['changed_files']['matter/src/metal/fem.metalinc']['candidate_sha256']
p=N/'matter/src/runtime.mm';before=p.read_text();after=before
old='''\\\"maximum_newton_records\\\":64'''
new='''\\\"default_maximum_newton_records\\\":64'''
assert after.count(old)==1;after=after.replace(old,new)
old='''\\\"scope\\\":\\\"all_newton_iterations_on_one_control_step\\\",\\\"physical_operator_changed\\\":false'''
new='''\\\"scope\\\":\\\"all_newton_iterations_on_one_control_step\\\",\\\"maximum_trace_records\\\":512,\\\"physical_operator_changed\\\":false'''
assert after.count(old)==1;after=after.replace(old,new)
old='''                            "fem_residual_armijo_trace_stopped={\\\"schema\\\":1,\\\"root\\\":%u,\\\"reason\\\":\\\"record_cap_64\\\"}\\n",
                            request.controlStep);'''
new='''                            "fem_residual_armijo_trace_stopped={\\\"schema\\\":2,\\\"root\\\":%u,\\\"reason\\\":\\\"record_cap\\\",\\\"maximum_trace_records\\\":%u}\\n",
                            request.controlStep, maximumTraceRecords);'''
assert after.count(old)==1;after=after.replace(old,new);p.write_text(after)
(O/'trace-labels.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/matter/src/runtime.mm',tofile='b/matter/src/runtime.mm')))
changed={p:sha(N/p) for p in m['changed_files']}
f['source_sha256'].update(changed);f['base_snapshot']=str(S)
f['scope']='Source137 plus corrected, still opt-in coupon cache admission from overlay138. Elastic starch retains original tangent evaluation; only eligible paper projections are reused. Same physical operator/material/history/gates. Corrects two stale trace-cap labels: original event now explicitly describes default64 and selected all event/cap-stop reports512. Cache full-history parity and repeated-fallback trajectory remain separate experiments.'
(N/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(N/p)==h,p
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),
    admission_overlay_manifest_sha256=sha(A/'manifest.json'),changed_files=changed,
    trace_label_erratum='source137 actual selected cap512 had two stale64 labels; corrected here without changing observer budget or numerical execution',physical_steps=0),indent=2)+'\n')
targets=['numi-matter-cardboard-probe','numi-matter-fem-reference-check',
    'numi-matter-fem-material-frame-check','numi-matter-fem-regional-material-check',
    'numi-matter-cardboard-coupled-free-box-operator-check','numi-matter-cardboard-gpu-check']
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-139'),'--cmake-source-relative','.','--jobs','4']
for target in targets:cmd += ['--target',target]
(B/'build-command-139.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build139-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,
    targets=len(targets),source_build_sha256=sha(R/'build-cardboard-provenance-139/source-build.json')),indent=2)+'\n')
print('build139',code,flush=True);raise SystemExit(code)
