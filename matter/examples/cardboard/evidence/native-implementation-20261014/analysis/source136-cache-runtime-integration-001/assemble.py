from pathlib import Path
import hashlib, json, shutil, subprocess

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
A=B/'analysis/cache-projection-reuse-runtime-overlay-136-001'
S=B/'snapshots/source-cauchy-positive-mass-admission-135'
N=B/'snapshots/source-cache-projection-reuse-runtime-136'
O=B/'analysis/source136-cache-runtime-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (A/'manifest.json').is_file(), 'await sealed overlay'
review=json.loads((A/'logs/source-review-report.json').read_text())
assert review['original_production_cache_kernel_exact_match']
assert review['sibling_kernel_matches_source134_after_only_admission_guard_removed']
assert all(review['runtime_invariants'].values())
f=json.loads((S/'frozen-source.json').read_text())
u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():
    assert sha(S/p)==h,p
for p,h in review['source_file_sha256'].items():
    assert sha(S/p)==h,p
for p,h in review['candidate_file_sha256'].items():
    assert sha(A/'candidate'/p)==h,p
assert not N.exists()
O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True)
(O/'clone.log').write_text(q.stdout+q.stderr)
assert q.returncode==0
for p,h in review['candidate_file_sha256'].items():
    shutil.copy2(A/'candidate'/p,N/p)
    f['source_sha256'][p]=h
f['base_snapshot']=str(S)
f['scope']='Source135 plus opt-in reuse of the identical compensated/scaled generic15 accepted projection across nine tangent columns. Strict bounded coupon admission; default cache builder unchanged. Full-history parity remains to be executed; no speed or physical qualification.'
(N/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
for p,h in {**f['source_sha256'],**u['files']}.items():
    assert sha(N/p)==h,p
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
(O/'integration.json').write_text(json.dumps(dict(
    snapshot=str(N), base_snapshot=str(S),
    overlay_manifest_sha256=sha(A/'manifest.json'),
    source_review_sha256=sha(A/'logs/source-review-report.json'),
    changed_files=review['candidate_file_sha256'], physical_steps=0),indent=2)+'\n')
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-136'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe']
(B/'build-command-136.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build136-runner.log').open('x') as out:
    code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,
    source_build_sha256=sha(R/'build-cardboard-provenance-136/source-build.json')),indent=2)+'\n')
print('build136',code,flush=True)
raise SystemExit(code)
