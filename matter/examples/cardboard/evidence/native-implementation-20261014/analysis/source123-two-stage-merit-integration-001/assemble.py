from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-native-tet858-vtccd-traces-122';N=B/'snapshots/source-two-stage-material-merit-123';O=B/'analysis/source123-two-stage-merit-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
assert not N.exists() and not O.exists();O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
p='matter/src/metal/common.metalinc';s=(N/p).read_text();before='const bool armijoAccepted = scaledCorrectionPolicy\n';after='''// First reduce the existing raw residual gate. Once that gate is
            // satisfied, resolve the dimensionless correction with the frozen
            // row-scaled merit. A scaled-only merit can stall far from the root;
            // a raw-only merit can reject useful near-root correction steps.
            // Both stopping gates above and the local budget remain unchanged.
            const bool useScaledMerit = scaledCorrectionPolicy && residualConverged;
            const bool armijoAccepted = useScaledMerit
''';assert s.count(before)==1;s=s.replace(before,after);(N/p).write_text(s)
(O/'common.patch').write_text(subprocess.run(['git','diff','--no-index','--',str(S/p),str(N/p)],capture_output=True,text=True).stdout)
selected={p:sha(N/p) for p in old['source_sha256']};extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source122 plus candidate two-stage local Armijo merit for accurate bilinear15 only: raw merit while existing raw residual gate fails, frozen row-scaled merit after it passes. No local/global tolerance, budget, physical material or acceptance-certificate change.','source_sha256':selected},indent=2)+'\n')
(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),cpu_prototype_manifest_sha256=sha(B/'analysis/exact-material-merit-prototype-003/manifest.json'),changes={p:dict(source122=old['source_sha256'][p],source123=h) for p,h in selected.items() if old['source_sha256'][p]!=h},physical_steps=0,native_qualification=False),indent=2)+'\n');(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-123'),'--cmake-source-relative','.','--jobs','4']
for target in ['numi-matter-cardboard-bilinear15-capture-diagnostic-check','numi-matter-cardboard-source392-tet858-point-check','numi-matter-cardboard-regional-bending-check']:cmd+=['--target',target]
(B/'build-command-123.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build123-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-123/source-build.json')),indent=2)+'\n');print('build123',code,flush=True);raise SystemExit(code)
