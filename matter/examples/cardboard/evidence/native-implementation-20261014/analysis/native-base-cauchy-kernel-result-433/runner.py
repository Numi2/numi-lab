from pathlib import Path
import datetime, hashlib, json, subprocess, time
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
D=B/'analysis/native-base-cauchy-kernel-check-133-001'
O=B/'analysis/native-base-cauchy-kernel-result-433'
P=B/'analysis/native-residual-fd-sweep-419-001/replay-input.bin'
W=B/'solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack'
J=R/'build-cardboard-provenance-133/source-build.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads(J.read_text()); c=json.loads((D/'compiled-program.json').read_text())
assert j['success'] and c['exit_code']==0
protected={Path(p):h for p,h in j['artifacts_sha256'].items()}
for p in [J,D/'compiled-program.json',D/'native_cauchy_kernel_check.mm',D/'native_cauchy_kernel_check',P,W]:protected[p]=sha(p)
assert sha(D/'native_cauchy_kernel_check')==c['binary_sha256']
assert sha(D/'native_cauchy_kernel_check.mm')==c['source_sha256']
O.mkdir()
cmd=[str(D/'native_cauchy_kernel_check'),str(W),str(P),str(O/'result')]
prediction=dict(question='Do the actual GPU transpose, masked forward action and compensated Cauchy reduction match independent captured-state calculations?',prediction='Require material/element error <=1e-3 and assembled error <=1%, exact fixed/unselected masking, strict-cache failure isolation, completed-latch bypass and explicit rejection of nonzero field/nonfinite free RHS. This checker makes zero physical transactions.',command=cmd,protected_inputs={str(p):h for p,h in protected.items()},physical_transactions=0)
(O/'prediction.json').write_text(json.dumps(prediction,indent=2)+'\n')
t=time.monotonic(); failure=None
for p,h in protected.items():assert sha(p)==h,p
with (O/'stdout.log').open('x') as out,(O/'stderr.log').open('x') as err:
 try:q=subprocess.run(cmd,stdout=out,stderr=err,timeout=180);code=q.returncode;timed=False
 except subprocess.TimeoutExpired:code=None;timed=True
for p,h in protected.items():assert sha(p)==h,p
(O/'receipt.json').write_text(json.dumps(dict(exit_code=code,timed_out=timed,wall_seconds=time.monotonic()-t,protected_inputs_unchanged=True,physical_transactions=0,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('native433',code,timed,flush=True)
raise SystemExit(code if code is not None else 124)
