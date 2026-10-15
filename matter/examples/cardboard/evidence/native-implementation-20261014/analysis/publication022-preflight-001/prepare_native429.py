from pathlib import Path
import hashlib,json,struct,subprocess,datetime,time
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/native-residual-direction-replay-419-001';S=B/'analysis/native-residual-fd-sweep-419-001';A=B/'analysis/captured-base-state-cauchy-419-001';O=B/'analysis/native-base-cauchy-replay-result-429';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'manifest.json')=='03c713a1aeeb57544d47430c7bd519b6128ea4b8243142d9e798cd114f1e0348'
m=json.loads((A/'manifest.json').read_text())
for row in m['files']:assert sha(A/row['path'])==row['sha256'],row['path']
base=(S/'replay-input.bin').read_bytes();E,N=2,882;start=48+E*N*80;size=E*N*16;profiles={}
for name in ['base_column_equilibrated_cauchy','base_raw_cauchy','base_column_equilibrated_trust_1e-06_delta']:
 v=(A/(name+'.full-node-delta-v.float32x4.bin')).read_bytes();assert len(v)==N*16
 payload=base[:start]+v+bytes(N*16)+base[start+size:];p=P/(name+'.input.bin');p.write_bytes(payload);assert payload[:start]==base[:start] and payload[start+size:]==base[start+size:];profiles[name]=dict(payload=p.name,sha256=sha(p),direction_sha256=sha(A/(name+'.full-node-delta-v.float32x4.bin')))
(P/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.native-alternative-direction-replay.v1',source125_build_sha256=sha(Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json')),base_sweep_manifest_sha256=sha(S/'manifest.json'),cauchy_manifest_sha256=sha(A/'manifest.json'),profiles=profiles,files={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file()},scope='Only correction direction changed; exact production125 candidate/accepted node/material history/dt/params/frames/constraints. Original captured action not compared with a new direction. All signed trial samples recompute production geometry, material forces and residual. Zero physical transactions.'),indent=2)+'\n')
O.mkdir();build_path=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json');build=json.loads(build_path.read_text());protected={Path(p):h for p,h in build['artifacts_sha256'].items()};protected[build_path]=sha(build_path)
for p,h in protected.items():assert sha(p)==h,p
m=json.loads((P/'manifest.json').read_text());commands={}
for name,row in profiles.items():
 cmd=[str(P/'build/native_residual_direction'),str(B/'solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack'),str(P/row['payload']),str(O/name)];commands[name]=cmd
(O/'prediction.json').write_text(json.dumps(dict(question='Can an operator-adjoint Cauchy alternative decrease the exact failed center nonlinear residual?',prediction='CPU predictions from b at current center: scaled direction residual ratio0.7025; raw0.8678. Test actual production source125 signed trial geometry/constitutive validity and residual; require exact center parity with424. No accepted transaction or convergence claim.',package_manifest_sha256=sha(P/'manifest.json'),commands=commands),indent=2)+'\n')
results={};t=time.monotonic()
for name,cmd in commands.items():
 for mode,argv in [('preflight',cmd+['--preflight-only']),('native',cmd)]:
  with (O/(name+'.'+mode+'.stdout.log')).open('x') as out,(O/(name+'.'+mode+'.stderr.log')).open('x') as err:
   try:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180);r=dict(exit_code=q.returncode,timed_out=False)
   except subprocess.TimeoutExpired:r=dict(exit_code=None,timed_out=True)
  results[name+'.'+mode]=r
  if r['exit_code']!=0:break
 for p,h in protected.items():assert sha(p)==h,p
for name,row in m['files'].items():assert sha(P/name)==row['sha256'],name
(O/'receipt.json').write_text(json.dumps(dict(commands=results,wall_seconds=time.monotonic()-t,physical_transactions=0,prepared_inputs_unchanged=True,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n');(O/'runner.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n');print('native429',results,flush=True)
