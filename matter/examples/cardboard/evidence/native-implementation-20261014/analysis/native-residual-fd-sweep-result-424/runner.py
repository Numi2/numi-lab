from pathlib import Path
import datetime,hashlib,json,subprocess,time
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/native-residual-fd-sweep-419-001';O=B/'analysis/native-residual-fd-sweep-result-424';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build_path=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json')
build=json.loads(build_path.read_text())
protected={Path(p):h for p,h in build['artifacts_sha256'].items()}
protected[build_path]=sha(build_path)
for p,h in protected.items():assert sha(p)==h,p
assert sha(P/'manifest.json')=='8fdc04c247e4fb2b8b28cd43b9ca139c345765ac68ea9888f7dc6745de6be024'
m=json.loads((P/'manifest.json').read_text())
for name,row in m['files'].items():assert sha(P/name)==row['sha256'],name
cmd=m['native_argv'][:-1]+[str(O/'result')]
(O/'prediction.json').write_text(json.dumps(dict(question='Does the captured control1 Newton48 residual have a resolved native derivative as representable perturbations decrease?',prediction='Production source125 candidate preparation, geometry validation, internal force and residual kernels replay exact capture419 center and signed corrections across binary32 step sizes. Same accepted node/material history, dt and parameters. Check exact retained endpoint parity before interpreting new smaller steps. Invalid endpoints and near-zero noise remain inconclusive.',package_manifest_sha256=sha(P/'manifest.json'),command=cmd),indent=2)+'\n')
start=time.monotonic();results={}
for name,argv in [('preflight',cmd[:-1]+['/tmp/cardboard424-preflight-unused','--preflight-only']),('native',cmd)]:
 with (O/f'{name}.stdout.log').open('x') as out,(O/f'{name}.stderr.log').open('x') as err:
  try:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180);results[name]=dict(exit_code=q.returncode,timed_out=False)
  except subprocess.TimeoutExpired:results[name]=dict(exit_code=None,timed_out=True)
 if results[name]['exit_code']!=0:break
for name,row in m['files'].items():assert sha(P/name)==row['sha256'],name
for p,h in protected.items():assert sha(p)==h,p
(O/'receipt.json').write_text(json.dumps(dict(commands=results,wall_seconds=time.monotonic()-start,prepared_package_unchanged=True,physical_transactions=0,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('native424',results,flush=True)
