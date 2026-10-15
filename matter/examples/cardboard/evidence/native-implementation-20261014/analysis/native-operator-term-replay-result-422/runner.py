from pathlib import Path
import datetime,hashlib,json,subprocess,time
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/native-operator-term-replay-419-001';O=B/'analysis/native-operator-term-replay-result-422';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest.json')=='3be02f1a7509836fa6ccc6b5528aba5b2c68f40589dbf7bc54d1745ee8f4dd5f'
m=json.loads((P/'manifest.json').read_text())
for name,row in m['files'].items():assert sha(P/name)==row['sha256'],name
cmd=m['native_argv'][:-1]+[str(O/'result')]
(O/'prediction.json').write_text(json.dumps(dict(question='Which element/frame/gather/mass layer explains the source125 exact control1 Newton48 action?',prediction='Exact source125 production element/gather kernels, cache and fresh paths; unchanged captured candidate, history, direction and dt. Zero physical transactions. Require comparison against captured action before interpreting independent terms.',package_manifest_sha256=sha(P/'manifest.json'),command=cmd),indent=2)+'\n')
start=time.monotonic();results={}
for name,argv in [('preflight',cmd[:-1]+['/tmp/cardboard422-preflight-unused','--preflight-only']),('native',cmd)]:
 with (O/f'{name}.stdout.log').open('x') as out,(O/f'{name}.stderr.log').open('x') as err:
  try:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180);results[name]=dict(exit_code=q.returncode,timed_out=False)
  except subprocess.TimeoutExpired:results[name]=dict(exit_code=None,timed_out=True)
 if results[name]['exit_code']!=0:break
for name,row in m['files'].items():assert sha(P/name)==row['sha256'],name
(O/'receipt.json').write_text(json.dumps(dict(commands=results,wall_seconds=time.monotonic()-start,prepared_package_unchanged=True,physical_transactions=0,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('native422',results,flush=True)
