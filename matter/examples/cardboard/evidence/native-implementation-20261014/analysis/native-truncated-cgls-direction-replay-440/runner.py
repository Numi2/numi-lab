from pathlib import Path
import datetime, hashlib, json, subprocess, time
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
A=B/'analysis/captured419-truncated-cgls-directions-001'
P=B/'analysis/native-residual-direction-replay-419-001'
O=B/'analysis/native-truncated-cgls-direction-replay-440'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'manifest.json')=='37d9bfcf6706f38d08f5cfc89a2f60c0df74b9f3da35e78ada43df4c82fa6b4e'
m=json.loads((A/'manifest.json').read_text())
for p,row in m['files'].items():assert sha(A/p)==row['sha256'],p
r=json.loads((A/'report.json').read_text())
program=json.loads((P/'compiled-program.json').read_text())
protected={P/p:h for p,h in program['files'].items()}
build_path=R/'build-cardboard-provenance-125/source-build.json'
build=json.loads(build_path.read_text())
protected.update({Path(p):h for p,h in build['artifacts_sha256'].items()})
protected[build_path]=sha(build_path)
for p,h in protected.items():assert sha(p)==h,p
selected=[('fp64',32),('fp64',64),('fp64',128),('emulated_fp32',16),('emulated_fp32',128)]
O.mkdir();commands={}
for arithmetic,k in selected:
    name=f'{arithmetic}_iter{k:03d}'
    row=r['runs']['outputs'][arithmetic][str(k)]
    commands[name]=[r['native_replay_commands']['helper'],
        r['native_replay_commands']['compiled_package'],
        str(A/row['payload_file']),str(O/name)]
(O/'prediction.json').write_text(json.dumps(dict(
    question='Can short CGLS directions reduce the captured actual nonlinear residual?',
    prediction='Compare exact-CSR FP64 CGLS at32/64/128 and FP32 emulation controls16/128 with source125 native residual. Linear residual ratios decrease through128; native direction and nonlinear curve remain unknown. Recompute signed trial geometry, projections and force/residual. Zero physical transactions and no original-Newton certificate transfer.',
    package_manifest_sha256=sha(A/'manifest.json'),commands=commands,
    helper_program_sha256=sha(P/'compiled-program.json'),
    source_build_sha256=sha(build_path)),indent=2)+'\n')
results={};start=time.monotonic()
for name,cmd in commands.items():
    for mode,argv in [('preflight',cmd+['--preflight-only']),('native',cmd)]:
        with (O/(name+'.'+mode+'.stdout.log')).open('x') as out,(O/(name+'.'+mode+'.stderr.log')).open('x') as err:
            try:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180);v=dict(exit_code=q.returncode,timed_out=False)
            except subprocess.TimeoutExpired:v=dict(exit_code=None,timed_out=True)
        results[name+'.'+mode]=v
        if v['exit_code']!=0:break
    for p,h in protected.items():assert sha(p)==h,p
for p,row in m['files'].items():assert sha(A/p)==row['sha256'],p
(O/'receipt.json').write_text(json.dumps(dict(commands=results,
    wall_seconds=time.monotonic()-start,physical_transactions=0,
    prepared_inputs_unchanged=True,helper_and_build_bindings_unchanged=True,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):
    dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('native440',results,flush=True)
raise SystemExit(0 if all(v['exit_code']==0 for v in results.values()) else 2)
