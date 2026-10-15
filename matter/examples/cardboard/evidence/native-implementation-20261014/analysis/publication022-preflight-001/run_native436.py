from pathlib import Path
import datetime, hashlib, json, subprocess, time

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
A=B/'analysis/captured419-regularized-directions-001'
P=B/'analysis/native-residual-direction-replay-419-001'
O=B/'analysis/native-regularized-direction-replay-436'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A/'manifest.json')=='339ae206918d3ef99d0a42859de94948299c39ea2114afd30981689c62acc734'
m=json.loads((A/'manifest.json').read_text())
for row in m['artifacts']:
    assert sha(A/row['path'])==row['sha256'],row['path']
report=json.loads((A/'report.json').read_text())
build_path=R/'build-cardboard-provenance-125/source-build.json'
build=json.loads(build_path.read_text())
protected={Path(p):h for p,h in build['artifacts_sha256'].items()}
protected[build_path]=sha(build_path)
program=json.loads((P/'compiled-program.json').read_text())
for p,h in protected.items():assert sha(p)==h,p
O.mkdir()
commands={name:[str(P/'build/native_residual_direction'),
    str(B/'solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack'),
    str(A/row['replay_payload_file']),str(O/name)]
    for name,row in report['results'].items()}
assert len(commands)==4
(O/'prediction.json').write_text(json.dumps(dict(
    question='Do the regularized exact-center proposals reduce the actual production nonlinear residual?',
    prediction='Test all four proposals at the source125 accepted history and candidate, using the unchanged material projections, kinematics and residual. Linear predicted ratios are .867773 raw, .214562/.066046/.032429 regularized. Recompute every signed endpoint; no physical transaction and no claim that the original Newton certificate certifies a new direction.',
    package_manifest_sha256=sha(A/'manifest.json'),
    program_binding_sha256=sha(P/'compiled-program.json'),
    source_build_sha256=sha(build_path),commands=commands),indent=2)+'\n')
results={}
t=time.monotonic()
for name,cmd in commands.items():
    for mode,argv in [('preflight',cmd+['--preflight-only']),('native',cmd)]:
        with (O/(name+'.'+mode+'.stdout.log')).open('x') as out,(O/(name+'.'+mode+'.stderr.log')).open('x') as err:
            try:
                q=subprocess.run(argv,stdout=out,stderr=err,timeout=180)
                r=dict(exit_code=q.returncode,timed_out=False)
            except subprocess.TimeoutExpired:
                r=dict(exit_code=None,timed_out=True)
        results[name+'.'+mode]=r
        if r['exit_code']!=0:break
    for p,h in protected.items():assert sha(p)==h,p
for row in m['artifacts']:
    assert sha(A/row['path'])==row['sha256'],row['path']
(O/'receipt.json').write_text(json.dumps(dict(commands=results,
    wall_seconds=time.monotonic()-t,physical_transactions=0,
    prepared_inputs_unchanged=True,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={
    str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p))
    for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('native436',results,flush=True)
raise SystemExit(0 if all(r['exit_code']==0 for r in results.values()) else 2)
