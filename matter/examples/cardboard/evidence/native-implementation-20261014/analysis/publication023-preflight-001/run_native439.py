from pathlib import Path
import datetime, hashlib, json, subprocess, time
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-repeated-cauchy-fallback-137'
D=R/'build-cardboard-provenance-137'
O=B/'native/source137-direct-operator-abi-439'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((D/'source-build.json').read_text())['success']
query=['ctest','--test-dir',str(D),'--show-only=json-v1']
tests=json.loads(subprocess.check_output(query,text=True))['tests']
targets={'numi-matter-fem-reference-check','numi-matter-fem-material-frame-check',
    'numi-matter-fem-regional-material-check',
    'numi-matter-cardboard-coupled-free-box-operator-check',
    'numi-matter-cardboard-gpu-check'}
selected=[t for t in tests if t.get('command') and
    Path(t['command'][0]).name in targets and '--cpu-only' not in t['command']]
assert len(selected)==5
O.mkdir()
(O/'selection.json').write_text(json.dumps(dict(query=query,tests=selected),indent=2)+'\n')
results={};start=time.monotonic()
for i,t in enumerate(selected):
    name=t['name'].replace('.','-');case=O/name
    cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(S/'matter/tools/cardboard_source_bound_run.py'),
        '--source-snapshot',str(S),'--runtime-dir',str(D),
        '--build-provenance',str(D/'source-build.json'),'--case',str(case),
        '--timeout','240','--question','Do the five direct operator consumers retain their native regressions after binding zero fallback metrics at buffer26?',
        '--prediction','Expect each existing synthetic native checker to pass with cache modes0/1/2 and fallback metrics zero. This verifies direct ABI compatibility and selected numerical checks; it does not qualify the full coupon, cardboard calibration or shipping assembly.',
        '--input-file',str(B/'analysis/cauchy-operator-tools-abi-overlay-137-001/manifest.json'),
        '--']+t['command']
    (O/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    with (O/(name+'-runner.log')).open('x') as out:
        q=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
    results[t['name']]=dict(exit_code=q.returncode,
        receipt_sha256=sha(case/'receipt.json'))
    print(t['name'],q.returncode,flush=True)
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.direct_operator_abi_suite.v1',
    results=results,source_build_sha256=sha(D/'source-build.json'),
    wall_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    classification='Five synthetic native regression tests; no complete coupon or physical box qualification'),indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):
    dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*'))
    if p.is_file() and 'inputs' not in p.relative_to(O).parts}),indent=2)+'\n')
raise SystemExit(0 if all(r['exit_code']==0 for r in results.values()) else 2)
