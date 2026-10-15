from pathlib import Path
import datetime,hashlib,json,subprocess
b=Path('/Users/home/cardboard-implementation-20261014')
case=b/'solver/accurate-source15-first-failure-capture-293'
out=b/'analysis/accurate-failure-capture-timeout-293-postmortem'
assert not out.exists();out.mkdir()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((case/'prediction.json').read_text())
r=json.loads((case/'receipt.json').read_text());assert r['timed_out'] and r['source_inputs_unchanged'] is None
bad=[]
for name,expected in p['inputs_sha256'].items():
    sealed=case/'inputs'/name
    assert h(sealed)==expected,name
    rel=Path(name)
    if rel.parts[0]=='source':live=Path(p['source_snapshot'])/Path(*rel.parts[1:])
    elif rel.parts[0]=='runtime':live=Path(p['runtime_dir'])/Path(*rel.parts[1:])
    elif rel.parts[0]=='executable':live=Path(p['argv'][0])
    elif rel.parts[0]=='external':
        candidates=[Path(x) for x in p['external_inputs'] if Path(x).name==rel.name]
        assert len(candidates)==1;live=candidates[0]
    else:raise ValueError(name)
    if not live.is_file() or h(live)!=expected:bad.append(str(live))
assert not bad,bad
ps=subprocess.check_output(['ps','-p','11280','-o','pid=,state=,rss=,command='],text=True).strip()
captures=list(case.glob('operator-capture*'))
report=dict(schema='numi.cardboard.timed-out-capture-postmortem.v1',
    audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_receipt_sha256=h(case/'receipt.json'),original_receipt=r,
    original_prediction_sha256=h(case/'prediction.json'),
    sealed_input_count=len(p['inputs_sha256']),all_sealed_and_live_inputs_match_at_postmortem=True,
    mismatched_inputs=bad,process_snapshot=ps,operator_capture_files=[str(x) for x in captures],
    qualification=False,scope='Separate postmortem only. Original exception receipt is unchanged; it did not complete its post-run input scan. No exact failed material capture exists. Zombie RSS0 is not an active GPU worker.')
assert not captures
(out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(postmortem=str(out),sealed_inputs=len(p['inputs_sha256']),captures=len(captures))))
