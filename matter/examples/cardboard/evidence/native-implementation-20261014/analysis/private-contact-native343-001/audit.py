from pathlib import Path
import argparse,json,hashlib
parser=argparse.ArgumentParser();parser.add_argument('number',type=int);parser.add_argument('output',type=Path);a=parser.parse_args();b=Path('/Users/home/cardboard-implementation-20261014');command=json.loads((b/f'native-{a.number}-command.json').read_text());case=Path(command[command.index('--case')+1]);receipt=json.loads((case/'receipt.json').read_text());assert receipt['source_inputs_unchanged'] and not receipt['timed_out']
rows=[]
for line in(case/'stdout.log').read_text().splitlines():
 if line.startswith('{'):
  r=json.loads(line)
  if r.get('schema')=='numi.matter.private_self_sweep_case.v1':rows.append(r)
assert len(rows)==16 and len({r['case']for r in rows})==16
for r in rows:
 assert r['passed']==(r['actual']['code']==r['expected']['code'])
 for field in r['input_buffers']['fields']:
  raw=bytes.fromhex(field['hex']);assert len(raw)==field['byte_count']
a.output.mkdir(exist_ok=False);(a.output/'cases.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in rows));neg=[r for r in rows if r['expected']['code']!=0]
prior=b/'analysis/private-mortar-tiling-only-325-001/cases.jsonl';old={r['case']:r for r in map(json.loads,prior.read_text().splitlines())};changes=[dict(case=r['case'],expected=r['expected']['code'],previous=old[r['case']]['actual']['code'],current=r['actual']['code'])for r in rows if old[r['case']]['actual']['code']!=r['actual']['code']]
report=dict(schema='numi.cardboard.private_contact_adjudication.v1',run=a.number,case=str(case),receipt=receipt,cases=len(rows),passed=sum(r['passed']for r in rows),negative_controls_reject=all(r['passed']for r in neg),failures=[dict(case=r['case'],expected=r['expected'],actual=r['actual'])for r in rows if not r['passed']],changes_vs325=changes,cases_sha256=hashlib.sha256((a.output/'cases.jsonl').read_bytes()).hexdigest(),accepted_FEM_transactions=0,physical_assembly_qualified=False,scope='Direct production contact/private CCD kernel cases only; not a complete FEM trajectory or full contact qualification unless every declared case passes.')
(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/'audit.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(cases=len(rows),passed=report['passed'],negative_controls_reject=report['negative_controls_reject'],changes=changes),indent=2))
