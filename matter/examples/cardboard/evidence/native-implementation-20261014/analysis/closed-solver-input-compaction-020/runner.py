from pathlib import Path
import json,hashlib,subprocess,datetime
B=Path('/Users/home/cardboard-implementation-20261014');S=B/'analysis/closed-solver-input-compaction-019';O=B/'analysis/closed-solver-input-compaction-020';O.mkdir(exist_ok=False)
selected=['current-failure-22-state-capture-153','rationalized-paper-control2-newton63-capture-304','inactive-field-admission-augmentation-on-335','inactive-field-admission-augmentation-off-336','two-stage-coupon-first-descent-capture-415','base-cauchy-disabled-parity-434','base-cauchy-first-failure-enabled-435','cache-reuse-full-history-parity-437','repeated-cauchy-full-coupon-438']
s=(S/'compact_case.py').read_text().replace("OUT = BASE / 'analysis/closed-solver-input-compaction-019'","OUT = BASE / 'analysis/closed-solver-input-compaction-020'").replace('number > 414','number > 440').replace("number = parse_case_number(case.name)\n", "number = parse_case_number(case.name)\n    if case.name not in "+repr(selected)+":\n        raise RuntimeError('case outside explicit task-owned closed-case manifest')\n")
(O/'compact_case.py').write_text(s);(O/'restore_inputs.py').write_text((S/'restore_inputs.py').read_text().replace('closed-solver-input-compaction-019','closed-solver-input-compaction-020'));(O/'selected.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exact_cases':selected,'retained_raw_cases':'419,441through458 and allactive/incomplete/timedoutcases','preservation':'archive then hash/size/modes verification and full streaming restore before unlink originals; original source snapshots, native binaries, receipts/predictions and all output histories retained'},indent=2)+'\n')
records=[]
for name in selected:
 with (O/(name+'.stdout.log')).open('x') as out,(O/(name+'.stderr.log')).open('x') as err:
  p=subprocess.run(['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(O/'compact_case.py'),name],stdout=out,stderr=err)
 records.append({'case':name,'exit_code':p.returncode});print(name,p.returncode,flush=True)
 if p.returncode:break
(O/'batch-summary.json').write_text(json.dumps({'records':records,'all_success':all(r['exit_code']==0 for r in records),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n');(O/'runner.py').write_bytes(Path(__file__).read_bytes())
