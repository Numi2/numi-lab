from pathlib import Path
import gzip,json,statistics,hashlib
A=Path('artifacts/crow-v15-supported-balance-identification-20261110');P=Path('artifacts/crow-v15-second-clearance-eval-20261110/second-j-original.json.gz');h=hashlib.sha256(P.read_bytes()).hexdigest()
with gzip.open(P,'rt') as f:d=json.load(f)['payload']
assert d['body_names'][0]=='crow_body' and d['body_names'][30]=='crow_course_perch'
r={}
for lo,hi in [(4246,4345),(4346,4445)]:
 x=[z['q'][0]-z['body_states'][30*13] for z in d['frames'] if lo<=z['step']<=hi];r[f'{lo}-{hi}']={'samples':len(x),'minimum_m':min(x),'median_m':statistics.median(x),'maximum_m':max(x)}
out={'replay_sha256':h,'root_minus_perch_world_x':r,'note':'Used development J, release event at 4346. Prior supported grip had large toe overlap, so this checks only the local root reference, not physical or landing qualification.'};(A/'route-reference-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
