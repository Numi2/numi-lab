from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-load-gated-pitch-feedback-20261113');s=collections.defaultdict(list)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='stage':s[int(x[1])].append({'t':int(x[2]),'open':bool(int(x[3])),'support':bool(int(x[4])),'posture':bool(int(x[5])),'streak':int(x[6]),'ready':bool(int(x[7])),'lift':float(x[8]),'toe_abs':float(x[9]),'toe_speed':float(x[10])})
out={}
for e,rows in s.items():
 inc=[(a,b) for a,b in zip(rows,rows[1:]) if b['lift']>a['lift']+1e-8];first=next((x for x in rows if x['lift']>0),None);out[str(e)]={'stages':len(rows),'first_nonzero_lift_zero_based':first['t'] if first else None,'max_lift_rad':max(x['lift'] for x in rows),'peak_target_step_rad':max(abs(a['lift']-b['lift']) for a,b in zip(rows,rows[1:])),'lift_increments':len(inc),'increases_without_support':sum(not b['support'] for a,b in inc),'increases_without_posture':sum(not b['posture'] for a,b in inc),'increases_without_toe_open':sum(not b['open'] for a,b in inc),'first_lift_open_streak':first['streak'] if first else None}
(A/'stage-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
