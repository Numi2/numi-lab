from pathlib import Path
import gzip,json,math,sys,hashlib
A=Path('artifacts/crow-v15-corrected-reset-flight-20261114');folder,reference=sys.argv[1:3];p=A/folder;turn={(x['lane'],x['step']):x for x in json.load(gzip.open(reference,'rt'))};straight={}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=(x['lane'],x['step']);assert k not in straight;straight[k]=x
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
rows=[]
for e in (6,7):
 bad=[t for t in range(1,601) if (e,t) not in turn or (e,t) not in straight or any(turn[e,t][k]!=straight[e,t][k] for k in ('q','v','policy_action','contacts'))];missing=[t for t in range(600,1201) if (e,t) not in turn or (e,t) not in straight]
 vals=[(t,wrap(yaw(turn[e,t]['q'])-yaw(straight[e,t]['q'])),turn[e,t]['q'][1]-straight[e,t]['q'][1]) for t in range(600,1201) if (e,t) in turn and (e,t) in straight];extreme=(min(vals,key=lambda x:x[1]) if e==6 else max(vals,key=lambda x:x[1])) if vals else None
 rows.append({'lane':e,'parity_through_600_mismatches':len(bad),'first_mismatches':bad[:5],'missing_comparison_steps':len(missing),'signed_yaw_extreme_tick_rad_and_lateral_m':extreme,'signed_yaw_gate':extreme is not None and (extreme[1]<=-.20 if e==6 else extreme[1]>=.20),'max_abs_lateral_deviation_m':max((abs(x[2]) for x in vals),default=None)})
out={'rows':rows,'reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),'passed':all(x['parity_through_600_mismatches']==0 and x['missing_comparison_steps']==0 and x['signed_yaw_gate'] for x in rows),'scope':'matched same-seed scripted control; not disturbed-start, biological or mission qualification'};(p/'matched-turn-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
