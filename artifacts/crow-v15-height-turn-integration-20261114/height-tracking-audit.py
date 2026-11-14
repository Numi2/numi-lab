from pathlib import Path
import gzip,hashlib,json,statistics
A=Path('artifacts/crow-v15-height-turn-integration-20261114');p=A/'turn-20261126-rebound-turn/run.stderr.gz';state={};duplicates=[]
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
   if k in state:duplicates.append(k)
   state[k]=x
rows=[]
for e in (6,7):
 first,final=(.55,.80) if e==6 else (.80,.55);windows=[]
 for lo,hi,goal in ((1000,1150,first),(1550,1800,final)):
  vals=[state[e,t]['q'][2] for t in range(lo,hi+1) if (e,t) in state]
  windows.append({'ticks':[lo,hi],'target_m':goal,'samples':len(vals),'mean_m':statistics.mean(vals),'mean_absolute_error_m':statistics.mean(abs(v-goal) for v in vals),'max_absolute_error_m':max(abs(v-goal) for v in vals)})
 peak=max(((state[e,t]['q'][2],t) for t in range(450,1801) if (e,t) in state));rows.append({'lane':e,'peak_height_m_and_step':peak,'windows':windows,'mean_height_change_m':windows[1]['mean_m']-windows[0]['mean_m'],'satisfies_straight_switch_error_gate':all(w['mean_absolute_error_m']<=.05 for w in windows)})
out={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows,'duplicate_states':duplicates,'scope':'retrospective application of straight-flight height windows to turn integration; development seed'}
(A/'height-tracking-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
