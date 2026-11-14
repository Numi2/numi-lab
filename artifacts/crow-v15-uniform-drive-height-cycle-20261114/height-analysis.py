from pathlib import Path
import gzip,json,math,statistics,sys
A=Path('artifacts/crow-v15-uniform-drive-height-cycle-20261114');P=A/sys.argv[1];states={};errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
   if k in states:errors.append('duplicate '+str(k))
   states[k]=x
base=json.loads((P/'analysis.json').read_text());rows=[]
for e in range(8):
 windows=[]
 for lo,hi,goal in ((650,780,.55),(1050,1200,.80),(1650,1800,.55)):
  vals=[states[e,t]['q'][2] for t in range(lo,hi+1) if (e,t) in states]
  ok=len(vals)==hi-lo+1 and all(math.isfinite(x) for x in vals)
  windows.append({'intervals':[lo,hi],'target_m':goal,'samples':len(vals),'mean_m':statistics.mean(vals) if ok else None,'mean_absolute_error_m':statistics.mean(abs(x-goal) for x in vals) if ok else None})
 valid=all(w['mean_m'] is not None for w in windows);up=windows[1]['mean_m']-windows[0]['mean_m'] if valid else None;down=windows[2]['mean_m']-windows[1]['mean_m'] if valid else None
 good=valid and windows[1]['mean_absolute_error_m']<=.05 and windows[2]['mean_absolute_error_m']<=.05 and up>=.12 and down<=-.12
 rows.append({'lane':e,'windows':windows,'climb_mean_m':up,'descent_mean_m':down,'passed':good})
out={'rows':rows,'base_gates_pass':base['passed_all_eight'],'duplicate_state_errors':errors,'passed_cycle':not errors and base['passed_all_eight'] and all(x['passed'] for x in rows),'scope':'used seed, fixed within-episode command; not general flight or landing qualification'}
(P/'height-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
