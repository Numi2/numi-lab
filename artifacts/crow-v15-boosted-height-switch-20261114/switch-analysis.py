from pathlib import Path
import gzip,json,math,statistics,sys
A=Path('artifacts/crow-v15-boosted-height-switch-20261114');P=A/sys.argv[1];meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];states={};targets={};errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step']
   if k in states:errors.append('duplicate state '+str(k))
   states[k]=x
  elif line.startswith('CROW_DRIVE_TARGET '):
   z=line.split();t,e=map(int,z[1:3]);k=e,t
   if k in targets:errors.append('duplicate target '+str(k))
   targets[k]=list(map(float,z[3:]))
rows=[]
for e in (6,7):
 first,final=(.55,.80) if e==6 else (.80,.55);windows=[]
 for lo,hi,goal in ((1000,1150,first),(1550,1800,final)):
  vals=[states[e,t]['q'][2] for t in range(lo,hi+1) if (e,t) in states];ok=len(vals)==hi-lo+1 and not meta['physical_failure_by_environment'][e]
  windows.append({'ticks':[lo,hi],'target_m':goal,'samples':len(vals),'mean_m':statistics.mean(vals) if ok else None,'mean_absolute_error_m':statistics.mean(abs(v-goal) for v in vals) if ok else None})
 errs=[];expected=0
 for t in range(1,min(1800,meta['steps_by_environment'][e])+1):
  sec=(t-2)*.02
  if 24.5<=sec<=35.0:
   expected+=1;a=states.get((e,t-1));b=targets.get((e,t))
   if a is not None and b is not None:
    delta=max(-.009,min(.006,.03*(final-a['q'][2])-.003));errs.append(abs((b[8]+b[9])/2-.35-(.70+delta)*math.sin(2*math.pi*6.4*(sec-7.42))))
 valid=all(w['mean_m'] is not None for w in windows);change=windows[1]['mean_m']-windows[0]['mean_m'] if valid else None
 rows.append({'lane':e,'windows':windows,'height_change_m':change,'late_target_expected_samples':expected,'late_target_samples':len(errs),'late_target_max_error':max(errs,default=None),'passed':valid and all(w['mean_absolute_error_m']<=.05 for w in windows) and (change>=.12 if e==6 else change<=-.12) and len(errs)==expected and expected>0 and max(errs)<=.0003})
base=json.loads((P/'analysis.json').read_text());out={'rows':rows,'base_physical_checks_pass':all(r['pass'] for r in base['rows']),'no_duplicate_records':not errors,'errors':errors,'passed_switch_screen':all(r['passed'] for r in rows) and all(r['pass'] for r in base['rows']) and not errors,'scope':'Used seed within-episode height switches; not qualification of general flight or landing.'};(P/'switch-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
