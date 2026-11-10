from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-toe-launch-control-20261110');rows=[];lanes=[{} for _ in range(8)]
for line in gzip.open(A/'control.stderr.gz','rt'):
 if line.startswith('CROW_SUPPORT_SAMPLE '):
  d=json.loads(line.split(' ',1)[1]);lanes[d['lane']][d['step']]=d
for i,lane in enumerate(lanes):
 assert all(t in lane for t in range(100,201));seq=[];bad=[];maxroll=maxpitch=0.
 for t in range(100,201):
  d=lane[t];force={'left':0.,'right':0.}
  for c in d['contacts']:
   a,b=c['a_name'],c['b_name'];imp=c['impulses'][0]
   if 'locomotion_ground' not in (a,b) or imp<=1e-7:continue
   bird=b if a=='locomotion_ground' else a
   if bird in ('crow_left_foot','crow_right_foot') or '_toe' in bird:
    side='left' if '_left_' in bird else 'right';force[side]+=imp
   else:bad.append({'step':t,'body':bird})
  if max(force.values())>0:seq.append('left' if force['left']>force['right'] else 'right')
  else:seq.append('none')
  x,y,z,w=d['q'][3:7];maxpitch=max(maxpitch,abs(math.asin(max(-1,min(1,2*(w*y-z*x))))));maxroll=max(maxroll,abs(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))))
 changes=sum(a!=b for a,b in zip(seq,seq[1:]) if a!='none' and b!='none');start,end=lane[100]['q'],lane[200]['q'];rows.append({'lane':i,'net_forward_m':end[0]-start[0],'net_lateral_m':end[1]-start[1],'max_abs_roll_deg':math.degrees(maxroll),'max_abs_pitch_deg':math.degrees(maxpitch),'left_dominant_endpoints':seq.count('left'),'right_dominant_endpoints':seq.count('right'),'unloaded_endpoints':seq.count('none'),'dominant_load_changes':changes,'nonfoot_ground_contacts':bad})
result={'window':'2.00 to 4.00 seconds, 101 accepted endpoints','lanes':rows,'scope':'contact-load and body-displacement observations only; load changes alone do not establish a natural gait or foot-contact-point slip'};(A/'walk-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(r['lane'],'forward_m',round(r['net_forward_m'],4),'left/right',r['left_dominant_endpoints'],r['right_dominant_endpoints'],'changes',r['dominant_load_changes'],'nonfoot',len(r['nonfoot_ground_contacts']))
