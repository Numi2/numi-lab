from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-corrected-wing-trim-screen-20261110')
for name in ['raise_026','raise_028','raise_030']:
 e=json.loads((A/(name+'-analysis.json')).read_text())['first_episode'];samples=[]
 for l in gzip.open(A/(name+'.stderr.gz'),'rt'):
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if 1000<=x['step']<=1500 and x['step']<=e['steps_by_environment'][x['lane']]:samples.append(x)
 lanes=[]
 for lane in range(16):
  mode=lane%5;rate=[0,.03,-.03,0,0][mode];target=[.85,.85,.85,1.05,.65][mode];rows=[]
  for x in samples:
   if x['lane']!=lane:continue
   q=x['q'];v=x['v'];a,b,c,w=q[3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));desired=rate*(x['step']*.02-6);rows.append([abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))),abs(-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]),abs(math.cos(yaw)*v[0]+math.sin(yaw)*v[1]-.7),abs(q[2]-target),q[2],math.cos(yaw)*v[0]+math.sin(yaw)*v[1]])
  m=[max((r[i] for r in rows),default=None) for i in range(4)];ok=len(rows)==6 and not e['physical_failure_by_environment'][lane] and all(x<=lim for x,lim in zip(m,[.15,.20,.20,.15]));lanes.append({'lane':lane,'mode':mode,'max_heading_lateral_speed_height_errors':m,'height_range':[min(r[4] for r in rows),max(r[4] for r in rows)] if rows else None,'speed_range':[min(r[5] for r in rows),max(r[5] for r in rows)] if rows else None,'pass':ok})
 out={'lanes':lanes,'pass_count':sum(x['pass'] for x in lanes),'limits':[.15,.20,.20,.15],'scope':'Exploratory tracking screen, not a qualification.'};(A/(name+'-metrics.json')).write_text(json.dumps(out,indent=2)+'\n');print(name,'passes',out['pass_count'],'lane0',lanes[0])
