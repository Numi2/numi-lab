from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-isolated-grip-revised-body-20261110');trace={};contacts={};outcomes={}
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='trace':trace[(int(v[1]),int(v[2]),int(v[3]))]={'phase':int(v[4]),'front_target':float(v[5]),'rear_target':float(v[6]),'up_ns':float(v[9]),'x':float(v[10]),'z':float(v[11]),'vx':float(v[12]),'qx':float(v[13]),'qy':float(v[14]),'vy':float(v[15]),'vz':float(v[16]),'toes':list(map(float,v[17:25]))}
  elif v[0]=='contact':
   i,arm,t=int(v[1]),int(v[2]),int(v[3]);side='left' if '_left_' in v[4] else 'right';part='rear' if 'rear_toe' in v[4] else 'front';nx=float(v[5]);nz=float(v[6]);imp=float(v[7]);k=(i,arm,t);z=contacts.setdefault(k,{s:{'front':0.,'rear':0.,'up':0.} for s in ('left','right')})
   if part=='front' and nx<-.5:z[side]['front']+=imp
   if part=='rear' and nx>.5:z[side]['rear']+=imp
   if nz<-.5:z[side]['up']+=-nz*imp
  elif v[0]=='case_result':outcomes[(int(v[1]),int(v[2]))]=' '.join(v[3:])
def longest(xs):
 best=cur=0
 for x in xs:cur=cur+1 if x else 0;best=max(best,cur)
 return best
rows=[]
for i in range(8):
 for arm in range(4):
  vals=[]
  for t in range(150):
   q=trace.get((i,arm,t));x=contacts.get((i,arm,t),{s:{'front':0.,'rear':0.,'up':0.} for s in ('left','right')})
   if q is None:continue
   grip=all(x[s][k]>.00005 for s in x for k in ('front','rear'));tilt=math.acos(max(-1.,min(1.,1-2*(q['qx']**2+q['qy']**2))));speed=math.sqrt(q['vx']**2+q['vy']**2+q['vz']**2);loaded=all(x[s]['up']>.00005 for s in x) and sum(x[s]['up'] for s in x)>=.5*.45004*9.81*.005 and tilt<=.35 and speed<=.12
   vals.append((t,grip,loaded,q['phase']))
  grip_steps=[t for t,g,_,_ in vals if g];release_steps=[t for t,g,_,ph in vals if ph==4 and not g];rows.append({'case':i,'arm':arm,'geometry':'original_5x11cm' if arm<2 else 'square_6x6cm','closure_enabled':bool(arm%2),'outcome':outcomes.get((i,arm),'missing'),'steps_observed':len(vals),'longest_bilateral_opposed_run':longest([g for _,g,_,_ in vals]),'longest_bilateral_toe_support_run':longest([x for _,_,x,_ in vals]),'last25_bilateral_toe_support':len(vals)>=25 and all(x for _,_,x,_ in vals[-25:]),'first_opposed_step':grip_steps[0] if grip_steps else None,'first_absent_opposed_after_release':release_steps[0] if release_steps else None})
out={'rows':rows,'scope':'toe-only contacts; this probe does not record nonfoot contacts and therefore cannot verify the full supported landing criterion','opposing_threshold_ns':.00005,'support_upward_impulse_threshold_ns':.5*.45004*9.81*.005};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for arm in range(4):
 z=[r for r in rows if r['arm']==arm];print(arm,'completed',sum(r['steps_observed']==150 for r in z),'opposed 3+',sum(r['longest_bilateral_opposed_run']>=3 for r in z),'support 25+',sum(r['longest_bilateral_toe_support_run']>=25 for r in z),'terminal toe support',sum(r['last25_bilateral_toe_support'] for r in z))
