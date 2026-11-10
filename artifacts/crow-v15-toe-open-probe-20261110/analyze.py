from pathlib import Path
import json,statistics
A=Path('artifacts/crow-v15-toe-open-probe-20261110');frames={0:{},1:{}};contacts={0:{},1:{}};states={}
for line in (A/'trace.log').read_text().splitlines():
 p=line.split()
 if p[0]=='trace':
  arm,t=int(p[1]),int(p[2]);v=list(map(float,p[3:]));frames[arm][t]=v
 elif p[0]=='contact':
  _,_,arm,t,name,imp,sep,nx,nz=p;arm,t=int(arm),int(t);contacts[arm].setdefault(t,[]).append((name,float(imp),float(sep),float(nx),float(nz)))
 elif p[0]=='case_result':states[int(p[2])]=' '.join(p[3:])
out={}
for arm in (0,1):
 window=range(120,160);angles=[];bilat=opposed=nonfoot=0;rear=[];body=[]
 for t in window:
  fs=frames[arm][t];angles.extend(abs(z) for z in fs[13:21]);body.append(fs[:3]);arr=contacts[arm].get(t,[]);groups={side:{'foot':0.,'front':0.,'rear':0.} for side in ('left','right')}
  for name,imp,sep,nx,nz in arr:
   side=next((s for s in groups if '_'+s+'_' in name),None)
   if side and ('_foot' in name or '_toe' in name):
    if nz<-.5:groups[side]['foot']+=imp*(-nz)
    if 'front_toe' in name and nx<-.5:groups[side]['front']+=imp
    if 'rear_toe' in name and nx>.5:groups[side]['rear']+=imp
   elif imp>1e-8:nonfoot+=1
  bilat+=all(groups[s]['foot']>.001 for s in groups);opposed+=all(groups[s][g]>.00005 for s in groups for g in ('front','rear'));rear.append(sum(groups[s]['rear'] for s in groups))
 out[str(arm)]={'mode':'hold' if arm==0 else 'ramp_to_zero','status':states.get(arm),'samples':len(frames[arm]),'last_40_mean_absolute_toe_angle_rad':statistics.mean(angles),'bilateral_upward_impulse_samples':bilat,'opposed_positive_impulse_samples':opposed,'mean_rear_opposing_impulse_Ns':statistics.mean(rear),'nonfoot_positive_contacts':nonfoot,'body_x_range_m':[min(x[0] for x in body),max(x[0] for x in body)],'body_z_range_m':[min(x[2] for x in body),max(x[2] for x in body)]}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
