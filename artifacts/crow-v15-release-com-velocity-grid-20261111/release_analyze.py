from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-release-com-velocity-grid-20261111');tr={};c={};toes={}
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='trace':tr[(int(v[1]),int(v[3]))]=v[4:]
  elif v[0]=='interval_contact':toes.setdefault((int(v[1]),int(v[3])),[]).append(float(v[5]))
  elif v[0]=='allcontact':
   i,t=int(v[1]),int(v[3]);an,bn=v[4:6];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;up=max((1 if body==bn else -1)*nz*imp,0);d=c.setdefault((i,t),{'left':0.,'right':0.,'other':0.});side=next((x for x in ('left','right') if '_'+x+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):d[side]+=up
   else:d['other']+=imp
rows=[];geom={r['case']:r for r in json.loads((A/'analysis.json').read_text())['rows']}
for i in range(1,17):
 rec=[]
 for t in range(1100):
  v=tr.get((i,t));
  if not v:continue
  qx,qy=float(v[9]),float(v[10]);tilt=math.acos(max(-1.,min(1.,1-2*(qx*qx+qy*qy))));vel=[float(v[8]),float(v[11]),float(v[12])];speed=math.sqrt(sum(z*z for z in vel));d=c.get((i,t),{'left':0.,'right':0.,'other':0.});support=min(d['left'],d['right'])>.0002/16 and d['left']+d['right']>=.5*.45004*9.81*.02/16 and d['other']<=1e-7 and tilt<=.35 and speed<=.12;imp=toes.get((i,t),[]);opened=len(imp)==8 and max(imp)<=1e-7 and max(abs(float(x)) for x in v[13:21])<.05;rec.append((t,int(v[0]),support,opened))
 tail=rec[-50:];g=geom[i];ok=len(rec)==1100 and len(tail)==50 and all(z[1]==4 and z[2] and z[3] for z in tail) and g['missing_grip_pose_samples']==0 and g['minimum_grip_sat_gap_m'] is not None and g['minimum_grip_sat_gap_m']>=-.0005;rows.append({'lane':i,'first_open_supported':next((z[0] for z in rec if z[1]==4 and z[2] and z[3]),None),'terminal_50_open_supported':sum(z[1]==4 and z[2] and z[3] for z in tail),'grip_gap_m':g['minimum_grip_sat_gap_m'],'strict_pass':ok})
(A/'release-analysis.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
