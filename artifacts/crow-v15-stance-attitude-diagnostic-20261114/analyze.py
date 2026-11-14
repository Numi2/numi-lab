from pathlib import Path
import gzip,hashlib,json,math
A=Path(__file__).resolve().parent;src=A.parent/'crow-v15-combined-hip28-pitch14-20261114/trace.log.gz';ticks={170,200,230,250,260,270,280,285,290,295,300,305,310,315,320};rows={}
with gzip.open(src,'rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0] not in ('step','balance','pitchcontrol') or x[1]!='1' or int(x[2]) not in ticks:continue
  t=int(x[2]);r=rows.setdefault(t,{'zero_based_tick':t})
  if x[0]=='step':
   q=list(map(float,x[x.index('q')+1:x.index('v')]));a,b,c,w=q[3:7];r.update({'roll_rad':math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)),'pitch_rad':math.asin(max(-1.,min(1.,2*(w*b-c*a)))),'left_hip_roll_rad':q[39],'right_hip_roll_rad':q[40]})
  elif x[0]=='balance':r['balance_observation']=list(map(float,x[3:]))
  else:r['pitch_control_observation']=list(map(float,x[3:]))
h=hashlib.sha256(src.read_bytes()).hexdigest();out={'source_sha256':h,'rows':[rows[t] for t in sorted(rows)],'scope':'accepted-state attitude and logged previous-state feedback; association, not a causal or substep-effort measurement'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
