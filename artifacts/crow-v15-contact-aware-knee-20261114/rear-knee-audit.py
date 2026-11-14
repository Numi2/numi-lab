from pathlib import Path
import gzip,json
A=Path(__file__).resolve().parent;stage={};events={};knee={};sensors={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0] not in ('stage','event','rearKnee','clearanceSensors'):continue
  k=(int(x[1]),int(x[2]));v=list(map(float,x[3:]));{'stage':stage,'event':events,'rearKnee':knee,'clearanceSensors':sensors}[x[0]][k]=v
bad=[];active=False;interventions=[]
for t in range(1,450):
 s=stage[1,t];prev=stage[1,t-1];ev=events[1,t-1];old=knee[1,t-1][0];now=knee[1,t][0];foot,rear=sensors[1,t-1];active |= bool(prev[4]) and bool(ev[0]);gate=active and bool(s[0]) and bool(s[1]) and bool(s[2]);target=old
 if gate:
  if foot<.002 and old>.13:target=max(.13,old-.004)
  elif ev[10]<.003:
   if rear<.0025 and foot>.003:target=min(.15,old+.004);interventions.append(t)
   else:target=max(old,min(.13,old+.004))
  elif ev[10]>.005:target=max(0,old-.004)
 if abs(now-target)>2e-7:bad.append([t,target,now])
vals=[v[0] for k,v in knee.items() if k[0]==1];out={'mismatches':bad,'intervention_ticks':interventions,'max_correction_rad':max(vals),'pass':not bad and bool(interventions) and max(vals)<=.150001,'scope':'reconstructs feedback from logged previous accepted-state geometry; not a physical range sensor'};(A/'rear-knee-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
