from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-pitch-gain-clearance-20261114');stage={};events={};knee={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  key=(int(x[1]),int(x[2])) if x[0] in ('stage','event','rearKnee') else None
  if x[0]=='stage':stage[key]=list(map(float,x[3:]))
  elif x[0]=='event':events[key]=list(map(float,x[3:]))
  elif x[0]=='rearKnee':knee[key]=float(x[3])
active=False;bad=[];maxstep=0;samples=0
for t in range(1,450):
 s=stage[1,t];prev=stage[1,t-1];event=events[1,t-1];active |= bool(prev[4]) and bool(event[0]);old=knee[1,t-1];now=knee[1,t];gate=active and bool(s[0]) and bool(s[1]) and bool(s[2]);clearance=event[10];target=old
 if gate and clearance<.003:target=min(.13,old+.004)
 elif gate and clearance>.005:target=max(0,old-.004)
 if abs(now-target)>2e-7:bad.append({'tick':t,'expected':target,'actual':now,'gate':gate,'clearance':clearance})
 maxstep=max(maxstep,abs(now-old));samples+=active
out={'active_samples':samples,'max_correction_rad':max(knee.values()),'min_correction_rad':min(knee.values()),'max_step_rad':maxstep,'mismatches':bad,'pass':samples>0 and not bad and max(knee.values())<=.130001 and maxstep<=.004001,'note':'Uses prior observed clearance/contact; state/contact epochs not independently synchronized.'};(A/'rear-knee-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
