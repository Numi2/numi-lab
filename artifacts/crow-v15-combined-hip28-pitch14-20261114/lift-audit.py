from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-combined-hip28-pitch14-20261114');stage={};events={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='stage':stage[int(x[1]),int(x[2])]=list(map(float,x[3:]))
  elif x[0]=='event':events[int(x[1]),int(x[2])]=list(map(float,x[3:]))
active=False;bad=[];samples=0;max_step=0
for t in range(1,450):
 s=stage[1,t];prev=stage[1,t-1];event=events[1,t-1];active |= bool(prev[4]) and bool(event[0])
 if not active:continue
 samples+=1;old=prev[5];now=s[5];gate=bool(s[0]) and bool(s[1]) and bool(s[2]);clearance=event[10];target=old
 if gate and clearance<.003:target=min(.28,old+.005)
 elif gate and clearance>.005:target=max(0,old-.005)
 max_step=max(max_step,abs(now-old))
 if abs(now-target)>2e-7:bad.append({'tick':t,'expected':target,'actual':now,'gate':gate,'clearance':clearance})
out={'active_samples':samples,'max_abs_step_rad':max_step,'mismatches':bad,'pass':samples>0 and not bad,'note':'Checks feedback against previous recorded contact/clearance observations; not synchronized physical contact.'};(A/'lift-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
