from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-release-perturbation-screen-20261112")
C=A/"release";B=A/"control"
def traces(path):
 states={};trigger={}
 with gzip.open(path/"trace.log.gz","rt") as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=="step":
    e,t=int(z[1]),int(z[2])+1;i=z.index("q");j=z.index("v");states[e,t]=(z[i+1:j],z[j+1:])
   if z[0]=="event":
    e,t=int(z[1]),int(z[2])+1
    if int(z[4])>=50 and e not in trigger:trigger[e]=t
 return states,trigger
cs,trig=traces(C);bs,_=traces(B)
cut={e:trig.get(e,550) for e in range(8)}
def sub(path,lanes,cuts):
 d={}
 with gzip.open(path/"substeps.log.gz","rt") as f:
  for line in f:
   z=line.split()
   if z and z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4])
    if e in lanes and t<=cuts[e]:
     assert (e,t,k) not in d
     d[e,t,k]=z[4:]
 return d
cc=sub(C,set(range(8)),cut)
maxcut=max(cut.values());bc=sub(B,set(range(8)),{e:maxcut for e in range(8)})
rows=[]
for e in range(8):
 b=e;n=cut[e]
 state_bad=[t for t in range(1,n+1) if cs.get((e,t))!=bs.get((b,t))]
 contact_bad=[(t,k) for t in range(1,n+1) for k in range(32) if cc.get((e,t,k)) is None or cc.get((e,t,k))!=bc.get((b,t,k))]
 rows.append({"lane":e,"baseline_lane":b,"last_pre_shift_interval":n,"states_compared":n,"contacts_compared":n*32,"state_mismatch_examples":state_bad[:3],"contact_mismatch_examples":contact_bad[:3],"pass":not state_bad and not contact_bad})
out={"rows":rows,"pass":all(x["pass"] for x in rows),"scope":"exact pre-shift comparison; controller switch follows support streak 50"}
(A/"prefix-parity.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
if not out["pass"]:raise SystemExit("prefix mismatch")
