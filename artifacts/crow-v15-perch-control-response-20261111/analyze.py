from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-perch-control-response-20261111");B=Path("artifacts/crow-v15-matched-balance-grip-screen-20261111/results.txt.gz")
def load(p):
 trace={};response={};contacts={}
 with gzip.open(p,"rt") as f:
  for line in f:
   v=line.split()
   if not v:continue
   if v[0]=="trace":trace[int(v[3])]=v[4:]
   elif v[0]=="response":response[int(v[1])]=list(map(float,v[2:]))
   elif v[0]=="interval_contact":contacts[int(v[3]),int(v[4])]=float(v[5])
 return trace,response,contacts
runs=[load(A/f"pulse-{i}.txt.gz") for i in range(5)];old=load(B)[0]
diff=[(i,t) for i,(tr,_,_) in enumerate(runs) for t in range(100) if tr.get(t)!=old.get(t)]
assert not diff and all(len(t)==len(r)==240 for t,r,c in runs),diff[:20]
out={"exact_prefix_samples_per_run":100,"runs":{}}
for i,(trace,response,contacts) in enumerate(runs):
 grip=[t for t in range(240) if all((contacts.get((t,k),0)>.0002 if k in (3,7) else sum(contacts.get((t,z),0) for z in ((0,1,2) if k==0 else (4,5,6)))>.0002) for k in (0,3,4,7))]
 vals={}
 for t in (99,129,150,180,200,239):
  v=response[t];base=runs[0][1][t];vals[str(t)]={"pitch_rad":v[0],"pitch_rate":v[1],"com_x":v[2],"delta_from_control":[v[j]-base[j] for j in range(3)]}
 out["runs"][str(i)]={"bilateral_grip_ticks":len(grip),"post_pulse_grip_ticks":sum(t>=130 for t in grip),"samples":vals}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
