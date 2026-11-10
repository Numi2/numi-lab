from pathlib import Path
import json
A=Path("artifacts/crow-v15-free-toe-force-observer-20261110");O=Path("artifacts/crow-v15-free-toe-impulse-20261110");old={};data={};outcomes={}
for line in (O/"results.txt").read_text().splitlines():
 v=line.split()
 if v and v[0]=="trace" and int(v[2])==0:old[(int(v[1]),int(v[3]))]=v[4:]
for line in (A/"results.txt").read_text().splitlines():
 v=line.split()
 if not v:continue
 if v[0]=="trace":
  c,a,t,n=map(int,v[1:5]);p=float(v[5]);q=list(map(float,v[6:14]));vel=list(map(float,v[14:22]));data.setdefault((c,a),[]).append((t,n,p,q,vel,v[4:5]+v[6:]))
 elif v[0]=="case_result":outcomes[(int(v[1]),int(v[2]))]=" ".join(v[3:])
diffs=[(c,r[0]) for c in range(4) for r in data.get((c,0),[]) if old.get((c,r[0]))!=r[5]];rows=[]
for c in range(4):
 for a in range(4):
  z=data.get((c,a),[]);tail=[r for r in z if r[0]>=25];row={"case":c,"arm":a,"outcome":outcomes.get((c,a)),"samples":len(z),"max_candidates":max((r[1] for r in z),default=-1),"max_observed_normal_impulse":max((r[2] for r in z),default=None),"max_angle_after_25":max((abs(x) for r in tail for x in r[3]),default=None)};row["pass"]=not diffs and len(z)==100 and row["max_observed_normal_impulse"]<=1e-8 and row["max_angle_after_25"]<.05;rows.append(row)
(A/"analysis.json").write_text(json.dumps({"baseline_parity_exact":not diffs and all(len(data.get((c,0),[]))==100 for c in range(4)),"differences":diffs[:20],"rows":rows},indent=2)+"\n")
print("baseline parity",not diffs and all(len(data.get((c,0),[]))==100 for c in range(4)),"differences",len(diffs))
for a in range(4):
 z=[r for r in rows if r["arm"]==a];print("arm",a,"pass",sum(r["pass"] for r in z),"max observed impulse",max(r["max_observed_normal_impulse"] for r in z),"max angle",max(r["max_angle_after_25"] for r in z))
