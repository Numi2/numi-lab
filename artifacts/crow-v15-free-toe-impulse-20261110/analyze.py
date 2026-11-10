from pathlib import Path
import json
A=Path("artifacts/crow-v15-free-toe-impulse-20261110");data={};outcomes={}
for line in (A/"results.txt").read_text().splitlines():
 v=line.split()
 if not v:continue
 if v[0]=="trace":
  c,a,t,n=map(int,v[1:5]);q=list(map(float,v[5:13]));vel=list(map(float,v[13:21]));data.setdefault((c,a),[]).append((t,n,q,vel))
 elif v[0]=="case_result":outcomes[(int(v[1]),int(v[2]))]=" ".join(v[3:])
rows=[]
for c in range(4):
 for a in range(4):
  z=data.get((c,a),[]);tail=[r for r in z if r[0]>=25];row={"case":c,"arm":a,"outcome":outcomes.get((c,a)),"samples":len(z),"max_contacts":max((r[1] for r in z),default=-1),"max_angle_after_25":max((abs(x) for r in tail for x in r[2]),default=None),"max_speed_after_25":max((abs(x) for r in tail for x in r[3]),default=None)};row["pass"]=len(z)==100 and row["max_contacts"]==0 and row["max_angle_after_25"]<.05;rows.append(row)
(A/"analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
for a in range(4):
 z=[r for r in rows if r["arm"]==a];print("arm",a,"pass",sum(r["pass"] for r in z),"max contacts",max(r["max_contacts"] for r in z),"max angle",max(r["max_angle_after_25"] for r in z))
