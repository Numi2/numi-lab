from pathlib import Path
import gzip,json,subprocess,statistics
A=Path("artifacts/crow-v15-corvid-takeoff-force-comparison-20261112")
C=Path("artifacts/crow-v15-coordinated-takeoff-screen-20261112/autonomous-20261126")
R=Path("artifacts/crow-v15-autonomous-walk-hold-journey-20261111/autonomous-20261126")
m=json.loads((R/"stderr-parts.json").read_text());out={}
for label,p in (("candidate",C),("baseline",R)):
 proc=subprocess.Popen(["cat",*[str(p/x["name"]) for x in m["parts"]]],stdout=subprocess.PIPE) if label=="baseline" else None
 sub={}
 with gzip.open(proc.stdout if proc else p/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if 340<=t<=360:sub[e,t,k]=list(map(float,z[6:]))
 if proc:assert proc.wait()==0
 assert len(sub)==8*21*16
 rows=[];bw=.45006*9.81;dt=.00125
 for e in range(8):
  x=[sub[e,t,k] for t in range(340,361) for k in range(16)];loaded=[i for i,z in enumerate(x) if z[2]+z[3]>1e-7];assert loaded
  feet=[[z[j]/dt/bw for z in x] for j in (0,1)]
  def smooth(v):return [sum(v[i:i+20])/20 for i in range(len(v)-19)]
  rows.append({"lane":e,"per_foot_unfiltered_peak_vertical_bodyweights":[max(v) for v in feet],"per_foot_25ms_average_peak_vertical_bodyweights":[max(smooth(v)) for v in feet],"combined_unfiltered_peak_vertical_bodyweights":max(a+b for a,b in zip(*feet)),"combined_upward_impulse_Ns":sum(z[0]+z[1] for z in x),"last_loaded_step_substep":[340+loaded[-1]//16,loaded[-1]%16],"samples":len(x)})
 out[label]=rows
out["limits"]="Solver substep impulse over 1.25ms, upward component and 25ms box-average, not a 40Hz zero-phase-filtered force-plate vector. No direct biological pass/fail or muscle-force inference. Used seed and one model, paired starts, no population claim."
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
