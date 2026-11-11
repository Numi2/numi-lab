from pathlib import Path
import gzip,heapq,json
D=Path("artifacts/crow-v15-walk-impact-comparison-20261111")
p=Path("artifacts/crow-v15-autonomous-sustained-walk-control-20261111/autonomous-20261126/run.stderr.gz");by={}
with gzip.open(p,"rt") as f:
 for line in f:
  if not line.startswith("CROW_TWO_TOE_SUBSTEP "):continue
  x=line.split();step,sub,lane=map(int,x[1:4])
  if 210<=step<=300 and lane in (0,3,6):
   y=list(map(float,x[6:]));by.setdefault(lane,[]).append((y[0]+y[1],step,sub,y[0],y[1]))
result={"source":str(p),"window_steps":[210,300],"nominal_weight_impulse_Ns":.45006*9.81*.02/16,"rows":[{"lane":lane,"peak_upward_impulses":[{"total_Ns":a,"step":t,"substep":k,"left_Ns":l,"right_Ns":r} for a,t,k,l,r in heapq.nlargest(5,rows)]} for lane,rows in sorted(by.items())],"limitation":"Selected lanes and time window on a used development start. Correlation of impulse with motion does not prove the cause of instability or establish a solver defect."}
(D/"analysis.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
