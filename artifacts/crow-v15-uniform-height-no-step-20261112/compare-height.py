from pathlib import Path
import json
A=Path("artifacts/crow-v15-uniform-height-no-step-20261112");B=Path("artifacts/crow-v15-uniform-height-screen-20261112")
a=json.loads((A/"analysis.json").read_text());b=json.loads((B/"analysis.json").read_text());rows=[]
for x,y in zip(a,b):
 out={"lane":x["lane"],"pulse_target_m":y["pulse_target_m"],"no_step_failure":x["failure"],"pulse_failure":y["failure"]}
 for phase in ("before","pulse","recovery"):
  if x[phase] and y[phase]:out[phase]={k:y[phase][k]-x[phase][k] for k in ("height_m","vertical_speed_m_s","forward_speed_m_s","yaw_rad")}
 rows.append(out)
(A/"height-differences.json").write_text(json.dumps(rows,indent=2)+"\n")
for r in rows: print(r)
