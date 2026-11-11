from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-uniform-release-perturbation-20261111");wanted={0,29,59,89,119,129,134,139,149};data={}
with gzip.open(A/"high/trace.log.gz","rt") as f:
 for line in f:
  a=line.split()
  if not a or a[0] not in ("step","event"):continue
  e,t=int(a[1]),int(a[2])
  if e not in (5,6,7) or t not in wanted:continue
  d=data.setdefault((e,t),{})
  if a[0]=="step":
   q=list(map(float,a[a.index("q")+1:a.index("v")]));v=list(map(float,a[a.index("v")+1:]));d.update(root_x_m=q[0],root_z_m=q[2],root_vx_m_s=v[0])
  else:d.update(release=bool(int(a[3])),support_streak=int(a[4]),left_up_Ns=float(a[5]),right_up_Ns=float(a[6]),left_front_normal_x_Ns=float(a[7]),left_rear_normal_x_Ns=float(a[8]),right_front_normal_x_Ns=float(a[9]),right_rear_normal_x_Ns=float(a[10]))
out={"rows":[{"lane":e,"interval_zero_based":t,**v} for (e,t),v in sorted(data.items())],"limits":"Native control-boundary last-substep signals are descriptive, not complete-substep geometry; lane 5 passed, lanes 6 and 7 fell before release."};(A/"first-fall-diagnostic.json").write_text(json.dumps(out,indent=2)+"\n")
