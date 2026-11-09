#!/usr/bin/env python3
"""Compare template-inverse cell-colored FP64 reference with frozen ordered evaluator."""
import hashlib,json,pathlib,shutil,subprocess
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
old=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-n128-001/numi_contact_fem_fp64_reference_n128")
new=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-template-001/numi_contact_fem_fp64_reference_template")
src=root/"tests/contact/fem_plane_fp64_reference_compact.cpp"
out=p/"template-reference-trajectory-parity-001.json"
if out.exists():raise SystemExit("attempt already exists")
include=root/"tests/contact/fem_plane_fp64_reference.cpp"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (old,new,src,include)}
rows=[]
for n in (2,8):
 if shutil.disk_usage(p).free<3.5*1024**3:raise SystemExit("free-space guard")
 runs=[]
 for name,exe in (("ordered",old),("compact",new)):
  command=[str(exe),str(n),"0.000002","0.01"]
  try:r=subprocess.run(command,capture_output=True,text=True,timeout=600)
  except subprocess.TimeoutExpired:raise SystemExit("parity run timed out")
  if r.returncode:raise SystemExit(name+" failed: "+r.stderr)
  runs.append(json.loads(r.stdout))
  path=p/("template-reference-parity-n"+str(n)+"-"+name+"-001.json")
  if path.exists():raise SystemExit("attempt exists")
  path.write_text(r.stdout)
 x,y=runs
 assert x["mesh_divisions"]==y["mesh_divisions"]==n and x["h"]==y["h"] and len(x["trace"])==len(y["trace"])==5001
 assert all(a[0]==b[0] for a,b in zip(x["trace"],y["trace"]))
 assert x["contact_area_weights_m2"]==y["contact_area_weights_m2"]
 assert x["mass_kg"]==y["mass_kg"] and x["area_m2"]==y["area_m2"]
 scaleF=max(x["peak_force_N"],1e-30);scaleI=max(x["impulse_Ns"],1e-30)
 v={"divisions":n,"max_force_relative_to_ordered_peak":max(abs(a[1]-b[1]) for a,b in zip(x["trace"],y["trace"]))/scaleF,
    "max_cumulative_impulse_relative_to_ordered_total":max(abs(a[2]-b[2]) for a,b in zip(x["trace"],y["trace"]))/scaleI,
    "peak_force_relative_difference":abs(x["peak_force_N"]-y["peak_force_N"])/scaleF,
    "total_impulse_relative_difference":abs(x["impulse_Ns"]-y["impulse_Ns"])/scaleI,
    "same_area_weights_and_total_mass":True}
 v["passes"]=all(v[k]<=1e-8 for k in ("max_force_relative_to_ordered_peak","max_cumulative_impulse_relative_to_ordered_total","peak_force_relative_difference","total_impulse_relative_difference"))
 rows.append(v);print(json.dumps(v),flush=True)
 if not v["passes"]:break
out.write_text(json.dumps({"scope":"same equations at n=2 and n=8; does not qualify larger meshes or temporal error","input_sha256":hashes,"runs":rows},indent=2)+"\n")
raise SystemExit(0 if len(rows)==2 and all(v["passes"] for v in rows) else 1)
