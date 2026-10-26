from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
r=Path(sys.argv[1]);c=r/"candidate.npz"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert c.exists()
base=D/"barycentric-connected-clearance-170/combined-clearance-394/run-001/candidate.npz"
a=np.load(base,allow_pickle=False);b=np.load(c,allow_pickle=False)
assert np.array_equal(a["source_O_faces"],b["source_O_faces"])
assert np.array_equal(a["area_reference_037_source_O_faces"],b["area_reference_037_source_O_faces"])
assert np.array_equal(a["area_reference_037_points_world_m_f64"],b["area_reference_037_points_world_m_f64"])
for n,out in [("audit_connected_outer_geometry_130.py","audit-130"),("audit_skin_bone_surfaces_159.py","audit-159")]:
 cmd=[PY,"/tmp/numi-human-geometry-slot.py",PY,str(D/n),str(c),str(r/out)]
 if out=="audit-159":cmd+=["--max-seconds","600"]
 with (r/(out+".log")).open("x") as log:
  cp=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert cp.returncode==0,(n,cp.returncode)
 if out=="audit-130":
  j=json.loads((r/out/"summary.json").read_text())
  if j["exact_PLC_surface_pairs"]!=0:
   print(json.dumps({"status":"FAIL_EXACT_SELF","summary":j}));sys.exit(2)
j=json.loads((r/"audit-159/report.json").read_text());assert j["complete"]
result=dict(candidate_sha256=sha(c),self_intersections=0,surface_pairs=sum(x["result"]["count"] for x in j["banks"] if x.get("result")),audit130_sha256=sha(r/"audit-130/summary.json"),audit159_sha256=sha(r/"audit-159/report.json"),native_admission=False,accepted_physical_steps=0)
(r/"root-exact-audit-result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
