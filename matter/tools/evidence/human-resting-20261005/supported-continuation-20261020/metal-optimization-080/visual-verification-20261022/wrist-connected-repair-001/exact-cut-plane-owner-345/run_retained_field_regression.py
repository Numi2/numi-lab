from pathlib import Path
import importlib.util,numpy as np,json,hashlib,time
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"exact-cut-plane-owner-345"
s=importlib.util.spec_from_file_location("geom",D/"tools/human_skin_patch_geometry.py");g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=[("left323",E/"bilateral-healthy-forearm-boundaries-323/left",0,True),("right342-as-disk",E/"right-healthy-forearm-muscle-envelope-342",0,False),("right342-diagnostic-genus1",E/"right-healthy-forearm-muscle-envelope-342",1,True),("corrugated340",E/"right-healthy-forearm-muscle-envelope-340",1,False)]
rows=[]
for name,p,genus,expected in cases:
 started=time.monotonic();m=np.load(p/"extracted-interface.npz",allow_pickle=False);f=np.load(p/"conditioned-field.npz",allow_pickle=False);row={"name":name,"mesh_sha256":sha(p/"extracted-interface.npz"),"field_sha256":sha(p/"conditioned-field.npz"),"expected_pass":expected,"expected_genus":genus}
 try:
  opened,mask,top=g.remove_exact_planar_cap(m["vertices_local_float32"],m["faces"],axis=2,coordinate=float(f["field_cap_z_local_m"]),expected_genus=genus)
  assert np.array_equal(mask,m["cap_face_mask"]), "emitted cap mask disagreement"
  row.update(passed=True,removed_faces=int(mask.sum()),boundary_edges=len(top["boundary_loops"][0]),genus=top["genus"],euler_characteristic=top["euler_characteristic"])
 except ValueError as exc:row.update(passed=False,error=str(exc))
 assert row["passed"]==expected,row
 row["elapsed_seconds"]=time.monotonic()-started;rows.append(row);print(json.dumps(row),flush=True)
report={"source":str(D/"tools/human_skin_patch_geometry.py"),"source_sha256":sha(D/"tools/human_skin_patch_geometry.py"),"cases":rows,"physical_admission":False,"scope":"Exact emitted-plane removal and topology gate only; right342 diagnostic acceptance at declared genus1 is not hand admission; requested hand genus0 rejects."}
(D/"retained-field-regression.json").write_text(json.dumps(report,indent=2)+"\n")
