"""Compose separately checked topology and coordinates, checking shared lineage."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=D/"D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz"
TOPO=D/"D/connected-quality-patch-534/run-001/candidate.npz"
COORD=D/"D/connected-local-backtrack-530/current-scaled-001/backtrack-373/pass-012/candidate.npz"
FIT=D/"barycentric-connected-clearance-170/connected-local-fit-529/current-scaled-001/candidate.npz"
OWNER=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
PINS={BASE:"c1c0b63ee448703ef3240dbcc251e650022d7f76f475bc75db9cae040d1e3a6c",
TOPO:"20c6f2a40f1bcf21cab5c63d94e113f4b09f782454d1819295482a751d49e14b",
COORD:"ee504ce47167ce04fa242743fabd777351887f740b7f70bf5e7ac4a248fe5fff",
FIT:"84832ab3222b7d9eb851a9e7d6b966c66e9e8c13994292c90b23e588218e2963",
OWNER:"5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identical(a,b):
 a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError("source pin mismatch: "+str(p))
 outdir=Path(__file__).parent/"run-001";outdir.mkdir(exist_ok=False)
 with np.load(BASE,allow_pickle=False) as b,np.load(TOPO,allow_pickle=False) as t,np.load(COORD,allow_pickle=False) as c,np.load(FIT,allow_pickle=False) as fit:
  if set(b.files)!=set(c.files):raise ValueError("coordinate branch changed source keys")
  if not all(identical(b[k],c[k])for k in b.files if k!="points_world_m"):
   raise ValueError("coordinate branch changed noncoordinate provenance")
  if not identical(b["points_world_m"],t["points_world_m"]):raise ValueError("topology branch moved points")
  payload={k:np.array(t[k],copy=True)for k in t.files}
  spec=importlib.util.spec_from_file_location("frozen408_for536",OWNER)
  owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
  x=np.asarray(b["points_world_m"]);y=np.asarray(c["points_world_m"])
  if x.dtype!=np.float32 or y.dtype!=np.float32:raise ValueError("requires emitted F32 points")
  f=np.asarray(t["source_O_faces"]);anchors=np.asarray(b["typed_coarse_anchor_output_rows_018"])
  support=np.asarray(fit["local_witness_cluster_222_support_rows"])
  A,rhs,edge_receipt=owner.build_edge_gradient_constraints(x.astype(float)*1000.,f,support,.5)
  source,_=owner.compose_source_reference(b,D/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz")
  trials=[]
  for alpha in [1.,.5,.25,.125]:
   p=(x.astype(float)+alpha*(y.astype(float)-x.astype(float))).astype(np.float32)
   area=owner.full_f32_area_gate(p,x.astype(float)*1000.,np.asarray(t["area_reference_037_points_world_m_f64"])*1000.,f,.1)
   d=(p.astype(float)-x.astype(float))*1000.
   slack=np.asarray(A@d[support].reshape(-1)).ravel()-rhs
   edgepass=bool(np.min(slack,initial=np.inf)>=-1e-7)
   cap=float(np.linalg.norm(p.astype(float)-source,axis=1).max()*1000.)
   anchorpass=identical(p[anchors],x[anchors])
   rec=dict(alpha=alpha,area_gate=area,edge_gradient_pass=edgepass,
            min_edge_slack_mm=float(slack.min(initial=np.inf)),source_max_mm=cap,anchors_unchanged=anchorpass)
   trials.append(rec)
   if area["pass"] and edgepass and cap<=12.000001 and anchorpass:
    payload["points_world_m"]=p
    dest=outdir/"candidate.npz";np.savez_compressed(dest,**payload)
    report=dict(status="CONSTRUCTION_COMPOSITION_PENDING_EXACT_AUDITS",
      candidate=str(dest),candidate_sha256=sha(dest),source_pins={str(k):v for k,v in PINS.items()},
      coordinate_branch_noncoordinate_arrays_bitwise_equal_to_base=True,
      topology_branch_coordinates_bitwise_equal_to_base=True,selected_alpha=alpha,trials=trials,
      edge_build_receipt=edge_receipt,physical_transfer_qualified=False,
      geometry_admitted=False,native_admission=False,accepted_physical_steps=0)
    break
  else:report=dict(status="COMPOSITION_FAILED_PREPARATION_GATES",trials=trials,native_admission=False,accepted_physical_steps=0)
 (outdir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items()if k not in ["trials","source_pins","edge_build_receipt"]}))
 if "candidate" not in report:raise SystemExit(1)
if __name__=="__main__":main()
