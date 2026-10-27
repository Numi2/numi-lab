"""Keep only mesh-connected coordinate components with bankwise nonincreasing audited crossings.
Offline construction proposal only: a fresh full545 check and strict546 finalization remain mandatory.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
PRED="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def loadnpz(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def labels_for_changes(x,y,f):
 if x.dtype!=np.float32 or y.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or y.shape!=x.shape or not np.isfinite(x).all()or not np.isfinite(y).all():raise ValueError("invalid Float32 points")
 if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or np.any(f<0)or np.any(f>=len(x)):raise ValueError("invalid faces")
 changed=np.any(x!=y,axis=1)
 edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges=edges[changed[edges].all(axis=1)]
 g=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(x),len(x))).tocsr()
 _,lab=connected_components(g,directed=False)
 lab[~changed]=-1
 owners=np.max(lab[f],axis=1)
 for tri,owner in zip(f,owners):
  if np.any((lab[tri]>=0)&(lab[tri]!=owner)):raise AssertionError("face spans mutable components")
 return lab,owners
def bank_pairs(report,candidate_hash,nfaces,expected_keys=None):
 if report.get("candidate_sha256")!=candidate_hash or report.get("complete")is not True or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185 or len(report.get("banks",[]))!=185 or report.get("exact_predicate_sha256")!=PRED:raise ValueError("incomplete/unpinned 185-bank report")
 result={}
 for b in report["banks"]:
  key=(b["body"],b["stable_bone_id"])
  if key in result:raise ValueError("duplicate bank")
  r=b.get("result")
  if r is None and b.get("skin_aabb_active")==0 and b.get("bank_aabb_active")==0:
   r={"count":0,"triangle_pairs":[]}
  if not isinstance(r,dict):raise ValueError("missing result without certified empty AABB")
  pairs=[tuple(v)for v in r.get("triangle_pairs",[])]
  if len(pairs)!=r.get("count")or len(set(pairs))!=len(pairs):raise ValueError("pair count/duplicate mismatch")
  if any(len(p)!=2 or any(type(v)is not int for v in p)or p[0]<0 or p[0]>=nfaces or p[1]<0 for p in pairs):raise ValueError("invalid pair")
  result[key]=set(pairs)
 if expected_keys is not None and set(result)!=set(expected_keys):raise ValueError("typed bank set changed")
 return result
def component_deltas(owners,before,after):
 keys=set(before)
 if set(after)!=keys:raise ValueError("bank set mismatch")
 deltas={int(k):{b:0 for b in keys}for k in np.unique(owners)if k>=0}
 for key in keys:
  for sign,rows in ((-1,before[key]),(1,after[key])):
   for face,_ in rows:
    k=int(owners[face])
    if k>=0:deltas[k][key]+=sign
  # Faces without mutable coordinates cannot acquire/lose geometric pairs.
  if {p for p in before[key]if owners[p[0]]<0}!={p for p in after[key]if owners[p[0]]<0}:raise ValueError("unchanged face pair mismatch")
 keep={k for k,ds in deltas.items()if sum(ds.values())<0 and all(v<=0 for v in ds.values())}
 return deltas,keep
def run(a):
 s=readj(a.baseline_state);roles={k:Path(s[k])for k in ("candidate","point_summary","surface_report")}
 for k,p in roles.items():
  if sha(p)!=s.get(k+"_sha256"):raise ValueError("baseline pin mismatch "+k)
 check=readj(a.target_check)
 if check.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS"or check.get("self_pairs")!=0 or check.get("contained_pairs")!=0 or check.get("arithmetic_pass")is not True or check.get("baseline_sha256")!=s["candidate_sha256"]:raise ValueError("target lacks complete zero-self/inside arithmetic check")
 target=Path(check["candidate_path"]);th=sha(target)
 if th!=check.get("candidate_sha256"):raise ValueError("target pin mismatch")
 bs=readj(roles["point_summary"])
 ts=readj(a.target_check.parent/"audit-130/summary.json")
 for summary,h in ((bs,s["candidate_sha256"]),(ts,th)):
  if summary.get("candidate_sha256")!=h or summary.get("complete_surface_audit")is not True or summary.get("exact_PLC_surface_pairs")!=0 or summary.get("inside_node_bank_pairs")!=0 or summary.get("surface_predicate_sha256")!=PRED:raise ValueError("point audit incomplete")
 b=loadnpz(roles["candidate"]);t=loadnpz(target)
 if set(b)!=set(t)or any(not same(b[k],t[k])for k in b if k!="points_world_m"):raise ValueError("nonpoint payload changed")
 x=b["points_world_m"];y=t["points_world_m"];f=b["source_O_faces"];anchors=b["typed_coarse_anchor_output_rows_018"]
 if len(anchors)!=237 or not same(x[anchors],y[anchors]):raise ValueError("anchors changed")
 labels,owners=labels_for_changes(x,y,f)
 bp=bank_pairs(readj(roles["surface_report"]),s["candidate_sha256"],len(f))
 tr=a.target_check.parent/"audit-159/report.json"
 tp=bank_pairs(readj(tr),th,len(f),bp)
 deltas,keep=component_deltas(owners,bp,tp)
 if not keep:raise ValueError("no bankwise improving component")
 rows=np.flatnonzero(np.isin(labels,list(keep)))
 output={k:v.copy()for k,v in b.items()};output["points_world_m"][rows]=y[rows]
 assert same(output["points_world_m"][anchors],x[anchors])
 a.out.mkdir(exist_ok=False,parents=True);np.savez_compressed(a.out/"candidate.npz",**output)
 records=[{"component":k,"nodes":int(np.count_nonzero(labels==k)),"retained":k in keep,"delta":sum(ds.values()),"per_bank_delta":{str(b):d for b,d in ds.items()if d}}for k,ds in sorted(deltas.items())]
 predicted=sum(map(len,bp.values()))+sum(sum(deltas[k].values())for k in keep)
 report={"status":"COMPONENT_PROPOSAL_PENDING_FRESH_FULL_AUDITS","baseline_state":str(a.baseline_state),"baseline_state_sha256":sha(a.baseline_state),"base_candidate_sha256":s["candidate_sha256"],"target":str(target),"target_sha256":th,"target_check":str(a.target_check),"target_check_sha256":sha(a.target_check),"target_surface_report_sha256":sha(tr),"candidate_sha256":sha(a.out/"candidate.npz"),"components":records,"predicted_surface_pairs":predicted,"selected_node_rows":rows.tolist(),"outside_selected_coordinates_bitwise_baseline":True,"nonpoint_payload_bitwise_baseline":True,"typed_anchors":237,"native_admission":False,"candidate_selected":False,"accepted_physical_steps":0,"source_sha256":sha(__file__),"limitations":["Endpoint pair composition is a proposal only; changed cross-component self intersections and containment require fresh full545/130/159.","No LP feasibility, optimality, physical skin/fat, mass, motion or native qualification claimed."]}
 (a.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"predicted":predicted,"kept_components":sorted(keep),"nodes":len(rows)}),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ("baseline-state","target-check","out"):p.add_argument("--"+k,type=Path,required=True)
 run(p.parse_args())
if __name__=="__main__":main()
