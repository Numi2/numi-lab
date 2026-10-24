import json, hashlib, pathlib
import numpy as np
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"connected-skin-exact-self-audit-018"
A=json.loads((OUT/"report.json").read_text())
BASE=E/"E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
specs={
 "016-run-001":E/"connected-skin-ray-envelope-016/run-001/outer-envelope-proposal.npz",
 "017-run-002":E/"connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz",
 "017-run-003":E/"connected-skin-safe-flow-017/run-003/connected-rest-proposal.npz",
}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
with np.load(BASE,allow_pickle=False) as z:
 bx=z["points_world_m"].copy(); f=z["physical_outer_faces"].copy()
out={"schema":"connected-skin-normal-flip-vs-exact-self-pairs-018-v1","audit_report_sha256":sha(OUT/"report.json"),"baseline_sha256":sha(BASE),"note":"negative face-normal cosine is an orientation diagnostic, not an intersection predicate; exact pair sets come from the 315 audit report"}
for name,path in specs.items():
 with np.load(path,allow_pickle=False) as z:
  x=z["points_world_m"].copy(); cf=z["physical_outer_faces"].copy()
 if not np.array_equal(f,cf): raise RuntimeError(name+" topology mismatch")
 a=bx[f].astype(np.float64); b=x[f].astype(np.float64)
 n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]); n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
 dot=np.einsum("ij,ij->i",n0,n1)
 den=np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1)
 good=den>0; cos=np.ones(len(f)); cos[good]=dot[good]/den[good]
 flips=set(map(int,np.flatnonzero(good&(cos<=0))))
 pairs=[tuple(map(int,p)) for p in A["candidates"][name].get("added_exact_self_pairs",[])]
 touch=[p for p in pairs if p[0] in flips or p[1] in flips]
 both=[p for p in pairs if p[0] in flips and p[1] in flips]
 touched_faces={q for p in pairs for q in p}
 out[name]={
  "candidate_sha256":sha(path),
  "baseline_negative_cosine_face_rows":sorted(flips),
  "negative_cosine_face_count":len(flips),
  "exact_added_pair_count":len(pairs),
  "added_pairs_touching_at_least_one_negative_cosine_face_count":len(touch),
  "added_pairs_touching_both_negative_cosine_faces_count":len(both),
  "unique_faces_in_added_pairs_count":len(touched_faces),
  "unique_added_pair_faces_with_negative_cosine_count":len(flips&touched_faces),
  "added_pairs_touching_negative_cosine_face_rows": [list(p) for p in touch],
  "added_pairs_touching_both_negative_cosine_face_rows": [list(p) for p in both],
 }
path=OUT/"normal-flip-overlap-018.json"
path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(path),"sha256":sha(path),"candidates":{k:{x:v[x] for x in ("negative_cosine_face_count","exact_added_pair_count","added_pairs_touching_at_least_one_negative_cosine_face_count","added_pairs_touching_both_negative_cosine_faces_count","unique_added_pair_faces_with_negative_cosine_count")} for k,v in out.items() if k not in ("schema","note","audit_report_sha256","baseline_sha256")}},sort_keys=True))
