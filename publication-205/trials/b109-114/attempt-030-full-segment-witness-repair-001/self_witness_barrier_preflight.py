from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,sys,json
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
W=ROOT/"D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027"
D=W/"b109-114/attempt-030-full-segment-witness-repair-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
TRIAL=W/"b109-114/attempt-029-target-ranked-local-fit-001/candidate-alpha-1p0.npz"
AUDIT=ROOT/"audit_skin_bone_surfaces_159.py"
SELF=ROOT/"D/publication-198-stage-001/scripts/owners/self_separation_026.py"
PAIRS=[(34537,34698),(34537,34701),(34698,34744),(34701,34744)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rattri(X,F,f):return [tuple(Fraction.from_float(float(v)) for v in q) for q in X[F[f]]]
def bary(p,t):
 p=np.asarray([float(v) for v in p]);t=np.asarray([[float(v) for v in r] for r in t]);v0=t[1]-t[0];v1=t[2]-t[0];v2=p-t[0];d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01;v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;return [float(1-v-w),float(v),float(w)]
def exactpoint(p):return [[int(x.numerator),int(x.denominator)] for x in p]
def main():
 with np.load(BASE,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 with np.load(TRIAL,allow_pickle=False) as z:trial={k:z[k].copy() for k in z.files}
 F=np.asarray(base["source_O_faces"],np.int64);X0=np.asarray(base["points_world_m"],np.float32);X1=np.asarray(trial["points_world_m"],np.float32)
 if not np.array_equal(F,trial["source_O_faces"]):raise ValueError("unexpected topology change")
 aud=mod("self_alpha1_exact_audit",AUDIT);exact=aud.exact_module()
 entries=[]
 for fa,fb in PAIRS:
  ta=rattri(X1,F,fa);tb=rattri(X1,F,fb);pts=[]
  for p in exact.triangle_intersection_points(ta,tb):
   if p not in pts:pts.append(p)
  pts=sorted(pts)
  if len(pts)!=2:raise ValueError(f"expected exact crossing segment for {fa}/{fb}, got {len(pts)} points")
  entries.append({"skin_face_rows":[fa,fb],"face_node_rows":[F[fa].astype(int).tolist(),F[fb].astype(int).tolist()],"source_parent_face_rows":[int(trial["refinement_face_parent_face_row"][fa]),int(trial["refinement_face_parent_face_row"][fb])],"exact_intersection_vertex_count":2,"endpoints":[{"exact_world_m":exactpoint(p),"face_a_barycentric":bary(p,ta),"face_b_barycentric":bary(p,tb)} for p in pts],"segment_length_m":float(np.linalg.norm(np.asarray([float(v) for v in pts[1]])-np.asarray([float(v) for v in pts[0]])))})
 sep=mod("self_sep_alpha1",SELF)
 rows,records=sep.build_pair_rows(X0.astype(np.float64)*1000,F,np.asarray(PAIRS,np.int64),margin_fraction=.25)
 receipt={"schema":"numi.human.b109-114.alpha1-exact-self-witnesses-and-baseline-barriers.v1","base_path":str(BASE),"base_sha256":sha(BASE),"self_witness_seed_path":str(TRIAL),"self_witness_seed_sha256":sha(TRIAL),"pair_ids":entries,"baseline_barrier":{"helper_path":str(SELF),"helper_sha256":sha(SELF),"margin_fraction":.25,"row_count":len(rows),"records":records,"rows":[{"nodes":r["nodes"].astype(int).tolist(),"coefficients":r["coefficients"].tolist(),"lower":float(r["lower"]),"label":r["label"]} for r in rows]},"construction_only":True,"physical_steps":0}
 out=D/"alpha1-self-witness-barrier-preflight.json";out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"report":str(out),"report_sha256":sha(out),"pairs":[{"faces":x["skin_face_rows"],"segment_length_m":x["segment_length_m"]} for x in entries],"baseline_barrier_row_count":len(rows),"baseline_pair_records":[{"faces":x["faces"],"kind":x["kind"],"closest_distance_mm":x.get("closest_distance_mm"),"support_gap_mm":x.get("support_gap_mm"),"preserved_margin_mm":x["preserved_margin_mm"]} for x in records]},indent=2))
if __name__=="__main__":main()
