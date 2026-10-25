#!/usr/bin/env python3
import json,hashlib,importlib.util,sys,time,pathlib
import numpy as np
from scipy import sparse
ROOT=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069")
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
RUN=ROOT/"coupled-layer-trial-128/run-003"
BASE=ROOT/"coupled-layer-clearance-133/run-002"
OUT=BASE/"post-audit-001"; OUT.mkdir(parents=True,exist_ok=False)
PRED=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PRED_IMPORTS=E/"incremental-exact-audit-owner-253/src"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
sys.path.insert(0,str(PRED_IMPORTS));pred=loadmod("numilab_human.cardiac_cavity_intersections",PRED)
t0=time.monotonic()
z0=np.load(RUN/"layer-candidate.npz",allow_pickle=False)
z=np.load(BASE/"joint-layer-candidate.npz",allow_pickle=False)
O=np.asarray(z["outer_points_f32"],np.float32); I=np.asarray(z["inner_points_f32"],np.float32); D=np.asarray(z["deep_points_f32"],np.float32)
F=np.asarray(z["outer_faces"],np.int64); n=len(O); nf=len(F)
need_faces=[np.asarray(z[k],np.int64) for k in ("outer_faces","inner_faces","deep_faces")]
assert np.array_equal(O,z0["outer_points_f32"])
assert np.array_equal(F,z0["outer_faces"])
all3=np.vstack((O,I,D))
def self_audit(X,Faces,name):
 rec,_=pred.float32_surface_records(X,np.asarray(Faces,np.int64),source_seam_groups=())
 out=pred._audit_self_pairs_touching_faces(rec,set(range(len(Faces))))
 pairs=np.asarray(out["triangle_pairs"],np.int64).reshape(-1,2)
 np.save(OUT/f"{name}-pairs.npy",pairs)
 return {"count":int(len(pairs)),"complete":bool(out.get("complete_surface_audit",False)),
         "aabb_candidates":int(out.get("aabb_candidate_pairs",-1)),
         "allowed_shared":int(out.get("allowed_shared_vertex_or_edge_pairs",-1)),
         "pairs_file":f"{name}-pairs.npy","pairs_sha256":sha(OUT/f"{name}-pairs.npy")}
def cross_audit(A,FA,B,FB,name):
 ra,_=pred.float32_surface_records(A,np.asarray(FA,np.int64),source_seam_groups=())
 rb,_=pred.float32_surface_records(B,np.asarray(FB,np.int64),source_seam_groups=())
 out=pred._audit_pair(ra,rb,same_surface=False)
 pairs=np.asarray(out["triangle_pairs"],np.int64).reshape(-1,2)
 np.save(OUT/f"{name}-pairs.npy",pairs)
 return {"count":int(len(pairs)),"aabb_candidates":int(out["aabb_candidate_pairs"]),
         "pairs_file":f"{name}-pairs.npy","pairs_sha256":sha(OUT/f"{name}-pairs.npy")}
audits={}
for label,X,Faces in (("O",O,F),("I",I,F),("D",D,F),
                      ("skin_shell",all3,z["skin_shell_faces"]),
                      ("fat_shell",all3,z["fat_shell_faces"])):
 t=time.monotonic();a=self_audit(X,Faces,label);a["seconds"]=time.monotonic()-t;audits[label]=a
for label,A,FA,B,FB in (("O_I",O,F,I,F),("I_D",I,F,D,F),("O_D",O,F,D,F)):
 t=time.monotonic();a=cross_audit(A,FA,B,FB,label);a["seconds"]=time.monotonic()-t;audits[label]=a
inputs=np.load(BASE/"joint-lp-inputs.npz",allow_pickle=False)
Aub=sparse.load_npz(BASE/"joint-lp-A-ub.npz").tocsr()
x=np.asarray(z["LP_solution"],np.float64)
b=np.asarray(inputs["b_ub"],np.float64)
slack=b-Aub@x
nd=6*n
dactual=np.concatenate(((I.astype(np.float64)-z0["inner_points_f32"].astype(np.float64)).reshape(-1)*1000,
                        (D.astype(np.float64)-z0["deep_points_f32"].astype(np.float64)).reshape(-1)*1000))
Ageo=sparse.csr_matrix(inputs["geometric_A_ub"])
bgeo=np.asarray(inputs["geometric_b_ub"],np.float64)
geoslack=bgeo-Ageo@dactual
meta=json.loads((BASE/"geometric-witness-rows.json").read_text())
worst=np.argsort(geoslack)[:10]
# Baseline O normals and affected column checks in exact emitted F32.
O64=O.astype(np.float64);I64=I.astype(np.float64);D64=D.astype(np.float64)
cross=np.cross(O64[F[:,1]]-O64[F[:,0]],O64[F[:,2]]-O64[F[:,0]])
vn=np.zeros((n,3),np.float64)
for c in range(3): np.add.at(vn,F[:,c],cross)
lens=np.linalg.norm(vn,axis=1)
affected=np.asarray(z["active_vertex_rows"],np.int64)
assert np.all(lens[affected]>0)
vn[affected]/=lens[affected,None]
oi=np.einsum("ij,ij->i",vn[affected],O64[affected]-I64[affected])*1000
idg=np.einsum("ij,ij->i",vn[affected],I64[affected]-D64[affected])*1000
def area_stats(X):
 tri=X[F].astype(np.float64); c=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 area=.5*np.linalg.norm(c,axis=1)
 dot=np.einsum("ij,ij->i",c,cross)
 cos=dot/(np.linalg.norm(c,axis=1)*np.linalg.norm(cross,axis=1))
 return {"min_area_m2":float(area.min()),"nonpositive_area":int(np.sum(~np.isfinite(area)|(area<=0))),
         "min_orientation_cosine_vs_baseline":float(cos.min()),"flipped_vs_baseline":int(np.sum(cos<=0))}
primal_tol=1e-7
report={
 "status":"POST_AUDIT_OF_FROZEN_LP_CANDIDATE",
 "solver_completion_status":"not serialized by as-run solver; primal x and full sparse model retained",
 "candidate_sha256":sha(BASE/"joint-layer-candidate.npz"),
 "attempt_source_sha256":sha(ROOT/"coupled-layer-clearance-133/run_joint_clearance_133_attempt_002.py"),
 "input_sha256":sha(RUN/"layer-candidate.npz"),
 "model_sha256":{"A_ub":sha(BASE/"joint-lp-A-ub.npz"),"inputs":sha(BASE/"joint-lp-inputs.npz"),
                 "witness_rows":sha(BASE/"geometric-witness-rows.json")},
 "lp_primal":{"max_full_model_violation_mm":float(max(0.,np.max(-slack,initial=0.))),
              "min_full_model_slack_mm":float(np.min(slack)),
              "feasible_at_1e-7":bool(np.max(-slack,initial=0.)<=primal_tol),
              "max_emitted_F32_geometric_violation_mm":float(max(0.,np.max(-geoslack,initial=0.))),
              "min_emitted_F32_geometric_slack_mm":float(np.min(geoslack)),
              "emitted_F32_geometric_feasible_at_1e-7":bool(np.max(-geoslack,initial=0.)<=primal_tol),
              "objective_value":float(np.dot(np.asarray(inputs["c"],np.float64),x))},
 "audits":audits,
 "affected_column_ordering_f32_mm":{"count":int(len(affected)),
      "min_OI":float(oi.min()),"below_0_05mm":int(np.sum(oi<0.05-1e-6)),
      "min_ID":float(idg.min()),"below_0_05mm_ID":int(np.sum(idg<0.05-1e-6))},
 "surface":{"O":area_stats(O64),"I":area_stats(I64),"D":area_stats(D64),
      "max_I_move_mm":float(np.linalg.norm(I64-z0["inner_points_f32"].astype(np.float64),axis=1).max()*1000),
      "max_D_move_mm":float(np.linalg.norm(D64-z0["deep_points_f32"].astype(np.float64),axis=1).max()*1000)},
 "worst_emitted_geometry_rows":[{"row":int(i),"slack_mm":float(geoslack[i]),"meta":meta[int(i)]}
                                for i in worst],
 "wall_seconds":time.monotonic()-t0
}
report["all_layer_and_shell_self_pairs_zero"]=all(audits[k]["count"]==0 for k in ("O","I","D","skin_shell","fat_shell"))
report["all_cross_layer_pairs_zero"]=all(audits[k]["count"]==0 for k in ("O_I","I_D","O_D"))
report["construction_geometry_gates_pass"]=bool(report["lp_primal"]["feasible_at_1e-7"] and
 report["lp_primal"]["emitted_F32_geometric_feasible_at_1e-7"] and
 report["all_layer_and_shell_self_pairs_zero"] and report["all_cross_layer_pairs_zero"] and
 report["affected_column_ordering_f32_mm"]["below_0_05mm"]==0 and
 report["affected_column_ordering_f32_mm"]["below_0_05mm_ID"]==0)
report["limits"]=[
 "This post-audit reuses one emitted LP candidate and does not re-solve.",
 "The optimality/termination status was not serialized; primal feasibility is recomputed from retained x and A/b.",
 "Static geometric construction only; no TetGen, volume-mesh quality, mass transfer, bone query, native physics, or physical admission."
]
(OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("status","lp_primal","audits","affected_column_ordering_f32_mm","surface","all_layer_and_shell_self_pairs_zero","all_cross_layer_pairs_zero","construction_geometry_gates_pass","wall_seconds")},sort_keys=True))

