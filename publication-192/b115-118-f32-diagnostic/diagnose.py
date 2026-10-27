from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
C=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
W=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/topology"
D=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/explicit-fit"
OUT=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/float32-diagnostic-001"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
TARGET=[13,58,98,102,116]; FLOOR=1e-6; FTOL=1e-8
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
OUT.mkdir(parents=True,exist_ok=True)
z=npz(C);q=npz(W/"patch-candidate.npz");fit=npz(D/"private-constrained-fit-patch.npz")
P=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["source_O_faces"],np.int64)
Qp=np.asarray(q["points_world_m_f32"],np.float32);PF=np.asarray(q["patch_faces"],np.int64);par=np.asarray(q["face_parent_candidate_rows"],np.int64)
_,_,_,_,M,off,pos,_,_,_=loadmod("diag_helper",HELPER).read_bank_inputs(BANKDIR,E)
skin=(P.astype(np.float64)-pos[115])@M[115].T-off[115]
init=(Qp.astype(np.float64)-pos[115])@M[115].T-off[115]
disp=np.asarray(fit["node_displacement_body_local_m_f64"],np.float64); dbl=init+disp
em=np.asarray(fit["points_world_m_f32"],np.float32)
pnodes=skin[F[par]]; cnodes=dbl[PF]
pn=np.cross(pnodes[:,1]-pnodes[:,0],pnodes[:,2]-pnodes[:,0]); cn=np.cross(cnodes[:,1]-cnodes[:,0],cnodes[:,2]-cnodes[:,0])
plen=np.linalg.norm(pn,axis=1); dsign=np.einsum("ij,ij->i",pn,cn)/np.maximum(plen**2,1e-300)
darea=np.linalg.norm(cn,axis=1)/np.maximum(plen,1e-300)
pw=P[F[par]].astype(np.float64); ew=em[PF].astype(np.float64)
pnw=np.cross(pw[:,1]-pw[:,0],pw[:,2]-pw[:,0]); cnw=np.cross(ew[:,1]-ew[:,0],ew[:,2]-ew[:,0])
pnw2=np.einsum("ij,ij->i",pnw,pnw); fsign=np.einsum("ij,ij->i",pnw,cnw)/np.maximum(pnw2,1e-300)
farea=np.linalg.norm(cnw,axis=1)/np.maximum(np.sqrt(pnw2),1e-300)
regen=((dbl+off[115])@M[115]+pos[115])
round_um=np.linalg.norm(regen-em.astype(np.float64),axis=1)*1e6
checks={
 "faces_identical":bool(np.array_equal(PF,fit["patch_faces"])),
 "parent_rows_identical":bool(np.array_equal(par,fit["face_parent_candidate_rows"])),
 "node_count_matches":bool(len(disp)==len(Qp)),
 "parent_indices_valid":bool(np.all((par>=0)&(par<len(F)))),
 "emitted_inverted_rows_reproduced":bool(TARGET==np.flatnonzero(fsign<=0).astype(int).tolist())
}
rows=[]
for i in TARGET:
 rows.append({"child_row":i,"patch_vertex_rows":PF[i].astype(int).tolist(),"parent_candidate_face_row":int(par[i]),
 "source_parent_vertex_rows":F[par[i]].astype(int).tolist(),
 "double_signed_projected_area_ratio":float(dsign[i]),
 "double_constraint_residual_after_1e-6_floor":float(dsign[i]-FLOOR),
 "double_unsigned_area_ratio":float(darea[i]),
 "double_inside_configured_ftol_of_floor":bool(abs(dsign[i]-FLOOR)<=FTOL),
 "emitted_f32_signed_orientation_ratio":float(fsign[i]),
 "emitted_f32_unsigned_area_ratio":float(farea[i]),
 "emitted_inverted":bool(fsign[i]<=0),
 "max_f32_vertex_rounding_error_um":float(round_um[PF[i]].max())})
old=json.loads((D/"report.json").read_text());opt=old["optimizer"]
report={"schema":"numi.human.wide-fit.float32-inversion-diagnostic.v1",
 "purpose":"Read-only rowwise comparison of saved SLSQP double coordinates and emitted Float32; no fit or geometry audit.",
 "pins":{"candidate":{"path":str(C),"sha256":sha(C)},"topology":{"path":str(W/"patch-candidate.npz"),"sha256":sha(W/"patch-candidate.npz")},
 "fit_patch":{"path":str(D/"private-constrained-fit-patch.npz"),"sha256":sha(D/"private-constrained-fit-patch.npz")},
 "fit_report":{"path":str(D/"report.json"),"sha256":sha(D/"report.json")},
 "fit_script":{"path":str(D/"constrained_exterior_fit.py"),"sha256":sha(D/"constrained_exterior_fit.py")}},
 "constraints":{"area_ratio_floor":FLOOR,"SLSQP_ftol":FTOL,"double_min_constraint_reported":opt["final_min_shape_constraint"],
 "SLSQP_success":opt["success"],"iterations":opt["iterations"]},
 "face_mapping_checks":checks,"affected_faces":rows,
 "interpretation":None}
near=any(r["double_inside_configured_ftol_of_floor"] for r in rows)
if not all(checks.values()): report["interpretation"]="Indexing or row alignment mismatch demonstrated; attribution not trustworthy."
elif near: report["interpretation"]="No indexing mismatch: fit and topology patch faces and parent rows match, and all five reported inversions reproduce. The double signed area constraints sit within SLSQP ftol of the 1e-6 floor, so solver termination accepts near-degenerate faces; Float32 emission then changes orientation sign. Both tolerance and rounding contribute."
else: report["interpretation"]="No indexing mismatch. See per-face residuals; emitted Float32 reverses orientation after the double constraint check."
rp=OUT/"receipt.json";rp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"receipt":str(rp),"sha256":sha(rp),"checks":checks,"constraints":report["constraints"],"faces":rows,"interpretation":report["interpretation"]},indent=2,allow_nan=False))
