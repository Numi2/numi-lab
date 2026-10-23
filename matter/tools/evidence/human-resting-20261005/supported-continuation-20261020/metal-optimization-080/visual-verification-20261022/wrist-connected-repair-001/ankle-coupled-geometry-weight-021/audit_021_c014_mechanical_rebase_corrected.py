#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"ankle-coupled-geometry-weight-021";C14=E/"joint-bmap-refinement-candidate-011-mechanical119-integration-001/source-C-reference-offset-diagnostic-014-exact-f32-residual-001"
DETAIL=E/"finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz";CAND=OUT/"ankle-coupled-021-candidate.npz"
MAN=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
BASE=OUT/"ankle-coupled-021-C014-mechanical-rebase-diagnostic.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 report=json.loads(BASE.read_text());S=np.load(C14/"coarse-surface.npz");C=np.load(C14/"coarse-to-source-C014.npz");L=np.load(C14/"candidate-layers-C014.npz");G=np.load(DETAIL);Q=np.load(CAND)
 n=len(L["outer_world"]);tet=np.asarray(L["tetrahedra"],np.int64);F=np.asarray(G["faces"],np.int64)
 V0=np.loadtxt(E/"native-pose-checks/connected-hand-registration069i/registered-skin.csv",delimiter=",",skiprows=1,usecols=(0,2,3,4),dtype=np.float32)[:,1:4]
 V1=np.asarray(Q["reference_positions_m"],np.float32);sv=np.asarray(S["source_vertex"],int);vk=np.asarray(S["vertex_kind"],int);direct=vk==0
 cf=np.asarray(C["source_face"],int);b=np.asarray(C["source_barycentric"],float)
 C0=np.empty((n,3));C1=np.empty_like(C0);C0[direct]=V0[sv[direct]];C1[direct]=V1[sv[direct]]
 tri=F[cf[~direct]];C0[~direct]=np.einsum("ni,nij->nj",b[~direct],V0[tri]);C1[~direct]=np.einsum("ni,nij->nj",b[~direct],V1[tri])
 residual=np.asarray(C["source_residual_vector_m"],float);O0=np.asarray(L["outer_world"],np.float32).astype(float);I0=np.asarray(L["inner_world"],np.float32).astype(float);D0=np.asarray(L["deep_world"],np.float32).astype(float)
 O1=(C1+residual).astype(np.float32).astype(float);I1=(O1+I0-O0).astype(np.float32).astype(float);D1=(O1+D0-O0).astype(np.float32).astype(float)
 def det(X):
  x=X[tet];a=x[:,1]-x[:,0];v=x[:,2]-x[:,0];w=x[:,3]-x[:,0];d=np.einsum("ij,ij->i",a,np.cross(v,w));e=np.zeros(len(x))
  for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):e=np.maximum(e,np.einsum("ij,ij->i",x[:,i]-x[:,j],x[:,i]-x[:,j]))
  return d,d/np.maximum(e,1e-30)**1.5
 X0=np.vstack([O0,I0,D0]).astype(np.float32).astype(float);X1=np.vstack([O1,I1,D1]).astype(np.float32).astype(float);j0,q0=det(X0);j1,q1=det(X1)
 ratio=np.divide(j1,j0,out=np.full_like(j1,np.nan),where=np.abs(j0)>1e-30);flip=(j0*j1<=0)
 rows=np.asarray(Q["source_vertex_ids"],int);mask=np.zeros(61950,bool);mask[rows]=True
 affected=(direct&mask[sv])|((~direct)&np.any(mask[F[cf]],axis=1))|(np.linalg.norm(C1-C0,axis=1)>1e-12)
 nodes=np.r_[np.flatnonzero(affected),n+np.flatnonzero(affected),2*n+np.flatnonzero(affected)];ct=np.any(np.isin(tet,nodes),axis=1)
 lay=np.asarray(L["tetra_layer"]);reg=np.asarray(L["tetra_region"])
 bylayer={}
 for k in np.unique(lay):
  m=lay==k;bylayer[str(int(k))]={"count":int(m.sum()),"baseline_min_abs_normalized_det":float(np.min(abs(q0[m]))),"candidate_min_abs_normalized_det":float(np.min(abs(q1[m]))),"baseline_abs_floor_failures_1e-5":int(np.sum(abs(q0[m])<1e-5)),"candidate_abs_floor_failures_1e-5":int(np.sum(abs(q1[m])<1e-5)),"candidate_sign_pattern_flips":int(np.sum(flip[m]))}
 byreg={}
 for r in np.unique(reg):
  m=(reg==r)&ct
  if m.any():byreg[str(r)]={"affected_tets":int(m.sum()),"sign_flips":int(np.sum(flip[m])),"minimum_J_ratio":float(np.nanmin(ratio[m])),"candidate_min_abs_normalized_det":float(np.min(abs(q1[m])))}
 sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
 from numilab_human.skin_source_payload_preflight import decode_payload
 payload=decode_payload((E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin").read_bytes())
 bind=np.asarray(payload["bindings_u"][:,0],int);names={int(a["core_body_index"]):str(a["myosim_body"]) for a in json.loads(MAN.read_text())["source"]["anchors"]};slots=[names[int(i)] for i in bind]
 W0=np.asarray(G["weights"],np.float32);W1=W0.copy();W1[rows]=Q["full_weights_after"];wc=np.empty((n,86));wc[direct]=W1[sv[direct]];wc[~direct]=np.einsum("ni,nij->nj",b[~direct],W1[tri]);donor=np.array([slots[int(i)] for i in np.argmax(wc,axis=1)])
 delta=np.linalg.norm(C1-C0,axis=1)*1000;per=[]
 for i in np.flatnonzero(affected):
  touched=[int(sv[i])] if direct[i] else sorted(map(int,F[cf[i]][mask[F[cf[i]]]]))
  per.append({"coarse_row":int(i),"kind":int(vk[i]),"donor":str(donor[i]),"source_ids_touched":touched,"C_delta_mm":float(delta[i]),"retained_residual_m":residual[i].tolist(),"residual_delta_if_outer_frozen_mm":-float(delta[i])})
 old=report["mechanical_jacobian_diagnostic"]
 report["mechanical_jacobian_diagnostic"]={"orientation_interpretation":"Native prism tetrahedra have authored alternating sign by face permutation. Raw negative is not inversion; compare each candidate determinant with that same stored tet's baseline sign.","baseline_raw_negative_by_authored_order":int(np.sum(j0<0)),"candidate_raw_negative_by_authored_order":int(np.sum(j1<0)),"candidate_sign_pattern_flips":int(np.sum(flip)),"sign_flip_tet_ids":np.flatnonzero(flip).astype(int).tolist(),"candidate_abs_normalized_det_below_1e-5":int(np.sum(abs(q1)<1e-5)),"baseline_abs_normalized_det_below_1e-5":int(np.sum(abs(q0)<1e-5)),"candidate_min_abs_normalized_det":float(np.min(abs(q1))),"baseline_min_abs_normalized_det":float(np.min(abs(q0))),"minimum_signed_J_ratio_candidate_over_baseline":float(np.nanmin(ratio)),"minimum_J_ratio_tet":int(np.nanargmin(ratio)),"changed_tet_count":int(ct.sum()),"changed_tet_sign_flips":int(np.sum(flip&ct)),"layer_stats":bylayer,"changed_region_stats":byreg,"raw_initial_diagnostic":old,"physical_layer_admission":False}
 report["coarse_mapping_response"]["affected_rows_and_source_witnesses"]=per
 report["coarse_mapping_response"]["affected_row_count_recomputed"]=int(affected.sum())
 report["coarse_mapping_response"]["note"]="Residual is frozen and O is recomputed as exact C plus residual. The per-row counterfactual residual delta assumes holding old O fixed and equals -deltaC."
 report["status"]="diagnostic_candidate_only_not_admitted_orientation_corrected"
 report["supersedes"]={"report":str(BASE),"reason":"Corrected determinant interpretation for authored alternating prism tetrahedron signs; raw negative signs do not imply inversion. Shape floor is applied to absolute normalized determinants."}
 out=OUT/"ankle-coupled-021-C014-mechanical-rebase-corrected.json"
 if out.exists():raise SystemExit("refusing overwrite existing output")
 out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n");json.loads(out.read_text())
 print(json.dumps({"report":str(out),"sha256":sha(out),"affected_coarse_rows":int(affected.sum()),"per_affected_row":per,"jacobian":report["mechanical_jacobian_diagnostic"]},indent=2))
if __name__=="__main__":main()
