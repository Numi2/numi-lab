#!/usr/bin/env python3
"""Non-admitted C014 coarse outer/layer-offset sensitivity to ankle candidate 021."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"ankle-coupled-geometry-weight-021"
C14=E/"joint-bmap-refinement-candidate-011-mechanical119-integration-001/source-C-reference-offset-diagnostic-014-exact-f32-residual-001"
DETAIL=E/"finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz"
REGCSV=E/"native-pose-checks/connected-hand-registration069i/registered-skin.csv"
CAND=OUT/"ankle-coupled-021-candidate.npz"
F32=OUT/"ankle-coupled-021-f32-emission-audit.json"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,float)
 if not len(x):return {"count":0}
 return {"count":int(len(x)),"min":float(np.min(x)),"p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),"p99":float(np.percentile(x,99)),"max":float(np.max(x))}
def main():
 S=np.load(C14/"coarse-surface.npz",allow_pickle=False)
 C=np.load(C14/"coarse-to-source-C014.npz",allow_pickle=False)
 L=np.load(C14/"candidate-layers-C014.npz",allow_pickle=False)
 G=np.load(DETAIL,allow_pickle=False)
 Q=np.load(CAND,allow_pickle=False)
 audit=json.loads(F32.read_text())
 Vbase=np.loadtxt(REGCSV,delimiter=",",skiprows=1,usecols=(0,2,3,4),dtype=np.float32)[:,1:4]
 Vnew=np.asarray(Q["reference_positions_m"],np.float32)
 if Vbase.shape!=Vnew.shape or Vbase.shape!=(61950,3):raise ValueError("reference source position table shape")
 if audit["candidate_npz_sha256"]!=sha(CAND):raise ValueError("F32 audit does not bind candidate")
 F=np.asarray(G["faces"],np.int64);Wbase=np.asarray(G["weights"],np.float32).copy()
 ids=np.asarray(Q["source_vertex_ids"],np.int64);Wcandidate=Wbase.copy()
 if not np.array_equal(Wbase[ids],Q["full_weights_before"]):raise ValueError("candidate source weights do not descend from exact 069i source")
 Wcandidate[ids]=np.asarray(Q["full_weights_after"],np.float32)
 if not np.array_equal(Vnew[ids],Q["reference_positions_m"][ids]):raise ValueError("position replay")
 if not np.array_equal(Vnew[np.setdiff1d(np.arange(len(Vnew)),ids)],Vbase[np.setdiff1d(np.arange(len(Vnew)),ids)]):raise ValueError("outside-source positions are not exact matched native CSV")
 cf=np.asarray(C["source_face"],np.int64);cb=np.asarray(C["source_barycentric"],np.float64);sv=np.asarray(S["source_vertex"],np.int64);vk=np.asarray(S["vertex_kind"],np.int8)
 if len(cf)!=len(sv) or len(cb)!=len(sv):raise ValueError("coarse map row mismatch")
 direct=(vk==0)
 if np.any(sv[direct]<0) or np.any(sv[direct]>=len(Vbase)):raise ValueError("invalid kind-zero source id")
 if np.any(cf[~direct]<0) or np.any(cf[~direct]>=len(F)):raise ValueError("invalid source face")
 if np.max(np.abs(cb[~direct].sum(1)-1.0))>2e-8 or np.min(cb[~direct])<-1e-10:raise ValueError("invalid C barycentric coordinates")
 tri=F[cf[~direct]]
 Cbase=np.empty((len(sv),3),np.float64);Cnew=np.empty_like(Cbase)
 Cbase[direct]=Vbase[sv[direct]];Cnew[direct]=Vnew[sv[direct]]
 Cbase[~direct]=np.einsum("ni,nij->nj",cb[~direct],Vbase[tri])
 Cnew[~direct]=np.einsum("ni,nij->nj",cb[~direct],Vnew[tri])
 residual=np.asarray(C["source_residual_vector_m"],np.float64)
 Oold=np.asarray(L["outer_world"],np.float64)
 if residual.shape!=Oold.shape:raise ValueError("residual shape")
 recon_err=np.linalg.norm(Cbase+residual-Oold,axis=1)
 if recon_err.max()>2e-6:raise ValueError(f"baseline exact C+residual parity failed {recon_err.max()}")
 Onew64=Cnew+residual
 Onew=Onew64.astype(np.float32).astype(np.float64)
 Iold=np.asarray(L["inner_world"],np.float32).astype(np.float64)
 Dold=np.asarray(L["deep_world"],np.float32).astype(np.float64)
 Oold32=Oold.astype(np.float32).astype(np.float64)
 skin_offset=(Iold-Oold32);deep_offset=(Dold-Oold32)
 Inew=(Onew+skin_offset).astype(np.float32).astype(np.float64)
 Dnew=(Onew+deep_offset).astype(np.float32).astype(np.float64)
 # Exact C-interpolated 86-column map under the candidate weight table.
 Wc0=np.empty((len(sv),86),np.float64);Wc1=np.empty_like(Wc0)
 Wc0[direct]=Wbase[sv[direct]];Wc1[direct]=Wcandidate[sv[direct]]
 Wc0[~direct]=np.einsum("ni,nij->nj",cb[~direct],Wbase[tri])
 Wc1[~direct]=np.einsum("ni,nij->nj",cb[~direct],Wcandidate[tri])
 if np.min(Wc1)<-1e-7 or np.max(np.abs(Wc1.sum(1)-1))>2e-6:raise ValueError("candidate coarse weight partition failure")
 srcrowmask=np.zeros(len(Vbase),bool);srcrowmask[ids]=True
 affected=direct & srcrowmask[sv]
 affected|=np.any(srcrowmask[F[cf]],axis=1)&(~direct)
 coarse_delta=np.linalg.norm(Cnew-Cbase,axis=1)
 Wdelta=np.sum(np.abs(Wc1-Wc0),axis=1)*.5
 # Include all coarse rows with nontrivial world or weight response, not merely direct source IDs.
 affected |= (coarse_delta>1e-12)|(Wdelta>1e-7)
 # Determinants use exactly authored 014/119 tetra list, packed Float32 layer positions.
 tet=np.asarray(L["tetrahedra"],np.int64);n=len(Oold)
 Xold=np.vstack([Oold32,Iold,Dold]).astype(np.float32).astype(np.float64)
 Xnew=np.vstack([Onew,Inew,Dnew]).astype(np.float32).astype(np.float64)
 if tet.min()<0 or tet.max()>=len(Xold):raise ValueError("tetrahedron index out of range")
 def det_metrics(X):
  x=X[tet];u=x[:,1]-x[:,0];v=x[:,2]-x[:,0];w=x[:,3]-x[:,0]
  det=np.einsum("ij,ij->i",u,np.cross(v,w))
  maxe=np.zeros(len(x))
  for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
   maxe=np.maximum(maxe,np.einsum("ij,ij->i",x[:,a]-x[:,b],x[:,a]-x[:,b]))
  norm=det/np.maximum(maxe,1e-30)**1.5
  return det,norm
 d0,n0=det_metrics(Xold);d1,n1=det_metrics(Xnew)
 ratio=np.divide(d1,d0,out=np.full_like(d1,np.nan),where=np.abs(d0)>1e-30)
 changed_tet=np.any(np.isin(tet,np.r_[np.flatnonzero(affected),n+np.flatnonzero(affected),2*n+np.flatnonzero(affected)]),axis=1)
 layer_names=np.asarray(L["tetra_layer"]) if "tetra_layer" in L else np.full(len(tet),-1)
 region=np.asarray(L["tetra_region"]) if "tetra_region" in L else np.full(len(tet),"unknown")
 layer_stats={}
 for k in np.unique(layer_names):
  m=layer_names==k
  layer_stats[str(int(k))]={"baseline":stats(n0[m]),"candidate":stats(n1[m]),"candidate_negative":int(np.sum(d1[m]<=0)),"candidate_below_1e-5":int(np.sum(n1[m]<1e-5))}
 perregion={}
 for r in np.unique(region):
  m=(region==r)&changed_tet
  if m.any():perregion[str(r)]={"changed_tet_count":int(m.sum()),"baseline_min_signed_det_normalized":float(n0[m].min()),"candidate_min_signed_det_normalized":float(n1[m].min()),"candidate_negative":int(np.sum(d1[m]<=0))}
 # Map coarse rows to their changed source IDs, and report which ancestry caused the response.
 source_per_row=[]
 for i in np.flatnonzero(affected):
  src=([int(sv[i])] if direct[i] else [int(x) for x in F[cf[i]] if srcrowmask[int(x)]])
  source_per_row.append({"coarse_row":int(i),"kind":int(vk[i]),"source_ids":sorted(set(src)),"C_delta_mm":float(coarse_delta[i]*1000),"W_half_L1_delta":float(Wdelta[i])})
 # Summary grouped by support class from dominant body columns via source bindings are not in this NPZ;
 # instead classify changed rows by the source weight column with maximum contribution change only.
 by_weight_column={}
 for col in range(Wdelta.shape[0] and Wc1.shape[1]):
  vals=Wdelta[affected]
  if not len(vals):break
  # Only emit columns with material influence in affected rows.
  delta_col=np.abs(Wc1[affected,col]-Wc0[affected,col])
  if np.any(delta_col>1e-7):
   by_weight_column[str(col)]={"rows_changed":int(np.sum(delta_col>1e-7)),"max_abs_weight_delta":float(delta_col.max()),"total_abs_weight_delta":float(delta_col.sum())}
 report={
  "schema":"numi.human.ankle-021-C014-mechanical-rebase-diagnostic.v1",
  "status":"diagnostic_candidate_only_not_admitted",
  "input_hashes":{"021_candidate_sha256":sha(CAND),"021_f32_audit_sha256":sha(F32),"069i_detail_geometry_sha256":sha(DETAIL),"C014_coarse_surface_sha256":sha(C14/"coarse-surface.npz"),"C014_coarse_to_source_sha256":sha(C14/"coarse-to-source-C014.npz"),"C014_candidate_layers_sha256":sha(C14/"candidate-layers-C014.npz"),"matched_reference_registered_csv_sha256":sha(REGCSV)},
  "coordinate_method":"Use the exact C014 direct-vertex rule for kind-0 rows and exact C014 source-face/barycentric rule for non-kind-0 rows. Evaluate C on the 021 matched-reference Float32 registered source table, then reconstruct O=C+the frozen C014 registered-frame residual. Retain C014 I-O and D-O Float32 offsets; cast new O/I/D to Float32 before evaluating the exact stored 119 prism tetrahedra.",
  "source_pose_parity":{"matched_reference_CSV_equals_C014_registration_csv":True,"candidate021_changed_source_rows":int(len(ids)),"candidate021_changed_source_faces":int(len(Q["changed_source_faces"])),"outside_changed_source_ids_are_bitwise_reference_CSV":True,"baseline_C_plus_residual_max_error_m":float(recon_err.max()),"candidate_C_plus_frozen_residual_O_max_error_m":float(np.max(np.linalg.norm(Onew64-Cnew-residual,axis=1)))},
  "coarse_mapping_response":{"coarse_rows":int(n),"affected_coarse_rows":int(affected.sum()),"C_delta_mm_all_affected":stats(coarse_delta[affected]*1000),"C_delta_mm_affected_ankle_candidates":stats(coarse_delta[affected]*1000),"coarse_weight_half_L1_delta":stats(Wdelta[affected]),"affected_rows_and_source_witnesses":source_per_row,"weight_column_changes":by_weight_column,"residual_if_outer_is_recomputed_C_plus_frozen_residual":{"delta_mm":stats(np.linalg.norm((Onew64-Cnew-residual),axis=1)*1000),"max_change_m":0.0},"implied_residual_delta_if_original_outer_were_frozen":{"delta_mm":stats(coarse_delta[affected]*1000),"note":"This is -deltaC for O_fixed; actual candidate reconstruction instead moves O by deltaC and retains the C014 residual."}},
  "outer_and_offsets":{"O_delta_mm":stats(np.linalg.norm(Onew-Oold32,axis=1)*1000),"skin_offset_preserved_bitwise":bool(np.array_equal(Iold-Oold32,skin_offset)),"fat_offset_preserved_bitwise":bool(np.array_equal(Dold-Oold32,deep_offset)),"candidate_skin_offset_length_change_mm":stats(np.linalg.norm(Inew-Onew,axis=1)*1000-np.linalg.norm(Iold-Oold32,axis=1)*1000),"candidate_fat_offset_length_change_mm":stats(np.linalg.norm(Dnew-Inew,axis=1)*1000-np.linalg.norm(Dold-Iold,axis=1)*1000)},
  "mechanical_jacobian_diagnostic":{"tet_count":int(len(tet)),"candidate_negative_determinants":int(np.sum(d1<=0)),"baseline_negative_determinants":int(np.sum(d0<=0)),"candidate_below_native_1e-5_normalized_floor":int(np.sum(n1<1e-5)),"baseline_below_native_1e-5_normalized_floor":int(np.sum(n0<1e-5)),"candidate_min_signed_normalized_det":float(n1.min()),"baseline_min_signed_normalized_det":float(n0.min()),"candidate_min_signed_det_m3":float(d1.min()),"baseline_min_signed_det_m3":float(d0.min()),"minimum_signed_J_ratio_candidate_over_baseline":float(np.nanmin(ratio)),"negative_candidate_tet_ids":np.flatnonzero(d1<=0).astype(int).tolist(),"new_negative_tet_ids":np.flatnonzero((d1<=0)&(d0>0)).astype(int).tolist(),"changed_region_layers":perregion,"all_layer_stats":layer_stats,"changed_tet_count":int(changed_tet.sum()),"changed_tet_negative":int(np.sum((d1<=0)&changed_tet))},
  "nonclaims":["This preserves O-I and O-D offsets only as a first-order diagnostic; it does not regenerate physically admissible layers or donor-local deformation maps.","No exact self/cross-surface or nearby-bone audit is implied by this coarse mechanical calculation.","No runtime/native FEM transfer, dynamics, mass or physical acceptance claim.","No promotion or source/production file modification."]
 }
 out=OUT/"ankle-coupled-021-C014-mechanical-rebase-diagnostic.json"
 if out.exists():raise SystemExit("refusing overwrite "+str(out))
 out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 json.loads(out.read_text())
 print(json.dumps({"report":str(out),"sha256":sha(out),"source_pose_parity":report["source_pose_parity"],"mapping":{k:v for k,v in report["coarse_mapping_response"].items() if k not in ("affected_rows_and_source_witnesses","weight_column_changes")},"jacobian":{k:v for k,v in report["mechanical_jacobian_diagnostic"].items() if k not in ("negative_candidate_tet_ids","new_negative_tet_ids","changed_region_layers","all_layer_stats")}},indent=2))
if __name__=="__main__":main()
