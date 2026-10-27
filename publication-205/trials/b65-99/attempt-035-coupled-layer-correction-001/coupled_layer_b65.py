#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
B=S/"per-bank-workers-20261027/b65-99"
BASE=S/"per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz"
LAYER=BASE.with_name("skin-fat-layers-541.npz")
OUTER=B/"attempt-032-shape-pass-exact-diagnostic-001/candidate-rejected-layer-gate.npz"
PREV=B/"attempt-032-shape-pass-exact-diagnostic-001/exact-diagnostic-report.json"
FIT=B/"attempt-031-partial-inside-slack-layer-fit-001/fit_806_reconstruct_layers_partial_inside_slack.py"
TRANSFER=B/"attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
OUTDIR=B/"attempt-035-coupled-layer-correction-001"
BASE_PIN="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
OUTER_PIN="5c98028c91d1adac1052c660d7e1c7715ef3ed2aeb643fe282bfa6c5a1292107"
STEP_MM=0.20
MAX_ITER=2
THICKNESS_REL_TOL=0.10
THICKNESS_ABS_TOL_MM=0.02
VOLUME_POSITIVE_FLOOR=0.20
COND_THRESHOLD=1e-5
FEAS_TOL_MM3=1e-7

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def tet_vol_grad(points,sign):
 p0,p1,p2,p3=np.asarray(points,np.float64)
 v=float(sign*np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6.0)
 g1=sign*np.cross(p2-p0,p3-p0)/6.0
 g2=sign*np.cross(p3-p0,p1-p0)/6.0
 g3=sign*np.cross(p1-p0,p2-p0)/6.0
 g0=-g1-g2-g3
 return v,np.asarray([g0,g1,g2,g3])

def tet_volume(points,sign):
 p0,p1,p2,p3=np.asarray(points,np.float64)
 return float(sign*np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6.0)

def main():
 OUTDIR.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 pins={str(p):sha(p) for p in (BASE,LAYER,OUTER,PREV,FIT,TRANSFER,HELPER,AUDIT_TOOL)}
 if pins[str(BASE)]!=BASE_PIN or pins[str(OUTER)]!=OUTER_PIN:raise RuntimeError("candidate/base pin mismatch")
 fit=loadmod("b65layerfit",FIT);transfer=loadmod("b65layerflags",TRANSFER)
 with np.load(BASE,allow_pickle=False) as z:Z={k:np.asarray(z[k]).copy() for k in z.files}
 with np.load(OUTER,allow_pickle=False) as z:C={k:np.asarray(z[k]).copy() for k in z.files}
 with np.load(LAYER,allow_pickle=False) as z:L={k:np.asarray(z[k]).copy() for k in z.files}
 P32=np.asarray(Z["points_world_m"],np.float32);O32=np.asarray(C["points_world_m"],np.float32);F=np.asarray(Z["source_O_faces"],np.int64)
 O0=np.asarray(L["outer_points_f32"],np.float32).astype(np.float64);I0=np.asarray(L["inner_points_f32"],np.float32).astype(np.float64);D0=np.asarray(L["deep_points_f32"],np.float32).astype(np.float64)
 if not np.array_equal(O0,P32.astype(np.float64)):raise RuntimeError("baseline layer/candidate outer mismatch")
 if not np.array_equal(F,C["source_O_faces"]):raise RuntimeError("outer candidate changed topology")
 topo=json.load(open(B/"attempt-002/fan-hole-ancestry-check.json"));prep=json.load(open(B/"attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json"))
 rows=sorted(set(map(int,topo["two_ring_patch_summary"][0]["face_rows"]))|set(map(int,topo["fan_expansion"][0]["fan_face_rows"]))|set(map(int,prep["measured_expansion"]["added_face_rows"])))
 boundary,boundary_nodes,patchmeta=fit.patch_topology(F,rows)
 patchnodes=np.unique(F[np.asarray(rows,np.int64)].reshape(-1));boundary=np.asarray(sorted(map(int,boundary_nodes)),np.int64)
 free=np.asarray(sorted(set(map(int,patchnodes))-set(map(int,boundary))),np.int64)
 if len(rows)!=248 or len(patchnodes)!=156 or len(boundary)!=62:raise RuntimeError("pinned 248 disk patch identity changed")
 layer_rows=np.flatnonzero(np.any(np.isin(F,patchnodes),axis=1)).astype(np.int64)
 oriented=np.asarray(L["outer_faces"],np.int64)
 if not np.array_equal(oriented,F[:,[0,2,1]]):raise RuntimeError("layer winding mismatch")
 sd=np.asarray(L["skin_thickness_transferred_m"],np.float64);fd=np.asarray(L["fat_thickness_transferred_m"],np.float64)
 # Initial trial is the 541 offset carried along with the fixed emitted outer candidate.
 dm=O32.astype(np.float64)-P32.astype(np.float64)
 Icarry=(np.asarray(L["inner_points_f64"],np.float64)+dm).astype(np.float32).astype(np.float64)
 Dcarry=(np.asarray(L["deep_points_f64"],np.float64)+dm).astype(np.float32).astype(np.float64)
 Ocur=O32.astype(np.float64)
 old_skin=transfer.face_flags(O0,I0,oriented[layer_rows],sd)
 old_fat=transfer.face_flags(I0,D0,oriented[layer_rows],fd)
 skin_target=np.linalg.norm(I0-O0,axis=1)*1000.0
 fat_target=np.linalg.norm(D0-I0,axis=1)*1000.0
 skin_tol=np.maximum(THICKNESS_ABS_TOL_MM,THICKNESS_REL_TOL*skin_target)
 fat_tol=np.maximum(THICKNESS_ABS_TOL_MM,THICKNESS_REL_TOL*fat_target)
 skin_low=np.maximum(0.0,skin_target-skin_tol);skin_high=skin_target+skin_tol
 fat_low=np.maximum(0.0,fat_target-fat_tol);fat_high=fat_target+fat_tol
 free_map={int(n):i for i,n in enumerate(free)};nfree=len(free);nxyz=6*nfree
 def vid(layer,node,axis):
  j=free_map.get(int(node))
  return None if j is None else (0 if layer=="I" else 3*nfree)+3*j+int(axis)
 # Build the exact appender tetra order for all layer faces, using mm coordinates.
 def make_specs(Omm,Imm,Dmm):
  specs=[]
  for layer in ("skin","fat"):
   for rr in layer_rows:
    src=oriented[int(rr)];a,b,c=sorted(map(int,src))
    inv=(int(src[0]>src[1])+int(src[0]>src[2])+int(src[1]>src[2]))&1
    sg=1.0 if inv else -1.0
    out,inn=(("O","I") if layer=="skin" else ("I","D"))
    nodesets=[[(out,a),(out,b),(out,c),(inn,c)],
      [(out,a),(out,b),(inn,b),(inn,c)],
      [(out,a),(inn,a),(inn,b),(inn,c)]]
    signs=[sg,-sg,sg]
    for ti,(labels,sign) in enumerate(zip(nodesets,signs)):
     pts=[]
     for lev,node in labels:
      arr={"O":Omm,"I":Imm,"D":Dmm}[lev]
      pts.append(arr[node])
     val,grads=tet_vol_grad(pts,sign)
     maxedge=max(float(np.linalg.norm(pts[i]-pts[j])) for i in range(4) for j in range(i+1,4))
     vargrads={}
     for (lev,node),g in zip(labels,grads):
      if lev not in ("I","D"):continue
      key=vid(lev,node,0)
      if key is None:continue
      base=0 if lev=="I" else 3*nfree
      col=base+3*free_map[int(node)]
      vargrads[col]=vargrads.get(col, np.zeros(3))+g
     specs.append({"layer":layer,"face":int(rr),"tet":int(ti),"volume":val,"grad":vargrads,"maxedge_mm":maxedge})
  return specs
 O0mm=O0*1000.;I0mm=I0*1000.;D0mm=D0*1000.
 Ocurmm=Ocur*1000.;Icur=Icarry.copy();Dcur=Dcarry.copy()
 def flags_for(Ix,Dx):
  sf=transfer.face_flags(Ocur,Ix,oriented[layer_rows],sd);ff=transfer.face_flags(Ix,Dx,oriented[layer_rows],fd)
  return sf,ff
 def make_floors(specs):
  base_specs=make_specs(O0mm,I0mm,D0mm)
  if len(base_specs)!=len(specs):raise RuntimeError("tet inventory mismatch")
  floors=[];basevol=[];base_det=[]
  for k,(s0,scur) in enumerate(zip(base_specs,specs)):
   v0=float(s0["volume"]);ndet=float((old_skin if s0["layer"]=="skin" else old_fat)["normalized_det"][np.searchsorted(layer_rows,s0["face"]),s0["tet"]])
   floor=VOLUME_POSITIVE_FLOOR*v0 if v0>0 else v0
   if v0>0:
    cond=ndet if ndet<COND_THRESHOLD else COND_THRESHOLD
    cond_vol=cond*(s0["maxedge_mm"]**3)/6.0
    floor=max(floor,cond_vol)
   floors.append(float(floor));basevol.append(v0);base_det.append(ndet)
  return np.asarray(floors),np.asarray(basevol),np.asarray(base_det)
 specs0=make_specs(Ocurmm,Icur*1000.,Dcur*1000.)
 floors,base_vols,base_ndet=make_floors(specs0)
 report={"schema":"numi.human.bank65_99.coupled-layer-correction.v1","status":"STARTED","physical_or_native_admission":False,
  "pins":pins,"fixed_outer_candidate_sha256":OUTER_PIN,"patch":{"face_count":len(rows),"node_count":len(patchnodes),"boundary_node_count":len(boundary),
   "movable_layer_node_count":len(free),"layer_face_count":len(layer_rows),"is_single_disk":patchmeta["is_single_disk"]},
  "method":"At fixed outer Float32 points, adjust only patch interior I/D nodes by a two-step relinearized HiGHS L1 fit. Hard constraints cover baseline-positive tet volume retention, nonworsening signed volume for baseline-invalid tets, baseline/new conditioning floors, per-node 541 thickness norms, and fixed shared O/I/D boundary.",
  "limits":{"max_steps":MAX_ITER,"per_step_component_trust_mm":STEP_MM,"skin_fat_thickness_relative_tolerance":THICKNESS_REL_TOL,
   "skin_fat_thickness_absolute_tolerance_mm":THICKNESS_ABS_TOL_MM,"positive_tet_volume_floor_fraction":VOLUME_POSITIVE_FLOOR,
   "normalized_det_floor":COND_THRESHOLD,"exact_margin_tolerance_mm3":FEAS_TOL_MM3},"iterations":[]}
 currentI=Icur.copy();currentD=Dcur.copy();allpass=False
 for iteration in range(1,MAX_ITER+1):
  specs=make_specs(Ocurmm,currentI*1000.,currentD*1000.)
  G=np.zeros((len(specs),nxyz),np.float64)
  for i,spec in enumerate(specs):
   for col,g in spec["grad"].items():G[i,col:col+3]=g
  vcur=np.asarray([s["volume"] for s in specs],np.float64)
  # Linearized signed-volume and conditioning constraints.
  avol=-G; bvol=vcur-floors
  # Linearized two-sided per-node thickness intervals.
  tr=[];tc=[];tv=[];tb=[];row=0
  Omm=Ocurmm;Imm=currentI*1000.;Dmm=currentD*1000.
  for node in free:
   n=int(node);ii=free_map[n]
   sv=Imm[n]-Omm[n];sn=float(np.linalg.norm(sv));svdir=sv/max(sn,1e-15)
   low=float(skin_low[n]);high=float(skin_high[n])
   # -u.dI <= current_norm-low; +u.dI <= high-current_norm
   for sign,rhs in ((-1.0,sn-low),(1.0,high-sn)):
    for axis in range(3):
     tr.append(row);tc.append(3*ii+axis);tv.append(sign*svdir[axis])
    tb.append(rhs);row+=1
   fv=Dmm[n]-Imm[n];fn=float(np.linalg.norm(fv));fvdir=fv/max(fn,1e-15)
   for sign,rhs in ((-1.0,fn-float(fat_low[n])),(1.0,float(fat_high[n])-fn)):
    for axis in range(3):
     tr.append(row);tc.append(3*nfree+3*ii+axis);tv.append(sign*fvdir[axis])
     tr.append(row);tc.append(3*ii+axis);tv.append(-sign*fvdir[axis])
    tb.append(rhs);row+=1
  Ath=sparse.coo_matrix((tv,(tr,tc)),shape=(row,nxyz)).tocsr()
  # L1 objective auxiliaries.
  Iden=sparse.eye(nxyz,format="csr");Zvol=sparse.csr_matrix((G.shape[0],nxyz));Zth=sparse.csr_matrix((Ath.shape[0],nxyz))
  Avol_lp=sparse.hstack((sparse.csr_matrix(avol),Zvol),format="csr")
  Ath_lp=sparse.hstack((Ath,Zth),format="csr")
  Aabs1=sparse.hstack((Iden,-Iden),format="csr");Aabs2=sparse.hstack((-Iden,-Iden),format="csr")
  A=sparse.vstack((Avol_lp,Ath_lp,Aabs1,Aabs2),format="csr")
  bvec=np.r_[bvol,np.asarray(tb),np.zeros(2*nxyz)]
  obj=np.r_[np.zeros(nxyz),np.ones(nxyz)]
  bounds=[(-STEP_MM,STEP_MM)]*nxyz+[(0,None)]*nxyz
  lp=linprog(obj,A_ub=A,b_ub=bvec,bounds=bounds,method="highs",options={"time_limit":30.0,"presolve":True})
  rec={"iteration":iteration,"solver_success":bool(lp.success),"solver_status":int(lp.status),"message":str(lp.message),
   "tet_constraint_count":int(len(specs)),"variable_count":int(nxyz),"LP_rows":int(A.shape[0]),
   "baseline_positive_tets":int(np.count_nonzero(base_vols>0)),"baseline_nonpositive_tets":int(np.count_nonzero(base_vols<=0)),
   "initial_min_volume_margin_mm3":float(np.min(vcur-floors))}
  if not lp.success:
   rec["status"]="LP_INFEASIBLE_OR_FAILED";report["iterations"].append(rec);break
  dx=np.asarray(lp.x[:nxyz],np.float64)
  trialI=currentI.copy();trialD=currentD.copy()
  trialI[free]+=dx[:3*nfree].reshape(nfree,3)/1000.
  trialD[free]+=dx[3*nfree:].reshape(nfree,3)/1000.
  trialI[boundary]=I0[boundary];trialD[boundary]=D0[boundary]
  trialI=trialI.astype(np.float32).astype(np.float64);trialD=trialD.astype(np.float32).astype(np.float64)
  trialI[boundary]=I0[boundary];trialD[boundary]=D0[boundary]
  ss,ff=flags_for(trialI,trialD)
  new_specs=make_specs(Ocurmm,trialI*1000.,trialD*1000.)
  vnew=np.asarray([s["volume"] for s in new_specs],np.float64)
  margins=vnew-floors
  volume_fail=np.flatnonzero(margins < -FEAS_TOL_MM3)
  old_auth=np.r_[old_skin["authored_bad"],old_fat["authored_bad"]]
  new_auth=np.r_[ss["authored_bad"],ff["authored_bad"]]
  old_np=np.concatenate([old_skin["nonpositive"],old_fat["nonpositive"]],axis=0)
  new_np=np.concatenate([ss["nonpositive"],ff["nonpositive"]],axis=0)
  old_cd=np.concatenate([old_skin["conditioned"],old_fat["conditioned"]],axis=0)
  new_cd=np.concatenate([ss["conditioned"],ff["conditioned"]],axis=0)
  auth_add=np.flatnonzero(new_auth&~old_auth)
  np_add=np.argwhere(new_np&~old_np);cd_add=np.argwhere(new_cd&~old_cd)
  skinlen=np.asarray(ss["normalized_det"]);fatlen=np.asarray(ff["normalized_det"])
  actual_skin=np.linalg.norm(trialI[free]-Ocur[free],axis=1)*1000.
  actual_fat=np.linalg.norm(trialD[free]-trialI[free],axis=1)*1000.
  thickness_pass=bool(np.all(actual_skin>=skin_low[free]-1e-5) and np.all(actual_skin<=skin_high[free]+1e-5) and
                      np.all(actual_fat>=fat_low[free]-1e-5) and np.all(actual_fat<=fat_high[free]+1e-5))
  boundarypass=bool(np.array_equal(trialI[boundary].view(np.uint32),I0[boundary].view(np.uint32)) and np.array_equal(trialD[boundary].view(np.uint32),D0[boundary].view(np.uint32)))
  rec.update({"max_inner_correction_from_carried_mm":float(np.max(np.linalg.norm((trialI-Icarry)*1000.,axis=1))),
   "max_deep_correction_from_carried_mm":float(np.max(np.linalg.norm((trialD-Dcarry)*1000.,axis=1))),
   "emitted_f32_volume_gate":{"pass":len(volume_fail)==0,"failure_count":int(len(volume_fail)),
    "minimum_margin_mm3":float(margins.min(initial=np.inf)),
    "first_failures":[{"layer":new_specs[i]["layer"],"face_row":new_specs[i]["face"],"tet":new_specs[i]["tet"],
     "baseline_signed_volume_mm3":float(base_vols[i]),"current_signed_volume_mm3":float(vcur[i]),
     "new_signed_volume_mm3":float(vnew[i]),"required_floor_mm3":float(floors[i]),"margin_mm3":float(margins[i])} for i in volume_fail[:16]]},
   "emitted_f32_added_flags":{"authored_bad_face_count":int(len(auth_add)),
     "authored_bad_faces":[[("skin" if i<len(layer_rows) else "fat"),int(layer_rows[i%len(layer_rows)])] for i in auth_add[:16]],
     "nonpositive_tets":[[("skin" if i<len(layer_rows) else "fat"),int(layer_rows[i%len(layer_rows)]),int(j)] for i,j in np_add[:16]],
     "conditioned_tets":[[("skin" if i<len(layer_rows) else "fat"),int(layer_rows[i%len(layer_rows)]),int(j)] for i,j in cd_add[:16]],
     "authored_added_count":int(len(auth_add)),"nonpositive_added_count":int(len(np_add)),"conditioned_added_count":int(len(cd_add))},
   "thickness_gate":{"pass":thickness_pass,"skin_norm_min_mm":float(actual_skin.min()),"skin_norm_max_mm":float(actual_skin.max()),
     "fat_norm_min_mm":float(actual_fat.min()),"fat_norm_max_mm":float(actual_fat.max()),
     "max_skin_error_from_541_norm_mm":float(np.max(np.abs(actual_skin-skin_target[free]))),
     "max_fat_error_from_541_norm_mm":float(np.max(np.abs(actual_fat-fat_target[free])))},
   "shared_boundary_bitwise_pass":boundarypass,
   "lp_objective":float(lp.fun),"lp_iterations":int(lp.nit or 0)})
  rec["exact_cell_gate_pass"]=bool(len(volume_fail)==0 and len(auth_add)==0 and len(np_add)==0 and len(cd_add)==0 and thickness_pass and boundarypass)
  iterfile=OUTDIR/f"iteration-{iteration:02d}-layer-points.npz"
  np.savez_compressed(iterfile,inner_points_f32=trialI.astype(np.float32),deep_points_f32=trialD.astype(np.float32))
  rec["layer_points_artifact"]={"path":str(iterfile),"sha256":sha(iterfile)}
  report["iterations"].append(rec)
  currentI,currentD=trialI,trialD
  if rec["exact_cell_gate_pass"]:
   allpass=True;break
 # Save the last exact-F32 layer state, whether accepted by the local gates or rejected.
 layer_out=OUTDIR/("skin-fat-layers-local-corrected.npz" if allpass else "skin-fat-layers-rejected.npz")
 with np.load(LAYER,allow_pickle=False) as z:Lout={k:np.asarray(z[k]).copy() for k in z.files}
 Lout.update({"outer_points_f64":Ocur,"outer_points_f32":O32,"inner_points_f64":currentI,"inner_points_f32":currentI.astype(np.float32),
  "deep_points_f64":currentD,"deep_points_f32":currentD.astype(np.float32)})
 np.savez_compressed(layer_out,**Lout)
 report.update({"status":"LOCAL_LAYER_GATES_PASS" if allpass else "REJECTED_LOCAL_LAYER_GATES",
   "candidate_outer_unchanged":True,"fixed_outer_candidate_sha256":OUTER_PIN,"local_layer_artifact":{"path":str(layer_out),"sha256":sha(layer_out)},
   "outer_target_and_inside_metrics_inherited_from_exact_diagnostic":json.load(open(PREV))["target_65_99"]})
 self={"run":False,"reason":"coupled layer gate did not pass"}
 if allpass:
  exact_audit=loadmod("b65layer_exact_audit",AUDIT_TOOL);exact=exact_audit.exact_module()
  changed_nodes=np.flatnonzero(np.any(np.asarray(C["points_world_m"],np.float32).view(np.uint32)!=P32.view(np.uint32),axis=1))
  changed_faces=np.flatnonzero(np.any(np.isin(F,changed_nodes),axis=1))
  recs=fit.mesh_records(O32.astype(np.float64),F[changed_faces],changed_faces,np.arange(len(O32),dtype=np.int64),exact)
  internal=exact._audit_pair(recs,recs,same_surface=True)
  comp=np.asarray([r for r in range(len(F)) if r not in set(map(int,changed_faces))],np.int64)
  xyz=O32.astype(np.float64)[F[comp]];lo=O32.astype(np.float64)[F[changed_faces]].min(axis=(0,1));hi=O32.astype(np.float64)[F[changed_faces]].max(axis=(0,1))
  cl=xyz.min(axis=1);ch=xyz.max(axis=1);active=np.flatnonzero(np.all(ch>=lo,axis=1)&np.all(cl<=hi,axis=1));comp_rows=comp[active]
  crec=fit.mesh_records(O32.astype(np.float64),F[comp_rows],comp_rows,np.arange(len(O32),dtype=np.int64),exact)
  other=exact._audit_pair(recs,crec,same_surface=True)
  self={"run":True,"patch_internal_pair_count":int(internal["count"]),"patch_vs_complement_pair_count":int(other["count"]),
   "complement_aabb_active_face_count":int(len(comp_rows)),"pass":int(internal["count"])==0 and int(other["count"])==0}
 report["exact_skin_self_screen"]=self
 report["elapsed_seconds"]=time.monotonic()-started
 rp=OUTDIR/"coupled-layer-correction-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"report":str(rp),"report_sha256":sha(rp),"script_sha256":sha(Path(__file__)),
  "layer":str(layer_out),"layer_sha256":sha(layer_out),"iterations":report["iterations"],"self":self},indent=2,allow_nan=False))
if __name__=="__main__":main()

