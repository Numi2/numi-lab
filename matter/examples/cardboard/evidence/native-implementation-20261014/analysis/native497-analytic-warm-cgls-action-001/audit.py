#!/usr/bin/env python3
"""Independent analytic material-to-nodal action audit for the native497 warm CGLS total vector."""
import hashlib, json, pathlib, struct, sys, time, math
import numpy as np
Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/warm-cgls-control20-diagnostic-replay-497"
FD=CASE/"diagnostics/capture-operator-fd-capture/capture"
SJ=CASE/"diagnostics/capture-base-cgls-capture/capture.state.json"
SB=CASE/"diagnostics/capture-base-cgls-capture/capture.state.bin"
CG=CASE/"diagnostics/capture-base-cgls-capture/capture"
OP=Q/"analysis/native497-damped-lsqr-candidates-003"
OUT=Q/"analysis/native497-analytic-warm-cgls-action-001"
TOOLS=Q/"snapshots/source-portable-paper-reference-185/matter/tools"
sys.path.insert(0,str(TOOLS))
import cardboard_bilinear_fp64_reference as base
import cardboard_compensated_fp64_reference as strict

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def writej(p,x):
 p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")

def rowstats(expected,actual):
 e=np.asarray(expected,dtype=np.float64); a=np.asarray(actual,dtype=np.float64)
 if e.shape!=a.shape: raise ValueError((e.shape,a.shape))
 n=len(e); ef=e.reshape(n,-1); af=a.reshape(n,-1)
 diff=af-ef
 en=np.linalg.norm(ef,axis=1); dn=np.linalg.norm(diff,axis=1); an=np.linalg.norm(af,axis=1)
 rel=np.full(n,np.nan); np.divide(dn,en,out=rel,where=en>0)
 finite=np.isfinite(ef).all(1)&np.isfinite(af).all(1)
 pos=finite&(en>0)
 comp=np.abs(diff); denom=np.abs(ef)
 compmask=finite[:,None]&(denom>0)
 crel=np.full_like(comp,np.nan); np.divide(comp,denom,out=crel,where=compmask)
 idx=int(np.nanargmax(rel)) if pos.any() else None
 return {
  "rows":n,"finite_rows":int(finite.sum()),"nonfinite_rows":int((~finite).sum()),
  "zero_expected_norm_rows":int(np.sum(finite&(en==0))),
  "expected_global_l2":float(np.linalg.norm(ef[finite])) if finite.any() else None,
  "actual_global_l2":float(np.linalg.norm(af[finite])) if finite.any() else None,
  "difference_global_l2":float(np.linalg.norm(diff[finite])) if finite.any() else None,
  "global_relative_l2":float(np.linalg.norm(diff[finite])/max(np.linalg.norm(ef[finite]),1e-300)) if finite.any() else None,
  "max_row_absolute_l2":float(np.max(dn[finite])) if finite.any() else None,
  "max_row_expected_l2":float(np.max(en[finite])) if finite.any() else None,
  "max_row_relative_l2_nonzero_expected":float(np.nanmax(rel)) if pos.any() else None,
  "worst_row_index_nonzero_expected":idx,
  "max_component_absolute_error":float(np.max(comp[finite])) if finite.any() else None,
  "max_component_relative_error_nonzero_expected":float(np.nanmax(crel)) if compmask.any() else None,
  "component_relative_zero_expected_count":int(np.sum(finite[:,None]&(denom==0)))
 }

def csr_load(path):
 z=np.load(path,allow_pickle=False)
 return {"indptr":z["indptr"],"indices":z["indices"],"data":z["data"],"shape":tuple(int(v) for v in z["shape"])}

def csr_mv(m,x):
 x=np.asarray(x,dtype=np.float64); nrow,ncol=m["shape"]
 if x.shape!=(ncol,):raise ValueError("CSR vector shape")
 y=np.empty(nrow,dtype=np.float64)
 for i in range(nrow):
  lo,hi=int(m["indptr"][i]),int(m["indptr"][i+1])
  y[i]=np.dot(m["data"][lo:hi],x[m["indices"][lo:hi]])
 return y

def load_state():
 meta=json.loads(SJ.read_text()); raw=SB.read_bytes()
 if not meta["complete"] or len(raw)!=meta["payload"]["byte_count"] or sha(SB)!=meta["payload"]["sha256"]:raise ValueError("state binary integrity")
 arrays={}
 for d in meta["arrays"]:
  b=raw[d["offset"]:d["offset"]+d["byte_count"]]
  if len(b)!=d["byte_count"] or hashlib.sha256(b).hexdigest()!=d["sha256"]:raise ValueError("array integrity "+d["name"])
  shape=d["shape"]
  if d["dtype"]=="float32x4":arrays[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(shape+[4])
  elif d["dtype"]=="float32":arrays[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(shape)
  elif d["dtype"]=="record:NMFEMNodeStateGPU":arrays[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(shape+[20])
  elif d["dtype"]=="uint32":arrays[d["name"]]=np.frombuffer(b,"<u4").copy().reshape(shape)
 return meta,arrays

def read_cgls():
 raw=CG.read_bytes(); h=struct.unpack_from("<8I",raw)
 if h[:7]!=(0x534c4743,3,2,882,3528,64,13):raise ValueError("CGLS header mismatch "+str(h))
 U,E,N=h[4],h[2],h[3]; sizes=[U*16]*7+[U*4,E*16,E*16,E*4,E*4,E*N*3*16]
 names=["x","r","s","p","Ax","b","mask","environment","scalars","control","errors","selected","snapshot"]
 seg=[]; off=32
 for size in sizes:
  off=(off+15)&~15; seg.append(raw[off:off+size]);off+=size
 if off!=len(raw):raise ValueError("CGLS payload length")
 out={}
 for name,b in zip(names,seg):
  if name in ("x","r","s","p","Ax","b","mask"):out[name]=np.frombuffer(b,"<f4").copy().reshape(U,4)
  elif name=="environment" or name in ("control","errors","selected"):out[name]=np.frombuffer(b,"<u4").copy()
  elif name in ("scalars",):out[name]=np.frombuffer(b,"<f4").copy().reshape(E,4)
  else:out[name]=b
 out["header"]=h
 return out

def parse_capture():
 head=None; nodes={}; ts={}
 with FD.open() as f:
  for line in f:
   r=json.loads(line);k=r.get("record")
   if k=="header":head=r
   elif k=="node" and r["environment"]==0:nodes[int(r["node_id"])]=r
   elif k=="tetrahedron" and r["environment"]==0:
    c=r["center"];tid=int(r["tet_id"])
    ts[tid]={"id":tid,"mat":int(r["material_index"]),"nodes":np.asarray(r["nodes"],np.int64),
     "vol":float(np.float32(r["rest_volume_m3"])),"inv":np.asarray(r["inverse_rest_rows_per_m"],np.float32)[:,:3],
     "q":np.asarray(r["material_frame_quaternion_xyzw"],np.float32),
     "F":np.asarray(c["F_local"],np.float32).astype(np.float64)+np.asarray(c["F_local_low"],np.float32).astype(np.float64),
     "P":np.asarray(c["P_local_Pa"],np.float32).astype(np.float64),
     "old":np.asarray(c["accepted_state"],np.float32),
     "native_state":np.asarray(c["projected_state"],np.float32).astype(np.float64)+np.asarray(c.get("projected_state_low",np.zeros(15)),np.float32).astype(np.float64),
     "C":np.asarray(r["cached_algorithmic_tangent_local"]["matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"],np.float32).reshape(9,9),
     "native_dir":np.asarray(c["F_direction_local_per_s"],np.float32),
     "native_tangent":np.asarray(c["tangent_local_Pa_per_s"],np.float32),
     "native_element":np.asarray(c["operator_force_node_world_N_per_s"],np.float32),
     "inverse_rows":np.asarray(r["inverse_rest_rows_per_m"],np.float32)[:,:3],
     "volume_f32":np.float32(r["rest_volume_m3"])}
 if head is None or len(nodes)!=882 or len(ts)!=1920:raise ValueError("capture incomplete")
 return head,nodes,[ts[i] for i in range(1920)]

def quatR(q):
 q=np.asarray(q,np.float64);q=q/np.linalg.norm(q);x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def make_force(stress_world,tets,node_count):
 element=np.empty((len(tets),4,3),np.float64)
 for i,t in enumerate(tets):
  h=-t["vol"]*stress_world[i]@t["inv"].astype(np.float64).T
  element[i,1:]=h.T;element[i,0]=-element[i,1:].sum(axis=0)
 nodal=np.zeros((node_count,3),np.float64)
 ids=np.asarray([t["nodes"] for t in tets],np.int64)
 for s in range(4):np.add.at(nodal,ids[:,s],element[:,s])
 return element,nodal

def main():
 start=time.time()
 if not OUT.exists() or any(q.name not in ("audit.py","attempt-001.json") for q in OUT.iterdir()): raise SystemExit("refusing non-fresh output namespace")
 meta,sa=load_state();cg=read_cgls();header,nodes,tets=parse_capture()
 if (meta["root"],meta["newton_iteration"],header["root"],header["newton_iteration"])!=(20,63,20,63):raise ValueError("capture identity")
 if header["world_fingerprint_hex"]!=meta["world_fingerprint_u64_hex"] or header["execution_fingerprint_hex"]!=meta["device_program_fingerprint_u64_hex"]:raise ValueError("fingerprint mismatch")
 E,N,T=2,882,1920;dt=float(np.float32(header["dt_seconds"]))
 acc=sa["accepted_nodes"].reshape(E,N,20)[0];cand=sa["candidate_nodes"].reshape(E,N,20)[0]
 histories=sa["accepted_material_history"].reshape(E,T,15)[0].astype(np.float64)
 tags=cand[:,11];free=(tags!=1)&(tags!=3)&(acc[:,3]>0)
 opv=np.load(OP/"operator.vectors.npz",allow_pickle=False)
 free_ids=opv["free_node_ids"].astype(np.int64)
 if not np.array_equal(free_ids,np.flatnonzero(free)):raise ValueError("operator free-node ordering mismatch")
 xfull=cg["x"][:E*N].reshape(E,N,4)
 x=xfull[0,:,:3].astype(np.float64)
 if np.max(np.abs(x[~free]))!=0:raise ValueError("nonzero CGLS direction on constrained rows")
 xfree=x[free_ids].reshape(-1)
 mat=np.asarray([t["mat"] for t in tets],np.int32)
 paper=np.flatnonzero((mat==0)|(mat==1));glue=np.flatnonzero(mat==2)
 old=np.zeros((T,15),np.float32)
 old[paper]=np.asarray([tets[int(i)]["old"] for i in paper],np.float32)
 history_mismatch=int(np.count_nonzero(old[paper].view(np.uint32)!=histories[paper].astype(np.float32).view(np.uint32)))
 if history_mismatch:raise ValueError("accepted paper history differs from sealed state")
 params=np.asarray(header["parameters_by_environment"][0],np.float32).astype(np.float64)
 desc={int(d["index"]):d for d in header["materials"]}
 F=np.asarray([t["F"] for t in tets]); R=np.asarray([quatR(t["q"]) for t in tets])
 inverse=np.asarray([t["inv"] for t in tets]);vol=np.asarray([t["vol"] for t in tets])
 # Directional dF/dt in each tet's material frame from the total CGLS velocity correction.
 H=np.empty((T,3,3),np.float64)
 for i,t in enumerate(tets):
  ns=t["nodes"]
  edge=np.column_stack([x[ns[j]]-x[ns[0]] for j in (1,2,3)])
  H[i]=edge@inverse[i]@R[i]
 H0=np.empty_like(H)
 x0=np.asarray([np.asarray(nodes[i]["center"]["delta_velocity_m_s"],np.float32) for i in range(N)],np.float64)
 for i,t in enumerate(tets):
  ns=t["nodes"];edge=np.column_stack([x0[ns[j]]-x0[ns[0]] for j in (1,2,3)])
  H0[i]=edge@inverse[i]@R[i]
 z=np.full((T,15),np.nan);projinfo=[]; strictref=strict.Reference(base)
 for m in (0,1):
  ids=np.flatnonzero(mat==m);d=desc[m];off=int(d["parameter_offset"]);count=len(d["parameter_names"])
  p=np.tile(params[off:off+count],(len(ids),1));oldm=old[ids]
  for k in range(0,len(ids),48):
   jj=ids[k:k+48];zz,it,rn,cn,ok,why,prov=strictref.project_with_metrics(p[k:k+48],F[jj],oldm[k:k+48])
   z[jj]=zz
   for q,tid in enumerate(jj):
    projinfo.append({"tet":int(tid),"material":m,"ok":bool(ok[q]),"iterations":int(it[q]),"scaled_residual":float(rn[q]),"dynamic_correction":float(cn[q]),"reason":str(why[q]) if not ok[q] else "ok"})
 bad=[q for q in projinfo if not q["ok"]]
 if bad:raise ValueError("strict center projection failure; rows preserved in exception: "+json.dumps(bad[:16]))
 # Per-tet strict implicit tangent, captured FP32 cache matrix action, and original-direction native parity.
 dP_strict=np.full((T,3,3),np.nan);dP_cache=np.full((T,3,3),np.nan);dP0_cache=np.full((T,3,3),np.nan)
 for m in (0,1):
  ids=np.flatnonzero(mat==m);d=desc[m];off=int(d["parameter_offset"]);count=len(d["parameter_names"])
  p=np.tile(params[off:off+count],(len(ids),1))
  for k in range(0,len(ids),48):
   jj=ids[k:k+48]
   dP_strict[jj]=strict.tangent(base,p[k:k+48],F[jj],old[jj],z[jj],H[jj])
 dP_cache=np.einsum("tij,tj->ti",np.asarray([t["C"] for t in tets]),H.reshape(T,9)).reshape(T,3,3)
 dP0_cache=np.einsum("tij,tj->ti",np.asarray([t["C"] for t in tets]),H0.reshape(T,9)).reshape(T,3,3)
 params_glue=desc[2];go=int(params_glue["parameter_offset"]);mu=float(params[go+1]);lam=float(params[go+2])
 dP_strict[glue]=base.glue_tangent_action(F[glue],H[glue],np.full(len(glue),mu),np.full(len(glue),lam))
 dP_cache[glue]=np.einsum("tij,tj->ti",np.asarray([t["C"] for t in tets])[glue],H[glue].reshape(len(glue),9)).reshape(len(glue),3,3)
 dP0_cache[glue]=np.einsum("tij,tj->ti",np.asarray([t["C"] for t in tets])[glue],H0[glue].reshape(len(glue),9)).reshape(len(glue),3,3)
 cached_native_dir=np.asarray([t["native_tangent"] for t in tets],np.float64)
 local0= dP0_cache
 # Independent stress/frame/element assembly for cached direction and strict direction.
 world_strict=dP_strict@np.swapaxes(R,-1,-2)
 world_cache=dP_cache@np.swapaxes(R,-1,-2)
 world0=dP0_cache@np.swapaxes(R,-1,-2)
 elem_strict,nodal_strict=make_force(world_strict,tets,N)
 elem_cache,nodal_cache=make_force(world_cache,tets,N)
 elem0,nodal0=make_force(world0,tets,N)
 mass=acc[:,3].astype(np.float64)
 mass_action=(mass[:,None]*x)
 mass0=mass[:,None]*x0
 stif_strict=-dt*dt*nodal_strict
 stif_cache=-dt*dt*nodal_cache
 stif0=-dt*dt*nodal0
 analytic_strict=mass_action+stif_strict
 analytic_cache=mass_action+stif_cache
 actionA=csr_load(OP/"operator.operator.npz")
 actionM=csr_load(OP/"operator.mass.npz")
 actionK=csr_load(OP/"operator.dt2-stiffness.npz")
 yA=csr_mv(actionA,xfree)
 yM=csr_mv(actionM,xfree)
 yK=csr_mv(actionK,xfree)
 strictflat=analytic_strict[free_ids].reshape(-1)
 cacheflat=analytic_cache[free_ids].reshape(-1)
 massflat=mass_action[free_ids].reshape(-1)
 stifstrictflat=stif_strict[free_ids].reshape(-1)
 stifcacheflat=stif_cache[free_ids].reshape(-1)
 # Reconstruct capture relation A*X = A*x0 + (B - R); retain this as a distinct FP32-sidecar check.
 Ax0=csr_mv(actionA,(x0[free_ids].reshape(-1)))
 Xseg=cg["x"][:E*N].reshape(E,N,4)[0,:,:3].astype(np.float64)[free_ids].reshape(-1)
 Rseg=cg["r"][:E*N].reshape(E,N,4)[0,:,:3].astype(np.float64)[free_ids].reshape(-1)
 Bseg=cg["b"][:E*N].reshape(E,N,4)[0,:,:3].astype(np.float64)[free_ids].reshape(-1)
 Axseg=cg["Ax"][:E*N].reshape(E,N,4)[0,:,:3].astype(np.float64)[free_ids].reshape(-1)
 recon=Ax0+Bseg-Rseg
 # Original x0 native tangent/readback consistency; source node-CV capture provides x0.
 original_local0=np.asarray([t["native_tangent"] for t in tets],np.float64)
 native_element0=np.asarray([t["native_element"] for t in tets],np.float64)
 # Strict metrics per paper and overall, plus component-level CSR reconciliation.
 pids=paper
 P_strict=np.full((T,3,3),np.nan)
 for m in (0,1):
  ids=np.flatnonzero(mat==m);d=desc[m];off=int(d["parameter_offset"]);count=len(d["parameter_names"])
  P_strict[ids]=base.stress(np.tile(params[off:off+count],(len(ids),1)),F[ids],z[ids])
 nativeP=np.asarray([t["P"] for t in tets])
 report={
  "schema":"numi.cardboard.native497-analytic-warm-cgls-action.v1",
  "evidence_class":"CPU-only independent analytic directional-action reconstruction at one captured linearization; no native dispatch/qualification",
  "source_identity":{"root":20,"newton_iteration":63,"environment":0,"E":E,"N":N,"T":T,"dt_fp32":dt,
   "world_fingerprint":header["world_fingerprint_hex"],"execution_fingerprint":header["execution_fingerprint_hex"],
   "history_word_mismatches":history_mismatch,"free_node_count":int(free.sum()),"free_dof_count":int(free.sum()*3)},
  "bindings":{},
  "strict_projection":{"paper_tets":int(len(paper)),"passed":int(len(paper)-len(bad)),"failed":bad,
   "max_scaled_residual":max(q["scaled_residual"] for q in projinfo),"max_dynamic_correction":max(q["dynamic_correction"] for q in projinfo),
   "max_iterations":max(q["iterations"] for q in projinfo),"residual_gate":2e-12,"correction_gate":2e-11},
  "input_layer":{"CGLS_total_direction_l2_m_s":float(np.linalg.norm(xfree)),"current_Newton_FGMRES_correction_l2_m_s":float(np.linalg.norm(x0[free_ids])),
    "CGLS_total_vs_candidate_velocity_free_order":True,"initial_FGMRES_correction_source":"exact operator-FD node center delta_velocity_m_s (current femSolution), not cumulative candidate-minus-accepted velocity","free_node_ids_equal_sorted_capture_predicate":True,
    "CGLS_x_fixed_max_abs":float(np.max(np.abs(x[~free]))),"CGLS_segment_norms":{"Ax":float(np.linalg.norm(Axseg)),"b":float(np.linalg.norm(Bseg)),"r":float(np.linalg.norm(Rseg))}},
  "material_layer":{
    "strict_directional_P_action_vs_widened_FP32_cached_matrix":rowstats(dP_cache[paper],dP_strict[paper]),
    "native_original_direction_cache_action_vs_captured_local_tangent":rowstats(cached_native_dir[paper],dP0_cache[paper]),
    "center_stress_strict_vs_captured":rowstats(nativeP[paper],P_strict[paper]),
    "native_cached_matrix_widened_to_fp64":True,
    "zero_floor":"Rows with zero expected norm remain unresolved for component-relative metrics; no measured Metal absolute noise floor is inferred."},
  "element_frame_layer":{
    "strict_world_P_direction_vs_cached_world_P_direction":rowstats(world_cache[paper],world_strict[paper]),
    "cached_world_action_vs_captured_original_element_action":rowstats(native_element0[paper],elem0[paper]),
    "element_force_formula":"H=-V*dP_world*inverse_rest^T; force slots 1..3=H^T, slot0=-sum(slots1..3)",
    "frame_formula":"dP_world=dP_local*R^T from captured material-frame quaternion"},
  "assembled_action":{
    "mass_strict_vs_exact_operator_mass_CSR":rowstats(yM.reshape(-1,3),massflat.reshape(-1,3)),
    "cached_tangent_assembled_vs_exact_dt2_stiffness_CSR":rowstats(yK.reshape(-1,3),stifcacheflat.reshape(-1,3)),
    "strict_tangent_assembled_vs_exact_dt2_stiffness_CSR":rowstats(yK.reshape(-1,3),stifstrictflat.reshape(-1,3)),
    "strict_full_action_vs_exact_cached_operator_A_times_X":rowstats(yA.reshape(-1,3),strictflat.reshape(-1,3)),
    "cached_full_action_vs_exact_cached_operator_A_times_X":rowstats(yA.reshape(-1,3),cacheflat.reshape(-1,3)),
    "strict_material_delta_from_cached_full_action_l2_Ns":float(np.linalg.norm(strictflat-cacheflat)),
    "mass_action_l2_Ns":float(np.linalg.norm(massflat)),"strict_stiffness_action_l2_Ns":float(np.linalg.norm(stifstrictflat)),
    "cached_stiffness_action_l2_Ns":float(np.linalg.norm(stifcacheflat)),
    "capture_rhs_residual_relation":{"Ax0_l2_Ns":float(np.linalg.norm(Ax0)),"B_l2_Ns":float(np.linalg.norm(Bseg)),"R_l2_Ns":float(np.linalg.norm(Rseg)),
     "A_times_X_from_CSR_l2_Ns":float(np.linalg.norm(yA)),"Ax0_plus_B_minus_R_vs_CSR":rowstats(yA.reshape(-1,3),recon.reshape(-1,3)),
     "captured_q_workspace_vs_CSR_A_times_X_not_comparable": {"captured_segment": "q workspace is last CGLS Ap, not A times total X", "q_l2": float(np.linalg.norm(Axseg))}},
    "scope":"Selected environment free XYZ mechanical rows, using exact captured FP32-cache CSR as comparator. This is not an independent native replay; it tests analytic material/tet/frame assembly against the separately reconstructed cache operator."},
  "limits":["No physical/native run was made.","Captured CSR is a binary64 reconstruction of the captured FP32 tangent-cache operator, not an independent physical or constitutive operator.","The strict reference uses exact captured F/state history and analytic implicit material tangent, but its force assembly is a CPU reconstruction.","No finite-difference noise-floor or full-coupon convergence qualification."],
  "elapsed_seconds":time.time()-start
 }
 bindings=[FD,SJ,SB,CG,CASE/"prediction.json",CASE/"receipt.json",CASE/"stderr.log",CASE/"output/manifest.json",CASE/"output/compiled.nmatterpack",
  OP/"manifest.json",OP/"operator.json",OP/"operator.operator.npz",OP/"operator.mass.npz",OP/"operator.dt2-stiffness.npz",OP/"operator.vectors.npz",OP/"operator.topology.npz",
  TOOLS/"cardboard_bilinear_fp64_reference.py",TOOLS/"cardboard_compensated_fp64_reference.py",TOOLS/"cardboard_dd_fp64.py",
  Q/"snapshots/source-portable-paper-reference-185/matter/src/metal/base_cgls.metalinc",
  Q/"snapshots/source-portable-paper-reference-185/matter/src/metal/fem.metalinc"]
 report["bindings"]={str(p):sha(p) for p in bindings}
 report["arrays"]={"path":"arrays.npz","contains":["free_node_ids","warm_total_direction_free_xyz","strict_projected_center_state","strict_directional_P","cached_directional_P","reference_element_action_strict","reference_nodal_action_strict","reference_mass_action","reference_stiffness_action","reference_total_action","CSR_A_action","CGLS_b","CGLS_r","CGLS_last_q_Ap_workspace","CGLS_A_times_X_reconstructed"]}
 np.savez_compressed(OUT/"arrays.npz",free_node_ids=free_ids,warm_total_direction_free_xyz=x[free_ids],
  strict_projected_center_state=z[paper],strict_directional_P=dP_strict,cached_directional_P=dP_cache,
  reference_element_action_strict=elem_strict,reference_nodal_action_strict=nodal_strict,
  reference_mass_action=mass_action,reference_stiffness_action=stif_strict,reference_total_action=analytic_strict,
  CSR_A_action=yA,CGLS_b=Bseg,CGLS_r=Rseg,CGLS_last_q_Ap_workspace=Axseg,CGLS_A_times_X_reconstructed=recon)
 writej(OUT/"report.json",report)
 (OUT/"README.md").write_text("# Native497 analytic warm-CGLS action audit (CPU only)\n\nThis reconstructs the actual captured warm total direction through strict paper material tangents, captured-frame transforms, tet force scatter, mass, and stiffness assembly, then compares against the same-generation captured FP32-cache CSR operator. It is descriptive CPU evidence only; no native action, physical acceptance, or full-coupon qualification is claimed. See report.json and arrays.npz.\n")
 inv={}
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!="manifest.json":inv[p.name]={"bytes":p.stat().st_size,"sha256":sha(p)}
 writej(OUT/"manifest.json",{"schema":"numi.cardboard.analysis-package.v1","package":"native497-analytic-warm-cgls-action-001","scope":"CPU-only independent analytic directional-action reconstruction","files":inv})
 print(json.dumps({"output":str(OUT),"strict_projection":report["strict_projection"],"material":report["material_layer"],"assembled_action":report["assembled_action"]},indent=2))
if __name__=="__main__":main()
