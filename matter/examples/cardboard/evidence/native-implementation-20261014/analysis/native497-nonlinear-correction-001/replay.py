#!/usr/bin/env python3
import hashlib, json, pathlib, struct, sys, time
import numpy as np
Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/warm-cgls-control20-diagnostic-replay-497"
FD=CASE/"diagnostics/capture-operator-fd-capture/capture"
SJ=CASE/"diagnostics/capture-base-cgls-capture/capture.state.json"
SB=CASE/"diagnostics/capture-base-cgls-capture/capture.state.bin"
CG=CASE/"diagnostics/capture-base-cgls-capture/capture"
OUT=Q/"analysis/native497-nonlinear-correction-001"
TOOLS=Q/"snapshots/source-portable-paper-reference-185/matter/tools"
sys.path.insert(0,str(TOOLS))
import cardboard_bilinear_fp64_reference as base
import cardboard_compensated_fp64_reference as strict
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def save(p,x): pathlib.Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def sidecar():
 m=json.loads(SJ.read_text()); raw=SB.read_bytes()
 assert m["complete"] and len(raw)==m["payload"]["byte_count"] and sha(SB)==m["payload"]["sha256"]
 a={}
 for d in m["arrays"]:
  b=raw[d["offset"]:d["offset"]+d["byte_count"]]
  assert len(b)==d["byte_count"] and hashlib.sha256(b).hexdigest()==d["sha256"],d["name"]
  if d["dtype"]=="float32x4": a[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(d["shape"]+[4])
  elif d["dtype"]=="float32": a[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(d["shape"])
  elif d["dtype"]=="record:NMFEMNodeStateGPU": a[d["name"]]=np.frombuffer(b,"<f4").copy().reshape(d["shape"]+[20])
  elif d["dtype"]=="uint32": a[d["name"]]=np.frombuffer(b,"<u4").copy().reshape(d["shape"])
 return m,a
def cgls():
 raw=CG.read_bytes(); h=struct.unpack_from("<8I",raw)
 assert h[:7]==(0x534c4743,3,2,882,3528,64,13),h
 U,E,N=h[4],h[2],h[3]; sizes=[U*16]*7+[U*4,E*16,E*16,E*4,E*4,E*N*3*16]; seg=[]; off=32
 for size in sizes:
  off=(off+15)&~15; seg.append(raw[off:off+size]); off+=size
 assert off==len(raw)
 return {"h":h,"x":np.frombuffer(seg[0],"<f4").copy().reshape(U,4),
  "selected":np.frombuffer(seg[11],"<u4").copy(),"errors":np.frombuffer(seg[10],"<u4").copy(),"sha":sha(CG)}
def parse_fd():
 h=None; nodes={}; ts={}
 with FD.open() as f:
  for line in f:
   r=json.loads(line); k=r.get("record")
   if k=="header": h=r
   elif k=="node" and r["environment"]==0: nodes[int(r["node_id"])]=r
   elif k=="tetrahedron" and r["environment"]==0:
    c=r["center"]; tid=int(r["tet_id"])
    ts[tid]={"id":tid,"mat":int(r["material_index"]),"nodes":np.array(r["nodes"],np.int64),
      "vol":float(np.float32(r["rest_volume_m3"])),"inv":np.asarray(r["inverse_rest_rows_per_m"],np.float32)[:,:3],
      "q":np.asarray(r["material_frame_quaternion_xyzw"],np.float32),
      "F":np.asarray(c["F_local"],np.float32).astype(float)+np.asarray(c["F_local_low"],np.float32).astype(float),
      "P":np.asarray(c["P_local_Pa"],np.float32).astype(float),
      "oldj":np.asarray(c["accepted_state"],np.float32)}
 assert h and len(nodes)==882 and len(ts)==1920
 return h,nodes,[ts[i] for i in range(1920)]
def quatR(q):
 q=np.asarray(q,np.float64); q=q/np.linalg.norm(q); x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
 t0=time.time(); sm,sa=sidecar(); cg=cgls(); fh,nodes,tets=parse_fd()
 assert sm["root"]==20 and sm["newton_iteration"]==63 and fh["root"]==20 and fh["newton_iteration"]==63
 assert fh["world_fingerprint_hex"]==sm["world_fingerprint_u64_hex"] and fh["execution_fingerprint_hex"]==sm["device_program_fingerprint_u64_hex"]
 assert cg["selected"].tolist()==[1,0] and cg["errors"].tolist()==[0,16]
 E,N,T=2,882,1920; dt=float(np.float32(fh["dt_seconds"]))
 acc=sa["accepted_nodes"].reshape(E,N,20)[0]; cand=sa["candidate_nodes"].reshape(E,N,20)[0]
 old=sa["accepted_material_history"].reshape(E,T,15)[0].astype(float)
 x=cg["x"][:E*N].reshape(E,N,4)[0,:N,:3].astype(float)
 mass=acc[:,3].astype(float); vold=acc[:,4:7].astype(float); uold=acc[:,16:19].astype(np.float32)
 vc=cand[:,4:7].astype(np.float32); uc=cand[:,16:19].astype(np.float32); tags=cand[:,11].astype(np.float32)
 free=(tags!=1)&(tags!=3)&(mass>0)
 R0=np.zeros((N,3)); node_mismatch=0
 for i,r in nodes.items():
  words=np.asarray(r["center"]["raw_state_float4s"],np.float32).reshape(20)
  node_mismatch+=int(np.count_nonzero(words.view(np.uint32)!=cand[i].view(np.uint32)))
  R0[i]=np.asarray(r["center_residual_impulse_Ns"],np.float32)[:3]
 mat=np.array([t["mat"] for t in tets],np.int32)
 paper_history_ids=np.flatnonzero((mat==0)|(mat==1))
 oldj=np.asarray([tets[i]["oldj"] for i in paper_history_ids],np.float32)
 hist_mismatch=int(np.count_nonzero(oldj.view(np.uint32)!=old[paper_history_ids].astype(np.float32).view(np.uint32)))
 R=np.asarray([quatR(t["q"]) for t in tets])
 desc={int(d["index"]):d for d in fh["materials"]}
 penv=np.asarray(fh["parameters_by_environment"][0],np.float32).astype(float)
 Fcap=np.asarray([t["F"] for t in tets])
 # FP64 contraction on exact FP32 trial node words; alpha-zero geometry is a source-match check.
 def Fgeom(u,t):
  edge=np.column_stack([u[t["nodes"][j]]-u[t["nodes"][0]] for j in (1,2,3)])
  return (np.eye(3)+edge@t["inv"].astype(float))@quatR(t["q"])
 Fg0=np.asarray([Fgeom(uc.astype(float),t) for t in tets])
 geomerr=float(np.max(np.abs(Fg0-Fcap)))
 strictref=strict.Reference(base)
 info0={}; z0=np.full((T,15),np.nan)
 for m in (0,1):
  ids=np.flatnonzero(mat==m); d=desc[m]; off=int(d["parameter_offset"]); n=len(d["parameter_names"])
  p=np.tile(penv[off:off+n],(len(ids),1))
  for k in range(0,len(ids),48):
   ii=ids[k:k+48]; z,it,r,c,ok,why,prov=strictref.project_with_metrics(p[k:k+48],Fcap[ii],old[ii])
   z0[ii]=z
   for j,i in enumerate(ii): info0[int(i)]={"ok":bool(ok[j]),"it":int(it[j]),"r":float(r[j]),"c":float(c[j]),"why":str(why[j]) if not ok[j] else "ok"}
 bad0=[i for i,a in info0.items() if not a["ok"]]
 P0=np.full((T,3,3),np.nan)
 for m in (0,1):
  ids=np.flatnonzero(mat==m); good=ids[np.array([info0[int(i)]["ok"] for i in ids])]
  d=desc[m]; off=int(d["parameter_offset"]); n=len(d["parameter_names"]); p=np.tile(penv[off:off+n],(len(good),1))
  if len(good): P0[good]=base.stress(p,Fcap[good],z0[good])
 glue=np.flatnonzero(mat==2); d=desc[2]; go=int(d["parameter_offset"]); gp=penv[go:go+2]
 if len(glue): P0[glue]=base.glue_stress(Fcap[glue],np.full(len(glue),gp[0]),np.full(len(glue),gp[1]))
 def force(P):
  out=np.zeros((N,3))
  for t,p in zip(tets,P):
   if not np.isfinite(p).all(): continue
   h=-t["vol"]*p@t["inv"].astype(float).T; f=h.T; ns=t["nodes"]
   out[ns[1]]+=f[0]; out[ns[2]]+=f[1]; out[ns[3]]+=f[2]; out[ns[0]]-=f.sum(axis=0)
  return out
 f0=force(P0@np.swapaxes(R,1,2))
 paper=np.flatnonzero((mat==0)|(mat==1))
 def branch(z,ids):
  out={}
  for m in (0,1):
   ix=np.flatnonzero(mat[ids]==m)
   if not len(ix): continue
   ii=ids[ix]; d=desc[m]; off=int(d["parameter_offset"]); n=len(d["parameter_names"]); p=np.tile(penv[off:off+n],(len(ii),1)); zz=z[ix]
   q,_=base.hill(p,zz[:,6:12]); y=1+p[:,17]*p[:,19]*p[:,21]*zz[:,13]/p[:,10]+p[:,18]*p[:,20]*p[:,22]*zz[:,14]/p[:,10]-q
   dl=zz[:,12]-old[ii,12]; out[m]={"y":y,"dl":dl}
  return out
 b0=branch(z0[paper],paper); nativeP=np.asarray([t["P"] for t in tets]); dP=P0[paper]-nativeP[paper]
 center={"paper_tets":len(paper),"strict_pass_count":len(paper)-len(bad0),
  "strict_failures":[{"tet":i,**info0[i]} for i in bad0],
  "max_scaled_residual":max((v["r"] for v in info0.values()),default=0),
  "max_dynamic_correction":max((v["c"] for v in info0.values()),default=0),
  "max_iterations":max((v["it"] for v in info0.values()),default=0),
  "native_center_stress_max_abs_error_pa":float(np.nanmax(np.abs(dP))),
  "native_center_stress_global_frobenius_relative":float(np.linalg.norm(dP)/max(np.linalg.norm(nativeP[paper]),1e-300)),
  "captured_center_F_vs_node_geometry_max_abs":geomerr}
 curve=[]; alphas=[0,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1]
 for alpha in alphas:
  if alpha==0: vt=vc.astype(float); Ft=Fcap.copy()
  else:
   af=np.float32(alpha); vt32=np.asarray(vc+np.asarray(af*x,np.float32),np.float32)
   ut32=np.asarray(uold+np.asarray(np.float32(dt)*vt32,np.float32),np.float32)
   G=np.asarray([Fgeom(ut32.astype(float),t) for t in tets])
   Ft=Fcap+(G-Fg0); vt=vt32.astype(float)
  z=np.full((T,15),np.nan); info={}
  for m in (0,1):
   ids=np.flatnonzero(mat==m); d=desc[m]; off=int(d["parameter_offset"]); n=len(d["parameter_names"]); p=np.tile(penv[off:off+n],(len(ids),1))
   for k in range(0,len(ids),48):
    ii=ids[k:k+48]; zz,it,r,c,ok,why,prov=strictref.project_with_metrics(p[k:k+48],Ft[ii],old[ii]); z[ii]=zz
    for j,i in enumerate(ii): info[int(i)]={"ok":bool(ok[j]),"it":int(it[j]),"r":float(r[j]),"c":float(c[j]),"why":str(why[j]) if not ok[j] else "ok"}
  P=np.full((T,3,3),np.nan)
  for m in (0,1):
   ids=np.flatnonzero(mat==m); good=ids[np.array([info[int(i)]["ok"] for i in ids])]
   d=desc[m]; off=int(d["parameter_offset"]); n=len(d["parameter_names"]); p=np.tile(penv[off:off+n],(len(good),1))
   if len(good): P[good]=base.stress(p,Ft[good],z[good])
  if len(glue): P[glue]=base.glue_stress(Ft[glue],np.full(len(glue),gp[0]),np.full(len(glue),gp[1]))
  fi=force(P@np.swapaxes(R,1,2)); dm=np.zeros((N,3)); dm[free]=-mass[free,None]*(vt[free]-vc[free])
  di=np.zeros((N,3)); di[free]=dt*(fi[free]-f0[free])
  total=R0+dm+di; total[~free]=0; mo=R0+dm; mo[~free]=0; ino=R0+di; ino[~free]=0
  failed=[i for i,v in info.items() if not v["ok"]]; transitions={}
  if not failed and not bad0:
   b1=branch(z[paper],paper)
   for m,b in b0.items():
    c=b1[m]; transitions[str(m)]={"lambda_sign_flips":int(np.count_nonzero((b["dl"]>0)!=(c["dl"]>0))),
     "yield_sign_flips":int(np.count_nonzero((b["y"]>0)!=(c["y"]>0))),
     "lambda_positive":int(np.count_nonzero(c["dl"]>0)),"yield_negative":int(np.count_nonzero(c["y"]<0)),
     "yield_min":float(np.min(c["y"])),"yield_max":float(np.max(c["y"]))}
  valid=not failed and not bad0
  curve.append({"alpha":alpha,"strict_all_paper_projection_pass":not failed,
   "strict_fail_count":len(failed),"failed_tets":failed[:50],
   "max_scaled_residual":max((v["r"] for v in info.values()),default=0),
   "max_dynamic_correction":max((v["c"] for v in info.values()),default=0),
   "max_local_iterations":max((v["it"] for v in info.values()),default=0),
   "free_residual_l2_Ns":float(np.linalg.norm(total[free])) if valid else None,
   "mass_only_residual_l2_Ns":float(np.linalg.norm(mo[free])),
   "internal_only_residual_l2_Ns":float(np.linalg.norm(ino[free])) if valid else None,
   "mass_delta_l2_Ns":float(np.linalg.norm(dm[free])),
   "internal_delta_l2_Ns":float(np.linalg.norm(di[free])) if not failed else None,
   "branch_transitions":transitions})
 native=[]
 with (CASE/"stderr.log").open(errors="replace") as f:
  for line in f:
   k="fem_residual_armijo_iteration="; at=line.find(k)
   if at>=0:
    try:
     v=json.loads(line[at+len(k):])
     if v.get("root")==20 and v.get("newton_iteration")==63: native=v["environments"]; break
    except Exception: pass
 snap=Q/"snapshots/source-checkpoint-warm-breakdown-173"
 srcs=[snap/"matter/src/metal/fem.metalinc",snap/"matter/include/numi/matter/fem_compensated_deformation.h",
  snap/"include/metalrobo/compensated_geometry_gpu.h",snap/"include/metalrobo/compensated_translation_gpu.h",
  TOOLS/"cardboard_bilinear_fp64_reference.py",TOOLS/"cardboard_compensated_fp64_reference.py",
  TOOLS/"cardboard_compensated_fp64_base.py",TOOLS/"cardboard_dd_fp64.py"]
 ins=[FD,SJ,SB,CG,CASE/"output/manifest.json",CASE/"output/compiled.nmatterpack",
  CASE/"output/rejected-conditioning-000021/manifest.json",CASE/"stderr.log"]
 binds={str(p):sha(p) for p in ins+srcs}
 report={"schema":"numi.cardboard.native497-warm-cgls-nonlinear-residual-cpu-audit.v1",
  "evidence_class":"CPU-only strict-reference replay of one captured CGLS candidate; no native rerun or qualification",
  "source_case":str(CASE),"root":20,"newton_iteration":63,"environment":0,"bindings":binds,
  "reference":{"implementation":"source-portable-paper-reference-185 compensated FP64 strict API",
   "base_sha256":sha(TOOLS/"cardboard_bilinear_fp64_reference.py"),
   "strict_api_sha256":sha(TOOLS/"cardboard_compensated_fp64_reference.py"),
   "dd_helper_sha256":sha(TOOLS/"cardboard_dd_fp64.py"),"scaled_residual_gate":2e-12,
   "dynamic_correction_gate":2e-11,"iteration_ceiling":80},
  "identity":{"world_fingerprint":fh["world_fingerprint_hex"],"execution_fingerprint":fh["execution_fingerprint_hex"],
   "E":E,"N":N,"T":T,"dt_fp32":dt,"state_payload_sha256":sm["payload"]["sha256"],
   "cgls_sha256":cg["sha"],"cgls_header":list(cg["h"]),"selected":cg["selected"].tolist(),
   "errors":cg["errors"].tolist(),"free_node_count":int(free.sum()),"fixed_count":int((~free).sum()),
   "cgls_x_free_norm_m_s":float(np.linalg.norm(x[free])),
   "cgls_x_fixed_max_abs_m_s":float(np.max(np.abs(x[~free])))},
  "input_integrity":{"node_word_mismatch_count_fd_json_vs_sidecar":node_mismatch,
   "history_word_mismatch_count_fd_json_vs_sidecar":hist_mismatch,"center_geometry_reconstruction_max_abs_F":geomerr,
   "captured_center_residual_l2_Ns":float(np.linalg.norm(R0[free])),
   "trial_geometry":"FP32 candidate velocity and accepted reference-displacement update from exact captured words; double contraction of realized words. Alpha-zero anchored to captured F high+low and geometry-derived delta. Not a native trial capture."},
  "native_armijo_newton63":native,"center_projection_and_native_stress_comparison":center,"curve":curve,
  "interpretation":{"mass_delta":"-M*(v_trial-v_center) on free rows","constitutive_delta":"dt*(f_internal_trial-f_internal_center) from strict independent projection and source element-force/gather equations",
   "residual":"captured native center residual plus independently recomputed mass and constitutive deltas; alpha-independent loads cancel",
   "limits":["No GPU/native/physical step was run.","Trial material states are strict CPU reference projections; no native material-state trial capture exists for the CGLS direction.",
    "Any strict projection failure makes the full residual at that alpha inconclusive; failed rows remain listed.",
    "FP64 trial geometry/material evaluation is not bitwise Metal emulation and cannot certify native Armijo.",
    "One control20/Newton63 diagnostic linearization; not full coupon qualification."]},
  "elapsed_seconds":time.time()-t0}
 save(OUT/"report.json",report); save(OUT/"curve.json",curve)
 (OUT/"README.md").write_text("# Native497 warm-CGLS nonlinear residual replay (CPU only)\n\nEvaluates the captured control 20 / Newton 63 CGLS candidate at alpha 0, 1/128, ..., 1 with the strict compensated FP64 material reference. It reconstructs trial velocity/displacement from captured node words and CGLS x, recomputes paper projection/stress and glue stress, and separates mass and constitutive-force residual increments anchored to the native center residual.\n\nNo GPU, native rerun, source edit, or full-coupon qualification was performed. See report.json for hashes, strict gates, curve and limits.\n")
 files={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()}
 save(OUT/"manifest.json",{"schema":"numi.cardboard.analysis-package.v1","package":"native497-nonlinear-correction-001","files":files,"host":"macmini","scope":"CPU-only read-only replay"})
 print(json.dumps({"out":str(OUT),"center":center,"native":native,"curve":curve,"seconds":time.time()-t0},indent=2))
if __name__=="__main__": main()
