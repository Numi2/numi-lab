#!/usr/bin/env python3
import json,hashlib,time,sys,datetime
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/control19-newton3-resume-diagnostic-555"
CAP=CASE/"diagnostics/capture-operator-fd-capture/capture"
AUD=Q/"analysis/native555-actual19-material-replay-001"
REF=Q/"snapshots/source-stable-green-reference-206-buildfix-002/matter/tools"
PREP=Q/"analysis/source198-strict-layer-replay-preparation-001"
OUT=Q/"analysis/native555-residual-delta-audit-001"
sys.path.insert(0,str(PREP));sys.path.insert(0,str(REF))
import cardboard_cached_operator_audit as G
import strict_layers
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def f32(x): return np.asarray(x,dtype=np.float32).astype(np.float64)
def pair(d,k): return f32(d[k])+f32(d.get(k+"_low",np.zeros_like(d[k])))
def rel_metrics(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 err=a-b; na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));ne=float(np.linalg.norm(err))
 return {"a_norm_l2":na,"b_norm_l2":nb,"difference_norm_l2":ne,"difference_relative_to_b":ne/nb if nb>0 else None,"difference_max_abs_component":float(np.max(np.abs(err))) if err.size else 0.0,"zero_b_norm":nb==0.0}
OUT.mkdir(parents=True, exist_ok=True)
assert set(x.name for x in OUT.iterdir()) <= {"run.py", "attempt-001.json"}
assert sha(CAP)=="c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53"
H,NR,TR,IR=G.capture_records(CAP)
assert IR["production_inputs_bitwise_unchanged"]
assert (H["root"],H["newton_iteration"])==(19,3)
dt=float(np.float32(H["dt_seconds"]))
pred=json.loads((CASE/"prediction.json").read_text())
manifest=json.loads((CASE/"output/manifest.json").read_text())
results={};arrays={}
for env in (0,1):
 ns=sorted([r for r in NR if r["environment"]==env],key=lambda r:r["node_id"])
 ts=sorted([r for r in TR if r["environment"]==env],key=lambda r:r["tet_id"])
 assert len(ns)==882 and len(ts)==1920
 dat=np.load(AUD/f"environment-{env}/arrays.npz")
 ids=np.asarray([t["nodes"] for t in ts],dtype=np.int64)
 inv=np.stack([f32(t["inverse_rest_rows_per_m"]) for t in ts])
 vol=f32([t["rest_volume_m3"] for t in ts])
 rot=np.stack([G.rotation_from_xyzw(f32(t["material_frame_quaternion_xyzw"])) for t in ts])
 mass=f32([n["center"]["mass_kg"] for n in ns])
 tags=f32([n["center"]["constraint_tag"] for n in ns])
 free=~np.isin(tags,[1.0,3.0])
 Fcenter=np.stack([pair(t["center"],"F_local") for t in ts])
 Fworldcenter=np.stack([f32(t["center"]["F_world"]) for t in ts])
 nativePworldcenter=np.stack([f32(t["center"]["P_world_Pa"]) for t in ts])
 nativePlocalcenter=np.stack([f32(t["center"]["P_local_Pa"]) for t in ts])
 Pcenterloc=dat["s0_reference_stress"].copy()
 assert Pcenterloc.shape==(1920,3,3)
 Pcenterworld=Pcenterloc@np.swapaxes(rot,-1,-2)
 f_native_center=strict_layers.forces(nativePworldcenter,inv,vol)
 f_ref_center=strict_layers.forces(Pcenterworld,inv,vol)
 force_roundtrip=rel_metrics(nativePworldcenter,nativePlocalcenter@np.swapaxes(rot,-1,-2))
 # Check that captured local/world stresses follow F_local=F_world R and P_world=P_local R^T.
 Flocal_pred=Fworldcenter@rot
 f_relation=rel_metrics(Fcenter,Flocal_pred)
 state_metrics=[]; saved={}
 for side in range(4):
  sidx=side+1; eps=float(np.float32(H["epsilon_values"][side//2])); sign=1.0 if side%2==0 else -1.0
  side_t=[t["sides"][side] for t in ts]
  Fside=np.stack([pair(x,"F_local") for x in side_t])
  Fworldside=np.stack([f32(x["F_world"]) for x in side_t])
  nativePworldside=np.stack([f32(x["P_world_Pa"]) for x in side_t])
  nativePlocalside=np.stack([f32(x["P_local_Pa"]) for x in side_t])
  Pref_local=dat[f"s{sidx}_reference_stress"].copy()
  Pref_world=Pref_local@np.swapaxes(rot,-1,-2)
  dFlocal=Fside-Fcenter
  dFworld=Fworldside-Fworldcenter
  dPnative=nativePworldside-nativePworldcenter
  dPref=Pref_world-Pcenterworld
  # Native side residual samples are saved as complete impulse vectors; only xyz are mechanical rows.
  R0=np.stack([f32(n["center_residual_impulse_Ns"])[:3] for n in ns])
  Rs=np.stack([f32(n["epsilon_pair_residual_impulses_Ns"][side])[:3] for n in ns])
  dRcap=Rs-R0
  dv=np.stack([f32(n["epsilon_pair_realized_velocity_delta_m_s"][side]) for n in ns])
  du=np.stack([f32(n["epsilon_pair_realized_u_delta_m"][side]) for n in ns])
  # Reconstruct side F from the captured nodal displacement increment and immutable reference geometry.
  edge_du=np.stack([np.stack([du[ids[j,k+1]]-du[ids[j,0]] for k in range(3)],axis=1) for j in range(1920)])
  dFworld_nodes=edge_du@inv
  dFlocal_nodes=dFworld_nodes@rot
  # Reconstruct the velocity-gradient/Fdot change from the captured nodal velocity increment.
  edge_dv=np.stack([np.stack([dv[ids[j,k+1]]-dv[ids[j,0]] for k in range(3)],axis=1) for j in range(1920)])
  dFdotworld_nodes=edge_dv@inv
  dFdotlocal_nodes=dFdotworld_nodes@rot
  dFdotlocal=np.stack([f32(x["Fdot_local_per_s"]) for x in side_t])-np.stack([f32(t["center"]["Fdot_local_per_s"]) for t in ts])
  dFdotworld=np.stack([f32(x["Fdot_world_per_s"]) for x in side_t])-np.stack([Fdot for Fdot in [f32(t["center"]["Fdot_world_per_s"]) for t in ts]])
  f_relation_side=rel_metrics(dFlocal,dFlocal_nodes)
  fw_relation_side=rel_metrics(dFworld,dFworld_nodes)
  fdot_relation_side=rel_metrics(dFdotlocal,dFdotlocal_nodes)
  fdot_world_relation_side=rel_metrics(dFdotworld,dFdotworld_nodes)
  # Exact force-change assembly from captured native stress and independent FP64 stress.
  f_native_side=strict_layers.forces(nativePworldside,inv,vol)
  f_ref_side=strict_layers.forces(Pref_world,inv,vol)
  g_native=strict_layers.gather(f_native_side-f_native_center,ids,882)
  g_ref=strict_layers.gather(f_ref_side-f_ref_center,ids,882)
  mass_term=-mass[:,None]*dv
  dRnative_rebuilt=dt*g_native+mass_term
  dRref_exact=dt*g_ref+mass_term
  # Linearized FP64 center tangent applied to the realized side delta-F, plus the same actual mass increment.
  C=dat["reference_C"]
  dp_linear_local=np.einsum("tij,tj->ti",C,dFlocal.reshape(1920,9)).reshape(1920,3,3)
  dp_linear_world=dp_linear_local@np.swapaxes(rot,-1,-2)
  f_linear=strict_layers.forces(dp_linear_world,inv,vol)
  g_linear=strict_layers.gather(f_linear,ids,882)
  dRlinear=dt*g_linear+mass_term
  state_metrics.append({
   "side_index":side,"state_index":sidx,"epsilon":eps,"sign":int(sign),
   "reference_projection":json.loads((AUD/f"environment-{env}/report.json").read_text())["states"][sidx]["projection"],
   "captured_F_vs_nodal_du_local":f_relation_side,"captured_Fworld_vs_nodal_du":fw_relation_side,
   "captured_Fdotlocal_vs_nodal_dv":fdot_relation_side,"captured_Fdotworld_vs_nodal_dv":fdot_world_relation_side,
   "mass_change_l2_Ns":float(np.linalg.norm(mass_term[free])),
   "native_stress_stiffness_change_l2_Ns":float(np.linalg.norm((dt*g_native)[free])),
   "reference_stress_stiffness_change_l2_Ns":float(np.linalg.norm((dt*g_ref)[free])),
   "linearized_stiffness_change_l2_Ns":float(np.linalg.norm((dt*g_linear)[free])),
   "captured_native_delta_R_free_Ns":rel_metrics(dRcap[free],dRcap[free]),
   "native_P_gather_plus_mass_vs_captured_delta_R_free_Ns":rel_metrics(dRnative_rebuilt[free],dRcap[free]),
   "reference_exact_side_stress_vs_captured_native_delta_R_free_Ns":rel_metrics(dRref_exact[free],dRcap[free]),
   "native_delta_R_vs_reference_exact_free_Ns":rel_metrics(dRcap[free],dRref_exact[free]),
   "reference_linearized_vs_exact_free_Ns":rel_metrics(dRlinear[free],dRref_exact[free]),
   "native_vs_reference_side_stress_local_Pa":rel_metrics(np.stack([f32(x["P_local_Pa"]) for x in side_t]),Pref_local),
   "max_abs_native_residual_delta_Ns":float(np.max(np.abs(dRcap[free]))),
   "max_abs_ref_exact_residual_delta_Ns":float(np.max(np.abs(dRref_exact[free]))),
   "free_node_count":int(free.sum()),
  })
  saved[f"env{env}_side{side}_dR_capture"]=dRcap
  saved[f"env{env}_side{side}_dR_native_rebuilt"]=dRnative_rebuilt
  saved[f"env{env}_side{side}_dR_ref_exact"]=dRref_exact
  saved[f"env{env}_side{side}_dR_ref_linear"]=dRlinear
  saved[f"env{env}_side{side}_mass"]=mass_term
 results[str(env)]={"nodes":len(ns),"tets":len(ts),"free_nodes":int(free.sum()),"center_native_Pworld_from_local":force_roundtrip,"center_Flocal_from_Fworld":f_relation,"sides":state_metrics}
 arrays.update(saved)
np.savez_compressed(OUT/"residual_deltas.npz",**arrays)
inputs=[CAP,AUD/"bindings.json",AUD/"manifest.json",AUD/"summary.json",PREP/"strict_layers.py",Q/"snapshots/source-qualification-checkpoint-proof-221-003/matter/tools/cardboard_solver_environment.py",REF/"cardboard_cached_operator_audit.py",REF/"cardboard_bilinear_fp64_reference.py",REF/"cardboard_compensated_fp64_reference.py"]
report={"schema":"numi.cardboard.native555-residual-delta-audit.v1","classification":"CPU-only reconstructed side residual comparison from exact 555 capture; no production action or solver convergence claim","native_capture":{"path":str(CAP),"sha256":sha(CAP),"root":19,"newton":3,"dt_seconds":dt,"material_rate_exponent":json.loads((CASE/"output/manifest.json").read_text())["solver"]["maximum_rate_exponent"]},"independent_reference":{"source_snapshot":str(Q/"snapshots/source-stable-green-reference-206-buildfix-002"),"reference_sha256":sha(REF/"cardboard_bilinear_fp64_reference.py"),"strict_module_sha256":sha(REF/"cardboard_compensated_fp64_reference.py"),"stable_green_overlay":True},"method":{"delta_R_native_from_capture":"R_side_native - R_center_native, saved per node by production; free xyz only","delta_R_native_rebuilt":"dt*G(f(P_world_native_side)-f(P_world_native_center))-m*delta_v","delta_R_reference_exact":"dt*G(f(P_world_FP64_side)-f(P_world_FP64_center))-m*delta_v","delta_R_reference_linearized":"dt*G(f(C_FP64_center:deltaF_local_realized))-m*delta_v","geometry":"F_local = F_world R; P_world = P_local R^T; side delta-F independently reconstructed from captured nodal delta-u and immutable tet inverse-rest; Fdot delta from nodal delta-v","external_terms":"cancel in side-minus-center difference; this is a residual-difference reconstruction, not an absolute residual refit"},"environments":results,"inputs_sha256":{str(p):sha(p) for p in inputs},"finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
manifest={"schema":"numi.cardboard.native555-residual-delta-files.v1","files_sha256":{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},"inputs_sha256":report["inputs_sha256"]}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"envs":{e:[{"side":s["side_index"],"eps":s["epsilon"],"Frel":s["captured_F_vs_nodal_du_local"]["difference_relative_to_b"],"Fabs":s["captured_F_vs_nodal_du_local"]["difference_max_abs_component"],"Fdotrel":s["captured_Fdotlocal_vs_nodal_dv"]["difference_relative_to_b"],"nativeRebuiltrel":s["native_P_gather_plus_mass_vs_captured_delta_R_free_Ns"]["difference_relative_to_b"],"refVsNativereL":s["reference_exact_side_stress_vs_captured_native_delta_R_free_Ns"]["difference_relative_to_b"],"nativeVsRefRel":s["native_delta_R_vs_reference_exact_free_Ns"]["difference_relative_to_b"],"linearVsExactRel":s["reference_linearized_vs_exact_free_Ns"]["difference_relative_to_b"]} for s in x["sides"]] for e,x in results.items()}},sort_keys=True))
