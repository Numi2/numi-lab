#!/usr/bin/env python3
import json,hashlib,sys,datetime,time
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/control19-newton3-resume-diagnostic-555"
CAP=CASE/"diagnostics/capture-operator-fd-capture/capture"
AUD=Q/"analysis/native555-actual19-material-replay-001"
PREP=Q/"analysis/source198-strict-layer-replay-preparation-001"
REF=Q/"snapshots/source-stable-green-reference-206-buildfix-002/matter/tools"
OUT=Q/"analysis/native555-global-ideal-fd-tight-003"
sys.path.insert(0,str(PREP));sys.path.insert(0,str(REF))
import strict_layers
# Analysis-only tighter continuation target; equations and row scales remain unchanged.
strict_layers.ref.TIGHT_RESIDUAL=2.0e-15
strict_layers.ref.TIGHT_DYNAMIC_CORRECTION=2.0e-14
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def metrics(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);e=x-y
 nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y));ne=float(np.linalg.norm(e))
 return {"measured_l2":nx,"reference_l2":ny,"difference_l2":ne,"relative_to_reference":ne/ny if ny else None,"max_abs_component":float(np.max(np.abs(e))) if e.size else 0.0}
OUT.mkdir(parents=True, exist_ok=True)
assert set(x.name for x in OUT.iterdir()) <= {"run.py", "attempt-001.json"}
assert sha(CAP)=="c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53"
header,nodes_all,tets_all,integrity=strict_layers.geometry.capture_records(CAP)
nodes=sorted((x for x in nodes_all if x["environment"]==0),key=lambda x:x["node_id"])
tets=sorted((x for x in tets_all if x["environment"]==0),key=lambda x:x["tet_id"])
a=np.load(AUD/"environment-0/arrays.npz")
material=np.asarray(a["material"]);paper=np.flatnonzero(material!=2);glue=np.flatnonzero(material==2)
params_by_env=np.asarray(header["parameters_by_environment"][0],np.float32).astype(np.float64)
specs={m["index"]:m for m in header["materials"]}
p=np.stack([params_by_env[specs[int(material[i])]["parameter_offset"]:specs[int(material[i])]["parameter_offset"]+28] for i in paper])
old=np.asarray(a["accepted_paper_history"],float)
dt=float(np.float32(header["dt_seconds"]))
F0=np.asarray(a["F_center"],float); direction=np.stack([strict_layers.f32(t["center"]["F_direction_local_per_s"]) for t in tets])
C=np.asarray(a["reference_C"],float)
ids=np.asarray(a["tet_nodes"],np.int64); inv=np.stack([strict_layers.f32(t["inverse_rest_rows_per_m"]) for t in tets]);vol=strict_layers.f32([t["rest_volume_m3"] for t in tets])
rot=np.stack([strict_layers.geometry.rotation_from_xyzw(strict_layers.f32(t["material_frame_quaternion_xyzw"])) if t["center"]["dt_seconds_bulk_modulus_kind_frame_active"][3]==1 else np.eye(3) for t in tets])
mass=strict_layers.f32([n["center"]["mass_kg"] for n in nodes])
tags=strict_layers.f32([n["center"]["constraint_tag"] for n in nodes]);free=~np.isin(tags,[1,3])
dv=np.stack([strict_layers.f32(n["correction_delta_v_m_s"]) for n in nodes])
mu=np.full(len(glue),params_by_env[specs[2]["parameter_offset"]+1]);lam=np.full(len(glue),params_by_env[specs[2]["parameter_offset"]+2])
# Exact center reference stress and assembled derivative.
P0=np.asarray(a["s0_reference_stress"],float)
force0=strict_layers.forces(P0@np.swapaxes(rot,-1,-2),inv,vol)
g0=strict_layers.gather(force0,ids,len(nodes))
mass_action=mass[:,None]*dv
direction_flat=direction.reshape(len(tets),9)
dP=np.einsum("tij,tj->ti",C,direction_flat).reshape(len(tets),3,3)
force_tan=strict_layers.forces((dP@np.swapaxes(rot,-1,-2)),inv,vol)
g_tan=strict_layers.gather(force_tan,ids,len(nodes))
stiffness_action=-dt*dt*g_tan
Aaction=mass_action+stiffness_action
# Residual convention R = dt*internal_force - m*(v-vold), so dR/depsilon = -A*delta_v.
linear_dR=-Aaction
rows=[]
saved={"center_reference_stress":P0,"A_action":Aaction,"linear_residual_derivative":linear_dR,"mass_action":mass_action,"stiffness_action":stiffness_action,"free_mask":free}
powers=[12,14,16,18,20]
for power in powers:
 eps=2.0**(-power); endpoints=[]
 for sign in (1,-1):
  F=F0.copy();F[paper]=F0[paper]+sign*eps*dt*direction[paper]
  z,it,rn,cn,valid,reasons,passes=strict_layers.project(p,F[paper],old)
  if not bool(np.all(valid)): 
   endpoints.append({"ok":False,"sign":sign,"valid_rows":int(valid.sum()),"failed_rows":np.flatnonzero(~valid).tolist(),"max_residual":float(np.nanmax(rn)),"max_correction":float(np.nanmax(cn))})
   continue
  stress=np.empty_like(F);stress[paper]=strict_layers.ref.stress(p,F[paper],z)
  stress[glue]=strict_layers.ref.glue_stress(F[glue],mu,lam)
  world=stress@np.swapaxes(rot,-1,-2)
  force=strict_layers.forces(world,inv,vol);g=strict_layers.gather(force,ids,len(nodes))
  # Endpoint velocity shift is the same epsilon-scaled current Newton correction.
  residual=dt*g0 # replaced below; retained only to make the physical residual convention explicit
  R=dt*g-mass[:,None]*(np.stack([strict_layers.f32(n["center"]["v_m_s"]) for n in nodes])+
                        sign*eps*dv-np.stack([strict_layers.f32(n["accepted"]["v_m_s"]) for n in nodes]))
  sig_center=strict_layers.branch(p,old,a["s0_reference_state"])
  sig=strict_layers.branch(p,old,z)
  endpoints.append({"ok":True,"sign":sign,"stress":stress,"R":R,"iterations":int(np.max(it)),"max_scaled_residual":float(np.max(rn)),"max_dynamic_correction":float(np.max(cn)),"branch_crossing_rows":int(np.sum(np.any(sig!=sig_center,axis=1)))})
 if all(e["ok"] for e in endpoints):
  plus,minus=endpoints
  secant=(plus["R"]-minus["R"])/(2*eps)
  validmask=free
  m=metrics(secant[validmask],linear_dR[validmask])
  oneplus=metrics(((plus["R"]- (dt*g0-mass[:,None]*(np.stack([strict_layers.f32(n["center"]["v_m_s"]) for n in nodes])-np.stack([strict_layers.f32(n["accepted"]["v_m_s"]) for n in nodes]))))/eps)[validmask],linear_dR[validmask])
  oneminus=metrics((( (dt*g0-mass[:,None]*(np.stack([strict_layers.f32(n["center"]["v_m_s"]) for n in nodes])-np.stack([strict_layers.f32(n["accepted"]["v_m_s"]) for n in nodes]))) - minus["R"])/eps)[validmask],linear_dR[validmask])
  dRplus=plus["R"]-(dt*g0-mass[:,None]*(np.stack([strict_layers.f32(n["center"]["v_m_s"]) for n in nodes])-np.stack([strict_layers.f32(n["accepted"]["v_m_s"]) for n in nodes])))
  dRminus=minus["R"]-(dt*g0-mass[:,None]*(np.stack([strict_layers.f32(n["center"]["v_m_s"]) for n in nodes])-np.stack([strict_layers.f32(n["accepted"]["v_m_s"]) for n in nodes])))
  rows.append({"power":power,"epsilon":eps,"plus_minus_valid":True,"plus_projection":{"iterations_max":plus["iterations"],"max_scaled_residual":plus["max_scaled_residual"],"max_dynamic_correction":plus["max_dynamic_correction"],"branch_crossing_rows":plus["branch_crossing_rows"]},"minus_projection":{"iterations_max":minus["iterations"],"max_scaled_residual":minus["max_scaled_residual"],"max_dynamic_correction":minus["max_dynamic_correction"],"branch_crossing_rows":minus["branch_crossing_rows"]},"centered_global_residual_derivative":m,"one_sided_plus_global_derivative":oneplus,"one_sided_minus_global_derivative":oneminus,"plus_deltaR_l2":float(np.linalg.norm(dRplus[free])),"minus_deltaR_l2":float(np.linalg.norm(dRminus[free]))})
  saved[f"p{power}_plus_R"]=plus["R"];saved[f"p{power}_minus_R"]=minus["R"];saved[f"p{power}_secant"]=secant
 else:
  rows.append({"power":power,"epsilon":eps,"plus_minus_valid":False,"plus":{k:v for k,v in endpoints[0].items() if k!="R" and k!="stress"},"minus":{k:v for k,v in endpoints[1].items() if k!="R" and k!="stress"}})
np.savez_compressed(OUT/"assembled_fd.npz",**saved)
inputs=[CAP,CASE/"prediction.json",CASE/"receipt.json",CASE/"output/manifest.json",AUD/"bindings.json",AUD/"manifest.json",AUD/"environment-0/arrays.npz",AUD/"environment-0/report.json",Q/"analysis/native555-global-ideal-fd-002/report.json",Q/"analysis/native555-global-ideal-fd-002/assembled_fd.npz",PREP/"strict_layers.py",REF/"cardboard_bilinear_fp64_reference.py",REF/"cardboard_compensated_fp64_reference.py",REF/"cardboard_cached_operator_audit.py",Q/"snapshots/source-qualification-checkpoint-proof-221-003/matter/tools/cardboard_solver_environment.py"]
report={"schema":"numi.cardboard.native555-global-ideal-fd.v1","classification":"independent CPU assembled ideal-direction FP64 FD; no native replay, no tolerance or convergence qualification","capture_sha256":sha(CAP),"root":19,"newton":3,"environment":0,"nodes":len(nodes),"tets":len(tets),"paper_tets":len(paper),"glue_tets":len(glue),"free_nodes":int(free.sum()),"dt_seconds":dt,"epsilon_powers":powers,"projection_target":{"scaled_residual":2.0e-15,"dynamic_correction":2.0e-14,"max_base_iterations":80,"max_extra_continuation_iterations":16,"equations_changed":False,"tolerance_changed_only_for_diagnostic":True},"direction_source":"captured center F_direction_local_per_s and node correction_delta_v_m_s, same accepted history; ideal unquantized FP64 endpoints, not realized native FP32 endpoints","residual_convention":"R=dt*internal_force-m*(v-v_old); ideal dR/depsilon should equal -A*delta_v","reference_A_action_l2":float(np.linalg.norm(Aaction[free])),"mass_A_action_l2":float(np.linalg.norm(mass_action[free])),"stiffness_A_action_l2":float(np.linalg.norm(stiffness_action[free])),"linear_residual_derivative_l2":float(np.linalg.norm(linear_dR[free])),"epsilon_results":rows,"inputs_sha256":{str(p):sha(p) for p in inputs},"finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
manifest={"schema":"numi.cardboard.native555-global-ideal-fd-files.v1","files_sha256":{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"},"inputs_sha256":report["inputs_sha256"]}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT),"reference_A_action_l2":report["reference_A_action_l2"],"epsilon":[{"power":x["power"],"centered_rel":x.get("centered_global_residual_derivative",{}).get("relative_to_reference"),"plus_rel":x.get("one_sided_plus_global_derivative",{}).get("relative_to_reference"),"minus_rel":x.get("one_sided_minus_global_derivative",{}).get("relative_to_reference"),"valid":x["plus_minus_valid"],"maxr":max(x.get("plus_projection",{}).get("max_scaled_residual",0),x.get("minus_projection",{}).get("max_scaled_residual",0))} for x in rows]},sort_keys=True))
