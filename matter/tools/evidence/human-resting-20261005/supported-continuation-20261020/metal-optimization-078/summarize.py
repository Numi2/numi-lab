import json,pathlib,re,math
p=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078")
metrics=["root_xyz_m","center_of_mass_xyz_m","root_linear_velocity_xyz_m_s","root_linear_speed_m_s","max_generalized_acceleration","penetration_m","contact_count","min_J","max_J","max_displacement_m","max_anchor_reaction_N","passive_ligament_max_tension_N","passive_ligament_force_residual_N","passive_ligament_moment_residual_Nm","body_contact_normal_force_N","body_contact_force_residual_N","body_contact_moment_residual_Nm"]
summ={}
for run in p.iterdir():
 if not (run/"process.json").exists():continue
 t=(run/"stdout.log").read_text();m={}
 for line in t.splitlines():
  if not any(x in line for x in ["human_standing_progress=accepted","resting_FEM_terminal_audit=","resting_fascia_fem_terminal_audit="]):continue
  for k,v in re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)",line):
   if k in metrics:
    try:m[k]=[float(z) for z in v[1:-1].split(",")] if v.startswith("[") else float(v)
    except ValueError:pass
 rows=re.findall(r"resting_integrated_body=completed([^\n]*)",t)
 s=json.load(open(run/"process.json"));s.update(body=dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)",rows[-1])) if rows else {},terminal_metrics=m,fem_terminal_passed="resting_FEM_terminal_audit=passed" in t,fascia_terminal_passed="resting_fascia_fem_terminal_audit=passed" in t)
 json.dump(s,open(run/"summary.json","w"),indent=2);summ[run.name]=s
base=summ.get("global-icb-base-1",{}).get("terminal_metrics",{})
for name,s in summ.items():
 print(name,s["body"].get("wall_s"),s["body"].get("real_time_factor"),s["fem_terminal_passed"],s["fascia_terminal_passed"])
 if name.startswith("combined"):
  for k,a in base.items():
   b=s["terminal_metrics"].get(k)
   if b is None:continue
   a=a if isinstance(a,list) else [a];b=b if isinstance(b,list) else [b]
   error=max(abs(x-y)/max(max(abs(x),abs(y)),1e-5) for x,y in zip(a,b))
   if error>.01:print("  >1pct",k,error)
