import csv,json,re,sys,math,hashlib
from pathlib import Path
run=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
log=(run/"launcher.log").read_text()
completion=re.search(r"resting_integrated_body=completed simulated_s=([\d.e+-]+) wall_s=([\d.e+-]+) real_time_factor=([\d.e+-]+) physiology_body_clock=(\w+) root_assistance=(\w+)",log)
def rows(name):
 with (run/name).open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
phys=rows("resting-coupled.csv")
# Surface audit includes a string geometry_mode; preserve other numeric fields.
with (run/"resting-surface-audit.csv").open() as f:
 audit=[{k:(v if k=="geometry_mode" else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
supports=rows("resting-com-support-impulses.csv")
slot=int(re.search(r"resting_rope_requested=true contact_index=(\d+)",log).group(1))
rope=[x for x in supports if x["contact_index"]==slot];other=[x for x in supports if x["contact_index"]!=slot]
def bounds(items,key):
 values=[r[key] for r in items];assert values and all(math.isfinite(x) for x in values),key
 return {"min":min(values),"max":max(values)}
def absmax(items,key):return max(abs(r[key]) for r in items)
inv=json.loads((run/"invocation.json").read_bytes())
dt=float(inv["argv"][inv["argv"].index("--muscle-step-seconds")+1])
report={"run_path":str(run),"status":"COMPLETED_NATIVE_CHECKPOINT" if completion else "STOPPED_NATIVE_ATTEMPT","continuous_16s_pass":bool(completion and float(completion[1])>=16.),
"host":"SSH numi","device":"Apple M6","binary_sha256":inv["binary_sha256"],"invocation_sha256":sha(run/"invocation.json"),"log_sha256":sha(run/"launcher.log"),
"completion":None if not completion else {"simulated_s":float(completion[1]),"wall_s":float(completion[2]),"real_time_factor":float(completion[3]),"physiology_body_clock":completion[4],"root_assistance":completion[5]},
"trace":{"rows":len(phys),"time_s":bounds(phys,"time_s"),"strictly_increasing_time":all(a["time_s"]<b["time_s"] for a,b in zip(phys,phys[1:])),"complete_breaths":phys[-1]["breaths"],"complete_cardiac_filling_ejection_cycles":phys[-1]["complete_filling_ejection_cycles"],"lung_volume_ml":bounds(phys,"lung_volume_ml"),"airflow_ml_s":bounds(phys,"airflow_ml_s"),"diaphragm_mm":bounds(phys,"diaphragm_mm"),"rib_mm":bounds(phys,"rib_mm"),"PaO2_mmhg":bounds(phys,"PaO2_mmhg"),"PaCO2_mmhg":bounds(phys,"PaCO2_mmhg"),"blood_error_ml_max_abs":absmax(phys,"blood_error_ml"),"oxygen_balance_error_stpd_ml_max_abs":absmax(phys,"oxygen_balance_error_stpd_ml"),"co2_balance_error_stpd_ml_max_abs":absmax(phys,"co2_balance_error_stpd_ml"),"aortic_ejected_ml":phys[-1]["aortic_ejected_ml"],"pulmonary_ejected_ml":phys[-1]["pulmonary_ejected_ml"],"root_assistance_n_max_abs":absmax(phys,"root_assistance_n"),"root_assistance_nm_max_abs":absmax(phys,"root_assistance_nm")},
"rope":{"existing_contact_index":slot,"impulse_sample_scope":"last substep of each sampled segment; not a segment integral","last_substep_tension_force_n":{"min":min(x["normal_impulse_ns"]/dt for x in rope),"max":max(x["normal_impulse_ns"]/dt for x in rope)},"max_tangent_impulse_ns":max(absmax(rope,"tangent0_impulse_ns"),absmax(rope,"tangent1_impulse_ns")),"other_contact_max_normal_impulse_ns":max((r["normal_impulse_ns"] for r in other),default=0.)},
"surface_audit":{"frames":len(audit),"max_zero_area_triangles":max(r["mesh_zero_area_triangles"] for r in audit),"max_nonfinite_area_triangles":max(r["mesh_nonfinite_area_triangles"] for r in audit),"triangles_checked":bounds(audit,"mesh_triangles_checked"),"max_functional_volume_relative_error":max(r["max_functional_volume_relative_error"] for r in audit),"mass_kg":bounds(audit,"represented_body_mass_kg")},
"evidence_scope":"native suspension checkpoint with retained coupled state; not five-minute resting/intervention qualification, clinical validation, all-surface intersection qualification, or blinking"}
movie=run/"rope-suspension-reference.mov"
if movie.is_file():report["continuous_unretimed_native_movie"]={"path":str(movie),"bytes":movie.stat().st_size,"sha256":sha(movie)}
report["retained_trace_sha256"]={name:sha(run/name) for name in ("resting-coupled.csv","resting-surface-audit.csv","resting-com-support-impulses.csv")}
(run/"checkpoint-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
