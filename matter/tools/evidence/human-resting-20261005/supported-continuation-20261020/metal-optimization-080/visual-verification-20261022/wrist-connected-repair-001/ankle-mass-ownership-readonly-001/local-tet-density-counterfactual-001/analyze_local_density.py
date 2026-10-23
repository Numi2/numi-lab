import contextlib,hashlib,io,json,runpy
from pathlib import Path
import numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
O=E/"ankle-mass-ownership-readonly-001/local-tet-density-counterfactual-001"
O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
 q=runpy.run_path(str(E/"ankle-mass-ownership-readonly-001/reconstruct_mass_ownership.py"))
b,c=q["b"],q["c"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def layer(name,T,lo,hi):
 d0,_,rho0=q["layer"](b,name,T); d1,_,rho1=q["layer"](c,name,T)
 v0=q["vol"](b["x"],T); v1=q["vol"](c["x"],T)
 ch=np.any(np.any(b["x"][T]!=c["x"][T],axis=2),axis=1)
 rho=np.full(len(T),rho0); rho[ch]=rho0*v0[ch]/v1[ch]
 patch_rho=rho0*float(v0[ch].sum())/float(v1[ch].sum())
 badids=np.flatnonzero(ch & ((rho<lo)|(rho>hi)))
 m0=rho0*v0; m1=rho*v1
 obj={"changed_tet_count":int(ch.sum()),"changed_tet_ids":np.flatnonzero(ch).tolist(),
 "changed_face_ordinals":np.unique(np.flatnonzero(ch)%4619).tolist(),
 "baseline_density_kg_m3":float(rho0),"original_candidate_uniform_density_kg_m3":float(rho1),
 "counterfactual_changed_density_min_kg_m3":float(rho[ch].min()),"counterfactual_changed_density_max_kg_m3":float(rho[ch].max()),
 "counterfactual_full_density_min_kg_m3":float(rho.min()),"counterfactual_full_density_max_kg_m3":float(rho.max()),
 "material_bounds_kg_m3":[lo,hi],"changed_tet_violations":int(((rho[ch]<lo)|(rho[ch]>hi)).sum()),
 "all_tet_violations":int(((rho<lo)|(rho>hi)).sum()),
 "base_mass_kg":float(m0.sum()),"counterfactual_mass_kg":float(m1.sum()),
 "total_mass_delta_kg":float(m1.sum()-m0.sum()),"max_abs_element_mass_delta_kg":float(np.abs(m1-m0).max()),
 "sum_abs_element_mass_delta_kg":float(np.abs(m1-m0).sum()),
 "aggregate_changed_patch_required_density_kg_m3":float(patch_rho),
 "aggregate_changed_patch_mass_preservation_within_bounds":bool(lo<=patch_rho<=hi),
 "bounded_changed_patch_mass_kg":[float(lo*v1[ch].sum()),float(hi*v1[ch].sum())],
 "baseline_changed_patch_mass_kg":float(m0[ch].sum()),
 "violating_changed_tets":[{"tet_id":int(i),"face_ordinal":int(i%4619),"density_kg_m3":float(rho[i]),"volume_ratio_new_over_old":float(v1[i]/v0[i])} for i in badids]}
 return {"T":T,"m0":m0,"m1":m1,"rho":rho,"report":obj}
s=layer("skin",q["TS"],900.,1100.); f=layer("fat",q["TF"],850.,950.)
def lump(which):
 out=np.zeros(3*2332)
 for z in (s,f):
  m=z["m0"] if which=="old" else z["m1"]
  for j in range(4): np.add.at(out,z["T"][:,j],m/4)
 return out
old=lump("old"); new=lump("new"); delta=new-old
mass={"node_count":len(delta),"old_sum_kg":float(old.sum()),"counterfactual_sum_kg":float(new.sum()),
 "max_abs_nodal_delta_kg":float(np.abs(delta).max()),"max_relative_nodal_delta":float((np.abs(delta)/np.maximum(np.abs(old),1e-300)).max()),
 "rms_nodal_delta_kg":float(np.sqrt(np.mean(delta**2))),"nonzero_float64_delta_count":int(np.count_nonzero(delta)),
 "float32_node_masses_exact_equal":bool(np.array_equal(old.astype(np.float32),new.astype(np.float32))),
 "scope":"skin/fat layer lumped contributions only; other body/tissue masses excluded"}
base_report=E/"ankle-mass-ownership-readonly-001/mass-ownership-007-vs-016.json"
inp=json.loads(base_report.read_text())["inputs_sha256"]
bounds_src=E/"ankle-coupled-021-on-294-001/main294-commonmode-layer-candidate-001/layer-candidate-report.json"
report={"schema":"numi.human.local-tet-density-counterfactual.v1","status":"read_only_counterfactual_not_implemented",
 "method":{"rho_new_tet":"rho_old * V_old_tet / V_new_tet on changed tetrahedra; unchanged tetrahedra retain rho_old",
 "tet_mass":"rho * abs(det)/6, equally lumped to four incident nodes","topology":"same 007 to 016 connectivity; no app/material/runtime changes"},
 "inputs_sha256":inp,"source_report_sha256":sha(base_report),"bounds_report_sha256":sha(bounds_src),
 "layers":{"skin":s["report"],"fat":f["report"]},"nodal_mass":mass,
 "decision":{"all_counterfactual_tet_densities_within_material_bounds":s["report"]["all_tet_violations"]==0 and f["report"]["all_tet_violations"]==0,
 "local_element_mass_preservation_possible_within_bounds":s["report"]["all_tet_violations"]==0 and f["report"]["all_tet_violations"]==0,
 "material_override_created":False,"physical_admission":False}}
out=O/"local-tet-density-counterfactual.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(out),"report_sha256":sha(out),"script_sha256":sha(__file__),"layers":report["layers"],"nodal_mass":mass,"decision":report["decision"]},indent=2,sort_keys=True))
