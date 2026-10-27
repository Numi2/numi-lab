import importlib.util,json,hashlib
import numpy as np
from pathlib import Path
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546");R=D/"hard-face-feasibility-634/local-refinement-638/from-764-001";W=D/"hard-face-feasibility-634/fresh-fit-from715-001";F=W/"retention-582-001";base=R/"joint-fit-001/restored-retention-582-001/next-714-001/retention-582-001";B=D/"box-aware-sat-fit-637/run-001-from766/merit-631-001";module=R.parent/"parent_mapped_comparison642.py"
spec=importlib.util.spec_from_file_location("m",module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=base/"candidate.npz";y=F/"candidate.npz"; parent=B/"candidate.npz"; latest=json.loads((F/"final/state.json").read_text())
assert m.sha(y)==latest["candidate_sha256"] and latest["candidate_selected"] is False and latest["native_admission"] is False
with np.load(x,allow_pickle=False) as u,np.load(y,allow_pickle=False) as v:
 assert np.array_equal(u["source_O_faces"],v["source_O_faces"]) and np.array_equal(u["refinement_face_parent_face_row"],v["refinement_face_parent_face_row"])
 mp=np.asarray(v["refinement_face_parent_face_row"])
old,banks=m.inventory(base/"full-check/audit-159/report.json",x);new,banks2=m.inventory(F/"full-check/audit-159/report.json",y);p,banks3=m.inventory(B/"full-check/audit-159/report.json",parent);pm,banks4=m.inventory(F/"full-check/audit-159/report.json",y,mp)
assert banks==banks2==banks3==banks4 and len(old)==715 and len(new)==703 and len(p)==764
point=json.loads((F/"full-check/audit-130/summary.json").read_text());assert point["inside_node_bank_pairs"]==point["exact_PLC_surface_pairs"]==0
changes=[]
for key in sorted(banks):
 a=sum(t[:2]==key for t in old);b=sum(t[:2]==key for t in new)
 if a!=b:changes.append({"body":key[0],"bone":key[1],"before":a,"after":b,"delta":b-a})
out={"status":"SAME_TOPOLOGY_AND_PARENT_IDENTITY_COMPARISON_UNADMITTED","same_topology":True,"candidate_before_sha256":m.sha(x),"candidate_after_sha256":m.sha(y),"before_surface_pairs":len(old),"after_surface_pairs":len(new),"same_topology_pair_identities_removed":len(old-new),"same_topology_pair_identities_introduced":len(new-old),"original_parent_pairs":len(p),"after_distinct_parent_pairs":len(pm),"parent_pair_identities_removed":len(p-pm),"parent_pair_identities_introduced":len(pm-p),"bank_changes":changes,"contained_node_bank_pairs":0,"self_pairs":0,"native_admission":False,"accepted_physical_steps":0,"source_script_sha256":m.sha(__file__),"inventory_owner_sha256":m.sha(module)}
pout=F/"final/comparison-703.json";assert not pout.exists();pout.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2))
