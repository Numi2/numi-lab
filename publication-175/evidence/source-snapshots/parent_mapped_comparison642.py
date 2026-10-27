"""Compare 159 crossing identities through the exact immediate face-parent map; no physical admission."""
from pathlib import Path
import hashlib,json
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
R=D/"hard-face-feasibility-634/local-refinement-638/from-764-001"
B=D/"box-aware-sat-fit-637/run-001-from766/merit-631-001"
F=R/"joint-fit-001/restored-retention-582-001/next-714-001/retention-582-001"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
 if not ok:raise ValueError(msg)
def inventory(path,candidate,mapping=None):
 r=read(path);require(r["candidate_sha256"]==sha(candidate) and r["complete"] is True and r["bank_count"]==r["expected_bank_count"]==185,"159 report mismatch")
 result=set();banks=set()
 for b in r["banks"]:
  key=(int(b["body"]),int(b["stable_bone_id"]));require(key not in banks,"duplicate bank");banks.add(key);v=b.get("result") or {"count":0,"triangle_pairs":[]}
  require(v["count"]==len(v["triangle_pairs"]),"pair count mismatch")
  for f,t in v["triangle_pairs"]:
   row=int(mapping[int(f)]) if mapping is not None else int(f);result.add((key[0],key[1],row,int(t)))
 return result,banks
def main():
 parent=B/"candidate.npz";refined=R/"candidate.npz";final=F/"candidate.npz";state=read(F/"strict-from-refined/state.json");meta=read(R/"report.json")
 require(state["candidate_sha256"]==sha(final) and state["native_admission"] is False and state["accepted_physical_steps"]==0,"final state pin mismatch")
 require(meta["input_candidate_sha256"]==sha(parent) and meta["output_candidate_sha256"]==sha(refined),"refinement lineage mismatch")
 with np.load(parent,allow_pickle=False)as b,np.load(refined,allow_pickle=False)as r,np.load(final,allow_pickle=False)as f:
  m=r["refinement_face_parent_face_row"];require(np.array_equal(m,f["refinement_face_parent_face_row"]) and len(m)==len(f["source_O_faces"]) and np.all(m>=0) and np.all(m<len(b["source_O_faces"])),"invalid parent map")
 old,banks=inventory(B/"full-check/audit-159/report.json",parent);initial,banks1=inventory(R/"audit-159/report.json",refined,m);new,banks2=inventory(F/"full-check/audit-159/report.json",final,m)
 require(banks==banks1==banks2 and old==initial,"baseline and initial split do not have equivalent parent pair identities")
 point=read(F/"full-check/audit-130/summary.json");require(point["candidate_sha256"]==sha(final) and point["complete_surface_audit"] is True and point["exact_PLC_surface_pairs"]==0 and point["inside_node_bank_pairs"]==0,"final 130 mismatch")
 removed=old-new;introduced=new-old; per=[]
 for key in sorted(banks):
  a=sum(v[:2]==key for v in old);b=sum(v[:2]==key for v in new)
  if a!=b:per.append({"body":key[0],"bone_id":key[1],"before":a,"after":b,"delta":b-a})
 out=F/"strict-from-refined/parent-mapped-comparison.json";require(not out.exists(),"output exists")
 report={"status":"MATCHED_PARENT_PAIR_IDENTITY_COMPARISON_UNADMITTED","parent_candidate_sha256":sha(parent),"initial_refined_sha256":sha(refined),"final_candidate_sha256":sha(final),"parent_surface_report_sha256":sha(B/"full-check/audit-159/report.json"),"initial_surface_report_sha256":sha(R/"audit-159/report.json"),"final_surface_report_sha256":sha(F/"full-check/audit-159/report.json"),"baseline_parent_pairs":len(old),"initial_refined_parent_pairs":len(initial),"final_distinct_parent_pairs":len(new),"parent_pairs_removed":len(removed),"parent_pairs_introduced":len(introduced),"final_exact_child_surface_pairs":sum((v.get("result") or {}).get("count",0) for v in read(F/"full-check/audit-159/report.json")["banks"]),"final_contained_node_bank_pairs":0,"final_self_pairs":0,"per_bank":per,"face_parent_map":"immediate refinement face parent; duplicate child witnesses on one parent pair collapse","native_admission":False,"accepted_physical_steps":0,"source_script_sha256":sha(__file__),"limitations":["Counts describe audited triangle-pair identities at one emitted rest pose; they do not establish whole-body anatomy, swept-motion clearance, volume containment, or physical/native qualification."]}
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({k:report[k] for k in ("baseline_parent_pairs","final_distinct_parent_pairs","parent_pairs_removed","parent_pairs_introduced","final_exact_child_surface_pairs")}))
if __name__=="__main__":main()
