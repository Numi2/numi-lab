from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-bank43-selfcut-001/retained-631v2-component-5883-only-001")
OUT=Path(__file__).resolve().parent
STATE=BASE/"final-546/state.json"
CAND=BASE/"candidate.npz"
FULL=BASE/"full-check/report.json"
AUDIT=BASE/"full-check/audit-159/report.json"
AUDIT130=BASE/"full-check/audit-130/summary.json"
WANTED=((43,16),(109,114),(65,99))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def connected_components(face_rows,adj):
 remaining=set(map(int,face_rows)); out=[]
 while remaining:
  start=min(remaining); stack=[start]; comp={start}; remaining.remove(start)
  while stack:
   f=stack.pop()
   for g in adj[f]:
    if g in remaining:
     remaining.remove(g);comp.add(g);stack.append(g)
  out.append(sorted(comp))
 return out
def main():
 required=(STATE,CAND,FULL,AUDIT,AUDIT130)
 missing=[str(p) for p in required if not p.is_file()]
 if missing: raise FileNotFoundError(missing)
 state=json.loads(STATE.read_text()); full=json.loads(FULL.read_text())
 audit=json.loads(AUDIT.read_text()); point=json.loads(AUDIT130.read_text())
 cand_sha=sha(CAND)
 if state.get("candidate_sha256")!=cand_sha or Path(state["candidate"]).resolve()!=CAND.resolve():
  raise ValueError("state does not bind the exact candidate")
 if full.get("candidate_sha256")!=cand_sha or audit.get("candidate_sha256")!=cand_sha or point.get("candidate_sha256")!=cand_sha:
  raise ValueError("one or more exact reports are not candidate-bound")
 if audit.get("complete") is not True or audit.get("bank_count")!=185:
  raise ValueError("159 exact report is incomplete or lacks all 185 banks")
 if point.get("complete_surface_audit") is not True or point.get("exact_PLC_surface_pairs")!=0 or point.get("inside_node_bank_pairs")!=0:
  raise ValueError("130 self/inside baseline is not strict zero")
 with np.load(CAND,allow_pickle=False) as z:
  faces=np.asarray(z["source_O_faces"],dtype=np.int64)
  anchors=np.asarray(z["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if anchors.shape!=(237,) or len(np.unique(anchors))!=237:
  raise ValueError("expected exact 237 unique typed anchors")
 edge_faces=defaultdict(list)
 for fi,tri in enumerate(faces):
  for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   edge_faces[tuple(sorted((int(u),int(v))))].append(int(fi))
 adj=[set() for _ in range(len(faces))]
 for fs in edge_faces.values():
  for f in fs:
   adj[f].update(g for g in fs if g!=f)
 target_pairs={}
 for bank in WANTED:
  rows=[]
  matches=[b for b in audit["banks"] if (int(b["body"]),int(b["stable_bone_id"]))==bank]
  if len(matches)!=1: raise ValueError(f"bank inventory mismatch for {bank}")
  result=matches[0].get("result") or {}
  for face,tri in result.get("triangle_pairs",[]):
   rows.append([int(face),int(bank[0]),int(bank[1]),int(tri)])
  if len(rows)!=int(result.get("count",-1)): raise ValueError(f"pair count mismatch for {bank}")
  target_pairs[bank]=rows
 bank_reports=[]
 all_rows=[]
 for bank,rows in target_pairs.items():
  byface=defaultdict(list)
  for r in rows: byface[r[0]].append(r)
  patches=[]
  for comp in connected_components(byface.keys(),adj):
   patch_rows=[r for face in comp for r in byface[face]]
   nodes=np.unique(faces[np.asarray(comp,dtype=np.int64)].ravel()).astype(np.int64)
   perface={str(f):len(byface[f]) for f in comp}
   patches.append({"bank":[int(bank[0]),int(bank[1])],"pair_count":len(patch_rows),
     "face_count":len(comp),"face_rows":comp,"per_face_pair_counts":perface,
     "node_count":int(len(nodes)),"node_rows":nodes.tolist(),
     "typed_anchor_overlap_count":int(np.intersect1d(nodes,anchors).size),
     "pair_identities_face_body_bone_triangle":patch_rows})
  patches.sort(key=lambda x:(-x["pair_count"],-x["face_count"],x["face_rows"][0]))
  bank_reports.append({"bank":[int(bank[0]),int(bank[1]),"stable_bone_id"],
   "pair_count":len(rows),"unique_crossed_faces":len(byface),
   "patch_count":len(patches),"patches":patches,
   "top_concentrated_faces":sorted(
     [{"face_row":int(f),"pair_count":len(v),"triangle_rows":sorted(r[3] for r in v)} for f,v in byface.items()],
     key=lambda x:(-x["pair_count"],x["face_row"]))[:8]})
  all_rows.extend(rows)
 combined_byface=defaultdict(list)
 for r in all_rows:combined_byface[r[0]].append(r)
 combined=[]
 for comp in connected_components(combined_byface.keys(),adj):
  patch_rows=[r for face in comp for r in combined_byface[face]]
  nodes=np.unique(faces[np.asarray(comp,dtype=np.int64)].ravel()).astype(np.int64)
  bc=Counter((r[1],r[2]) for r in patch_rows)
  combined.append({"pair_count":len(patch_rows),"face_count":len(comp),"face_rows":comp,
    "bank_pair_counts":[{"body":int(b),"stable_bone_id":int(k),"pair_count":int(n)} for (b,k),n in sorted(bc.items())],
    "node_count":int(len(nodes)),"node_rows":nodes.tolist(),
    "typed_anchor_overlap_count":int(np.intersect1d(nodes,anchors).size),
    "pair_identities_face_body_bone_triangle":patch_rows})
 combined.sort(key=lambda x:(-x["pair_count"],-x["face_count"],x["face_rows"][0]))
 source_paths=required+(Path(__file__).resolve(),)
 report={"schema":"numi.target_group_adjacency_analysis663.v1",
  "classification":"read-only grouping of exact candidate-bound audit-159 pair identities by edge-connected current skin faces; no solver or geometry recomputation",
  "candidate_path":str(CAND),"candidate_sha256":cand_sha,
  "state_path":str(STATE),"state_sha256":sha(STATE),
  "full_check_report_path":str(FULL),"full_check_report_sha256":sha(FULL),
  "audit159_path":str(AUDIT),"audit159_sha256":sha(AUDIT),
  "audit130_summary_path":str(AUDIT130),"audit130_summary_sha256":sha(AUDIT130),
  "candidate_domain":{"points":int(point.get("points",-1)),"faces":int(point.get("faces",-1)),
   "typed_anchors":int(len(anchors)),"exact_self_pairs":int(point.get("exact_PLC_surface_pairs",-1)),
   "inside_node_bank_pairs":int(point.get("inside_node_bank_pairs",-1))},
  "adjacency_rule":"two current source_O_faces rows are adjacent when they share an undirected mesh edge",
  "nonmanifold_edges_shared_by_more_than_two_faces":int(sum(len(v)>2 for v in edge_faces.values())),
  "selected_banks":bank_reports,
  "combined_edge_connected_patches":combined,
  "combined_totals":{"exact_pair_rows":len(all_rows),"unique_skin_faces":len(combined_byface),
    "unique_skin_nodes":int(len(np.unique(faces[np.asarray(sorted(combined_byface),dtype=np.int64)].ravel()))),
    "edge_connected_patch_count":len(combined),
    "typed_anchor_overlap_nodes":int(sum(x["typed_anchor_overlap_count"] for x in combined))},
  "suggested_joint_target_set":{"include_all_pair_rows":True,"pairs":len(all_rows),
    "patches":len(combined),"reason":"these rows are the complete current 159 crossings for the three requested banks, grouped into five edge-connected face patches; retain every exact typed pair row together rather than selecting individual pairs",
    "feasibility_claim":False,"requires_existing_joint_lp_and_exact_checks":True},
  "limitations":["This groups existing 159 witnesses only; no exact geometry was rerun.",
    "Edge-connected face patches are topological groupings, not independent solver components.",
    "No axis choices, hard-LP feasibility, Float32 acceptance, or anatomy/physical clearance are claimed.",
    "The baseline's audit130 reports zero self pairs and zero inside node-bank pairs; edge/solid clearance certification is not asserted."],
  "source_pins":{str(p):sha(p) for p in source_paths},
  "native_admission":False,"accepted_physical_steps":0}
 out=OUT/"target-groups.json"
 if out.exists(): raise FileExistsError(out)
 out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":"PASS","report":str(out),"report_sha256":sha(out),
  "candidate_sha256":cand_sha,"audit159_sha256":sha(AUDIT),
  "banks":[{"bank":x["bank"][:2],"pairs":x["pair_count"],"faces":x["unique_crossed_faces"],"patches":x["patch_count"]} for x in bank_reports],
  "total_pairs":len(all_rows),"combined_patches":len(combined)},sort_keys=True))
if __name__=="__main__": main()
