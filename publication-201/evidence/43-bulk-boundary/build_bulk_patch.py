from pathlib import Path
import hashlib,json
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
B=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
C=B/"candidate.npz";A=B/"audit-159/report.json";I=B/"audit-130/summary.json"
OLDPATCH=S/"per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy"
PIN="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af";sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(C)!=PIN:raise ValueError("baseline pin mismatch")
a=json.loads(A.read_text());inside=json.loads(I.read_text())
if inside.get("candidate_sha256")!=PIN or not inside.get("complete_surface_audit"):raise ValueError("inside report not complete/pinned")
with np.load(C,allow_pickle=False) as z:c={k:z[k] for k in z.files}
f=c["source_O_faces"]
target_bank=next((b for b in a["banks"] if int(b["body"])==43 and int(b["stable_bone_id"])==16),None)
if target_bank is None:raise ValueError("typed bank 43/16 missing")
pairs=sorted({(int(sf),int(bf)) for sf,bf in ((target_bank.get("result") or {}).get("triangle_pairs") or [])})
crossing_rows=sorted({q[0] for q in pairs})
inside_records=[w for w in inside["inside_witnesses"] if int(w.get("bank_body",-1))==43 and int(w.get("bank_stable_id",-1))==16 and w.get("classification")=="inside" and int(w.get("ambiguous",-1))==0]
inside_nodes=sorted({int(w["output_node"]) for w in inside_records})
if len(inside_nodes)==0:raise ValueError("no target inside nodes in audit130")
node_faces={}
for r,t in enumerate(f):
 for n in t:node_faces.setdefault(int(n),[]).append(r)
inside_star_rows=sorted({r for n in inside_nodes for r in node_faces.get(n,[])})
patch=set(crossing_rows)|set(inside_star_rows)
edgeowners={}
for r,t in enumerate(f):
 for j in range(3):
  u,v=int(t[j]),int(t[(j+1)%3]);edgeowners.setdefault((min(u,v),max(u,v)),[]).append(r)
def describe(rows):
 rows=set(rows);bedges=[];adj={};adjf={r:set() for r in rows};bad=[]
 for e,own in edgeowners.items():
  active=[r for r in own if r in rows]
  if len(active)>1:
   for i in active:
    for j in active:
     if i!=j:adjf[i].add(j)
  outside=[r for r in own if r not in rows]
  if active and outside:
   bedges.append({"edge":list(e),"inside_faces":active,"outside_faces":outside})
   u,v=e;adj.setdefault(u,[]).append(v);adj.setdefault(v,[]).append(u)
  elif active and len(own)==1:
   bedges.append({"edge":list(e),"inside_faces":active,"outside_faces":[]})
   u,v=e;adj.setdefault(u,[]).append(v);adj.setdefault(v,[]).append(u)
 for e,own in edgeowners.items():
  if any(r in rows for r in own) and len(own)!=2:bad.append({"edge":list(e),"owners":own})
 comps=[];seen=set()
 for s in sorted(adj):
  if s in seen:continue
  stack=[s];seen.add(s);co=[]
  while stack:
   u=stack.pop();co.append(u)
   for v in adj[u]:
    if v not in seen:seen.add(v);stack.append(v)
  comps.append(sorted(co))
 loops=[]
 for co in comps:
  if all(len(adj[n])==2 for n in co):
   s=min(co);prev=None;cur=s;loop=[]
   while True:
    loop.append(cur);nxt=next(v for v in adj[cur] if v!=prev)
    if nxt==s:break
    prev,cur=cur,nxt
   loops.append(loop)
 facecomps=[];seenf=set()
 for s in sorted(rows):
  if s in seenf:continue
  stack=[s];seenf.add(s);co=[]
  while stack:
   u=stack.pop();co.append(u)
   for v in adjf[u]:
    if v not in seenf:seenf.add(v);stack.append(v)
  facecomps.append(sorted(co))
 return {"face_rows":sorted(rows),"face_count":len(rows),"boundary_edges":bedges,"boundary_edge_count":len(bedges),"boundary_nodes":sorted(adj),"boundary_degree_histogram":{str(d):sum(len(v)==d for v in adj.values()) for d in sorted({len(v) for v in adj.values()})},"boundary_components":[sorted(x) for x in comps],"boundary_cycles":loops,"boundary_simple_loops":len(comps)==len(loops) and all(len(loops[i])==len(comps[i]) for i in range(min(len(loops),len(comps)))) ,"face_component_count":len(facecomps),"face_component_sizes":sorted(map(len,facecomps),reverse=True),"global_nonmanifold_or_open_edges":bad}
source_ptr=c["face_source_ancestry_indptr"];source_vals=c["face_source_ancestry_face_rows"];detail_ptr=c["face_fine_detail_ancestry_indptr"];detail_vals=c["face_fine_detail_ancestry_rows"];sub_ptr=c["face_fine_subface_ancestry_indptr"];sub_vals=c["face_fine_subface_ancestry_rows"];par_ptr=c["refinement_face_parent_ancestry_indptr"];par_vals=c["refinement_face_parent_ancestry_face_rows"]
def row_ids(ptr,vals,r):return list(map(int,vals[int(ptr[r]):int(ptr[r+1])]))
def lineage(rows):
 ret=[];bad=[]
 for r in sorted(rows):
  q={"face_row":r,"face_kind":int(c["face_kind"][r]),"source":row_ids(source_ptr,source_vals,r),"detail":row_ids(detail_ptr,detail_vals,r),"subface":row_ids(sub_ptr,sub_vals,r),"parent":row_ids(par_ptr,par_vals,r)};ret.append(q)
  if q["face_kind"]!=2 or not all(q[k] for k in ("source","detail","subface","parent")) or not set(q["detail"])<=set(q["source"]):bad.append(q)
 return ret,bad
stages=[];first=describe(patch);lin,bad=lineage(patch);first["lineage_complete"]=not bad;first["lineage_failures"]=bad;stages.append({"stage":"crossings_plus_inside_node_stars","topology":first})
if not first["boundary_simple_loops"] or first["face_component_count"]!=1:
 ring={r for e in first["boundary_edges"] for r in e["outside_faces"]}
 patch2=patch|ring
 second=describe(patch2);lin2,bad2=lineage(patch2);second["lineage_complete"]=not bad2;second["lineage_failures"]=bad2;second["added_one_ring_faces"]=sorted(ring-set(patch));stages.append({"stage":"one_edge_ring_expansion_if_needed","topology":second});patch=patch2;final=second;final_lineage=lin2;final_bad=bad2
else:final=first;final_lineage=lin;final_bad=bad
old=np.load(OLDPATCH,allow_pickle=False).astype(np.int64) if OLDPATCH.is_file() else np.zeros(0,dtype=np.int64)
report={"schema":"numi.human.b43-16.bulk-crossing-inside-star-patch.v1","baseline_path":str(C),"baseline_sha256":PIN,"audit159_sha256":sha(A),"audit130_sha256":sha(I),"target_bank":[43,16],"target_exact_pair_count":len(pairs),"target_unique_crossing_face_count":len(crossing_rows),"target_crossing_face_rows":crossing_rows,"inside_target_node_count":len(inside_nodes),"inside_target_node_rows":inside_nodes,"inside_target_witnesses":inside_records,"all_incident_face_count_for_inside_nodes":len(inside_star_rows),"inside_node_star_face_rows":inside_star_rows,"starting_union_face_count":len(set(crossing_rows)|set(inside_star_rows)),"prior_434_patch_face_count":int(len(old)),"prior_434_patch_equals_starting_union":bool(set(map(int,old))==set(crossing_rows)|set(inside_star_rows)),"prior_patch_overlap_count":int(len(set(map(int,old))&(set(crossing_rows)|set(inside_star_rows)))),"stages":stages,"final_topology":final,"final_patch_face_rows":final["face_rows"],"final_patch_face_count":final["face_count"],"final_source_union":sorted({x for q in final_lineage for x in q["source"]}),"final_detail_union":sorted({x for q in final_lineage for x in q["detail"]}),"final_subface_union":sorted({x for q in final_lineage for x in q["subface"]}),"final_parent_union":sorted({x for q in final_lineage for x in q["parent"]}),"final_per_face_ancestry":final_lineage,"final_ancestry_complete":not final_bad,"geometry_changed":False,"candidate_emitted":False}
status="BULK_PATCH_TOPOLOGY_READY_FOR_EXACT_BOUNDARY_CHECK" if final["face_component_count"]==1 and final["boundary_simple_loops"] and not final_bad else "BULK_PATCH_TOPOLOGY_NOT_READY"
report["status"]=status
O=Path(__file__).with_name("bulk-patch-topology.json");O.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:report[k] for k in ("status","target_exact_pair_count","target_unique_crossing_face_count","inside_target_node_count","all_incident_face_count_for_inside_nodes","starting_union_face_count","prior_434_patch_face_count","prior_434_patch_equals_starting_union","prior_patch_overlap_count","final_patch_face_count","final_source_union","final_detail_union","final_subface_union","final_ancestry_complete","final_topology")},indent=2,sort_keys=True))
