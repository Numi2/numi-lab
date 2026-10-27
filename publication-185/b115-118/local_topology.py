from pathlib import Path
import json, hashlib
from collections import defaultdict, deque
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
OUT=S/'per-bank-workers-20261027/b115-118/attempt-005/local-topology-report.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
a=json.loads(A.read_text()); bank=next(b for b in a['banks'] if (b['body'],b['stable_bone_id'])==(115,118)); pairs=bank['result']['triangle_pairs']
cross=sorted({int(p[0]) for p in pairs}); crossset=set(cross)
with np.load(C,allow_pickle=False) as z:
 P=z['points_world_m'].astype(np.float64); F=z['source_O_faces'].astype(np.int64)
 coarse=z['parent_coarse_face_row']; detail=z['parent_fine_detail_face_row']; sub=z['parent_fine_subface_row'];
 ancptr=z['face_source_ancestry_indptr']; ancrows=z['face_source_ancestry_face_rows'];
 fsubptr=z['face_fine_subface_ancestry_indptr']; fsubrows=z['face_fine_subface_ancestry_rows'];
 fdetailptr=z['face_fine_detail_ancestry_indptr']; fdetailrows=z['face_fine_detail_ancestry_rows'];
face_edges=[]; edgefaces=defaultdict(list); nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t: nodefaces[int(v)].append(fi)
 es=[]
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
  e=tuple(sorted((int(u),int(v)))); es.append(e); edgefaces[e].append(fi)
 face_edges.append(es)
def get_components(verts, by_vertex=False):
 vset=set(verts); unseen=set(vset); out=[]
 while unseen:
  start=min(unseen); unseen.remove(start); q=deque([start]); comp=[]
  while q:
   f=q.popleft(); comp.append(f)
   cand=set()
   if by_vertex:
    for v in F[f]:cand.update(nodefaces[int(v)])
   else:
    for e in face_edges[f]:cand.update(edgefaces[e])
   for g in cand&unseen:
    unseen.remove(g);q.append(g)
  out.append(sorted(comp))
 return out
cross_edge_components=get_components(cross)
cross_vertex_components=get_components(cross,True)
# Derive one-/two-ring face sets around crossings.
ring1=set(cross)
for f in cross:
 for v in F[f]:ring1.update(nodefaces[int(v)])
ring2=set(ring1)
for f in list(ring1):
 for v in F[f]:ring2.update(nodefaces[int(v)])
# Topology edge contacts between crossed face set and the one-ring complement.
components=[]
for ci,comp in enumerate(cross_edge_components):
 cset=set(comp); nodes=sorted(set(map(int,F[comp].reshape(-1))))
 ecounts=defaultdict(int)
 for f in comp:
  for e in face_edges[f]:ecounts[e]+=1
 bedges=[e for e,k in ecounts.items() if k==1]
 adjacent=set()
 for e in bedges: adjacent.update(edgefaces[e])
 adjacent.difference_update(cset)
 ringfaces=set(comp)
 for f in comp:
  for v in F[f]:ringfaces.update(nodefaces[int(v)])
 def uniq(vals): return sorted(set(map(int,vals)))
 parent_rows=uniq(coarse[comp]); detail_rows=uniq(detail[comp]); sub_rows=uniq(sub[comp])
 src=[]; fs=[]; fd=[]
 for f in comp:
  src.extend(ancrows[ancptr[f]:ancptr[f+1]])
  fs.extend(fsubrows[fsubptr[f]:fsubptr[f+1]])
  fd.extend(fdetailrows[fdetailptr[f]:fdetailptr[f+1]])
 # Mesh patch boundary information; edges at boundary between component-crossing faces and adjacent outside faces.
 components.append({"component":ci,"crossing_face_rows":comp,"crossing_pair_count":sum(1 for p in pairs if int(p[0]) in cset),
  "crossing_bone_face_rows":uniq(p[1] for p in pairs if int(p[0]) in cset),
  "crossing_node_rows":nodes,"crossing_edge_count":len(ecounts),"internal_edges":sum(k>1 for k in ecounts.values()),
  "crossing_patch_boundary_edge_count":len(bedges),"edge_adjacent_non_crossing_faces":len(adjacent),
  "one_vertex_ring_face_count":len(ringfaces),"two_vertex_ring_face_count":len(set(comp)|{g for f in ringfaces for v in F[f] for g in nodefaces[int(v)]}),
  "parent_coarse_face_rows":parent_rows,"fine_detail_face_rows":detail_rows,"fine_subface_rows":sub_rows,
  "source_ancestry_rows":uniq(src),"fine_subface_ancestry_rows":uniq(fs),"fine_detail_ancestry_rows":uniq(fd),
  "patch_boundary_edges": [list(e) for e in bedges]})
# Candidate source ancestry/face parent check on ring and count crossing faces with no preserved ancestry.
ring_ids=sorted(ring1)
report={"schema":"numi.human.bank115_118.local_topology.v1","candidate_path":str(C),"candidate_sha256":sha(C),"audit159_path":str(A),"audit159_sha256":sha(A),"bank":{"body":115,"stable_bone_id":118,"exact_cross_pair_count":len(pairs),"crossing_skin_face_count":len(cross),"crossing_skin_face_rows":cross,"crossing_bone_face_rows":sorted({int(p[1]) for p in pairs})},"crossing_face_edge_connected_component_count":len(cross_edge_components),"crossing_face_vertex_connected_component_count":len(cross_vertex_components),"crossing_face_vertex_components":[c for c in cross_vertex_components],"crossing_components":components,"rings":{"crossing_faces":len(crossset),"one_vertex_ring_faces":len(ring1),"two_vertex_ring_faces":len(ring2),"one_ring_faces_not_exact_crossing":len(ring1-crossset),"two_ring_faces_not_exact_crossing":len(ring2-ring1),"coarse_parent_rows_one_ring":sorted(set(map(int,coarse[ring_ids]))),"detail_ancestry_rows_crossing_faces":sorted({int(v) for f in cross for v in fdetailrows[fdetailptr[f]:fdetailptr[f+1]]})},"shape_probe":{"candidate_min_face_area_m2":float(np.min(np.linalg.norm(np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]),axis=1)*0.5)),"local_ring_min_face_area_m2":float(np.min(np.linalg.norm(np.cross(P[F[ring_ids,1]]-P[F[ring_ids,0]],P[F[ring_ids,2]]-P[F[ring_ids,0]]),axis=1)*0.5)),"outer_boundary_node_count":len(set(v for f in ring1-ring2 for v in F[f])) if ring1-ring2 else 0},"limitations":["Topology and source-ancestry only; no candidate remesh or physical admission."]}
OUT.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("bank","crossing_face_edge_connected_component_count","crossing_face_vertex_connected_component_count","crossing_face_vertex_components","rings","shape_probe")},indent=2))
