from pathlib import Path
import csv,json,hashlib
from collections import defaultdict,deque
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
W=ROOT/"combined-d071-b109-b61-001"
D=ROOT/"b109-114/attempt-031-alpha-scaled-progress-screen-001"
with np.load(W/"candidate.npz",allow_pickle=False) as z:
 F=np.asarray(z["source_O_faces"],np.int64);X=np.asarray(z["points_world_m"],np.float32)
 parent=np.asarray(z["refinement_face_parent_face_row"],np.int64)
a=json.loads((W/"audit-159/report.json").read_text())
b=next(v for v in a["banks"] if (int(v["body"]),int(v["stable_bone_id"]))==(109,114))
target_faces={int(f) for f,bb in b["result"]["triangle_pairs"]}
inside=[]
with (W/"audit-130/query/pairs.csv").open(newline="") as f:
 for r in csv.DictReader(f):
  if (r["bank_body"],r["bank_stable_id"],r["classification"])==("109","114","inside"):inside.append(int(r["query_row"]))
self_pairs=[tuple(map(int,r)) for r in np.load(D/"audit-130/surface-self-pairs.npy")]
seed=set(target_faces)
seed.update(f for f,tri in enumerate(F) if any(int(n) in set(inside) for n in tri))
seed.update(f for pair in self_pairs for f in pair)
edge_faces=defaultdict(list)
for fi,tri in enumerate(F):
 for j in range(3):edge_faces[tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))].append(fi)
adj=[set() for _ in range(len(F))]
for uses in edge_faces.values():
 for f in uses:
  adj[f].update(g for g in uses if g!=f)
def comps(faces):
 rem=set(faces);out=[]
 while rem:
  root=min(rem);rem.remove(root);q=deque([root]);c=[]
  while q:
   f=q.popleft();c.append(f)
   for g in adj[f]&rem:rem.remove(g);q.append(g)
  out.append(sorted(c))
 return sorted(out,key=lambda c:(-len(c),c[0]))
def boundary_info(faces):
 fs=set(faces);edge_count=defaultdict(int)
 for f in fs:
  t=F[f]
  for j in range(3):edge_count[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))]+=1
 bedges=[e for e,n in edge_count.items() if n==1]
 graph=defaultdict(set)
 for u,v in bedges:graph[u].add(v);graph[v].add(u)
 deg=Counter(len(v) for v in graph.values()) if False else {k:sum(len(v)==k for v in graph.values()) for k in sorted(set(map(len,graph.values())))}
 unseen=set(graph);loops=[]
 while unseen:
  r=min(unseen);unseen.remove(r);q=deque([r]);c=[]
  while q:
   u=q.popleft();c.append(u)
   for v in graph[u]&unseen:unseen.remove(v);q.append(v)
  loops.append(len(c))
 return {"boundary_edges":len(bedges),"boundary_vertices":len(graph),"boundary_degree_histogram":deg,"boundary_components":sorted(loops,reverse=True),"all_boundary_degree_two":all(len(v)==2 for v in graph.values())}
rows=[];front=set(seed)
for ring in range(0,5):
 patch=set(front) if ring==0 else set(front)
 if ring==0:patch=set(seed)
 else:
  patch |= prior
 prior=patch
 cs=comps(patch)
 rows.append({"ring":ring,"faces":len(patch),"nodes":int(len(np.unique(F[np.asarray(sorted(patch),np.int64)]))),"component_face_counts":[len(c) for c in cs],"component_boundaries":[boundary_info(c) for c in cs],"boundary":boundary_info(patch)})
 if ring<4:prior=set(patch)|set().union(*(adj[f] for f in patch))
receipt={"schema":"numi.human.b109-114.full-target-segment-patch-topology.v1","candidate_sha256":hashlib.sha256((W/"candidate.npz").read_bytes()).hexdigest(),"faces":len(F),"nodes":len(X),"target_raw_pair_count":len(b["result"]["triangle_pairs"]),"target_skin_face_count":len(target_faces),"inside_sample_nodes":inside,"inside_sample_count":len(inside),"self_witness_pairs":[list(x) for x in self_pairs],"seed_union_face_count":len(seed),"seed_union_component_counts":[len(c) for c in comps(seed)],"rings":rows,"construction_only":True,"physical_steps":0}
out=D/"patch-topology-preflight.json";out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"target_skin_faces":len(target_faces),"inside_nodes":len(inside),"seed_faces":len(seed),"rings":rows},indent=2))
