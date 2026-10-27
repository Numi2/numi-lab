from pathlib import Path
import json,hashlib
from collections import defaultdict,deque,Counter
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
D=S/'per-bank-workers-20261027/b43-16/attempt-001/local-diagnostic.json'
OUT=S/'per-bank-workers-20261027/b43-16/attempt-001/ring-expansion.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(C,allow_pickle=False) as z:F=np.asarray(z['source_O_faces'],np.int64)
base=json.loads(A.read_text());bank=next(b for b in base['banks'] if (b['body'],b['stable_bone_id'])==(43,16));pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']];cross={x[0] for x in pairs}
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
def boundary(patch):
 es=[]
 for e,fs in edgefaces.items():
  inside=[f for f in fs if f in patch];outside=[f for f in fs if f not in patch]
  if inside and outside:es.append(e)
  elif inside and len(fs)==1:es.append(e)
 g=defaultdict(list)
 for u,v in es:g[u].append(v);g[v].append(u)
 unseen=set(g);comps=[]
 while unseen:
  s=min(unseen);unseen.remove(s);q=[s];cc=[]
  while q:
   v=q.pop();cc.append(v)
   for w in g[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  comps.append(cc)
 return es,g,comps
patch=set(cross)
for f in cross:
 for v in F[f]:patch.update(nodefaces[int(v)])
steps=[]
for step in range(3):
 es,g,comps=boundary(patch);hist=dict(Counter(len(g[v]) for v in g));branched=sorted(v for v in g if len(g[v])>2);contained=cross&patch;contained_pairs=[p for p in pairs if p[0] in patch]
 steps.append({'step':step,'patch_face_count':len(patch),'patch_node_count':len({int(v) for f in patch for v in F[f]}),'crossing_skin_faces_contained':len(contained),'crossing_skin_faces_total':len(cross),'exact_cross_pairs_owned':len(contained_pairs),'exact_cross_pairs_total':len(pairs),'missing_crossing_face_rows':sorted(cross-patch),'boundary_edge_count':len(es),'boundary_node_count':len(g),'boundary_component_count':len(comps),'boundary_component_sizes_nodes':sorted([len(c) for c in comps]),'boundary_degree_histogram':hist,'branched_boundary_nodes':branched,'simple_single_loop':bool(len(comps)==1 and all(len(g[v])==2 for v in g))})
 if not branched or step==2:break
 # Expand the complete remaining incident face fan at every branched boundary vertex.
 add=set()
 for v in branched:add.update(nodefaces[int(v)])
 add-=patch
 steps[-1]['branched_vertex_incident_faces_added_next']=len(add)
 steps[-1]['branched_vertex_face_fan_sizes']={str(v):len(set(nodefaces[v])-patch) for v in branched}
 if len(patch)+len(add)>5000:
  steps[-1]['expansion_stopped']='5k-face cap';break
 patch.update(add)
report={'schema':'numi.human.bank43_16.topology_only_ring_expansion.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_sha256':sha(A),'local_diagnostic_sha256':sha(D),'typed_bank':[43,16],'expansion_policy':'At each stage, add every incident mesh face around all current boundary vertices of degree >2. At most two expansions; stop if patch exceeds 5,000 faces. No coordinates moved.','steps':steps,'geometry_or_clearance_conclusion':False,'limitations':['Boundary topology and current exact-crossing face ownership only. No construction, clearance proof, self audit, other-bank check, or physical/native admission.']}
OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
