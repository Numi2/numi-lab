from pathlib import Path
import json,csv,collections,numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
D=Path("/Users/numi/Work/numi-human-connected-surface-069");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
p=D/'repair-run-005/iteration-004';a=json.loads((p/'tet-edge-report.json').read_text())
z=np.load(E/'E/connected-volume-refinement-037/run-006/candidate.npz');T=z['tetrahedra'];M=z['material_ids'];X=np.load(p/'candidate.npz')['points_world_f32']
lookup={int(g):i for i,g in enumerate(z['global_node_ids_original_prefix'])}
names=['outer_skin','skin_fat_interface','deep_fat'];layers=[]
for i in range(len(X)):
 lo,hi=z['source_vertex_ancestry_indptr'][i:i+2]
 layers.append(tuple(sorted({names[lookup[int(g)]//2411]for g in z['source_vertex_ancestry_global_ids'][lo:hi]})))
emap=collections.defaultdict(set)
for tet,mat in zip(T,M):
 for i in range(4):
  for j in range(i):emap[tuple(sorted((int(tet[i]),int(tet[j]))))].add(int(mat))
groups={}
for w in a['witnesses']:groups[(w['edge_index'],w['body'],w['bone_stable_id'])]=w
bylayer=collections.Counter();bymat=collections.Counter();bybone=collections.Counter()
for w in groups.values():
 a,b=map(int,w['endpoint_local_nodes'])
 bylayer[' / '.join(sorted(set(layers[a]+layers[b])))]+=1
 bymat[str(sorted(emap[tuple(sorted((a,b)))]))]+=1
 bybone[(int(w['body']),int(w['bone_stable_id']))]+=1
rows=[]
for (body,bone),count in bybone.most_common():
 W=[w for w in groups.values()if w['body']==body and w['bone_stable_id']==bone]
 nodes=np.unique([n for w in W for n in w['endpoint_local_nodes']])
 rows.append({'body':body,'bone_stable_id':bone,'edge_bank_groups':count,'touched_nodes':len(nodes),'world_aabb_min_m':X[nodes].min(axis=0).tolist(),'world_aabb_max_m':X[nodes].max(axis=0).tolist()})
report={'candidate_sha256':__import__('hashlib').sha256((p/'candidate.npz').read_bytes()).hexdigest(),'groups':len(groups),'ancestral_layer_groups':dict(bylayer),'incident_material_groups':dict(bymat),'bone_groups':rows,'physical_steps':0,'geometry_admitted':False,'note':'Ancestral layer tags, not new source anatomy or proof of full-solid clearance.'}
(D/'connected-crossing-region-map.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k]for k in ['groups','ancestral_layer_groups','incident_material_groups']},indent=2));print(json.dumps(rows[:8],indent=2))
