from pathlib import Path
import hashlib, json
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=ROOT/'bone-clearance-fit-040-shared-columns/attempt-005'
AP=ROOT/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
PP=ROOT/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
ATT4=ROOT/'bone-clearance-fit-040-shared-columns/attempt-004'
PASS=ATT4/'best-valid-components.npz'
AUDIT=ATT4/'component-audit.json'
INP=ATT4/'fit-input.npz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT.mkdir(parents=True,exist_ok=True)
a=np.load(AP,allow_pickle=False); p=np.load(PP,allow_pickle=False); c=np.load(PASS,allow_pickle=False); raw=np.load(ATT4.joinpath(bytes((99,97,110,100,105,100,97,116,101,46,110,112,122)).decode()),allow_pickle=False); inp=np.load(INP,allow_pickle=False)
Xbase=np.asarray(a['points'],np.float32);T=np.asarray(a['tetrahedra'],np.int64);mat=np.asarray(p['material_ids'],np.int64);F=np.asarray(p['O_faces'],np.int64)
if not np.array_equal(T,p['tetrahedra']) or not np.array_equal(mat,p['material_ids']):raise ValueError('AP/PP volume mismatch')
Xstart=np.asarray(c['points_world_f32'],np.float32)
if Xstart.shape!=Xbase.shape:raise ValueError('pass-only candidate point shape mismatch')
oldcols=np.asarray(raw['movable_columns'],np.int64);oldcid=np.asarray(raw['component_id'],np.int64)
if oldcols.shape!=oldcid.shape:raise ValueError('old component labels mismatch')
component_cols={int(ci):set(oldcols[oldcid==ci].tolist()) for ci in np.unique(oldcid)}
# Direct O-face vertex adjacency, matching the fit's surface coupling graph.
E=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
adj=[set() for _ in range(2411)]
for u,v in E:adj[int(u)].add(int(v));adj[int(v)].add(int(u))
# Retry every component with an emitted-F32 q/J or descriptor energy/stress failure.
# One measured outer-face ring is added around each directly failing component.
plan=[
 {'name':'c0_energy','targets':[0],'coupled':[22]},
 {'name':'c1_inverted_energy','targets':[1],'coupled':[]},
 {'name':'c7_constraint_J','targets':[7],'coupled':[]},
 {'name':'c11_J','targets':[11],'coupled':[6,31]},
 {'name':'c16_energy_stress','targets':[16],'coupled':[]},
 {'name':'c19_stress_J','targets':[19],'coupled':[2]},
]
regions=[]
allvars=set()
for ri,s in enumerate(plan):
 target=set().union(*(component_cols[k] for k in s['targets']))
 # The additional surface ring is a variable ring; the next ring stays fixed.
 ring=set().union(*(adj[n] for n in target))-target
 free=target|ring
 # If a newly free vertex's incident tets couple another pre-existing variable
 # component, include its full component rather than freezing one coupled node.
 trows=np.flatnonzero(np.isin(T%2411,list(free)).any(axis=1))
 lkup=np.full(2411,-1,np.int64);lkup[oldcols]=oldcid
 coupled_found=set()
 for tr in trows:
  coupled_found.update(int(x) for x in np.unique(lkup[T[tr]%2411]) if x>=0)
 coupled_found-=set(s['targets'])
 include=set(s['coupled'])|coupled_found
 for k in include: free.update(component_cols[k])
 # Recompute support after coupled groups were added.
 trows=np.flatnonzero(np.isin(T%2411,list(free)).any(axis=1))
 frows=np.flatnonzero(np.isin(F,list(free)).any(axis=1))
 epcols=np.asarray(inp['constraint_endpoint_column_rows'],np.int64)
 ep=np.flatnonzero(np.isin(epcols, list(free)))
 if allvars&free: raise ValueError(f'expansion regions overlap: {sorted(allvars&free)[:20]}')
 allvars|=free
 regions.append({'region':ri,'name':s['name'],'target_components':s['targets'],'coupled_components':sorted(include),'seed_columns':sorted(target),'expanded_ring_columns':sorted(ring),'movable_columns':sorted(free),'face_rows':frows.tolist(),'tet_rows':trows.tolist(),'endpoint_constraint_rows':ep.tolist()})
# Check the final region partitions have no tet or edge terms crossing variable sets.
region_of=np.full(2411,-1,np.int64)
for q,r in enumerate(regions):region_of[np.asarray(r['movable_columns'],np.int64)]=q
for ti,tet in enumerate(T):
 ids=np.unique(region_of[tet%2411]);ids=ids[ids>=0]
 if len(ids)>1:raise ValueError(f'expanded regions share tetrahedron {ti}: {ids.tolist()}')
for edge in E:
 ids=np.unique(region_of[edge]);ids=ids[ids>=0]
 if len(ids)>1:raise ValueError(f'expanded regions share surface edge {edge.tolist()}: {ids.tolist()}')
# Compact ragged arrays.
def pack(name):
 offsets=[0];vals=[]
 for r in regions: vals.extend(r[name]);offsets.append(len(vals))
 return np.asarray(offsets,np.int64),np.asarray(vals,np.int64)
co,cv=pack('movable_columns');fo,fv=pack('face_rows');to,tv=pack('tet_rows');eo,ev=pack('endpoint_constraint_rows')
np.savez_compressed(OUT/'retry-input.npz',points_baseline_f32=Xbase,points_start_pass_only_f32=Xstart,tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),reference_points_m=inp['reference_points_m'],global_nodes=a['global_nodes'],region_column_offsets=co,region_columns=cv,region_face_offsets=fo,region_face_rows=fv,region_tet_offsets=to,region_tet_rows=tv,region_endpoint_offsets=eo,region_endpoint_rows=ev,endpoint_column_rows=epcols,constraint_source_rows=inp['constraint_source_rows'],constraint_endpoint_node_rows=inp['constraint_endpoint_node_rows'],constraint_hit_points_world_m=inp['constraint_plane_point_world_m'],constraint_normals_world=inp['constraint_normal_world'],constraint_margins_m=inp['constraint_clearance_m'],O_faces=F.astype(np.int32))
report={'schema':'numi.human.shared-column-attempt-005-expanded-regions.v1','status':'preflight_prepared_not_fit','input_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [AP,PP,PASS,AUDIT,INP]},'regions':[{k:v for k,v in r.items() if k not in ('face_rows','tet_rows','endpoint_constraint_rows')}|{'surface_face_count':len(r['face_rows']),'incident_tet_count':len(r['tet_rows']),'endpoint_constraints':len(r['endpoint_constraint_rows'])} for r in regions],'disjoint_variables':True,'outside_region_start':'immutable attempt-004 best-valid component assembly; not physically clear, pass-only only on measured q/J/constraint constraints for retained components','limitations':['Root descriptor-energy/stress helper not yet integrated.','The expanded fit is not emitted and is not a physical/native candidate.','Every emitted candidate still requires full F32 J/q/energy/Piola/layer and all-bank exact geometry gates.']}
(OUT/'preflight.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
