from pathlib import Path
import numpy as np,importlib.util,sys,hashlib,json,time,collections
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
O=E/'boundary-disk-construction-287'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
geo=module('geo287',O/'tools/human_skin_patch_geometry.py')
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
import numilab_human
ci=module('numilab_human.cardiac_cavity_intersections',E/'exact-plane-filter-owner-264/src/numilab_human/cardiac_cavity_intersections.py')
source=E/'bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz'
q=np.load(source,allow_pickle=False);X=q['outer_world_actual007_m'].astype(np.float32);F=q['surface_faces']
edges=collections.defaultdict(list)
for t in F:
 for a,b in zip(t,np.roll(t,-1)):edges[tuple(sorted((int(a),int(b))))].append((int(a),int(b)))
bound=[v[0] for v in edges.values() if len(v)==1]
nxt={}
for a,b in bound:
 if a in nxt:raise RuntimeError('nonmanifold open boundary')
 nxt[a]=b
loops=[]
while nxt:
 a=min(nxt);loop=[a];b=nxt.pop(a)
 while b!=a:
  loop.append(b);b=nxt.pop(b)
 loops.append(np.asarray(loop,dtype=np.int64))
caps=[];details=[];dome_parts=[]
physical_X=X.copy()
for loop in loops:
 cap,rep=geo.triangulate_boundary_disk(X,loop[::-1])
 loop_points=X[loop].astype(float)
 centered=loop_points-loop_points.mean(axis=0)
 _,_,ax=np.linalg.svd(centered,full_matrices=False)
 # Original source O is inward-wound, hence negative cap normal is outward.
 n=-np.cross(X[cap[:,1]]-X[cap[:,0]],X[cap[:,2]]-X[cap[:,0]]).sum(axis=0).astype(float)
 n/=np.linalg.norm(n)
 begin=len(X)
 boundary={tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1))}
 for level in range(4):
  es={tuple(sorted((int(a),int(b)))) for t in cap for a,b in zip(t,np.roll(t,-1))}
  selected={e:.5 for e in es if e not in boundary and (level==0 or np.linalg.norm(X[e[0]]-X[e[1]])>.002)}
  if not selected:break
  X,cap,_,_,_=geo.split_surface_edges(X,cap,selected)
 ids=np.arange(begin,len(X))
 uv=centered@ax[:2].T
 pts=(X[ids]-loop_points.mean(axis=0))@ax[:2].T
 distance=np.full(len(ids),np.inf)
 for a,b in zip(uv,np.roll(uv,-1,axis=0)):
  d=b-a;t=np.clip(((pts-a)@d)/(d@d),0,1)
  distance=np.minimum(distance,np.linalg.norm(pts-(a+t[:,None]*d),axis=1))
 weights=np.sqrt(distance/max(float(distance.max(initial=0)),1e-12))
 dome_parts.append((ids,n,weights))
 caps.append(cap);details.append({'loop':loop.tolist(),'report':rep,'temporary_interior_nodes':len(ids)})
C=np.vstack(caps)
base=X.copy();trials=[]
best=None
for amplitude in [-.0001,.0001,-.00025,.00025,-.0005,.0005,-.001,.001,-.002,.002,-.004,.004,-.008,.008,-.016,.016]:
 X=base.copy()
 for ids,n,weights in dome_parts:X[ids]+=amplitude*weights[:,None]*n
 X=X.astype(np.float32)
 capindex=ci._records([ci.float32_point_lattice_key(p) for p in X],C.tolist())
 index=ci._records([ci.float32_point_lattice_key(p) for p in X],F.tolist())
 sa=ci._audit_pair(capindex,capindex,same_surface=True)
 oa=ci._audit_record_pairs(ci._aabb_candidate_pairs(capindex,index,same_surface=False),same_surface=True)
 shell=np.vstack((F,C))
 edge_count=collections.Counter(tuple(sorted((int(a),int(b)))) for t in shell for a,b in zip(t,np.roll(t,-1)))
 bad_edges=sum(v!=2 for v in edge_count.values())
 trials.append({'amplitude_m':amplitude,'self':sa['count'],'outer':oa['count'],'bad_incidence_edges':bad_edges})
 score=(sa['count']+oa['count']+bad_edges,abs(amplitude))
 if best is None or score<best[0]:best=(score,X.copy(),amplitude)
 if sa['count']==0 and oa['count']==0 and bad_edges==0:break
X=best[1];selected_amplitude=best[2]
shell=np.vstack((F,C))
assert np.array_equal(X[:len(physical_X)],physical_X)
index=ci._records([ci.float32_point_lattice_key(p) for p in X],F.tolist())
capindex=ci._records([ci.float32_point_lattice_key(p) for p in X],C.tolist())
selfaudit=ci._audit_pair(capindex,capindex,same_surface=True)
outeraudit=ci._audit_record_pairs(ci._aabb_candidate_pairs(capindex,index,same_surface=False),same_surface=True)
v=float(np.einsum('ij,ij->i',X[shell[:,0]].astype(float),np.cross(X[shell[:,1]].astype(float),X[shell[:,2]].astype(float))).sum()/6)
parity=1 if v>0 else -1
out=O/'global-outer-dome-287.npz'
np.savez_compressed(out,points_world_m=X,shell_faces=shell if parity==1 else shell[:,[0,2,1]],physical_outer_faces=F,temporary_cap_faces=C,construction_closure_faces=C,face_role=np.r_[np.zeros(len(F),np.uint8),np.ones(len(C),np.uint8)],temporary_cap_face_rows=np.arange(len(F),len(shell)),source_to_plc_winding_parity=np.array([parity],np.int64),temporary_vertex_mask=np.arange(len(X))>=len(physical_X))
report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'shell_npz_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_loops':details,'physical_outer_faces':len(F),'temporary_cap_faces':len(C),'signed_source_volume_m3':v,'plc_winding_parity':parity,'cap_self':selfaudit,'cap_vs_physical_outer':outeraudit,'construction_pass':selfaudit['count']==0 and outeraudit['count']==0,'physical_admission':False,'anatomical_apertures_unchanged':True,'scope':'temporary cap self and cap-vs-O only; retained O not newly globally audited','dome_trials':trials,'temporary_interior_nodes':len(X)-len(physical_X),'physical_outer_vertices_bitwise_unchanged':True,'selected_amplitude_m':selected_amplitude}
(O/'global-outer-dome-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['construction_pass','physical_outer_faces','temporary_cap_faces','signed_source_volume_m3','plc_winding_parity']}))
print('cap_self',selfaudit['count'],'capO',outeraudit['count'])

