#!/usr/bin/env python3
"""Read-only topology/shape/LP-size preflight for a 43/16 centroid-fan repair.
No bone query, optimizer, candidate emission, or geometry audit is performed.
"""
import collections, hashlib, json, os, sys
import numpy as np

ROOT='/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638'
CAND=ROOT+'/partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
AUDIT=ROOT+'/partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
DIAG=ROOT+'/per-bank-workers-20261027/b43-16/attempt-001/local-diagnostic.json'
DISP=ROOT+'/per-bank-workers-20261027/b43-16/attempt-001/final-disposition.json'
PATCH=ROOT+'/per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy'
OUT=os.path.dirname(__file__)
EXPECTED={
 'candidate.npz':'7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af',
 'audit-159/report.json':'c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e',
 'attempt-001/local-diagnostic.json':'adcfeedb5219cb7547b377baa98cbfb83d7ecc64b08b2df1b153bef7ed1aedda',
 'attempt-001/final-disposition.json':'b651eef56622cb66eac52ec79eeef74cba9a4d869a8356caf044cb9fb39a94b6',
 'attempt-001/complement-filled-patch-face-rows.npy':'ad2fb608a536ab8396b781baab4b01f62fc8ae1da81ad533fcb6a4c281131085',
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def shape_metrics(points, child_faces, parent_points, parent_faces, child_parent):
 tri=np.asarray(points,dtype=np.float32)[child_faces].astype(np.float64)
 par=np.asarray(parent_points,dtype=np.float32)[parent_faces[child_parent]].astype(np.float64)
 pn=np.cross(par[:,1]-par[:,0],par[:,2]-par[:,0]); cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 pn2=np.einsum('ij,ij->i',pn,pn); plen=np.sqrt(np.maximum(pn2,1e-300))
 area_rel=np.linalg.norm(cn,axis=1)/plen
 orient=np.einsum('ij,ij->i',pn,cn)/np.maximum(pn2,1e-300)
 angles=[]; aspects=[]
 for t in tri:
  lens=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
  area=float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*0.5)
  aspects.append(max(lens)**2/(4*np.sqrt(3)*max(area,1e-30)))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
  angles.append(min(aa))
 return {'face_count':int(len(child_faces)),
         'minimum_unsigned_parent_area_ratio':float(area_rel.min()),
         'minimum_signed_orientation_ratio':float(orient.min()),
         'minimum_angle_degrees':float(min(angles)),
         'p05_angle_degrees':float(np.percentile(angles,5)),
         'maximum_aspect_measure':float(max(aspects)),
         'p95_aspect_measure':float(np.percentile(aspects,95)),
         'orientation_positive':bool(np.all(orient>0))}

def main():
 inputs={'candidate.npz':CAND,'audit-159/report.json':AUDIT,
         'attempt-001/local-diagnostic.json':DIAG,
         'attempt-001/final-disposition.json':DISP,
         'attempt-001/complement-filled-patch-face-rows.npy':PATCH}
 hashes={k:sha(v) for k,v in inputs.items()}
 if hashes!=EXPECTED: raise RuntimeError('pinned input hash mismatch: '+json.dumps(hashes,sort_keys=True))
 diag=json.load(open(DIAG)); disposition=json.load(open(DISP)); audit=json.load(open(AUDIT))
 parent_rows=np.load(PATCH,allow_pickle=False).astype(np.int64)
 with np.load(CAND,allow_pickle=False) as z:
  P=z['points_world_m']; F=z['source_O_faces']; anc_i=z['face_source_ancestry_indptr']; anc_r=z['face_source_ancestry_face_rows']
 if len(parent_rows)!=434 or len(np.unique(parent_rows))!=434: raise RuntimeError('patch row count/uniqueness mismatch')
 if np.any(parent_rows<0) or np.any(parent_rows>=len(F)): raise RuntimeError('patch row outside candidate faces')
 patch_set=set(map(int,parent_rows)); local=diag['patches'][0]
 crossing=set(map(int,local['crossing_skin_faces']))
 inside=set(int(x['node_row']) for x in local['inside_crossing_nodes'])
 if len(crossing)!=183 or len(inside)!=43 or not crossing.issubset(patch_set): raise RuntimeError('stored witness counts/patch inclusion mismatch')
 patch_faces=F[parent_rows]
 source_vertices=np.unique(patch_faces)
 edge_uses=collections.defaultdict(list); node_faces=collections.defaultdict(set)
 for fi,tri in zip(parent_rows,patch_faces):
  a,b,c=map(int,tri)
  for n in (a,b,c): node_faces[n].add(int(fi))
  for u,v in ((a,b),(b,c),(c,a)): edge_uses[tuple(sorted((u,v)))].append(int(fi))
 boundary_edges=[e for e,uses in edge_uses.items() if len(uses)==1]
 boundary_nodes=set(x for e in boundary_edges for x in e)
 degrees=collections.Counter()
 for u,v in boundary_edges: degrees[u]+=1;degrees[v]+=1
 edge_adjacency=collections.defaultdict(set)
 for uses in edge_uses.values():
  if len(uses)==2:
   a,b=uses;edge_adjacency[a].add(b);edge_adjacency[b].add(a)
 seen=set();components=[]
 for row in parent_rows:
  r=int(row)
  if r in seen: continue
  todo=[r];seen.add(r);n=0
  while todo:
   q=todo.pop();n+=1
   for w in edge_adjacency[q]:
    if w not in seen: seen.add(w);todo.append(w)
  components.append(n)
 euler=len(source_vertices)-len(edge_uses)+len(parent_rows)
 ancestry_counts=anc_i[parent_rows+1]-anc_i[parent_rows]
 if np.any(ancestry_counts<=0): raise RuntimeError('empty source-ancestry span in patch')
 if not inside.issubset(set(map(int,source_vertices))): raise RuntimeError('inside witness node outside patch')
 if inside & boundary_nodes: raise RuntimeError('inside witness lies on fixed outer boundary')

 # The active neighborhood is witness faces + all faces incident to the 43 inside
 # nodes, expanded by one shared-edge ring. Outer boundary vertices remain fixed.
 core=set(crossing)
 for n in inside: core.update(node_faces[n] & patch_set)
 active=set(core)
 for r in core: active.update(edge_adjacency[r])
 active_nodes=set(int(n) for r in active for n in F[r])
 movable_source=active_nodes-boundary_nodes
 movable_centers=set(active)
 movable_points=len(movable_source)+len(movable_centers)
 touched_parents=set(active)
 for n in movable_source:
  touched_parents.update(node_faces[n] & patch_set)
 affected_children=3*len(touched_parents)

 # Match the 115/118 center: half incenter barycentrics + half centroid. Test
 # emitted Float32 geometry, with untouched original nodes copied bitwise.
 local_nodes=np.asarray(sorted(source_vertices),dtype=np.int64)
 node_to_local={int(n):i for i,n in enumerate(local_nodes)}
 centers=np.empty((len(parent_rows),3),np.float64); bary=np.empty((len(parent_rows),3),np.float64)
 src_faces_local=np.empty((len(parent_rows),3),np.int64)
 for j,(fi,tri_ids) in enumerate(zip(parent_rows,patch_faces)):
  tri=P[tri_ids].astype(np.float64)
  opp=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])])
  if not np.isfinite(opp).all() or opp.sum()<=0: raise RuntimeError('invalid parent edge lengths')
  bary[j]=0.5*opp/opp.sum()+0.5/3.0
  centers[j]=bary[j]@tri
  src_faces_local[j]=[node_to_local[int(n)] for n in tri_ids]
 emitted_centers=centers.astype(np.float32)
 fan_points=np.vstack((P[local_nodes].astype(np.float32),emitted_centers)).astype(np.float32)
 nsrc=len(local_nodes); child_faces=[]; child_parent=[]
 for j,(a,b,c) in enumerate(src_faces_local):
  m=nsrc+j
  child_faces.extend(((a,b,m),(b,c,m),(c,a,m)))
  child_parent.extend((j,j,j))
 child_faces=np.asarray(child_faces,np.int64); child_parent=np.asarray(child_parent,np.int64)
 parent_points=P[local_nodes].astype(np.float32)
 parent_tri_points=P[F[parent_rows]].astype(np.float32)
 metrics=shape_metrics(fan_points,child_faces,parent_tri_points,np.arange(len(parent_rows)),child_parent)
 center_corner_collisions=[]
 for j,tri_ids in enumerate(patch_faces):
  for corner,n in enumerate(tri_ids):
   if np.array_equal(emitted_centers[j],P[int(n)]): center_corner_collisions.append({'parent_face_row':int(parent_rows[j]),'corner_node':int(n)})

 report={
  'schema':'numi.human.bank43_16.centroid_fan_prep.v1',
  'status':'PREFLIGHT_ONLY_NO_FIT_OR_BONE_OR_SELF_AUDIT',
  'inputs':{k:{'path':v,'sha256':hashes[k]} for k,v in inputs.items()},
  'candidate':{'point_count':int(len(P)),'face_count':int(len(F)),'dtype_points':str(P.dtype),'dtype_faces':str(F.dtype)},
  'patch':{'parent_face_count':int(len(parent_rows)),'unique_source_vertex_count':int(len(source_vertices)),
           'edge_count':int(len(edge_uses)),'boundary_edge_count':int(len(boundary_edges)),
           'boundary_node_count':int(len(boundary_nodes)),'boundary_degree_histogram':{str(k):int(v) for k,v in collections.Counter(degrees.values()).items()},
           'face_edge_components':components,'euler_characteristic':int(euler),
           'crossing_face_count':int(len(crossing)),'crossing_faces_in_patch':int(len(crossing&patch_set)),
           'exact_crossing_pair_count':int(local['crossing_pair_count']),'inside_source_node_count':int(len(inside)),
           'inside_nodes_not_on_boundary':int(len(inside-boundary_nodes)),
           'ancestry_parent_count':int(len(ancestry_counts)),
           'ancestry_entries_min_max':[int(ancestry_counts.min()),int(ancestry_counts.max())],
           'expanded_child_faces':int(3*len(parent_rows)),
           'ancestry_entries_if_each_child_inherits_parent_span':int(3*ancestry_counts.sum())},
  'active_region':{'core_parent_faces':int(len(core)),'one_edge_ring_parent_faces':int(len(active)),
                   'movable_source_nodes_excluding_fixed_boundary':int(len(movable_source)),
                   'movable_fan_centers':int(len(movable_centers)),
                   'total_movable_points':int(movable_points),
                   'movable_coordinate_dofs':int(3*movable_points),
                   'affected_parent_faces_for_shape_linearization':int(len(touched_parents)),
                   'affected_child_faces_for_shape_linearization':int(affected_children),
                   'fixed_boundary_nodes':int(len(boundary_nodes))},
  'sparse_lp_estimate':{'variables_coordinate_delta_plus_abs_slack':int(6*movable_points),
                        'absolute_value_rows':int(6*movable_points),
                        'crossing_endpoint_halfspace_rows':int(2*int(local['crossing_pair_count'])),
                        'known_inside_source_halfspace_rows':int(len(inside)),
                        'worst_case_point_inside_rows_for_every_movable_point':int(movable_points),
                        'conservative_bone_row_bound':int(2*int(local['crossing_pair_count'])+movable_points),
                        'linearized_area_rows':int(affected_children),
                        'total_linear_inequality_row_bound':int(6*movable_points+2*int(local['crossing_pair_count'])+movable_points+affected_children),
                        'estimated_nonzero_upper_bound':int(12*movable_points+9*2*int(local['crossing_pair_count'])+3*movable_points+9*affected_children),
                        'notes':['one 0.5 mm trust-region LP with HiGHS is proposed; no solver run occurred',
                                 '498 exact-pair endpoint rows are retained, with one halfspace per endpoint',
                                 'point-location adds at most one plane per movable point; exact local classification is deferred',
                                 'linearized child-area constraints are provisional; exact emitted Float32 area, angle, aspect, bone and self checks remain mandatory']},
  'emitted_baseline_shape':metrics,
  'center_float32_equals_parent_corner_count':int(len(center_corner_collisions)),
  'center_float32_equals_parent_corner_examples':center_corner_collisions[:8],
  'gates':{'constructor_reference_min_parent_area_ratio':0.10,'min_angle_degrees':0.5,'max_aspect_measure':50.0,
           'baseline_shape_gate_pass':bool(metrics['orientation_positive'] and metrics['minimum_unsigned_parent_area_ratio']>=0.10 and metrics['minimum_angle_degrees']>=0.5 and metrics['maximum_aspect_measure']<=50.0 and not center_corner_collisions),
           'fit_authorized_or_run':False,'candidate_emitted':False,'bone_or_self_audit_run':False},
  'limitations':['This is topology and shape preflight only; it does not check collision clearance or physical adequacy.',
                 'The LP size assumes a centroid fan over the saved one-disk patch and one-edge-ring movable support.',
                 'Exact pair-to-fan barycentric endpoint mapping and per-point bone normals must be reconstructed from the pinned exact witnesses before a solver run.',
                 'After any authorized fit, reject unless the exact emitted Float32 skin has strict net crossing reduction, zero new self pairs, all 185 bank no-worsening, ancestry/source/anchor and shape gates pass.']}
 out=OUT+'/preflight.json'
 with open(out,'w') as f: json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:report[k] for k in ('status','patch','active_region','sparse_lp_estimate','emitted_baseline_shape','center_float32_equals_parent_corner_count','gates')},indent=2,sort_keys=True))
 print('receipt',out)

if __name__=='__main__': main()
