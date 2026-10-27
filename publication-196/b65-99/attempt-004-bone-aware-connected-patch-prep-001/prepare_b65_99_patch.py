from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=C.parent/'audit-159/report.json'
TOPO=S/'per-bank-workers-20261027/b65-99/attempt-002/fan-hole-ancestry-check.json'
SUBULP=S/'per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/facewise-failure-witness.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=S/'per-bank-workers-20261027/b65-99/attempt-004-bone-aware-connected-patch-prep-001'
C_PIN='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
A_PIN='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
CLEARANCE_MM=0.05
MOVEMENT_CAP_MM=2.0
started=time.monotonic()

def json_fallback(x):
 if isinstance(x,set):return sorted(x)
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(f'unsupported JSON type {type(x).__name__}')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def patch_topology(faces):
 uses=Counter();adj=defaultdict(set)
 for f in np.asarray(faces,np.int64):
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);e=tuple(sorted((a,b)));uses[e]+=1;adj[a].add(b);adj[b].add(a)
 if any(n not in (1,2) for n in uses.values()):raise RuntimeError('selected patch has nonmanifold edge incidence')
 boundary=[e for e,n in uses.items() if n==1]
 deg=Counter();boundary_adj=defaultdict(set)
 for a,b in boundary:deg[a]+=1;deg[b]+=1;boundary_adj[a].add(b);boundary_adj[b].add(a)
 loops=0;seen=set()
 for v in boundary_adj:
  if v in seen:continue
  loops+=1;stack=[v];seen.add(v)
  while stack:
   x=stack.pop()
   for y in boundary_adj[x]:
    if y not in seen:seen.add(y);stack.append(y)
  if any(len(boundary_adj[x])!=2 for x in stack):pass
 comp=0;seenv=set()
 for v in adj:
  if v in seenv:continue
  comp+=1;stack=[v];seenv.add(v)
  while stack:
   x=stack.pop()
   for y in adj[x]:
    if y not in seenv:seenv.add(y);stack.append(y)
 euler=len(adj)-len(uses)+len(faces)
 return boundary,{'vertex_count':len(adj),'edge_count':len(uses),'face_count':len(faces),'boundary_edge_count':len(boundary),
  'boundary_vertex_count':len(boundary_adj),'boundary_loop_count':loops,'boundary_degree_histogram':{str(k):v for k,v in Counter(deg.values()).items()},
  'connected_components':comp,'euler_characteristic':euler,'is_single_disk':bool(comp==1 and loops==1 and euler==1 and all(v==2 for v in deg.values()))}

OUT.mkdir(parents=True,exist_ok=True)
report={'schema':'numi.human.bank65_99.bone-aware-connected-patch-preflight.v1','status':'STARTED',
 'scope':'Read-only local preflight. No optimizer, vertex edits, exact re-audit, all-bank audit, or self screen.',
 'bank':[65,99],'candidate_path':str(C),'audit159_path':str(A),'topology_path':str(TOPO),'sub_ulp_witness_path':str(SUBULP),
 'constraints_proposed':{'crossing_face_vertex_rule':'for each stored exact skin-face/stable-bone-triangle pair, constrain all three vertices of that skin face to the outward bone-triangle plane halfspace',
  'clearance_mm':CLEARANCE_MM,'per_node_movement_cap_mm':MOVEMENT_CAP_MM,'fixed_boundary':'63-edge simple patch loop from pinned two-ring/fan topology',
  'shape_guards_for_later_fit':{'same 241 source face identities','positive orientation','projected area ratio >= 0.10','min angle >= max(0.5 deg, 0.9 * input patch min angle)','max node displacement <= 2.0 mm'},
  'no contour roots will be inserted':'Use current patch vertices and face identities; no exact endpoints become nodes or edges.'}}
try:
 if sha(C)!=C_PIN or sha(A)!=A_PIN:raise RuntimeError('pinned candidate or audit changed')
 topology=json.load(open(TOPO));subulp=json.load(open(SUBULP))
 if topology.get('candidate_sha256')!=C_PIN or topology.get('target_bank')!=[65,99]:raise RuntimeError('two-ring/fan topology is not pinned to this target')
 if subulp.get('candidate_sha256')!=C_PIN:raise RuntimeError('sub-ULP receipt candidate pin mismatch')
 summary=topology['two_ring_patch_summary'][0];fan=topology['fan_expansion'][0]
 rows=sorted(set(map(int,summary['face_rows']))|set(map(int,fan['fan_face_rows'])))
 if len(rows)!=241 or int(topology['final_boundary_checks'][0]['patch_face_count'])!=len(rows):raise RuntimeError('pinned patch face inventory differs from topology receipt')
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P32=np.asarray(Z['points_world_m'],np.float32);P=P32.astype(np.float64);F=np.asarray(Z['source_O_faces'],np.int64)
 parent_faces=F[np.asarray(rows,np.int64)]
 boundary,topo=patch_topology(parent_faces)
 if not topo['is_single_disk'] or topo['boundary_edge_count']!=63:raise RuntimeError(f'patch topology preflight disagrees with pinned 63-edge disk: {topo}')
 bank_doc=next(x for x in json.load(open(A))['banks'] if (int(x['body']),int(x['stable_bone_id']))==(65,99))
 pairs=[tuple(map(int,p)) for p in bank_doc['result']['triangle_pairs']]
 cross_faces=sorted(set(f for f,_ in pairs))
 if len(pairs)!=148 or len(cross_faces)!=120 or not set(cross_faces).issubset(set(rows)):raise RuntimeError('stored crossing-pair inventory no longer matches local patch')
 ind=np.asarray(Z['face_source_ancestry_indptr'],np.int64);anc=np.asarray(Z['face_source_ancestry_face_rows'],np.int64)
 empty=[f for f in rows if ind[f+1]<=ind[f]]
 if empty:raise RuntimeError(f'candidate source-face CSR is empty for patch rows {empty[:8]}')
 helper=loadmod('b65_99_prepare_bank_helper',HELPER)
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 mask=(ids[:,0]==65)&(ids[:,1]==99);bone= np.asarray(T[mask],np.float64);stable=np.asarray(ids[mask,3],np.int64)
 if len(bone)!=342 or len(set(map(int,stable)))!=len(stable):raise RuntimeError('bank triangle inventory mismatch')
 bone_row={int(v):i for i,v in enumerate(stable)}
 raw_norm=np.cross(bone[:,1]-bone[:,0],bone[:,2]-bone[:,0]);norm_len=np.linalg.norm(raw_norm,axis=1)
 if np.any(norm_len==0):raise RuntimeError('degenerate input bone triangle')
 volume6=float(sum(float(a@np.cross(b,c)) for a,b,c in bone));outward=raw_norm/norm_len[:,None]*(1.0 if volume6>0 else -1.0)
 skin_local=(P-pos[65])@M[65].T-off[65]
 constraints=[];node_reqs=defaultdict(list);face_normals=defaultdict(list);face_stats=[];boundary_nodes=set(a for e in boundary for a in e)
 for sf,bid in pairs:
  bi=bone_row.get(bid)
  if bi is None:raise RuntimeError(f'unknown stored stable bone id {bid}')
  tri=skin_local[F[sf]];btri=bone[bi];n=outward[bi]
  d=((tri-btri[0])@n)*1000.0
  vio=np.maximum(0.0,CLEARANCE_MM-d)
  constraints.append({'skin_face':sf,'bone_triangle_id':bid,'triangle_vertices':list(map(int,F[sf])),
   'initial_signed_distance_mm':d.tolist(),'required_outward_mm_per_vertex':vio.tolist(),'max_required_mm':float(vio.max()),
   'all_three_already_outside_clearance':bool(np.all(d>=CLEARANCE_MM))})
  face_normals[sf].append(n)
  for local_i,node in enumerate(F[sf]):
   node_reqs[int(node)].append({'skin_face':sf,'bone_triangle_id':bid,'normal':n.tolist(),'signed_distance_mm':float(d[local_i]),'required_outward_mm':float(vio[local_i]),'fixed_boundary':int(node) in boundary_nodes})
 face_conflicts=[]
 for sf,ns in face_normals.items():
  dots=[float(ns[i]@ns[j]) for i in range(len(ns)) for j in range(i+1,len(ns))]
  if dots and min(dots)<0:face_conflicts.append({'skin_face':sf,'plane_count':len(ns),'minimum_normal_dot':min(dots)})
 node_conflicts=[];fixed_violators=[];all_node_violations=[]
 for node,reqs in node_reqs.items():
  if len(reqs)>1:
   dots=[float(np.asarray(reqs[i]['normal'])@np.asarray(reqs[j]['normal'])) for i in range(len(reqs)) for j in range(i+1,len(reqs))]
   if dots and min(dots)<0:node_conflicts.append({'node':node,'constraint_count':len(reqs),'minimum_normal_dot':min(dots)})
  vio=max(float(x['required_outward_mm']) for x in reqs)
  if vio>0:all_node_violations.append((node,vio))
  if int(node) in boundary_nodes and vio>1e-12:fixed_violators.append({'node':node,'max_required_outward_mm':vio,'constraints':[x for x in reqs if x['required_outward_mm']>0]})
 patch_pts=skin_local[parent_faces]
 angles=[];aspects=[];areas=[]
 for t in patch_pts:
  e=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
  cr=np.cross(t[1]-t[0],t[2]-t[0]);ar=float(np.linalg.norm(cr));areas.append(ar*0.5)
  if ar<=0:raise RuntimeError('degenerate Float32 source face in patch')
  aspects.append(max(e)**2/(2*np.sqrt(3)*ar))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))))
  angles.append(min(aa))
 dot_values=[float(np.dot(outward[bone_row[bid]], outward[bone_row[bid2]])) for sf in face_normals for i,bid in enumerate([b for f,b in pairs if f==sf]) for bid2 in [b for f,b in pairs if f==sf][i+1:]]
 fixed_distance_report=topology['final_boundary_checks'][0].get('minimum_node_distance')
 report.update({
  'status':'PREPARED_LIGHT_PREFLIGHT_COMPLETE',
  'inputs':{'candidate_sha256':sha(C),'audit159_sha256':sha(A),'two_ring_fan_receipt_sha256':sha(TOPO),'sub_ulp_failure_witness_sha256':sha(SUBULP),
   'bone_triangle_count':len(bone),'candidate_points':int(len(P)),'candidate_faces':int(len(F))},
  'patch':{'source_face_count':len(rows),'crossing_face_count':len(cross_faces),'stored_crossing_pair_count':len(pairs),
   'source_rows_sha256':hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest(),**topo,
   'boundary_candidate_node_rows':sorted(map(int,boundary_nodes)),'all_crossing_faces_in_patch':True,
   'all_patch_faces_have_source_face_csr':True,'immediate_parent_face_ids_preserved':True,
   'source_face_csr_id_count':int(sum(ind[f+1]-ind[f] for f in rows)),
   'nearest_pinned_boundary_distance_mm':fixed_distance_report.get('distance_mm') if fixed_distance_report else None},
  'shape_baseline':{'min_angle_degrees':float(min(angles)),'p05_angle_degrees':float(np.percentile(angles,5)),
   'max_aspect_measure':float(max(aspects)),'min_triangle_area_m2':float(min(areas)),'max_triangle_area_m2':float(max(areas)),
   'all_source_faces_nondegenerate':True},
  'all_three_vertex_halfspace_preflight':{'constraint_count':len(constraints),'constrained_unique_skin_nodes':len(node_reqs),
   'crossing_pairs_all_three_vertices_outside_clearance_initially':sum(bool(x['all_three_already_outside_clearance']) for x in constraints),
   'crossing_pairs_requiring_at_least_one_vertex_motion':sum(not bool(x['all_three_already_outside_clearance']) for x in constraints),
   'constraint_node_violations':len(all_node_violations),'max_individual_required_outward_mm':max((x[1] for x in all_node_violations),default=0.0),
   'nodes_exceeding_proposed_2mm_cap':sum(v> MOVEMENT_CAP_MM for _,v in all_node_violations),
   'fixed_boundary_nodes_touched_by_crossing_constraints':sum(int(v) in boundary_nodes for v in node_reqs),
   'fixed_boundary_nodes_with_violated_halfspace':len(fixed_violators),'fixed_boundary_violation_witnesses':fixed_violators,
   'crossing_faces_with_opposing_target_plane_normals':len(face_conflicts),'shared_skin_nodes_with_opposing_target_plane_normals':len(node_conflicts),
   'minimum_same_face_target_normal_dot':min(dot_values,default=None),
   'per_pair':constraints,'face_normal_conflicts':face_conflicts,'shared_node_normal_conflicts':node_conflicts},
  'sub_ulp_contour_root_caveat':{'distinct_roots_collapsed_after_float32':subulp['distinct_exact_roots_collapsed'],
   'exact_root_separation_m':subulp['exact_sub_ulp_witnesses'][1]['separation_from_anchor_m'],
   'endpoint_topology_strategy':'do not emit or triangulate exact root coordinates; move existing patch vertices only'},
  'next_attempt_contract':{'fit_will_start_from':str(C),'topology':'keep the existing 241-face patch and 63 fixed boundary edges; no extra contour vertices',
   'bone_constraints':'for every crossing face/stable bone triangle pair, require every vertex of the corresponding current skin face to satisfy that bone triangle outward plane; apply the same halfspace to those existing face vertices',
   'shape':'preserve all 241 face identities and CSR; positive parent orientation, area ratio >=0.10, min angle >= max(0.5 degrees, 90% baseline min angle), max movement 2.0 mm',
   'required_after_authorization':'SLSQP feasibility/fit, emitted Float32 shape screen, exact local bone/inside screen, exact neighboring-skin self screen, then all-bank screen if warranted',
   'none_of_these_postfit_checks_was_run':True}
 })
except Exception as e:
 report.update({'status':'PREPARED_LIGHT_PREFLIGHT_FAILED','failure':repr(e)})
report['elapsed_seconds']=time.monotonic()-started if 'started' in globals() else None
out=OUT/'preflight.json';out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False,default=json_fallback)+'\n')
print(json.dumps({k:report.get(k) for k in ['status','failure','inputs','patch','shape_baseline','all_three_vertex_halfspace_preflight','sub_ulp_contour_root_caveat','next_attempt_contract'] if k in report},indent=2,allow_nan=False,default=json_fallback))
print('receipt_path',out)
print('receipt_sha256',sha(out))
