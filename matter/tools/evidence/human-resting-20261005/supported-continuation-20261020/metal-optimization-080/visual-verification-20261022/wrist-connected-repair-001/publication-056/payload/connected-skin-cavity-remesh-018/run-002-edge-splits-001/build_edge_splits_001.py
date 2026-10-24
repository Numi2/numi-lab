#!/usr/bin/env python3
"""Test conforming exact-midpoint split candidates around one 018 quality witness."""
import collections,hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
T0=time.monotonic()
E=pathlib.Path(sys.argv[1]); OUT=E/'connected-skin-cavity-remesh-018'/'run-002-edge-splits-001';OUT.mkdir(parents=True,exist_ok=True)
C=E/'connected-skin-cavity-remesh-018';R=C/'run-002-holefix-001';W=R/'quality-witness.json'
basepath=E/'E'/'connected-skin-mechanical-transfer-013'/'boundary-refinement-002'/'candidate.npz'
candpath=E/'connected-skin-safe-flow-017'/'run-002'/'connected-rest-proposal.npz'
selpath=R/'cavity-selection.npz'
base=np.load(basepath,allow_pickle=False);cand=np.load(candpath,allow_pickle=False);sel=np.load(selpath,allow_pickle=False)
mspec=importlib.util.spec_from_file_location('cavity018',C/'prepare_local_cavity_018-holefix.py');m=importlib.util.module_from_spec(mspec);mspec.loader.exec_module(m)
esrc=E/'global-inner-p1-edge-support-403'/'refine_internal_o_edges_403.py';espec=importlib.util.spec_from_file_location('edge403',esrc);edge=importlib.util.module_from_spec(espec);espec.loader.exec_module(edge)
# Full input and selected TetGen cavity must share exact topology and material rows.
x=cand['points_world_m'];t=cand['tetrahedra'];attrs=base['region_attributes'];oldrows=sel['selected_input_tet_rows'];bf=sel['oriented_boundary_faces_global'];la=None
if not np.array_equal(t,base['tetrahedra']):raise RuntimeError('surface candidate tet rows differ from pinned baseline')
ln=m.read_tetgen_nodes(R/'cavity.1.node');raw,la=m.read_tetgen_tets(R/'cavity.1.ele');node_order=sorted(ln);node_row={v:i for i,v in enumerate(node_order)}
lx=np.asarray([ln[i] for i in node_order],np.float64).astype(np.float32);lt=np.asarray([[node_row[int(v)] for v in q] for q in raw],np.int64)
det=m.signed_dets(lx,lt);neg=np.flatnonzero(det<0);lt[neg,0],lt[neg,1]=lt[neg,1].copy(),lt[neg,0].copy()
# Rebuild exact pre-flip holefix global volume mesh.
bnodes=set(map(int,np.unique(bf)));selected_nodes=set(map(int,np.unique(t[oldrows])));keymap={x[g].tobytes():g for g in bnodes};allkey=collections.defaultdict(list)
for g,p in enumerate(x):allkey[p.tobytes()].append(g)
mapped=[];extras=[]
for p in lx:
 k=p.tobytes()
 if k in keymap:mapped.append(keymap[k])
 elif k in allkey:
  q=allkey[k]
  if not any(g in selected_nodes for g in q) or len(q)!=1:raise RuntimeError(f'ambiguous existing node alias {q[:8]}')
  mapped.append(q[0])
 else:mapped.append(len(x)+len(extras));extras.append(p.copy())
mapped=np.asarray(mapped,np.int64);xx=np.vstack((x,np.asarray(extras,np.float32).reshape(-1,3)))
keep=np.ones(len(t),bool);keep[oldrows]=False
child=mapped[lt];tout=np.vstack((t[keep],child));aout=np.r_[attrs[keep],la]
if np.any(m.signed_dets(xx,tout)<=0):raise RuntimeError('input merged mesh has nonpositive F32 cells')
if not np.array_equal(m.canonical_sort(m.whole_boundary_faces(tout)),m.canonical_sort(m.whole_boundary_faces(t))):raise RuntimeError('holefix merged mesh boundary mismatch before split')
# map the exact original cavity boundary rows into the local TetGen node table
local_of_global={int(g):i for i,g in enumerate(mapped)}
bfl=np.asarray([[local_of_global[int(v)] for v in f] for f in bf],np.int64)
# The witness defines one bad local TetGen child. Test all six original edges by descending length.
witness=json.loads(W.read_text());bad_local=int(witness['child_local_tet_row']);badnodes=lt[bad_local]
edge_specs=[]
for i,j in m.EDGE_PAIRS:
 a,b=sorted((int(badnodes[i]),int(badnodes[j])));ga,gb=sorted((int(mapped[a]),int(mapped[b])))
 edge_specs.append({'local':[a,b],'global':[ga,gb],'length_m':float(np.linalg.norm(lx[a].astype(np.float64)-lx[b].astype(np.float64)))})
edge_specs.sort(key=lambda z:(-z['length_m'],z['global']))

# Generic oriented-triangle splitting, keeping ancestry aligned and propagating a
# parent-face barycentric embedding when supplied.
def split_triangles(F, edge_mid, row_arrays=(), bary=None):
 F=np.asarray(F,np.int64).copy(); arrays=[np.asarray(a).copy() for a in row_arrays]
 B=None if bary is None else np.asarray(bary,np.float64).copy(); origin=np.arange(len(F),dtype=np.int64)
 for (u,v),mid in sorted(edge_mid.items()):
  outF=[];outA=[[] for _ in arrays];outB=[] if B is not None else None;outOrigin=[]
  for fi,f in enumerate(F):
   hit=None
   for k in range(3):
    a=int(f[k]);b=int(f[(k+1)%3]);c=int(f[(k+2)%3])
    if {a,b}=={u,v}:hit=(k,a,b,c);break
   if hit is None:
    outF.append(f)
    for z,arr in enumerate(arrays):outA[z].append(arr[fi])
    if B is not None:outB.append(B[fi])
    outOrigin.append(origin[fi]);continue
   k,a,b,c=hit; f1=np.asarray([a,mid,c],np.int64);f2=np.asarray([mid,b,c],np.int64)
   outF.extend((f1,f2))
   for z,arr in enumerate(arrays):outA[z].extend((arr[fi],arr[fi]))
   if B is not None:
    bm=(B[fi,k]+B[fi,(k+1)%3])*0.5
    # rotate bary rows into the cyclic (a,b,c) ordering selected above
    bp=np.asarray([B[fi,k],B[fi,(k+1)%3],B[fi,(k+2)%3]])
    b1=np.asarray([bp[0],bm,bp[2]]);b2=np.asarray([bm,bp[1],bp[2]])
    outB.extend((b1,b2))
   outOrigin.extend((origin[fi],origin[fi]))
  F=np.asarray(outF,np.int64);arrays=[np.asarray(a) for a in outA];origin=np.asarray(outOrigin,np.int64)
  if B is not None:B=np.asarray(outB,np.float64)
 return F,arrays,B,origin

def face_area(P,F):
 q=np.asarray(P[F],np.float64);return 0.5*np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
def canonical_oriented_set(F):return {tuple(q) for q in edge.canonical_oriented_faces(F)}
def eval_one(specs):
 edges=[tuple(z['global']) for z in specs]
 X2,T2,par,ends=edge.split_edges_globally(np.asarray(xx,np.float32),tout,edges)
 if not np.array_equal(X2[:len(xx)],xx):raise RuntimeError('old node prefix changed')
 emap={tuple(map(int,e)):int(len(xx)+i) for i,e in enumerate(ends)}
 if set(emap)!=set(edges):raise RuntimeError('midpoint edge endpoint map mismatch')
 det2=m.signed_dets(X2,T2);q2=m.normalized_quality(X2,T2)
 # Build expected shell by applying the same midpoint template to the original
 # emitted global boundary, and compare its complete oriented facet set.
 boundary0=edge.oriented_boundary_faces(tout)
 expected,_,_,_=split_triangles(boundary0,emap)
 boundary2=edge.oriented_boundary_faces(T2);bcmp=edge.compare_oriented_boundary(expected,boundary2)
 # Update the registered physical O and shell tables with inherited typed face ancestry.
 O=cand['physical_outer_faces']; shell=base['physical_shell_faces']; closure=base['construction_closure_faces']
 parent_arrays=[base['physical_O_parent_face_rows'],base['physical_O_parent_detailed_source_face_ids'],base['physical_O_parent_surface_ids'],base['physical_O_parent_source271_face_rows']]
 O2,oa,B2,origin=split_triangles(O,emap,parent_arrays,base['physical_O_parent_barycentric'])
 sh2,sha,_,shorig=split_triangles(shell,emap,[base['physical_shell_parent_face_rows']])
 cl2,_,_,_=split_triangles(closure,emap)
 oldarea=face_area(xx,O);newarea=face_area(X2,O2);sumby=np.bincount(origin,weights=newarea,minlength=len(O));arel=float(np.max(np.abs(sumby-oldarea)/np.maximum(oldarea,1e-30)))
 barysum=np.max(np.abs(B2.sum(axis=2)-1)); barymin=float(B2.min())
 surface_positive=bool(np.all(newarea>0))
 # Exact boundary-source parity against the updated registered shell table.
 shellcmp=edge.compare_oriented_boundary(sh2,boundary2)
 edge_rows=[]
 off=np.linalg.norm(X2[np.asarray([emap[e] for e in edges])].astype(np.float64)-np.asarray([0.5*(xx[a].astype(np.float64)+xx[b].astype(np.float64)) for a,b in edges]),axis=1)
 metrics={'split_edges_global':edges,'split_lengths_m':[z['length_m'] for z in specs],
  'output_nodes':len(X2),'output_tets':len(T2),'new_tets':len(T2)-len(tout),
  'incident_tets_split':int(np.count_nonzero(np.bincount(par,minlength=len(tout))>1)),
  'positive_f32':bool(np.isfinite(det2).all() and np.all(det2>0)),'nonpositive_f32_count':int(np.count_nonzero(det2<=0)),
  'min_normalized_quality_f32':float(q2.min()),'below_1e_5_count':int(np.count_nonzero(q2<1e-5)),
  'p01_normalized_quality_f32':float(np.quantile(q2,.01)),
  'global_boundary_parent_split_exact_oriented':bcmp,
  'registered_shell_boundary_exact_oriented':shellcmp,
  'physical_outer_face_count_before_after':[len(O),len(O2)],
  'physical_outer_parent_area_relative_error_max':arel,'physical_outer_faces_positive_area':surface_positive,
  'new_outer_barycentric_min':barymin,'new_outer_barycentric_sum_error_max':float(barysum),
  'midpoint_max_off_edge_distance_m':float(off.max()) if len(off) else 0.,
  'global_volume_before_m3':float((np.abs(m.signed_dets(xx,tout))/6).sum()),
  'global_volume_after_m3':float((np.abs(det2)/6).sum()),
  'global_volume_delta_m3':float((np.abs(det2)/6).sum()-(np.abs(m.signed_dets(xx,tout))/6).sum()),
  'old_coordinate_prefix_exact':True,'new_node_ancestry':'endpoint-pair/alpha=0.5 emitted; full source and original-parent CSR pending'}
 candidate={'points_world_m':X2,'tetrahedra':T2,'region_attributes':aout[par],
   'parent_merged_global_tet_rows':par,'physical_outer_faces':O2,
   'physical_O_parent_face_rows':oa[0],
   'physical_O_parent_detailed_source_face_ids':oa[1],
   'physical_O_parent_surface_ids':oa[2],
   'physical_O_parent_source271_face_rows':oa[3],
   'physical_O_parent_barycentric':B2,
   'physical_shell_faces':sh2,'physical_shell_parent_face_rows':sha[0],
   'construction_closure_faces':cl2,
   'physical_outer_node_ids':np.unique(np.r_[base['physical_outer_node_ids'],[emap[e] for e in edges if any(set(e).issubset(set(f)) for f in O)]]).astype(np.int64),
   'mechanical_midpoint_edge_endpoints':np.asarray(edges,np.int64),
   'mechanical_midpoint_global_node_rows':np.asarray([emap[e] for e in edges],np.int64),
   'replaced_018_input_tet_rows':oldrows}
 return metrics,candidate

results=[];chosen=None;chosen_specs=None;chosen_data=None
for z in edge_specs:
 print(json.dumps({'stage':'single_edge_try','edge':z['global'],'length_m':z['length_m']}),flush=True)
 met,data=eval_one([z]); met['split_edges_local']=[z['local']];met['split_edges_global']= [z['global']]
 met['candidate_meets_quality_and_boundary'] = bool(met['positive_f32'] and met['below_1e_5_count']==0 and met['global_boundary_parent_split_exact_oriented']['exact_unoriented_set'] and met['global_boundary_parent_split_exact_oriented']['oriented_mismatch_count']==0 and met['registered_shell_boundary_exact_oriented']['exact_unoriented_set'] and met['registered_shell_boundary_exact_oriented']['oriented_mismatch_count']==0 and met['physical_outer_faces_positive_area'] and met['physical_outer_parent_area_relative_error_max']<1e-6 and met['new_outer_barycentric_min']>=-1e-12 and met['new_outer_barycentric_sum_error_max']<1e-12)
 results.append(met)
 if met['candidate_meets_quality_and_boundary']:
  chosen=(met,data);chosen_specs=[z];break
# If all single-edge bisections fail, try bounded pairs of the four longest edges.
if chosen is None:
 from itertools import combinations
 for combo in combinations(edge_specs[:4],2):
  print(json.dumps({'stage':'edge_pair_try','edges':[z['global'] for z in combo]}),flush=True)
  met,data=eval_one(list(combo));met['split_edges_local']=[z['local'] for z in combo];met['split_edges_global']=[z['global'] for z in combo]
  met['candidate_meets_quality_and_boundary']=bool(met['positive_f32'] and met['below_1e_5_count']==0 and met['global_boundary_parent_split_exact_oriented']['exact_unoriented_set'] and met['global_boundary_parent_split_exact_oriented']['oriented_mismatch_count']==0 and met['registered_shell_boundary_exact_oriented']['exact_unoriented_set'] and met['registered_shell_boundary_exact_oriented']['oriented_mismatch_count']==0 and met['physical_outer_faces_positive_area'] and met['physical_outer_parent_area_relative_error_max']<1e-6 and met['new_outer_barycentric_min']>=-1e-12 and met['new_outer_barycentric_sum_error_max']<1e-12)
  results.append(met)
  if met['candidate_meets_quality_and_boundary']:
   chosen=(met,data);chosen_specs=list(combo);break
# Keep the best finite geometric trial even if quality did not clear, but never call it passing.
if chosen is None and results:
 candidates=[]
 # Recompute only the best-ranked trial and checkpoint it; geometry metrics remain in report.
 best=max(results,key=lambda z:(z['positive_f32'],z['min_normalized_quality_f32']))
 keys={tuple(e) for e in best['split_edges_global']}; specs=[z for z in edge_specs if tuple(z['global']) in keys]
 _,data=eval_one(specs);chosen=(best,data);chosen_specs=specs
if chosen:
 met,data=chosen
 np.savez_compressed(OUT/'edge-split-geometry.npz',**data)
 geomsha=hashlib.sha256((OUT/'edge-split-geometry.npz').read_bytes()).hexdigest()
else:geomsha=None
report={'schema':'numi.holefix018-conforming-edge-split-quality-trial.v1','status':'quality_and_boundary_pass' if chosen and chosen[0]['candidate_meets_quality_and_boundary'] else 'no_passing_bounded_edge_split_found','candidate_unadmitted':True,'input_baseline_sha256':m.sha256(basepath),'input_surface_candidate_sha256':m.sha256(candpath),'input_holefix_nodes_sha256':m.sha256(R/'cavity.1.node'),'input_holefix_elements_sha256':m.sha256(R/'cavity.1.ele'),'splitter_path':str(esrc),'splitter_sha256':hashlib.sha256(esrc.read_bytes()).hexdigest(),'source_domain':'original positive holefix merged mesh, before rejected 018 bistellar flip','bad_local_tet_row':bad_local,'bad_local_tet_nodes':badnodes.tolist(),'six_original_edges':edge_specs,'trials':results,'chosen_edges':[] if not chosen_specs else [z['global'] for z in chosen_specs],'candidate_geometry_path':None if geomsha is None else str(OUT/'edge-split-geometry.npz'),'candidate_geometry_sha256':geomsha,'full_original_parent_ancestry':'pending; no whole TetGen ancestry map','full_material_mass_ledger':'pending; no mass qualification','elapsed_s':time.monotonic()-T0}
(OUT/'edge-split-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'stage':'complete','status':report['status'],'trials':len(results),'chosen_edges':report['chosen_edges'],'geometry_sha256':geomsha,'elapsed_s':report['elapsed_s']},indent=2),flush=True)
