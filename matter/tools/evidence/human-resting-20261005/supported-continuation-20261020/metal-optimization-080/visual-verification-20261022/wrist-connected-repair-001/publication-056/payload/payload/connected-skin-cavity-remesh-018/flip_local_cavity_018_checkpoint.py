#!/usr/bin/env python3
"""Bounded boundary-preserving bistellar flip around one failed TetGen child."""
import collections, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
t0=time.monotonic(); E=pathlib.Path(sys.argv[1]); out=E/'connected-skin-cavity-remesh-018'/'run-002-flips-002';out.mkdir(parents=True,exist_ok=True)
R=E/'connected-skin-cavity-remesh-018'/'run-002-holefix-001'
script=E/'connected-skin-cavity-remesh-018'/'prepare_local_cavity_018-holefix.py'
spec=importlib.util.spec_from_file_location('cavity018',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
basepath=E/'E'/'connected-skin-mechanical-transfer-013'/'boundary-refinement-002'/'candidate.npz'
candpath=E/'connected-skin-safe-flow-017'/'run-002'/'connected-rest-proposal.npz'
base=np.load(basepath,allow_pickle=False);cand=np.load(candpath,allow_pickle=False);sel=np.load(R/'cavity-selection.npz',allow_pickle=False)
x=cand['points_world_m'];t=cand['tetrahedra'];oldrows=sel['selected_input_tet_rows'];bf=sel['oriented_boundary_faces_global'];roles=sel['boundary_role'];attrs=base['region_attributes']
ln=m.read_tetgen_nodes(R/'cavity.1.node'); raw,la=m.read_tetgen_tets(R/'cavity.1.ele'); order=sorted(ln);rr={v:i for i,v in enumerate(order)}
lx=np.asarray([ln[i] for i in order],np.float64).astype(np.float32);lt=np.asarray([[rr[int(v)] for v in q] for q in raw],np.int64)
d=m.signed_dets(lx,lt); neg=np.flatnonzero(d<0);lt[neg,0],lt[neg,1]=lt[neg,1].copy(),lt[neg,0].copy()
# Match all exact TetGen coordinates back to existing global rows or append Steiner rows.
bnodes=set(map(int,np.unique(bf))); selected_nodes=set(map(int,np.unique(t[oldrows]))); keymap={x[g].tobytes():g for g in bnodes};allkey=collections.defaultdict(list)
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
if not np.array_equal(m.canonical_sort(m.whole_boundary_faces(tout)),m.canonical_sort(m.whole_boundary_faces(t))):raise RuntimeError('preflip global boundary was not exact')
q=m.normalized_quality(lx,lt);bad=np.flatnonzero(q<1e-5)
if len(bad)!=1:raise RuntimeError(f'expected one known local quality failure, got {len(bad)}')
badid=int(bad[0])
# Local fixed boundary expressed in TetGen local row IDs.
g2l={int(g):i for i,g in enumerate(mapped)}
if len(g2l)!=len(mapped):raise RuntimeError('nonunique global-to-local TetGen node map')
bfl=np.asarray([[g2l[int(v)] for v in f] for f in bf],np.int64)
bset={tuple(sorted(map(int,f))) for f in bfl}
bedge=set()
for f in bfl:
 for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):bedge.add(tuple(sorted((int(a),int(b)))))
# Canonical boundary of any replacement patch must be identical.
def patch_boundary(cells):
 fs=np.stack(tuple(cells[:,fc] for fc in m.FACE_CORNERS),axis=1).reshape(-1,3)
 keys=np.sort(fs,axis=1); order0=np.lexsort(keys.T[::-1]);keys=keys[order0]
 starts=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1];sizes=np.diff(np.r_[starts,len(keys)])
 return {tuple(map(int,keys[s])) for s,n in zip(starts,sizes) if n==1}
def orient_positive(cells):
 cells=np.asarray(cells,np.int64).copy();dd=m.signed_dets(lx,cells);neg0=np.flatnonzero(dd<0)
 cells[neg0,0],cells[neg0,1]=cells[neg0,1].copy(),cells[neg0,0].copy();dd=m.signed_dets(lx,cells)
 if np.any(dd<=0):return None,dd
 return cells,dd
def replace_patch(patch_ids,newcells):
 oldcells=lt[np.asarray(patch_ids,np.int64)]
 if patch_boundary(oldcells)!=patch_boundary(newcells):return None
 rem=np.ones(len(lt),bool);rem[np.asarray(patch_ids,np.int64)]=False
 result=np.vstack((lt[rem],newcells))
 # Fail closed on duplicate tet rows, nonmanifold face incidence, or nonpositive Float32 determinants.
 canon=np.sort(result,axis=1);co=np.lexsort(canon.T[::-1]);can=canon[co]
 if np.any(np.all(can[1:]==can[:-1],axis=1)):return None
 try:m.sorted_face_groups(result)
 except ValueError:return None
 if np.any(m.signed_dets(lx,result)<=0):return None
 return result
# Pinned incident-face groups and edge owner index; build once, not per trial.
face,fo,key,orderf,starts,sizes=m.sorted_face_groups(lt)
face_owners={tuple(map(int,key[s])):fo[s:s+int(n)].copy() for s,n in zip(starts,sizes)}
edgekeys,edgeowners=m.edge_owner_index(lt,len(lx))
trials=[];accepted=[]
# Two possible 2->3 flips across the bad child's internal faces.
for fc in m.FACE_CORNERS:
 tri=tuple(sorted(map(int,lt[badid,fc])))
 own=face_owners.get(tri,np.empty(0,np.int64))
 if len(own)!=2 or badid not in own:continue
 if tri in bset:continue
 other=int(own[0] if int(own[1])==badid else own[1]);tids=[badid,other]
 if la[tids[0]]!=la[tids[1]]:trials.append({'kind':'2to3','face':tri,'neighbor':other,'status':'material_mismatch'});continue
 A,B,C=tri; d0=next(int(v) for v in lt[badid] if int(v) not in tri);e0=next(int(v) for v in lt[other] if int(v) not in tri)
 side1=float(m.signed_dets(lx,np.asarray([[A,B,C,d0]],np.int64))[0]);side2=float(m.signed_dets(lx,np.asarray([[A,B,C,e0]],np.int64))[0])
 if side1*side2>=0:trials.append({'kind':'2to3','face':tri,'neighbor':other,'status':'same_side_opposites','signed_side_dets':[side1,side2]});continue
 new,det=orient_positive(np.asarray([[d0,e0,A,B],[d0,e0,B,C],[d0,e0,C,A]],np.int64))
 if new is None:trials.append({'kind':'2to3','face':tri,'neighbor':other,'status':'nonpositive_new_child','determinants':det.tolist()});continue
 if patch_boundary(lt[tids])!=patch_boundary(new):trials.append({'kind':'2to3','face':tri,'neighbor':other,'status':'patch_boundary_mismatch'});continue
 qnew=m.normalized_quality(lx,new); result=replace_patch(tids,new)
 if result is None:trials.append({'kind':'2to3','face':tri,'neighbor':other,'status':'global_local_topology_or_positive_failure','min_new_q':float(qnew.min())});continue
 rec={'kind':'2to3','face':tri,'neighbor_tet':other,'removed_tets':tids,'new_tets':new.tolist(),'new_min_q':float(qnew.min()),'new_q':qnew.tolist(),'positive_f32':True,'local_boundary_preserved':True,'status':'quality_pass' if qnew.min()>=1e-5 else 'quality_below_floor'}
 trials.append(rec)
 if rec['status']=='quality_pass':accepted.append((rec,result))
# Three possible 3->2 flips around each bad-tet edge.
for a,b in m.EDGE_PAIRS:
 u,v=sorted((int(lt[badid,a]),int(lt[badid,b])))
 if (u,v) in bedge:continue
 code=u*len(lx)+v;lo=int(np.searchsorted(edgekeys,code,'left'));hi=int(np.searchsorted(edgekeys,code,'right'));ids=np.unique(edgeowners[lo:hi])
 if len(ids)!=3 or badid not in ids:continue
 opp=[]
 for j in ids:opp.append(next(int(z) for z in lt[int(j)] if z!=u and z!=v))
 if len(set(opp))!=3:trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'opposite_vertex_not_unique'});continue
 if len({int(la[int(j)]) for j in ids})!=1:trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'material_mismatch'});continue
 A,B,C=map(int,opp)
 # A 3-to-2 flip is legal only when the edge endpoints lie on opposite sides of ABC.
 su=float(m.signed_dets(lx,np.asarray([[A,B,C,u]],np.int64))[0]);sv=float(m.signed_dets(lx,np.asarray([[A,B,C,v]],np.int64))[0])
 if su*sv>=0:trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'same_side_edge_endpoints','signed_side_dets':[su,sv]});continue
 new,det=orient_positive(np.asarray([[A,B,C,u],[A,B,C,v]],np.int64))
 if new is None:trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'nonpositive_new_child','determinants':det.tolist()});continue
 if patch_boundary(lt[ids])!=patch_boundary(new):trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'patch_boundary_mismatch'});continue
 qnew=m.normalized_quality(lx,new);result=replace_patch(ids,new)
 if result is None:trials.append({'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'status':'global_local_topology_or_positive_failure','min_new_q':float(qnew.min())});continue
 rec={'kind':'3to2','edge':[u,v],'incident_tets':ids.tolist(),'new_tets':new.tolist(),'new_min_q':float(qnew.min()),'new_q':qnew.tolist(),'positive_f32':True,'local_boundary_preserved':True,'status':'quality_pass' if qnew.min()>=1e-5 else 'quality_below_floor'}
 trials.append(rec)
 if rec['status']=='quality_pass':accepted.append((rec,result))
# Select highest-quality passing flip; if none passes, preserve the best positive topological trial for a measured failure receipt.
positive=[r for r in trials if r.get('positive_f32') and r.get('local_boundary_preserved') and 'new_min_q' in r]
status='no_legal_quality_improving_flip'
chosen=None
if accepted:
 chosen=max(accepted,key=lambda z:z[0]['new_min_q']);status='single_flip_clears_quality_floor'
elif positive:
 rec=max(positive,key=lambda r:r['new_min_q']); status='best_single_flip_still_below_quality_floor'
 # Reconstruct the same result to support the conservative ancestry ledger.
 for attempt in trials:
  if attempt is not rec:continue
  if attempt['kind']=='2to3':ids=attempt['removed_tets']
  else:ids=attempt['incident_tets']
  cells=np.asarray(attempt['new_tets'],np.int64);res=replace_patch(ids,cells)
  chosen=(rec,res);break
outreport={'schema':'numi.local-internal-bistellar-flip-018.v1','status':status,'candidate_unadmitted':True,
 'baseline_sha256':m.sha256(basepath),'input_surface_candidate_sha256':m.sha256(candpath),
 'prior_tetgen_local_node_sha256':m.sha256(R/'cavity.1.node'),'prior_tetgen_local_element_sha256':m.sha256(R/'cavity.1.ele'),
 'bad_local_tet_row':badid,'bad_global_tet_row':int(keep.sum()+badid),'bad_tet_local_nodes':lt[badid].tolist(),
 'bad_tet_global_nodes':child[badid].tolist(),'baseline_min_local_q':float(q.min()),'baseline_below_floor_local_count':int((q<1e-5).sum()),
 'trial_count':len(trials),'trials':trials,'elapsed_s':None,'material_mass_ledger':'not asserted; no independently validated physical reference inverse'}
if chosen is None:
 outreport['elapsed_s']=time.monotonic()-t0
 (out/'flip-report.json').write_text(json.dumps(outreport,indent=2,sort_keys=True)+'\n');print(json.dumps(outreport,indent=2));sys.exit(0)
rec,lt2=chosen
# Validate all local cells, exact local cavity boundary, and global boundary after the flip.
q2=m.normalized_quality(lx,lt2);d2=m.signed_dets(lx,lt2)
if np.any(d2<=0):raise RuntimeError('accepted flip produced nonpositive F32 local child')
if patch_boundary(lt2)!=patch_boundary(lt):raise RuntimeError('accepted flip changed local cavity boundary')
child2=mapped[lt2];tout2=np.vstack((t[keep],child2));aout2=np.r_[attrs[keep],la]
if not np.array_equal(m.canonical_sort(m.whole_boundary_faces(tout2)),m.canonical_sort(m.whole_boundary_faces(t))):raise RuntimeError('accepted flip changed global boundary')
if np.any(m.signed_dets(xx,tout2)<=0):raise RuntimeError('accepted flip made global nonpositive cell')
outreport['global_min_q_after_flip']=float(m.normalized_quality(xx,tout2).min());outreport['global_below_floor_after_flip']=int((m.normalized_quality(xx,tout2)<1e-5).sum());outreport['global_nonpositive_after_flip']=int((m.signed_dets(xx,tout2)<=0).sum())
# Checkpoint the actual candidate geometry and quality BEFORE any ancestry work.
# This is a local construction diagnostic; the complete input-parent ledger remains pending.
local_old_ids=np.asarray(rec.get('removed_tets',rec.get('incident_tets')),np.int64)
if local_old_ids.ndim!=1 or not len(local_old_ids):raise RuntimeError('flip did not identify removed local cells')
local_new=np.asarray(rec['new_tets'],np.int64)
np.savez_compressed(out/'local-flip-geometry.npz',
    local_points_world_m=lx,local_tetrahedra=lt2,local_region_attributes=la,
    selected_original_input_tet_rows=oldrows,tetgen_local_nodes_global=mapped,
    flipped_old_local_tet_rows=local_old_ids,flipped_new_local_tets=local_new,
    points_world_m=xx.astype(np.float32),tetrahedra=tout2,region_attributes=aout2,
    physical_outer_faces=cand['physical_outer_faces'],
    physical_O_parent_face_rows=cand['physical_O_parent_face_rows'],
    protected_node_ids=cand['protected_node_ids'],
    replaced_input_tet_rows=oldrows)
outreport.update({
 'geometry_checkpoint':'local-flip-geometry.npz',
 'geometry_checkpoint_sha256':m.sha256(out/'local-flip-geometry.npz'),
 'flip_kind':rec['kind'],'flip_record':rec,
 'local_tet_count_after':int(len(lt2)),
 'local_min_q_after':float(q2.min()),'local_p01_q_after':float(np.quantile(q2,0.01)),
 'local_below_floor_after':int((q2<1e-5).sum()),
 'local_nonpositive_f32_after':int((d2<=0).sum()),
 'global_tet_count_after':int(len(tout2)),
 'global_min_q_after':float(m.normalized_quality(xx,tout2).min()),
 'global_p01_q_after':float(np.quantile(m.normalized_quality(xx,tout2),0.01)),
 'global_below_floor_after':int((m.normalized_quality(xx,tout2)<1e-5).sum()),
 'global_nonpositive_f32_after':int((m.signed_dets(xx,tout2)<=0).sum()),
 'global_boundary_exact':True,'physical_outer_faces_exact':True,
 'complete_original_parent_ancestry':'pending; TetGen output has no full parent map',
 'full_material_mass_ledger':'pending; this local topology delta is not mass qualification'})
(out/'flip-report.json').write_text(json.dumps(outreport,indent=2,sort_keys=True)+'\n')
print(json.dumps({'stage':'geometry_checkpoint','status':status,**{k:outreport[k] for k in ('geometry_checkpoint_sha256','flip_kind','local_min_q_after','local_below_floor_after','global_min_q_after','global_below_floor_after','global_nonpositive_f32_after')}},indent=2))
# Restrict overlap to the tiny removed/added bistellar patch. This only proves
# local retriangulation coverage and is not the missing original-parent ledger.
try:
 owner=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse'/'build_014-replay.py'
 if m.sha256(owner)!=m.OWNER_SHA256:raise RuntimeError('pinned geometric overlap owner changed')
 intersection=m.load_intersection_owner(owner,m.OWNER_SHA256)
 local_map,local_raw=m.local_overlap(lx,lt[local_old_ids],np.arange(len(local_old_ids),dtype=np.int64),lx,local_new,intersection)
 outreport['local_flip_patch_overlap']={**local_raw,
   'old_patch_tets':int(len(local_old_ids)),'new_patch_tets':int(len(local_new)),
   'parent_column_sum_error':float(np.max(np.abs(np.bincount(local_map['parent_local_rows'],weights=local_map['parent_fraction'],minlength=len(local_old_ids))-1))),
   'child_row_coverage_error':float(np.max(np.abs(local_map['child_coverage']-1))),
   'meaning':'local TetGen-patch retriangulation only; not original 013/014 parent ancestry or physical mass'}
 np.savez_compressed(out/'local-flip-patch-overlap.npz',
   parent_offsets=local_map['parent_offsets'],parent_local_rows=local_map['parent_local_rows'],
   child_fraction=local_map['child_fraction'],parent_fraction=local_map['parent_fraction'],
   intersection_volume_m3=local_map['intersection_volume'],
   child_coverage=local_map['child_coverage'],old_coverage=local_map['old_coverage'])
 outreport['local_flip_overlap_sha256']=m.sha256(out/'local-flip-patch-overlap.npz')
 outreport['local_flip_overlap_status']='measured; scope is flip patch only'
except Exception as exc:
 outreport['local_flip_overlap_status']='pending_or_failed'
 outreport['local_flip_overlap_error']=f'{type(exc).__name__}: {exc}'
outreport['elapsed_s']=time.monotonic()-t0
(out/'flip-report.json').write_text(json.dumps(outreport,indent=2,sort_keys=True)+'\n')
print(json.dumps({'stage':'final_report','status':outreport['status'],'local_flip_overlap_status':outreport['local_flip_overlap_status']},indent=2))
