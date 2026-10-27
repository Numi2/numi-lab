from __future__ import annotations
import hashlib, importlib.util, json, sys, traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent
PREFLIGHT=D/'preflight.json'
BATCH=D/'candidate-private.npz'
BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-008-witness-direction-backtrack-prepared-001/candidate-private-iteration-01.npz')
BASE_REPORT=BASE.parent/'report.json'
PARENT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
BANK_HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT.parent/'audit_skin_bone_surfaces_159.py'
REFRESH=ROOT/'surface-clearance-546/component-sat-selector-638/edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=D/'exact-screen-corrected-001'
EXPECTED_PREFLIGHT_SHA='6169a812653299153a6a3ad205e6305a263b4d614f414fe1a8eff4581c0a6dff'
EXPECTED_BATCH_SHA='097452436f69e4bbe2c12ba51d78fa8415076c8d786519b4baa701863e8c0e9b'
EXPECTED_BASE_SHA='8931f3f240f595f2df17e91cb4f4c82688721e778e81bcbc4acc7f6264d61d70'
EXPECTED_PARENT_SHA='beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f'cannot load {path}')
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def frac(x):return Fraction.from_float(float(x))
def csr_rows(ptr,vals,n):
 p=np.asarray(ptr);v=np.asarray(vals)
 if p.shape!=(n+1,) or int(p[0])!=0 or int(p[-1])!=len(v) or np.any(p[1:]<p[:-1]):raise ValueError('invalid ancestry CSR')
 return [tuple(map(int,v[int(p[i]):int(p[i+1])])) for i in range(n)]
def mesh_records(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)]
 out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids)
  n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise RuntimeError(f'exact Float32-emitted skin face {fid} is degenerate')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out
def canon_bone(pairset,ancestry):
 return {(int(parent),int(bone)) for f,bone in pairset for parent in ancestry[int(f)]}
def canon_self(pairset,ancestry):
 out=set()
 for f,g in pairset:
  for a in ancestry[int(f)]:
   for b in ancestry[int(g)]:out.add(tuple(sorted((int(a),int(b)))))
 return out
def pair_rows(rec):
 return sorted(set((int(a),int(b)) for a,b in (rec.get('triangle_pairs') or [])))
def classify_surface(points,faces,rows,complement_rows,vertex_ids,exact):
 patch_faces=faces[np.asarray(rows,np.int64)]
 patch_records=mesh_records(points,patch_faces,rows,vertex_ids,exact)
 internal=exact._audit_pair(patch_records,patch_records,same_surface=True)
 tri=np.asarray(points,np.float64)[patch_faces];lo=tri.min(axis=(0,1));hi=tri.max(axis=(0,1))
 patch_set=set(map(int,rows));edge_use={}
 for f in patch_set:
  a,b,c=map(int,faces[f])
  for u,v in ((a,b),(b,c),(c,a)):
   e=tuple(sorted((u,v)));edge_use[e]=edge_use.get(e,0)+1
 patch_boundary={e for e,n in edge_use.items() if n==1}
 boundary_adjacent=set()
 for f in map(int,complement_rows):
  a,b,c=map(int,faces[f])
  if any(tuple(sorted((u,v))) in patch_boundary for u,v in ((a,b),(b,c),(c,a))):boundary_adjacent.add(f)
 if len(complement_rows):
  ct=np.asarray(points,np.float64)[faces[np.asarray(complement_rows,np.int64)]]
  cl=ct.min(axis=1);ch=ct.max(axis=1)
  # Per-face bounds: reduce each triangle's 3 vertices, then test all 3 axes per face.
  active=np.flatnonzero(np.all(ch>=lo,axis=1)&np.all(cl<=hi,axis=1))
  comp_rows=np.asarray(complement_rows,np.int64)[active]
 else:comp_rows=np.asarray([],np.int64)
 active_set=set(map(int,comp_rows))
 if boundary_adjacent and (not active_set or not boundary_adjacent.issubset(active_set)):
  raise RuntimeError(f'AABB filter omitted shared-boundary neighbor(s): total={len(boundary_adjacent)} omitted={sorted(boundary_adjacent-active_set)[:12]}')
 comp_records=mesh_records(points,faces[comp_rows],comp_rows,vertex_ids,exact)
 external=exact._audit_pair(patch_records,comp_records,same_surface=True)
 return patch_records,{'patch_internal_pair_count':int(internal['count']),'patch_internal_pairs':pair_rows(internal),
  'aabb_active_complement_face_count':int(len(comp_rows)),'shared_boundary_adjacent_face_count':len(boundary_adjacent),
  'shared_boundary_adjacent_faces_in_aabb_active_count':len(boundary_adjacent & active_set),
  'aabb_bounds_axis_semantics':'min/max over triangle vertices axis=1; overlap predicates all tested per face over 3 axes',
  'patch_vs_complement_pair_count':int(external['count']),
  'patch_vs_complement_pairs':pair_rows(external),'raw_pair_union':sorted(set(pair_rows(internal))|set(pair_rows(external)))}
def all_locations(points,node_ids,prepared,exact):
 loc=Counter();inside=[]
 for n in map(int,node_ids):
  v=exact.point_location_prepared(tuple(frac(x) for x in points[n]),prepared)['location'];loc[v]+=1
  if v in ('inside','boundary'):inside.append(n)
 return {'counts':dict(loc),'inside_or_boundary_node_ids':sorted(inside)}
def screen_variant(points_world,faces,rows,bone_records,prepared_bone,exact,source_rows,parent_rows,baseline_source,baseline_parent,
                   base_pairs,base_inside,base_self,preflight,operations,variant_index):
 nnode=len(points_world);vertex_ids=np.arange(nnode,dtype=np.int64)
 # Work in the same body-65 frame and with the same exact owner used by the prior pinned screen.
 local=(np.asarray(points_world,np.float64)-preflight['_pose_pos'][65])@preflight['_pose_M'][65].T-preflight['_pose_off'][65]
 allfaces=np.asarray(faces,np.int64);rows=np.asarray(rows,np.int64)
 comp_rows=np.setdiff1d(np.arange(len(allfaces),dtype=np.int64),rows,assume_unique=False)
 patch_records,self_report=classify_surface(local,allfaces,rows,comp_rows,vertex_ids,exact)
 bgate=exact._audit_pair(patch_records,bone_records,same_surface=False)
 hitset=set((int(f),int(b)) for f,b in bgate['triangle_pairs'])
 baseline_hitset=set((int(f),int(b)) for f,b in base_pairs)
 src_ids=canon_bone(hitset,source_rows);src_added=sorted(src_ids-canon_bone(baseline_hitset,baseline_source));src_removed=sorted(canon_bone(baseline_hitset,baseline_source)-src_ids)
 par_ids=canon_bone(hitset,parent_rows);par_added=sorted(par_ids-canon_bone(baseline_hitset,baseline_parent));par_removed=sorted(canon_bone(baseline_hitset,baseline_parent)-par_ids)
 self_pairs=set(tuple(sorted(p)) for p in self_report['raw_pair_union'])
 base_self_pairs=set(tuple(sorted(p)) for p in base_self['raw_pair_union'])
 src_self=canon_self(self_pairs,source_rows);base_src_self=canon_self(base_self_pairs,baseline_source)
 par_self=canon_self(self_pairs,parent_rows);base_par_self=canon_self(base_self_pairs,baseline_parent)
 loc=all_locations(local,np.unique(allfaces[rows]),prepared_bone,exact)
 base_inside=set(map(int,base_inside));cur_inside=set(loc['inside_or_boundary_node_ids'])
 new_inside=sorted(cur_inside-base_inside);removed_inside=sorted(base_inside-cur_inside)
 report={'variant_index':variant_index,'selected_operation_face_rows':[list(map(int,op['face_rows'])) for op in operations],
  'exact_local_65_99_child_pair_count':int(bgate['count']),'exact_local_65_99_child_pairs':sorted(hitset),
  'baseline_child_pair_count':len(baseline_hitset),'child_pair_added_count':len(hitset-baseline_hitset),
  'child_pair_removed_count':len(baseline_hitset-hitset),'child_pair_added':sorted(hitset-baseline_hitset),
  'child_pair_removed':sorted(baseline_hitset-hitset),
  'canonical_source_bone_identity':{'before_count':len(canon_bone(baseline_hitset,baseline_source)),'after_count':len(src_ids),
    'added_count':len(src_added),'removed_count':len(src_removed),'net_delta':len(src_ids)-len(canon_bone(baseline_hitset,baseline_source)),
    'added':src_added,'removed':src_removed},
  'canonical_immediate_parent_bone_identity':{'before_count':len(canon_bone(baseline_hitset,baseline_parent)),'after_count':len(par_ids),
    'added_count':len(par_added),'removed_count':len(par_removed),'net_delta':len(par_ids)-len(canon_bone(baseline_hitset,baseline_parent)),
    'added':par_added,'removed':par_removed},
  'local_self':self_report,
  'canonical_source_self_identities':{'before_count':len(base_src_self),'after_count':len(src_self),
    'added_count':len(src_self-base_src_self),'removed_count':len(base_src_self-src_self),
    'added':[list(x) for x in sorted(src_self-base_src_self)],'removed':[list(x) for x in sorted(base_src_self-src_self)]},
  'canonical_immediate_parent_self_identities':{'before_count':len(base_par_self),'after_count':len(par_self),
    'added_count':len(par_self-base_par_self),'removed_count':len(base_par_self-par_self),
    'added':[list(x) for x in sorted(par_self-base_par_self)],'removed':[list(x) for x in sorted(base_par_self-par_self)]},
  'point_location_all_patch_nodes':loc,'inside_boundary_comparison':{'baseline_count':len(base_inside),'current_count':len(cur_inside),
    'new_ids':new_inside,'removed_ids':removed_inside,'no_new_inside_or_boundary':not new_inside},
  'canonical_parent_identity_improved_without_additions':bool(not src_added and len(src_removed)>0 and not par_added),
  'local_self_new_witnesses':sorted(self_pairs-base_self_pairs),
  'local_self_pass':len(self_pairs)==0}
 return report,local,patch_records

def run():
 report={'schema':'numi.human.bank65_99.topology-flip-batch-local-exact.v1','status':'STARTED',
  'geometry_or_native_admission':False,'candidate_modified':False,
  'scope':'Exact local 65/99 bone screen and patch-internal / AABB-selected complement self screen for the preflighted private 10-flip topology batch, with canonical source/immediate-parent identities and all-patch-node point locations. No full all-bank audit or physical admission.'}
 try:
  if sha(PREFLIGHT)!=EXPECTED_PREFLIGHT_SHA or sha(BATCH)!=EXPECTED_BATCH_SHA or sha(BASE)!=EXPECTED_BASE_SHA or sha(PARENT)!=EXPECTED_PARENT_SHA:raise RuntimeError('preflight/candidate/parent input pin mismatch')
  pre=json.loads(PREFLIGHT.read_text());base_doc=json.loads(BASE_REPORT.read_text())
  if pre.get('status')!='SHAPE_AND_LINEAGE_PREFLIGHT_PASSED_PENDING_EXACT_LOCAL_SCREEN':raise RuntimeError('preflight did not pass')
  if pre.get('selected_operation_count')!=10 or pre.get('selected_multi_parent_operation_count')!=2:raise RuntimeError('selected batch count changed')
  rows=np.asarray(base_doc['topology']['patch_face_rows'],np.int64)
  ops=pre['selected_operations']
  with np.load(BATCH,allow_pickle=False) as z:candidate={k:z[k].copy() for k in z.files}
  with np.load(BASE,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
  with np.load(PARENT,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
  # Pin detailed mask helpers and target bone bank source.
  audit_tool=loadmod('b65_99_topology_audit_owner',AUDIT_TOOL);bank_helper=loadmod('b65_99_topology_bank_helper',BANK_HELPER)
  refresh=loadmod('b65_99_topology_refresh_generation002',REFRESH)
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=bank_helper.read_bank_inputs(BANKDIR,E)
  bmask=(ids[:,0]==65)&(ids[:,1]==99);bone=np.asarray(T[bmask],np.float64);bone_ids=np.asarray(ids[bmask,3],np.int64)
  if len(bone)!=342:raise RuntimeError(f'65/99 target inventory changed: {len(bone)}')
  exact=audit_tool.exact_module();bone_records=audit_tool.records(bone,bone_ids,exact);prepared_bone=exact.prepare_point_location(bone_records)
  rows=sorted(map(int,base_doc['topology']['patch_face_rows']))
  base_pairs=sorted(set((int(f),int(b)) for f,b in base_doc['rounds'][0]['exact_local_bone_pairs']))
  baseline_source=csr_rows(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],len(base['source_O_faces']))
  baseline_parent=csr_rows(base['refinement_face_parent_ancestry_indptr'],base['refinement_face_parent_ancestry_face_rows'],len(base['source_O_faces']))
  parent_rows=csr_rows(candidate['refinement_face_parent_ancestry_indptr'],candidate['refinement_face_parent_ancestry_face_rows'],len(candidate['source_O_faces']))
  source_rows=csr_rows(candidate['face_source_ancestry_indptr'],candidate['face_source_ancestry_face_rows'],len(candidate['source_O_faces']))
  # Baseline local self comparison uses the exact same 248-face scope and current F32 coordinates.
  baseline_other=np.setdiff1d(np.arange(len(base['source_O_faces']),dtype=np.int64),np.asarray(rows,np.int64))
  base_local=(np.asarray(base['points_world_m'],np.float64)-pos[65])@M[65].T-off[65]
  base_records,base_self=classify_surface(base_local,np.asarray(base['source_O_faces'],np.int64),np.asarray(rows,np.int64),baseline_other,
    np.arange(len(base_local),dtype=np.int64),exact)
  base_loc=all_locations(base_local,np.unique(base['source_O_faces'][np.asarray(rows,np.int64)]),prepared_bone,exact)
  baselineself={**base_self,'canonical_source_self': [list(x) for x in sorted(canon_self(set(map(tuple,base_self['raw_pair_union'])),baseline_source))],
    'canonical_parent_self':[list(x) for x in sorted(canon_self(set(map(tuple,base_self['raw_pair_union'])),baseline_parent))]}
  changed_rows=set(int(f) for op in ops for f in op['face_rows'])
  variants=[];excluded=[];remaining=list(ops);final=None
  max_screen_rounds=2
  for vi in range(max_screen_rounds):
   if vi==0:cand=candidate
   else:
    compose=loadmod('b65_99_topology_composer534',ROOT/'connected-quality-patch-534/replay_surviving_flips_534.py')
    replay=loadmod('b65_99_topology_replay405',ROOT/'local-two-parent-retriangulation-405/cross_seam_search_405.py')
    cand=compose.apply_ops_batch(base,remaining,replay)
   q=loadmod('b65_99_topology_preflight_shape_module',D/'prepare_topology_flip_batch.py')
   measured=q.quality_after_topology(cand['points_world_m'],base['points_world_m'],base['source_O_faces'],cand['source_O_faces'],rows,remaining)
   shape_pass=bool(measured['orientation_positive'] and measured['minimum_unsigned_parent_area_ratio']>=float(pre['shape_gate']['area_ratio_floor']) and
      measured['minimum_angle_degrees']>=float(pre['shape_gate']['minimum_angle_floor_deg']) and measured['maximum_aspect_measure']<=float(pre['shape_gate']['maximum_aspect_cap']))
   if not shape_pass:
    variants.append({'variant_index':vi,'status':'SHAPE_FAILED_BEFORE_EXACT','shape':measured});break
   src_rows=csr_rows(cand['face_source_ancestry_indptr'],cand['face_source_ancestry_face_rows'],len(cand['source_O_faces']))
   par_rows=csr_rows(cand['refinement_face_parent_ancestry_indptr'],cand['refinement_face_parent_ancestry_face_rows'],len(cand['source_O_faces']))
   patchfaces=np.asarray(cand['source_O_faces'],np.int64)[np.asarray(rows,np.int64)]
   world=np.asarray(cand['points_world_m'],np.float32)
   # Verify generation-002 ancestry against every exact bank pair found in this variant.
   local=(world.astype(np.float64)-pos[65])@M[65].T-off[65]
   variant,_,_=screen_variant(world,cand['source_O_faces'],rows,bone_records,prepared_bone,exact,src_rows,par_rows,
      baseline_source,baseline_parent,base_pairs,base_loc['inside_or_boundary_node_ids'],base_self,{'_pose_M':M,'_pose_off':off,'_pose_pos':pos},remaining,vi)
   hitset=set((int(f),int(b)) for f,b in variant['exact_local_65_99_child_pairs'])
   surface={'banks':[{'body':65,'stable_bone_id':99,'result':{'triangle_pairs':sorted(hitset)}}]}
   lineage=refresh.validate_ancestry(cand,parent,surface)
   variant['shape']=measured;variant['generation002_lineage_refresh']=lineage
   variant['exact_pair_count_from_guard_screen']=len(hitset)
   variants.append(variant)
   round_changed_rows=set(int(f) for op in remaining for f in op['face_rows'])
   bad_ops=set()
   # Exclude only operations directly touching a newly-added canonical bone identity or self witness.
   added_src=set(tuple(x) for x in variant['canonical_source_bone_identity']['added'])
   added_par=set(tuple(x) for x in variant['canonical_immediate_parent_bone_identity']['added'])
   if added_src or added_par:
    for f,b in variant['exact_local_65_99_child_pairs']:
     f=int(f);b=int(b)
     if f not in round_changed_rows:continue
     if any((int(a),b) in added_src for a in src_rows[f]) or any((int(a),b) in added_par for a in par_rows[f]):
      for oi,op in enumerate(remaining):
       if f in set(map(int,op['face_rows'])):bad_ops.add(oi)
   newself=set(tuple(sorted(map(int,p))) for p in variant['local_self_new_witnesses'])
   for f,g in newself:
    for oi,op in enumerate(remaining):
     if set((f,g))&set(map(int,op['face_rows'])):bad_ops.add(oi)
   if not bad_ops or vi==max_screen_rounds-1:
    final={'variant_index':vi,'operation_count':len(remaining),'excluded_operation_indices':excluded}
    break
   removed=[remaining[i] for i in sorted(bad_ops)]
   excluded.extend([{'operation_index_in_initial_batch':ops.index(op),'face_rows':op['face_rows'],
     'reason':'directly implicated by newly added canonical bone identity or self witness'} for op in removed])
   remaining=[op for i,op in enumerate(remaining) if i not in bad_ops]
  final_status='PRIVATE_REJECTED_NO_CANONICAL_IMPROVEMENT'
  if final and variants:
   last=variants[-1]
   src_identity=last['canonical_source_bone_identity']
   parent_identity=last['canonical_immediate_parent_bone_identity']
   no_new_self=(not last['local_self_new_witnesses'] and last['local_self_pass'])
   no_new_inside=not last['inside_boundary_comparison']['new_ids']
   if src_identity['removed_count']>0 and src_identity['added_count']==0 and parent_identity['added_count']==0 and no_new_self and no_new_inside:
    final_status='PRIVATE_LOCAL_CANONICAL_IDENTITY_IMPROVEMENT_PENDING_ALL_BANKS'
   elif src_identity['added_count'] or parent_identity['added_count'] or not no_new_self or not no_new_inside:
    final_status='PRIVATE_REJECTED_NEW_CANONICAL_BONE_SELF_OR_INSIDE_WITNESS'
  report.update({'status':final_status,'candidate_modified':False,
   'input_pins':{'preflight_sha256':sha(PREFLIGHT),'private_batch_candidate_sha256':sha(BATCH),'baseline_candidate_sha256':sha(BASE),
     'parent_candidate_sha256':sha(PARENT),'audit_owner_sha256':sha(AUDIT_TOOL),'bank_input_helper_sha256':sha(BANK_HELPER),
     'generation002_refresh_sha256':sha(REFRESH),'target_bone_count':len(bone)},
   'topology_batch':{'initial_operation_count':len(ops),'initial_changed_face_rows':sorted(changed_rows),
     'initial_nonconflicting_face_node_disjoint':True,'multi_parent_operation_count':pre['selected_multi_parent_operation_count'],
     'saved_pair_count_touched':sum(int(op['crossing_pair_count_touched']) for op in ops),
     'operations':ops,'excluded_operations':excluded,'final_variant':final},
   'baseline_local_self':baselineself,'baseline_inside_or_boundary_nodes':base_loc['inside_or_boundary_node_ids'],
   'variants':variants,'geometry_or_native_admission':False,
   'all_bank_clearance':'NOT_RUN','full_surface_self':'NOT_RUN','native_physical_admission':'NOT_CLAIMED'})
  if final and final['operation_count']>0:
   OUT.mkdir(parents=True,exist_ok=True)
   finalops=remaining
   compose=loadmod('b65_99_topology_composer534_final',ROOT/'connected-quality-patch-534/replay_surviving_flips_534.py')
   replay=loadmod('b65_99_topology_replay405_final',ROOT/'local-two-parent-retriangulation-405/cross_seam_search_405.py')
   finalcand=candidate if len(finalops)==len(ops) else compose.apply_ops_batch(base,finalops,replay)
   idx=int(final['variant_index']);dest=OUT/f'candidate-exact-screen-round-{idx+1:02d}.npz';np.savez_compressed(dest,**finalcand)
   report['final_private_candidate_path']=str(dest);report['final_private_candidate_sha256']=sha(dest)
  elif not final:
   report['status']='EXACT_LOCAL_SCREEN_NO_FINAL_VARIANT'
 except Exception as e:
  report.update({'status':'EXACT_LOCAL_SCREEN_FAILED','failure':repr(e),'traceback':traceback.format_exc(),'candidate_modified':False})
 OUT.mkdir(parents=True,exist_ok=True)
 rp=OUT/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+chr(10))
 report['script_path']=str(Path(__file__));report['script_sha256']=sha(Path(__file__))
 rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+chr(10))
 print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','variants','topology_batch')},indent=2,sort_keys=True,allow_nan=False))
 print('report_path',rp,'report_sha256',sha(rp))

if __name__=='__main__':run()
