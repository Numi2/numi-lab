#!/usr/bin/env python3
"""Selective centroid-fan shape/topology preflight. Read-only; no bone tests/fit."""
import collections, json, os, numpy as np
import preflight
ROOT=preflight.ROOT; CAND=preflight.CAND; DIAG=preflight.DIAG; PATCH=preflight.PATCH; OUT=preflight.OUT

def triangle_stats(t):
 lens=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
 area=float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*0.5)
 ang=[]
 for i in range(3):
  u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
  ang.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
 asp=max(lens)**2/(4*np.sqrt(3)*max(area,1e-30))
 return area,min(ang),asp

def main():
 rows=np.load(PATCH,allow_pickle=False).astype(np.int64); d=json.load(open(DIAG)); p=d['patches'][0]
 cross=set(map(int,p['crossing_skin_faces'])); inside=set(int(x['node_row']) for x in p['inside_crossing_nodes'])
 with np.load(CAND,allow_pickle=False) as z: P=z['points_world_m']; F=z['source_O_faces']
 pf=F[rows]; edgeuses=collections.defaultdict(list);node_faces=collections.defaultdict(set)
 for fi,tri in zip(rows,pf):
  a,b,c=map(int,tri)
  for n in (a,b,c): node_faces[n].add(int(fi))
  for u,v in ((a,b),(b,c),(c,a)): edgeuses[tuple(sorted((u,v)))].append(int(fi))
 boundary_edges=[e for e,u in edgeuses.items() if len(u)==1]
 boundary=set(x for e in boundary_edges for x in e)
 # Half incenter + half centroid, identical formula to 115/118 constructor.
 per={}; centers={}; bary={}
 for fi,ids in zip(rows,pf):
  tri=P[ids].astype(np.float64)
  opp=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])])
  b=0.5*opp/opp.sum()+0.5/3.0; c=(b@tri).astype(np.float32).astype(np.float64)
  bary[int(fi)]=b;centers[int(fi)]=c
  pa,amin,asp=triangle_stats(tri)
  children=[]; ratios=[]; orients=[]
  pn=np.cross(tri[1]-tri[0],tri[2]-tri[0]);pn2=float(pn@pn)
  for a,bv in ((0,1),(1,2),(2,0)):
   t=np.asarray([tri[a],tri[bv],c],np.float64)
   ar,ang,aspc=triangle_stats(t); cn=np.cross(t[1]-t[0],t[2]-t[0])
   ratios.append(float(np.linalg.norm(cn)/max(np.sqrt(pn2),1e-300)))
   orients.append(float((pn@cn)/max(pn2,1e-300)))
   children.append({'min_angle':ang,'aspect':aspc,'area_ratio':ratios[-1],'orientation':orients[-1]})
  fan_ok=bool(min(x['min_angle'] for x in children)>=0.5 and max(x['aspect'] for x in children)<=50.0 and min(ratios)>=0.10 and min(orients)>0)
  per[int(fi)]={'parent_min_angle_degrees':amin,'parent_aspect':asp,'parent_area_m2':pa,
                'fan_min_angle_degrees':min(x['min_angle'] for x in children),
                'fan_max_aspect':max(x['aspect'] for x in children),'fan_min_area_ratio':min(ratios),
                'fan_min_orientation_ratio':min(orients),'fan_gate_pass':fan_ok,
                'intrinsic_angle_failure_for_any_interior_fan':bool(amin<1.0),
                'children':children}
 good={fi for fi,x in per.items() if x['fan_gate_pass']}; bad=set(per)-good
 intrinsic={fi for fi in bad if per[fi]['intrinsic_angle_failure_for_any_interior_fan']}
 crossing_good=cross&good; crossing_bad=cross&bad
 frozen=boundary|{int(n) for fi in bad for n in F[fi]}
 inside_frozen=inside&frozen
 # Move centers/source nodes only in a single edge-ring support over fan-good faces.
 core=set(crossing_good)
 for n in inside-inside_frozen: core.update(node_faces[n]&good)
 good_adj=collections.defaultdict(set)
 for uses in edgeuses.values():
  if len(uses)==2:
   a,b=uses
   if a in good and b in good: good_adj[a].add(b);good_adj[b].add(a)
 active=set(core)
 for fi in list(core): active.update(good_adj[fi])
 active &= good
 active_nodes=set(int(n) for fi in active for n in F[fi])
 movable_source=active_nodes-frozen
 movable_centers=active
 movable_points=len(movable_source)+len(movable_centers)
 touched_good=set(active)
 for n in movable_source: touched_good.update(node_faces[n]&good)
 affected_children=3*len(touched_good)
 patch_set=set(map(int,rows))
 if movable_source:
  movable_arr=np.fromiter(movable_source,dtype=np.int64)
  hit_mask=np.any(np.isin(F,movable_arr),axis=1)
  outside_patch_touched=set(map(int,np.flatnonzero(hit_mask)))-patch_set
 else: outside_patch_touched=set()
 # Construct the mixed topology in integer IDs and check disk/boundary preservation.
 ncenter=0; mixed=[]; face_parents=[]
 for fi,tri in zip(rows,pf):
  a,b,c=map(int,tri)
  if int(fi) in good:
   m=len(P)+ncenter;ncenter+=1
   mixed.extend(((a,b,m),(b,c,m),(c,a,m)));face_parents.extend([int(fi)]*3)
  else:
   mixed.append((a,b,c));face_parents.append(int(fi))
 eu=collections.defaultdict(int)
 for tri in mixed:
  a,b,c=map(int,tri)
  for x,y in ((a,b),(b,c),(c,a)): eu[tuple(sorted((x,y)))]+=1
 mixed_boundary=[e for e,n in eu.items() if n==1]
 mixed_v={x for tri in mixed for x in tri};chi=len(mixed_v)-len(eu)+len(mixed)
 # Existing whole-candidate faces outside the patch that touch any free source node.
 patch_set=set(map(int,rows));outside_patch_count=len(outside_patch_touched)
 target_crossings={fi for fi in crossing_good}
 # Pair-to-face rows are absent from the compact retained diagnostic, so only bounds are valid here.
 report={
  'schema':'numi.human.bank43_16.selective_centroid_fan_preflight.v1',
  'status':'SELECTIVE_PREFLIGHT_ONLY_NO_FIT_OR_BONE_OR_SELF_AUDIT',
  'parent_patch':{'faces':int(len(rows)),'crossing_faces':int(len(cross)),'exact_pairs_total':int(p['crossing_pair_count']),
                  'inside_nodes':int(len(inside)),'fixed_boundary_nodes':int(len(boundary))},
  'fan_face_classification':{'half_incenter_half_centroid_fan_good_faces':int(len(good)),
                             'fan_bad_excluded_faces':int(len(bad)),
                             'bad_faces_intrinsic_angle_proof':int(len(intrinsic)),
                             'bad_faces_nonintrinsic_for_this_center_only':int(len(bad-intrinsic)),
                             'crossing_faces_fan_good_targetable_by_face':int(len(crossing_good)),
                             'crossing_faces_excluded_unresolved':int(len(crossing_bad)),
                             'crossing_pair_count_on_excluded_faces':'unknown; retained diagnostic lacks pair-to-skin-face mapping',
                             'minimum_unresolved_pair_count_lower_bound':int(len(crossing_bad)),
                             'inside_nodes_frozen_by_bad_face_or_boundary':int(len(inside_frozen)),
                             'inside_nodes_not_frozen':int(len(inside-inside_frozen)),
                             'all_bad_face_source_rows':[{'face_row':int(fi),**per[fi]} for fi in sorted(bad)],
                             'intrinsic_angle_rule':'For any interior center, the two child angles at each original corner partition that source corner angle; if the parent minimum angle is <1 degree, at least one child is <0.5 degree.'},
  'selective_topology':{'fan_child_faces':int(3*len(good)),'unchanged_excluded_faces':int(len(bad)),
                        'mixed_output_faces':int(len(mixed)),'new_centers':int(ncenter),
                        'boundary_edges':int(len(mixed_boundary)),'expected_boundary_edges':int(len(boundary_edges)),
                        'euler_characteristic':int(chi),'boundary_preserved':bool(len(mixed_boundary)==len(boundary_edges)),
                        'source_ancestry_entries_if_each_child_inherits_parent':int(len(mixed)),
                        'excluded_source_faces_remain_bitwise_original':True},
  'selective_active_region':{'good_crossing_core_faces':int(len(core&crossing_good)),
                             'one_edge_ring_good_fan_faces':int(len(active)),
                             'movable_source_nodes_after_boundary_and_excluded_face_freeze':int(len(movable_source)),
                             'movable_centers':int(len(movable_centers)),'movable_points':int(movable_points),
                             'coordinate_dofs':int(3*movable_points),
                             'affected_fan_parent_faces_for_area_linearization':int(len(touched_good)),
                             'affected_new_child_faces_for_quality_gates':int(affected_children),
                             'old_candidate_faces_outside_patch_touched_by_free_nodes':int(outside_patch_count),
                             'outside_old_face_nonworsening_rows':int(outside_patch_count)},
  'sparse_lp_upper_bound':{'variables_delta_plus_l1_slack':int(6*movable_points),
                           'absolute_value_rows':int(6*movable_points),
                           'endpoint_rows_if_all_pairs_are_targetable':498,
                           'endpoint_rows_exactly_targetable':'unknown until pair-to-face witnesses are reconstructed',
                           'inside_node_rows_for_free_known_inside_nodes':int(len(inside-inside_frozen)),
                           'additional_bone_planes_upper_bound_one_per_movable_point':int(movable_points),
                           'area_rows_for_touched_new_children':int(affected_children),
                           'outside_old_face_nonworsening_rows':int(outside_patch_count),
                           'row_upper_bound_if_all_pairs_targetable':int(6*movable_points+498+movable_points+affected_children+outside_patch_count),
                           'estimated_nnz_upper_bound_if_all_pairs_targetable':int(12*movable_points+9*498+3*movable_points+9*affected_children+9*outside_patch_count)},
  'quality_gates':{'new_child_min_angle_degrees':0.5,'new_child_max_aspect':50.0,
                   'new_child_min_parent_relative_area_ratio':0.10,
                   'all_selected_fan_children_pass_before_fit':bool(all(per[fi]['fan_gate_pass'] for fi in good)),
                   'excluded_faces_frozen_and_unchanged':True,
                   'exact_postfit_gate_required':True},
  'limitations':['This is shape/topology accounting only; no target-bone point location, exact collision test, solver, or candidate emission occurred.',
                 'Crossing pairs are not attributed to individual skin faces in the retained compact witness, so feasible exact pair count remains unknown.',
                 'All inside nodes touching an excluded face are frozen and remain unresolved; these are not claimed cleared.',
                 'The one-edge-ring selective LP remains only a plan; exact witness-to-child barycentric mapping, point-location planes, sparse feasibility and exact emitted checks are unimplemented/unrun.',
                 'The 0.5 mm trust region and old-face linearization are proposed bounds, not achieved results.']}
 out=OUT+'/selective-preflight.json'
 with open(out,'w') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 summary={k:report[k] for k in ('status','fan_face_classification','selective_topology','selective_active_region','sparse_lp_upper_bound','quality_gates')}
 summary['fan_face_classification']={k:v for k,v in summary['fan_face_classification'].items() if k!='all_bad_faces_source_rows'}
 print(json.dumps(summary,indent=2,sort_keys=True))
 print('receipt',out)
if __name__=='__main__': main()
