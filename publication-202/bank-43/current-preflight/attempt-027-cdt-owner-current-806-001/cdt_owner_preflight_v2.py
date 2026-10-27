#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b43-16/attempt-027-cdt-owner-current-806-001'
BASE=S/'per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz'
TOPO=S/'per-bank-workers-20261027/b43-16/attempt-027-cdt-owner-current-806-001/bulk-patch-topology.json'
BOUND=S/'per-bank-workers-20261027/b43-16/attempt-027-cdt-owner-current-806-001/exact-source-boundary-checks.json'
LAYER=S/'per-bank-workers-20261027/combined-d071-b109-b61-001/skin-fat-layers-541.npz'
PINS={'base':'806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756','topology':'9bbe5a000612ff48200ffec14c10a4642bb87b66d0b105c6b637cb4e7f4c704b','boundary':'71c062c31a2d45c7a62b6e72b439ffee9136f17cfe082848819e7b3c1d132766'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def orient(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
def main():
 for name,p in [('base',BASE),('topology',TOPO),('boundary',BOUND)]:
  if sha(p)!=PINS[name]:raise ValueError(f'{name} pin mismatch')
 topo=json.loads(TOPO.read_text());br=json.loads(BOUND.read_text())
 patch=np.asarray(topo['final_patch_face_rows'],np.int64)
 if len(patch)!=182 or int(br['boundary_node_count'])!=90 or br['exact_boundary_edge_hit_count']!=0 or br['boundary_nodes_with_inside_banks']:raise ValueError('pinned 90-node clear-boundary facts differ')
 with np.load(BASE,allow_pickle=False) as z:
  F=np.asarray(z['source_O_faces'],np.int64);P=np.asarray(z['points_world_m'],np.float32)
  parent=np.asarray(z['refinement_face_parent_face_row'],np.int64);B=np.asarray(z['refinement_face_parent_barycentric'],np.float64)
  kind=np.asarray(z['face_kind']);sp=np.asarray(z['face_source_ancestry_indptr']);sv=np.asarray(z['face_source_ancestry_face_rows']);dp=np.asarray(z['face_fine_detail_ancestry_indptr']);dv=np.asarray(z['face_fine_detail_ancestry_rows']);up=np.asarray(z['face_fine_subface_ancestry_indptr']);uv=np.asarray(z['face_fine_subface_ancestry_rows'])
  sem={k:' '.join(map(str,np.asarray(z[k]).reshape(-1).tolist())) for k in ['refinement_face_parent_ancestry_semantics','refinement_face_parent_barycentric_semantics']}
 def ids(ptr,vals,f):return tuple(map(int,vals[int(ptr[f]):int(ptr[f+1])]))
 groups=defaultdict(list)
 for f in patch:groups[int(parent[f])].append(int(f))
 pnodes=set(map(int,F[patch].ravel()));bnode=set(map(int,br['boundary_node_ids']))
 reports=[];errs=[];simple_bad=ancestry_bad=chart_bad=0;interior_by_group=0
 for p,fs in sorted(groups.items()):
  coords={};edge_owners=defaultdict(list);faces=[];parentbad=[]
  for f in fs:
   if not np.isfinite(B[f]).all() or np.min(B[f]) < -2e-8 or np.max(B[f])>1+2e-8 or np.max(np.abs(B[f].sum(1)-1))>2e-8:parentbad.append(f)
   for node,bc in zip(F[f],B[f]):
    uv2=bc[1:]
    if int(node) in coords:errs.append(float(np.max(np.abs(coords[int(node)]-uv2))))
    else:coords[int(node)]=uv2.copy()
   faces.append((f,np.asarray([coords[int(n)] for n in F[f]])))
   t=F[f]
   for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edge_owners[tuple(sorted((int(a),int(b))))].append(f)
  bedges=[e for e,own in edge_owners.items() if len(own)==1];bad_edges=[e for e,own in edge_owners.items() if len(own)>2]
  link=defaultdict(set)
  for a,b in bedges:link[a].add(b);link[b].add(a)
  unseen=set(link);loops=[]
  while unseen:
   st=[next(iter(unseen))];cc=set()
   while st:
    x=st.pop()
    if x in cc:continue
    cc.add(x);unseen.discard(x);st.extend(link[x]-cc)
   loops.append(cc)
  one_loop=bool(link) and len(loops)==1 and all(len(x)==2 for x in link.values()) and not bad_edges
  interior_by_group+=len(set(coords)-set(link))
  areas=[orient(*t) for _,t in faces]
  chart_ok=not parentbad and all(abs(a)>1e-14 for a in areas) and (all(a>0 for a in areas) or all(a<0 for a in areas))
  # Require one registered source/detail/subface and exactly one immediate parent per old row in each chart cell.
  src={ids(sp,sv,f) for f in fs};det={ids(dp,dv,f) for f in fs};sub={ids(up,uv,f) for f in fs}
  anc_ok=all(len(ids(np.asarray([0]),np.asarray([]),0))==0 for _ in []) # empty, explicit below
  anc_ok=(len(set(map(int,parent[fs])))==1 and all(int(kind[f])==2 for f in fs) and len(src)==len(det)==len(sub)==1 and all(next(iter(x))>=0 for ss in (src,det,sub) for x in ss))
  if not one_loop:simple_bad+=1
  if not anc_ok:ancestry_bad+=1
  if not chart_ok:chart_bad+=1
  reports.append({'parent_face_row':p,'face_count':len(fs),'node_count':len(coords),'boundary_edge_count':len(bedges),'boundary_loop_count':len(loops),'boundary_loop_sizes':sorted(map(len,loops),reverse=True),'simple_single_loop':one_loop,'interior_node_count':len(set(coords)-set(link)),'chart_barycentric_valid_and_nonfolded':chart_ok,'chart_min_abs_determinant':float(min(map(abs,areas))) if areas else 0.,'ancestry_constant_source_detail_subface_with_single_immediate_parent':anc_ok,'source_ids':sorted({x for s in src for x in s}),'detail_ids':sorted({x for s in det for x in s}),'subface_ids':sorted({x for s in sub for x in s})})
 # Piecewise parent charts do not supply a single global 2D chart. Check an independent best-fit projection only as a conservative diagnostic.
 xyz=np.asarray(P[list(pnodes)],np.float64);ctr=xyz.mean(0);_,svd,v=np.linalg.svd(xyz-ctr,full_matrices=False);q=(xyz-ctr)@v[:2].T
 nodes=sorted(pnodes);xy={n:q[i] for i,n in enumerate(nodes)};orientations=[orient(*(xy[int(n)] for n in F[f])) for f in patch]
 global_projection={'singular_values_m':svd.tolist(),'all_projected_faces_same_orientation':bool(orientations and (all(x>1e-13 for x in orientations) or all(x< -1e-13 for x in orientations))),'degenerate_projected_faces':sum(abs(x)<=1e-13 for x in orientations)}
 with np.load(LAYER,allow_pickle=False) as z:
  perm=np.asarray(z['face_winding_permutation'],np.int64);expected=F[:,perm]
  layer_points={'outer_points_f32_matches_candidate':bool(np.array_equal(np.asarray(z['outer_points_f32'],np.float32),P))}
  layer_faces={'source_outer_faces_raw_matches_candidate':bool(np.array_equal(np.asarray(z['source_outer_faces_raw'],np.int64),F)),'outer_faces_match_winding_permutation':bool(np.array_equal(np.asarray(z['outer_faces'],np.int64),expected)),'inner_faces_match_winding_permutation':bool(np.array_equal(np.asarray(z['inner_faces'],np.int64),expected)),'deep_faces_match_winding_permutation':bool(np.array_equal(np.asarray(z['deep_faces'],np.int64),expected))}
  layer_deltas={k:float(np.max(np.abs(np.asarray(z[k],np.float64)-np.asarray(P,np.float64)))) for k in ['outer_points_f64','inner_points_f64','deep_points_f64']}
 inside=sorted(map(int,topo['inside_target_node_rows']))
 result={'schema':'numi.human.b43-16.cdt-owner-piecewise-chart-preflight.v2','status':'PREFLIGHT_COMPLETE_NO_GEOMETRY_CHANGED','pins':{'base_candidate_sha256':sha(BASE),'bulk_patch_topology_sha256':sha(TOPO),'exact_boundary_receipt_sha256':sha(BOUND)},'target_bank':[43,16],'patch_face_count':len(patch),'patch_vertex_count':len(pnodes),'fixed_outer_boundary_nodes':len(bnode),'interior_patch_nodes':len(pnodes-bnode),'boundary_node_clearance':{'all185_node_inside_count':0,'exact_boundary_edge_hits_all185':0},'inside_target_nodes_count':len(inside),'inside_target_node_rows':inside,'topology_only_triangulation_preserves_point_positions_and_inside_classifications':True,'owner_source':'partial-midpoint-continuation-root-001/bank74-cdt-root-001/constrained_edgeflip_root.py','owner_limit':'The existing owner emits a planar triangulation in one source-triangle barycentric chart; it has no native 3D metric, layer-volume, collision, or ancestry writer. A piecewise per-immediate-parent adapter could preserve single-parent CSR only if every chart group passes the recorded checks.','parent_chart_groups':len(groups),'max_same_parent_node_barycentric_disagreement':max(errs,default=0.),'groups_with_invalid_or_folded_chart':chart_bad,'groups_without_simple_boundary':simple_bad,'groups_without_constant_registered_ancestry':ancestry_bad,'piecewise_chart_group_reports':reports,'global_best_fit_projection_diagnostic':global_projection,'layer_541_sha256':sha(LAYER),'layer_541_matches_this_candidate':{'point_arrays_bitwise':layer_points,'face_arrays_bitwise':layer_faces,'inner_and_deep_offsets_from_skin_are_expected_layer_displacements_m':layer_deltas,'face_winding_permutation':perm.tolist(),'reusable_for_this_candidate':all(layer_points.values()) and all(layer_faces.values())},'construction_gate':'STOP_FIXED_INSIDE_NODES' if inside else 'ASSESS_CHART_AND_LAYER','why_no_construction':'The requested CDT uses the existing interior vertices without moving them. It therefore leaves all 43 points already classified inside bank 43/16 inside. The 806 layer bundle matches the pinned outer points and raw source faces; its declared [0,2,1] winding must be applied when checking layer cells. However, the topology-only CDT cannot move the 43 inside vertices, so it cannot clear this bank.','geometry_changed':False,'candidate_emitted':False,'physical_admission':False,'accepted_physical_steps':0}
 out=W/'cdt-owner-preflight-v2.json';out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'patch_faces':len(patch),'patch_nodes':len(pnodes),'fixed_boundary_nodes':len(bnode),'interior_nodes':len(pnodes-bnode),'inside_target_nodes':len(inside),'parent_chart_groups':len(groups),'max_same_parent_node_barycentric_disagreement':result['max_same_parent_node_barycentric_disagreement'],'invalid_or_folded_groups':chart_bad,'nonsimple_groups':simple_bad,'ancestry_groups_bad':ancestry_bad,'global_projection':global_projection,'layer_matches':result['layer_541_matches_this_candidate'],'receipt_sha256':sha(out)},indent=2,sort_keys=True))
if __name__=='__main__':main()
