from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=C.parent/'audit-159/report.json'
TOPO=S/'per-bank-workers-20261027/b65-99/attempt-002/fan-hole-ancestry-check.json'
ENDPOINTS=S/'per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/receipt.json'
FAILURE=S/'per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/facewise-failure-witness.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=S/'per-bank-workers-20261027/b65-99/attempt-004-bone-aware-connected-patch-prep-001'
C_PIN='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
A_PIN='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
CLEARANCE_MM=0.05
TRUST_STEP_MM=0.5
MAX_TOTAL_MM=3.0
started=time.monotonic()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_tri(t):return tuple(tuple(frac(x) for x in p) for p in np.asarray(t).reshape(3,3))
def exact_point(row):return tuple(Fraction(x) for x in row)
def barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]));axes=exact._axes(n)
 a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise RuntimeError('degenerate exact triangle during witness reconstruction')
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)
def patch_topology(F,rows):
 use=Counter();adj=defaultdict(set)
 for r in rows:
  f=F[int(r)]
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);e=tuple(sorted((a,b)));use[e]+=1;adj[a].add(b);adj[b].add(a)
 if any(v not in (1,2) for v in use.values()):raise RuntimeError('patch has nonmanifold edge incidence')
 boundary=[e for e,n in use.items() if n==1];bad=[e for e,n in use.items() if n not in (1,2)]
 b_adj=defaultdict(set);deg=Counter()
 for a,b in boundary:b_adj[a].add(b);b_adj[b].add(a);deg[a]+=1;deg[b]+=1
 loops=0;seen=set()
 for v in b_adj:
  if v in seen:continue
  loops+=1;stack=[v];seen.add(v)
  while stack:
   x=stack.pop()
   for y in b_adj[x]:
    if y not in seen:seen.add(y);stack.append(y)
 comp=0;seenv=set()
 for v in adj:
  if v in seenv:continue
  comp+=1;stack=[v];seenv.add(v)
  while stack:
   x=stack.pop()
   for y in adj[x]:
    if y not in seenv:seenv.add(y);stack.append(y)
 euler=len(adj)-len(use)+len(rows)
 info={'vertex_count':len(adj),'edge_count':len(use),'face_count':len(rows),'boundary_edge_count':len(boundary),
  'boundary_vertex_count':len(b_adj),'boundary_loop_count':loops,'boundary_degree_histogram':{str(k):int(v) for k,v in Counter(deg.values()).items()},
  'connected_components':comp,'euler_characteristic':int(euler),'is_single_disk':bool(comp==1 and loops==1 and euler==1 and all(v==2 for v in deg.values()))}
 return boundary,b_adj,info

def json_default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,set):return sorted(x)
 raise TypeError(type(x).__name__)

OUT.mkdir(parents=True,exist_ok=True)
report={'schema':'numi.human.bank65_99.witness-barycentric-preflight.v1','status':'STARTED','script_path':str(Path(__file__)),
 'scope':'Cached exact intersection witness reconstruction and combinatorial boundary expansion only. No optimizer or new exact intersection/self/all-bank audit.',
 'bank':[65,99],'clearance_mm':CLEARANCE_MM,'trust_step_mm':TRUST_STEP_MM,'max_cumulative_movement_mm':MAX_TOTAL_MM}
try:
 if sha(C)!=C_PIN or sha(A)!=A_PIN:raise RuntimeError('candidate or audit pin changed')
 top=json.load(open(TOPO));enddoc=json.load(open(ENDPOINTS));fail=json.load(open(FAILURE))
 if top.get('candidate_sha256')!=C_PIN or enddoc.get('candidate_sha256')!=C_PIN:raise RuntimeError('stored topology/endpoint candidate pin mismatch')
 Z={}
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P=np.asarray(Z['points_world_m'],np.float32).astype(np.float64);F=np.asarray(Z['source_O_faces'],np.int64)
 helper=loadmod('b65_99_witness_helper',HELPER);audit_tool=loadmod('b65_99_witness_audit',AUDIT_TOOL);exact=audit_tool.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 bmask=(ids[:,0]==65)&(ids[:,1]==99);bone=np.asarray(T[bmask],np.float64);bone_ids=np.asarray(ids[bmask,3],np.int64)
 bmap={int(x):i for i,x in enumerate(bone_ids)};bone_norm=np.cross(bone[:,1]-bone[:,0],bone[:,2]-bone[:,0]);lens=np.linalg.norm(bone_norm,axis=1)
 if len(bone)!=342 or np.any(lens==0):raise RuntimeError('bone triangle inventory or area check failed')
 bone_norm/=lens[:,None];signed6=float(sum(float(a@np.cross(b,c)) for a,b,c in bone));bone_norm*=1.0 if signed6>0 else -1.0
 bone_keys=[tuple(exact.float32_point_lattice_key(x) for x in tri) for tri in bone]
 edge_to_faces=defaultdict(list)
 for bi,ks in enumerate(bone_keys):
  for i,j in ((0,1),(1,2),(2,0)):edge_to_faces[tuple(sorted((ks[i],ks[j])))].append(bi)
 vtx_to_faces=defaultdict(list)
 for bi,ks in enumerate(bone_keys):
  for k in ks:vtx_to_faces[k].append(bi)
 def outward_at(bary,bi):
  zero=[i for i,x in enumerate(bary) if x==0];incident={bi}
  if len(zero)>=2:
   vi=next(i for i,x in enumerate(bary) if x==1);incident.update(vtx_to_faces[bone_keys[bi][vi]])
  elif len(zero)==1:
   ij=[i for i in range(3) if i!=zero[0]];incident.update(edge_to_faces[tuple(sorted((bone_keys[bi][ij[0]],bone_keys[bi][ij[1]])))])
  n=bone_norm[sorted(incident)].sum(axis=0);ln=float(np.linalg.norm(n))
  if ln<1e-10:raise RuntimeError('cancelling adjacent bone normals at witness')
  return n/ln,int(len(incident))
 base_summary=top['two_ring_patch_summary'][0];fan=top['fan_expansion'][0]
 base_rows=sorted(set(map(int,base_summary['face_rows']))|set(map(int,fan['fan_face_rows'])))
 if len(base_rows)!=241:raise RuntimeError(f'expected 241-face topological patch; got {len(base_rows)}')
 base_boundary,base_badj,base_info=patch_topology(F,base_rows)
 if not base_info['is_single_disk'] or base_info['boundary_edge_count']!=63:raise RuntimeError(f'base patch changed: {base_info}')
 base_boundary_nodes=set(base_badj)
 face_ancestry_ind=np.asarray(Z['face_source_ancestry_indptr'],np.int64)
 empty=[r for r in base_rows if face_ancestry_ind[r+1]<=face_ancestry_ind[r]]
 if empty:raise RuntimeError(f'base patch CSR ancestry empty: {empty[:8]}')
 skin_local=(P-pos[65])@M[65].T-off[65]
 witness_receipts=enddoc['exact_intersections']['exact_pair_endpoint_feature_witnesses']
 witness_rows=[];node_normals=defaultdict(list);fixed_only_nodes=set();expand_nodes=set();all_fixed_points=[];over_cap_witnesses=[];unique_points=set();max_move_lb=0.0
 for rec in witness_receipts:
  sf=int(rec['skin_face']);bid=int(rec['bone_triangle']);bi=bmap[bid]
  s_tri=exact_tri(skin_local[F[sf]]);b_tri=exact_tri(bone[bi])
  for ord,(raw,skin_feature) in enumerate(zip(rec['exact_local_endpoints_m'],rec['skin_features'])):
   q=exact_point(raw);sw=barycentric(q,s_tri,exact);bw=barycentric(q,b_tri,exact)
   if sum(sw)!=1 or min(sw)<0 or max(sw)>1 or sum(bw)!=1 or min(bw)<0 or max(bw)>1:
    raise RuntimeError(f'cached witness barycentric outside triangle for {(sf,bid,ord)}')
   reconstructed=tuple(sum(sw[i]*s_tri[i][k] for i in range(3)) for k in range(3))
   if reconstructed!=q:raise RuntimeError(f'cached witness barycentric reconstruction differs for {(sf,bid,ord)}')
   normal,normal_face_count=outward_at(bw,bi)
   face_nodes=tuple(map(int,F[sf]));weights=tuple(float(x) for x in sw)
   support=[(face_nodes[i],sw[i]) for i in range(3) if sw[i]!=0]
   boundary_weight=sum((w for node,w in support if node in base_boundary_nodes),Fraction(0))
   movable_weight=Fraction(1)-boundary_weight
   needed=(CLEARANCE_MM/float(movable_weight)) if movable_weight>0 else None
   boundary_support=[n for n,w in support if n in base_boundary_nodes]
   if needed is None:
    all_fixed_points.append({'skin_face':sf,'bone_triangle_id':bid,'endpoint_ordinal':ord,'support_nodes':[n for n,w in support],
     'boundary_support_nodes':boundary_support})
    fixed_only_nodes.update(boundary_support);expand_nodes.update(boundary_support)
   else:
    max_move_lb=max(max_move_lb,needed)
    if needed>MAX_TOTAL_MM and boundary_support:
     over_cap_witnesses.append({'skin_face':sf,'bone_triangle_id':bid,'endpoint_ordinal':ord,'support_nodes':[n for n,w in support],
      'boundary_support_nodes':boundary_support,'equal_support_lower_bound_mm':needed})
     expand_nodes.update(boundary_support)
   qf=np.asarray([float(x) for x in q],np.float64)
   witness_rows.append({'skin_face':sf,'bone_triangle_id':bid,'endpoint_ordinal':ord,'point_key':[str(x) for x in q],
    'source_face_nodes':list(face_nodes),'barycentric_exact':[str(x) for x in sw],
    'barycentric_float64':list(weights),'support_nodes':[n for n,w in support],
    'fixed_boundary_barycentric_weight_exact':str(boundary_weight),'movable_barycentric_weight_exact':str(movable_weight),
    'minimum_equal_support_step_mm':needed,'bone_normal':normal.tolist(),'adjacent_bone_face_count':normal_face_count,
    'skin_feature_from_receipt':skin_feature})
   unique_points.add(tuple(q))
   for node,w in support:node_normals[node].append((normal,bid,sf,float(w)))
 # Expand only around fixed support nodes that block a witness under the 3 mm total trust cap.
 rows=set(base_rows);growth=[]
 if expand_nodes:
  for r,f in enumerate(F):
   if r in rows:continue
   if any(int(v) in expand_nodes for v in f):rows.add(r);growth.append(r)
 expanded_rows=sorted(rows)
 exp_boundary,exp_badj,exp_info=patch_topology(F,expanded_rows)
 if not exp_info['is_single_disk']:raise RuntimeError(f'measured witness-support expansion did not preserve a disk: {exp_info}')
 exp_boundary_nodes=set(exp_badj)
 empty_exp=[r for r in expanded_rows if face_ancestry_ind[r+1]<=face_ancestry_ind[r]]
 # Re-evaluate witness supports against expanded fixed nodes.
 residual_fixed=[];partial_fixed=0;post_lb=0.0
 for w in witness_rows:
  # exact bary list is retained as text; float64 is adequate for this support audit.
  q=[Fraction(x) for x in w['barycentric_exact']]
  nodes=w['source_face_nodes'];fixed=sum((q[i] for i,n in enumerate(nodes) if n in exp_boundary_nodes),Fraction(0))
  movable=Fraction(1)-fixed
  if movable==0:residual_fixed.append(w)
  elif fixed>0:partial_fixed+=1
  else:pass
  if movable>0:post_lb=max(post_lb,CLEARANCE_MM/float(movable))
 node_conflicts=[]
 for node,rows0 in node_normals.items():
  if len(rows0)>1:
   dots=[float(rows0[i][0]@rows0[j][0]) for i in range(len(rows0)) for j in range(i+1,len(rows0))]
   if dots and min(dots)<-0.9:
    node_conflicts.append({'node':node,'constraint_count':len(rows0),'minimum_normal_dot':min(dots),
      'bone_triangle_ids':sorted(set(x[1] for x in rows0))})
 other_bank_pairs=[]
 for bdoc in json.load(open(A))['banks']:
  if (int(bdoc['body']),int(bdoc['stable_bone_id']))==(65,99):continue
  bres=bdoc.get('result')
  if not isinstance(bres,dict):continue
  for sf,bid in bres.get('triangle_pairs',[]):
   if int(sf) in set(expanded_rows):other_bank_pairs.append([int(bdoc['body']),int(bdoc['stable_bone_id']),int(sf),int(bid)])
 row_hash=lambda rr:hashlib.sha256(np.asarray(rr,np.int64).tobytes()).hexdigest()
 report.update({'status':'WITNESS_BARYCENTRICS_VERIFIED_BOUNDARY_EXPANSION_PREPARED',
  'pins':{'candidate_sha256':sha(C),'audit159_sha256':sha(A),'two_ring_fan_topology_receipt_sha256':sha(TOPO),
   'exact_endpoint_graph_receipt_sha256':sha(ENDPOINTS),'sub_ulp_failure_witness_sha256':sha(FAILURE)},
  'opposing_plane_finding':{'skin_node_from_all_three_preflight':21007,'bone_triangle_ids':[174,175],
   'normal_dot':-0.9848370105232628,'shared_float32_vertices_between_bone_triangles':0,'shared_bone_edges':0,
   'classification':'distinct non-edge-adjacent bone triangles with nearly opposing outward normals; supports witness-local constraints instead of imposing both planes on every shared skin vertex'},
  'cached_witnesses':{'pair_count':len(witness_receipts),'endpoint_count':len(witness_rows),'unique_exact_endpoint_points':len(unique_points),
   'all_barycentrics_exactly_reconstruct':True,'exact_local_root_ids_116_117_f32_collapse_separation_m':fail['exact_sub_ulp_witnesses'][1]['separation_from_anchor_m'],
   'endpoint_nodes_are_not_emitted_or_inserted':True,'max_equal_support_motion_lower_bound_mm_before_expansion':max_move_lb,
   'fixed_only_witness_endpoint_count_before_expansion':len(all_fixed_points),'fixed_only_support_nodes_before_expansion':sorted(fixed_only_nodes),
   'witness_endpoint_count_over_3mm_support_bound_before_expansion':len(over_cap_witnesses),
   'support_nodes_selected_for_measured_expansion':sorted(expand_nodes),'witness_node_opposing_normal_conflicts_count':len(node_conflicts),'witness_node_opposing_normal_conflicts':node_conflicts},
  'base_patch':{'source_face_count':len(base_rows),'source_rows_sha256':row_hash(base_rows),**base_info,
   'all_source_face_csr_rows_present':True,'fixed_boundary_node_count':len(base_boundary_nodes)},
  'measured_expansion':{'source_face_count':len(expanded_rows),'source_rows_sha256':row_hash(expanded_rows),
   'added_face_count':len(growth),'added_face_rows':sorted(growth),**exp_info,'new_fixed_boundary_node_count':len(exp_boundary_nodes),
   'fixed_only_witness_endpoints_after_expansion':len(residual_fixed),'witnesses_with_partial_fixed_support_after_expansion':partial_fixed,
   'support_nodes_expanded':sorted(expand_nodes),'over_cap_witness_count_before_expansion':len(over_cap_witnesses),
   'max_equal_support_motion_lower_bound_mm_after_expansion':post_lb,'all_added_faces_have_source_face_csr':not empty_exp,
   'added_face_rows_missing_ancestry':empty_exp,'stored_other_bank_crossing_pairs_in_expanded_patch':other_bank_pairs},
  'fit_contract':{'method':'per-witness exact barycentric halfspaces on current crossing endpoints, no endpoint emission; local trust-region objective, re-audit exact Float32 patch against all 342 bank triangles each iteration, add only new exact witness constraints, cap at three iterations',
   'trust_step_mm':TRUST_STEP_MM,'cumulative_patch_displacement_cap_mm':MAX_TOTAL_MM,
   'fixed_boundary':'the measured expanded patch boundary only; do not fix witness-only support nodes inside patch',
   'shape_gates':{'source face CSR and face identities preserved','positive parent-relative orientation','area ratio >= 0.10',
    'min angle >= max(0.5 degrees, 90% of preflight patch baseline)','max movement <= 3.0 mm'},
   'conditional_gates':'stop before self check unless mapped parent-identity crossings are strictly below the original 148 identities and there are no inside/boundary vertices; if improved, run exact local patch self and neighboring-skin self; no native/physical admission',
   'fit_or_exact_reaudit_run':False,'sub_ulp_coordinates_inserted':False}})
except Exception as e: report.update({'status':'WITNESS_BARYCENTRIC_PREFLIGHT_FAILED','failure':repr(e)})
report['elapsed_seconds']=time.monotonic()-started;report['script_sha256']=sha(Path(__file__))
out=OUT/'witness-preflight.json';out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False,default=json_default)+'\n')
print(json.dumps({k:report.get(k) for k in ['status','failure','opposing_plane_finding','cached_witnesses','base_patch','measured_expansion','fit_contract'] if k in report},indent=2,allow_nan=False,default=json_default))
print('receipt_path',out,'sha256',sha(out))
