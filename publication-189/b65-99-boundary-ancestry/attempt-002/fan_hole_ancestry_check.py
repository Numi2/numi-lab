#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import ast,hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
PREV=S/'per-bank-workers-20261027/b65-99/attempt-001/topology-boundary-check.json'
HELPER=S/'per-bank-workers-20261027/b43-16/attempt-002/boundary_feasibility.py'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
W=S/'per-bank-workers-20261027/b65-99/attempt-002'
OUT=W/'fan-hole-ancestry-check.json'
EXPECTED_C='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
EXPECTED_A='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
BANK=[65,99]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def ff(x):return Fraction.from_float(float(x))
tree=ast.parse(HELPER.read_text())
wanted={'dot','sub','add','mul','cross','in_tri3','orient2','on_seg2','seg_intersect2','in_tri2','exact_segment_triangle','point_seg_dist','point_tri_dist','seg_seg_dist','seg_tri_dist'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(HELPER),'exec'),globals())
start=time.monotonic()
assert sha(C)==EXPECTED_C and sha(A)==EXPECTED_A
with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float32).astype(np.float64);F=Z['source_O_faces'].astype(np.int64)
audit=mod('bank65_fanhole_audit',AUDIT);exact=audit.exact_module();h=mod('bank65_fanhole_helper',audit.BANKHELPER)
bankdir=audit.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=h.read_bank_inputs(bankdir,audit.E)
mask=(ids[:,0]==BANK[0])&(ids[:,1]==BANK[1]);BT=np.asarray(T[mask],float);BID=np.asarray(ids[mask,3],np.int64)
local=(P-pos[BANK[0]])@M[BANK[0]].T-off[BANK[0]]
brec=audit.records(BT,BID,exact)
BTfrac=[tuple(tuple(ff(x) for x in v) for v in tri) for tri in BT]
allbanks=json.loads(A.read_text())['banks']
br=next(x for x in allbanks if [int(x['body']),int(x['stable_bone_id'])]==BANK)
pairs=[tuple(map(int,x)) for x in br['result']['triangle_pairs']]
seed=set(sf for sf,bf in pairs)
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
adj=defaultdict(set)
for fs in edgefaces.values():
 for f in fs:adj[f].update(g for g in fs if g!=f)
# Edge-connected crossing components.
unseen=set(seed);components=[]
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];cc=[]
 while q:
  f=q.pop();cc.append(f)
  for g in adj[f]:
   if g in unseen and g in seed:unseen.remove(g);q.append(g)
 components.append(set(cc))
components.sort(key=lambda x:min(x))
def boundary(p):
 rows=[]
 for e,fs in edgefaces.items():
  ii=[f for f in fs if f in p];oo=[f for f in fs if f not in p]
  if ii and oo:rows.append((e,ii,oo))
  elif ii and len(fs)==1:rows.append((e,ii,[]))
 graph=defaultdict(list)
 for e,ii,oo in rows:graph[e[0]].append(e[1]);graph[e[1]].append(e[0])
 loops=[];un=set(graph)
 while un:
  st=min(un);un.remove(st);q=[st];cc=[]
  while q:
   v=q.pop();cc.append(v)
   for w in graph[v]:
    if w in un:un.remove(w);q.append(w)
  loops.append(sorted(cc))
 return rows,graph,loops
# Reconstruct the two contiguous edge rings independently for every original crossing component.
patches=[set(c) for c in components]
for _ in range(2):
 nxt=[]
 for p in patches:
  be,_,_=boundary(p);addfaces={f for _,_,oo in be for f in oo};nxt.append(p|addfaces)
 patches=nxt
seed_rings=[{'component_id':i,'faces':len(p),'face_rows':sorted(p)} for i,p in enumerate(patches)]
# Exactly one component was found; fan-expand all non-2-valence boundary nodes of the 2-ring patch once.
pre=[]
fan_added=[]
for i,p in enumerate(patches):
 be,bg,loops=boundary(p);branches=sorted(n for n in bg if len(bg[n])!=2)
 additions=set(f for n in branches for f in nodefaces[n])-p
 fan_added.append(additions)
 pre.append({'component_id':i,'face_count':len(p),'boundary_edges':len(be),'boundary_nodes':len(bg),'boundary_loop_node_counts':sorted(map(len,loops)),'boundary_degree_histogram':dict(Counter(len(bg[n]) for n in bg)),'branch_nodes':[[int(n),len(bg[n])] for n in branches],'fan_faces_added':len(additions),'fan_face_rows':sorted(additions)})
# Separate components remain separate. Only one was present; retain a guard for future multi-component reuse.
fan_conflict=None
for i in range(len(patches)):
 for j in range(i+1,len(patches)):
  pi=patches[i]|fan_added[i];pj=patches[j]|fan_added[j]
  if pi&pj or any(g in pj for f in pi for g in adj[f]):fan_conflict={'components':[i,j],'reason':'fan_expansion_would_join_components'}
fanpatches=[p|fan_added[i] for i,p in enumerate(patches)]
def complement_components(p):
 rem=set(range(len(F)))-p;un=set(rem);comps=[];which={}
 while un:
  st=min(un);un.remove(st);q=[st];cc=[]
  while q:
   f=q.pop();cc.append(f)
   for g in adj[f]:
    if g in un and g in rem:un.remove(g);q.append(g)
  idx=len(comps);comps.append(sorted(cc))
  for f in cc:which[f]=idx
 return comps,which
def hole_candidates(p):
 be,bg,loops=boundary(p);comps,which=complement_components(p)
 # only map cycles when the patch boundary is a disjoint union of simple loops
 if not loops or any(len(bg[n])!=2 for n in bg):
  return be,bg,loops,comps,which,[],{'eligible':False,'reason':'fan boundary still branched; cycles not individually attributable'}
 node_loop={n:i for i,loop in enumerate(loops) for n in loop}
 loop_edges=Counter();loop_comp=defaultdict(set);comp_loop=defaultdict(set);comp_edges=Counter()
 for e,ins,outs in be:
  li=node_loop[e[0]]
  loop_edges[li]+=1
  cids={which[f] for f in outs if f in which}
  loop_comp[li].update(cids)
  for cid in cids:comp_loop[cid].add(li);comp_edges[cid]+=1
 eligible=[]
 for cid,faces in enumerate(comps):
  loops_for=comp_loop.get(cid,set())
  if len(faces)<=12 and len(loops_for)==1:
   li=next(iter(loops_for))
   if loop_comp[li]=={cid} and comp_edges[cid]==loop_edges[li] and len(loops[li])<=12:
    eligible.append({'complement_component_id':cid,'face_count':len(faces),'boundary_loop_id':li,'boundary_node_count':len(loops[li]),'face_rows':faces})
 return be,bg,loops,comps,which,eligible,{'eligible':True,'complement_component_count':len(comps),'small_hole_fill_threshold_faces':12,'small_hole_fill_threshold_loop_nodes':12,'boundary_loop_count':len(loops)}
fill_checks=[];filled=[set(p) for p in fanpatches]
for i,p in enumerate(fanpatches):
 be,bg,loops,comps,which,eligible,meta=hole_candidates(p)
 fill_checks.append({'component_id':i,'fan_patch_faces':len(p),'boundary_loop_count':len(loops),'boundary_loop_node_counts':sorted(map(len,loops)),'boundary_degree_histogram':dict(Counter(len(bg[n]) for n in bg)),'small_hole_fill_analysis':meta,'eligible_holes':eligible})
 for hrec in eligible:filled[i].update(hrec['face_rows'])
# Exact local target-boundary and stored neighbor checks only if boundary is a union of simple loops.
def check_boundary(p,ci):
 be,bg,loops=boundary(p);deg=dict(Counter(len(bg[n]) for n in bg));simple=bool(loops) and all(len(bg[n])==2 for n in bg)
 result={'component_id':ci,'patch_face_count':len(p),'boundary_edge_count':len(be),'boundary_node_count':len(bg),'boundary_loop_count':len(loops),'boundary_loop_node_counts':sorted(map(len,loops)),'boundary_degree_histogram':deg,'all_boundary_loops_simple':simple}
 if not simple:
  result['checks_run']=False;result['obstruction']='boundary remains branched or empty after fan and unambiguous small-hole fill';return result
 bn=set(bg);inside=[];node_rows=[]
 for n in sorted(bn):
  locres=exact.point_location(tuple(ff(x) for x in local[n]),brec)
  ds=[point_tri_dist(local[n],tr) for tr in BT];j=int(np.argmin(ds))
  rec={'skin_node_row':int(n),'exact_location':locres.get('location'),'exact_ray':locres.get('ray'),'exact_crossings':locres.get('crossings'),'nearest_target_bone_face_row':int(BID[j]),'distance_mm':float(ds[j]*1000)}
  node_rows.append(rec)
  if rec['exact_location'] in ('inside','boundary'):inside.append(rec)
 edge_hits=[];edge_rows=[]
 for e,ii,oo in be:
  u,v=map(int,e);lo=np.minimum(local[u],local[v]);hi=np.maximum(local[u],local[v])
  cand=np.flatnonzero(np.all(BT.max(1)>=lo,axis=1)&np.all(BT.min(1)<=hi,axis=1))
  hits=[]
  for j in cand:
   r=exact_segment_triangle(tuple(ff(x) for x in local[u]),tuple(ff(x) for x in local[v]),BTfrac[j])
   if r['hit']:hits.append({'target_bone_face_row':int(BID[j]),'kind':r['kind']})
  if hits:edge_hits.append({'edge':[u,v],'inside_faces':ii,'outside_faces':oo,'hits':hits})
  ds=[seg_tri_dist(local[u],local[v],tr) for tr in BT];j=int(np.argmin(ds))
  edge_rows.append({'edge':[u,v],'nearest_target_bone_face_row':int(BID[j]),'distance_mm':float(ds[j]*1000)})
 bedges={tuple(sorted(e)) for e,_,_ in be};incident={f for n in bn for f in nodefaces[n]}
 inbf={f for _,ii,_ in be for f in ii};outbf={f for _,_,oo in be for f in oo}
 neighbor_inside=[];neighbor_boundary=[]
 for bank in allbanks:
  key=[int(bank['body']),int(bank['stable_bone_id'])]
  if key==BANK or not bank.get('result'):continue
  for sf,bf in bank['result']['triangle_pairs']:
   sf=int(sf);bf=int(bf);t=F[sf];sn=sorted(set(map(int,t))&bn);se=[]
   for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
    e=tuple(sorted((int(a),int(b))))
    if e in bedges:se.append(list(e))
   rel=[]
   if sf in p:rel.append('in_patch')
   if sf in inbf:rel.append('inside_boundary_face')
   if sf in outbf:rel.append('outside_boundary_adjacent_face')
   if sf in incident:rel.append('boundary_node_incident_face')
   if se:rel.append('shares_boundary_edge')
   if sn:rel.append('shares_boundary_node')
   if rel:
    rec={'typed_bank':key,'skin_face_row':sf,'bone_face_row':bf,'relation':rel,'shared_boundary_nodes':sn,'shared_boundary_edges':se}
    neighbor_inside.append(rec)
    if sf in incident or sn or se:neighbor_boundary.append(rec)
 result.update({'checks_run':True,'boundary_inside_or_on_target_nodes':len(inside),'inside_or_on_witnesses':inside,'minimum_node_distance':min(node_rows,key=lambda x:x['distance_mm']),'exact_target_bone_boundary_edge_hits':len(edge_hits),'exact_target_bone_boundary_edge_hit_witnesses':edge_hits,'minimum_edge_distance':min(edge_rows,key=lambda x:x['distance_mm']),'stored_neighbor_crossings_in_patch':len(neighbor_inside),'stored_neighbor_crossings_touching_boundary':len(neighbor_boundary),'stored_neighbor_boundary_witnesses':neighbor_boundary[:200],'boundary_node_records':node_rows,'boundary_edge_records':edge_rows})
 return result
final_boundary=[check_boundary(p,i) for i,p in enumerate(filled)]
# Clarify -1 marker rows among the stored bank crossings via all available CSR ancestry channels.
def csr_rows(sf,ptr,rows):
 if ptr not in Z or rows not in Z:return None
 lo,hi=map(int,Z[ptr][sf:sf+2]);return list(map(int,Z[rows][lo:hi]))
minus_rows=[]
for sf in sorted(seed):
 if int(Z['parent_source_face_marker'][sf])!=-1:continue
 minus_rows.append({'skin_face_row':int(sf),'parent_source_face_marker':int(Z['parent_source_face_marker'][sf]),'parent_coarse_face_row':int(Z['parent_coarse_face_row'][sf]),'parent_fine_detail_face_row':int(Z['parent_fine_detail_face_row'][sf]),'parent_fine_subface_row':int(Z['parent_fine_subface_row'][sf]),'source_face_ancestry_rows':csr_rows(sf,'face_source_ancestry_indptr','face_source_ancestry_face_rows'),'fine_detail_ancestry_rows':csr_rows(sf,'face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows'),'fine_subface_ancestry_rows':csr_rows(sf,'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),'immediate_parent_face_rows':csr_rows(sf,'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows')})
semantics={}
for k in ('face_source_ancestry_semantics','face_fine_detail_ancestry_semantics','face_fine_subface_ancestry_semantics','refinement_face_parent_ancestry_semantics'):
 if k in Z:semantics[k]=str(Z[k].item())
multi_sets=Counter(tuple(r['source_face_ancestry_rows'] or []) for r in minus_rows)
report={'schema':'numi.human.bank65_99.fan_hole_and_ancestry_check.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'previous_topology_receipt':str(PREV),'previous_topology_receipt_sha256':sha(PREV),'target_bank':BANK,'initial_crossing_component_count':len(components),'two_ring_patch_summary':seed_rings,'fan_expansion':pre,'fan_expansion_component_conflict':fan_conflict,'small_hole_fill_checks':fill_checks,'final_boundary_checks':final_boundary,'ancestry_minus_one_audit':{'crossing_faces_with_parent_source_marker_minus_one':len(minus_rows),'rows':minus_rows,'source_ancestry_set_counts':[[list(k),v] for k,v in multi_sets.most_common()],'semantics':semantics,'conclusion':'Alternative direct source/fine ancestry is present where CSR rows are nonempty; -1 scalar markers alone do not mean missing provenance. Review rows with empty CSR ancestry as unresolved.'},'geometry_changed':False,'whole_body_or_all_bank_audit_run':False,'native_admission':False,'accepted_physical_steps':0,'limitations':['Topology-only fan/fill; no coordinates or source arrays edited.','Complement fill admits only disconnected complement components of at most 12 faces, attached to exactly one simple boundary loop of at most 12 nodes with no other loop attachment.','Stored neighboring-bank pairs come from audit-159; no neighboring-bank rerun.','Distance is Float64 diagnostic only; exact point location and edge-segment contact are rational predicates on stored transformed coordinates.','This does not establish tissue thickness or physical admission.'],'elapsed_seconds':time.monotonic()-start}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
summary={'status':'COMPLETE_BOUNDED_FAN_HOLE_CHECK','fan':pre,'hole_fill':fill_checks,'final_boundary':[{k:v for k,v in r.items() if k not in ('boundary_node_records','boundary_edge_records','stored_neighbor_boundary_witnesses','inside_or_on_witnesses','exact_target_bone_boundary_edge_hit_witnesses')} for r in final_boundary],'ancestry_minus_one':{'count':len(minus_rows),'unique_source_ancestry_sets':[[list(k),v] for k,v in multi_sets.most_common()],'empty_source_ancestry_count':sum(not r['source_face_ancestry_rows'] for r in minus_rows),'semantics':semantics},'elapsed_seconds':report['elapsed_seconds']}
print(json.dumps(summary,indent=2,allow_nan=False))
