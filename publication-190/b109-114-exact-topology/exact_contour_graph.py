#!/usr/bin/env python3
"""Exact rational crossing-contour reconstruction for separate 109/114 patches."""
from pathlib import Path
from collections import defaultdict,Counter,deque
from fractions import Fraction
import hashlib,importlib.util,json,sys,time,traceback
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
C=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz')
A=C.parent/'audit-159/report.json'
PATCHES=S/'per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/patch-components'
OUT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-002-exact-contours')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
CAND_PIN='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
AUDIT_PIN='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
BODY=109; BONE=114
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_tri(tri):return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))
def barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
 axes=exact._axes(n);a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise ValueError('degenerate exact triangle during barycentric projection')
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)
def feature_for(bary,vertex_ids,vertex_keys,side,body,bone):
 zeros=[i for i,w in enumerate(bary) if w==0]
 if len(zeros)>=2:
  i=next(i for i,w in enumerate(bary) if w==1)
  return ('skin-vertex',int(vertex_ids[i])) if side=='skin' else ('bone-vertex',int(body),int(bone),vertex_keys[i])
 if len(zeros)==1:
  i,j=[x for x in range(3) if x!=zeros[0]]
  if side=='skin':
   vi,vj=int(vertex_ids[i]),int(vertex_ids[j]);lo,hi=sorted((vi,vj));t=bary[j] if vj==hi else bary[i]
   return ('skin-edge',lo,hi,t)
  ki,kj=vertex_keys[i],vertex_keys[j];lo,hi=sorted((ki,kj));t=bary[j] if kj==hi else bary[i]
  return ('bone-emitted-edge',int(body),int(bone),lo,hi,t)
 return None
def ratstr(x):return f'{x.numerator}/{x.denominator}'
def keyjson(x):
 if isinstance(x,Fraction):return ratstr(x)
 if isinstance(x,tuple):return [keyjson(v) for v in x]
 return x
def tri_degenerate(t,exact):return not any(exact._cross(exact._sub(t[1],t[0]),exact._sub(t[2],t[0])))
def graph_components(adj):
 unseen=set(adj);out=[]
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];c=[]
  while q:
   v=q.pop();c.append(v)
   for w in adj[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  out.append(sorted(c))
 return out
started=time.monotonic()
report={'schema':'numi.human.bank109_114.exact_rational_crossing_contours.v1','status':'STARTED','candidate_path':str(C),'candidate_sha256':None,'audit_path':str(A),'audit_sha256':None,'patch_component_paths':[str(PATCHES/'component-00-face-rows.npy'),str(PATCHES/'component-01-face-rows.npy')],'exact_predicate_path':str(EXACT),'exact_predicate_sha256':None,'scope':'Exact Float32-metre pair intersections and feature-canonical rational endpoint graphs; no geometry moved and no full audit.'}
try:
 if sha(C)!=CAND_PIN or sha(A)!=AUDIT_PIN:raise RuntimeError('pinned candidate/audit changed')
 report['candidate_sha256']=sha(C);report['audit_sha256']=sha(A);report['exact_predicate_sha256']=sha(EXACT)
 at=loadmod('bank109_exact_audit',AUDIT_TOOL);helper=loadmod('bank109_exact_helper',HELPER);exact=at.exact_module()
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P32=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 mask=(ids[:,0]==BODY)&(ids[:,1]==BONE);BID=ids[mask,3].astype(np.int64);BT=np.asarray(T[mask],np.float64)
 row_for_bone={int(fid):i for i,fid in enumerate(BID)}
 bone_local=[exact_tri(t) for t in BT]
 bone_vertex_keys=[[exact.float32_point_lattice_key(v) for v in tri] for tri in BT]
 skin_local=(P32.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
 audit_doc=json.loads(A.read_text());bank=next(x for x in audit_doc['banks'] if (int(x['body']),int(x['stable_bone_id']))==(BODY,BONE))
 pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']]
 if len(pairs)!=187:raise RuntimeError(f'expected 187 saved exact pair rows, got {len(pairs)}')
 patch_faces=[np.load(PATCHES/f'component-{i:02d}-face-rows.npy',allow_pickle=False).astype(np.int64) for i in (0,1)]
 face_to_component={int(f):i for i,fs in enumerate(patch_faces) for f in fs}
 if len(face_to_component)!=sum(len(x) for x in patch_faces):raise RuntimeError('patch face arrays overlap')
 crossing_skin=sorted({sf for sf,bf in pairs})
 missing_patch=[sf for sf in crossing_skin if sf not in face_to_component]
 if missing_patch:raise RuntimeError(f'exact crossing skin faces missing from patch arrays: {missing_patch}')
 endpoint_rows=[];pair_segments=[];degenerate_pairs=[];point_touches=[];nonsegment_pairs=[];exact_intersection_count=0
 for pi,(sf,bone_id) in enumerate(pairs):
  bi=row_for_bone[bone_id];ts=exact_tri(skin_local[F[sf]]);tb=bone_local[bi]
  sd=tri_degenerate(ts,exact);bd=tri_degenerate(tb,exact)
  if sd or bd:
   degenerate_pairs.append({'pair_index':pi,'skin_face_row':sf,'bone_face_row':bone_id,'skin_triangle_degenerate':sd,'bone_triangle_degenerate':bd,
    'skin_local_triangle_float64_m':skin_local[F[sf]].tolist(),'bone_local_triangle_float64_m':BT[bi].tolist()})
   continue
  raw=exact.triangle_intersection_points(ts,tb);uniq=sorted(set(raw))
  if not uniq:continue
  exact_intersection_count+=1
  if len(uniq)<2:
   point_touches.append({'pair_index':pi,'skin_face_row':sf,'bone_face_row':bone_id,'exact_point_local_m':[ratstr(x) for x in uniq[0]]})
   continue
  p0=uniq[0];direction=next(exact._sub(p,p0) for p in uniq[1:] if p!=p0)
  if not all(not any(exact._cross(direction,exact._sub(p,p0))) for p in uniq[1:]):
   nonsegment_pairs.append({'pair_index':pi,'skin_face_row':sf,'bone_face_row':bone_id,'unique_intersection_point_count':len(uniq),'points_local_m':[[ratstr(x) for x in p] for p in uniq]})
   continue
  axis=max(range(3),key=lambda k:max(p[k] for p in uniq)-min(p[k] for p in uniq))
  endpoints=[min(uniq,key=lambda p:p[axis]),max(uniq,key=lambda p:p[axis])]
  ids0=[]
  skin_ids=[int(v) for v in F[sf]]
  for ordinal,p in enumerate(endpoints):
   sb=barycentric(p,ts,exact);bb=barycentric(p,tb,exact)
   if min(sb)<0 or max(sb)>1 or min(bb)<0 or max(bb)>1:
    raise RuntimeError(f'exact endpoint outside triangle pair {(sf,bone_id)}')
   sfk=feature_for(sb,skin_ids,None,'skin',BODY,BONE)
   bfk=feature_for(bb,None,bone_vertex_keys[bi],'bone',BODY,BONE)
   if sfk is None and bfk is None:
    raise RuntimeError(f'exact endpoint has no canonical feature key {(sf,bone_id,ordinal)}')
   world=tuple(sum(sb[j]*frac(P32[skin_ids[j],k]) for j in range(3)) for k in range(3))
   ids0.append(len(endpoint_rows))
   endpoint_rows.append({'id':len(endpoint_rows),'pair_index':pi,'skin_face_row':sf,'bone_face_row':bone_id,'ordinal':ordinal,'local':p,'skin_bary':sb,'bone_bary':bb,'skin_feature':sfk,'bone_feature':bfk,'world':world})
  pair_segments.append({'pair_index':pi,'skin_face_row':sf,'bone_face_row':bone_id,'endpoint_ids':ids0})
 # Identity joins are exact source-feature joins only; no spatial tolerance or coordinate-bin welding.
 parent=list(range(len(endpoint_rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  ra,rb=find(a),find(b)
  if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
 feature_owner={}
 for r in endpoint_rows:
  for feat in (r['skin_feature'],r['bone_feature']):
   if feat is not None:
    if feat in feature_owner:union(r['id'],feature_owner[feat])
    else:feature_owner[feat]=r['id']
 roots={i:find(i) for i in range(len(endpoint_rows))};by_root=defaultdict(list)
 for r in endpoint_rows:by_root[roots[r['id']]].append(r)
 conflicts=[]
 for root,rows in by_root.items():
  if len({r['local'] for r in rows})!=1 or len({r['world'] for r in rows})!=1:
   conflicts.append({'root':root,'endpoint_record_ids':[r['id'] for r in rows],'local_points':[[ratstr(x) for x in r['local']] for r in rows],'world_points':[[ratstr(x) for x in r['world']] for r in rows]})
 per_component=[]
 for ci,faces in enumerate(patch_faces):
  faceset=set(map(int,faces)); segs=[s for s in pair_segments if s['skin_face_row'] in faceset]
  adj=defaultdict(list);edge_mult=Counter();collapsed=[]
  for s in segs:
   a,b=(roots[i] for i in s['endpoint_ids'])
   if a==b:collapsed.append({'pair_index':s['pair_index'],'skin_face_row':s['skin_face_row'],'bone_face_row':s['bone_face_row'],'endpoint_root':a});continue
   e=tuple(sorted((a,b)));edge_mult[e]+=1;adj[a].append(b);adj[b].append(a)
  comps=graph_components(adj)
  degrees=Counter(len(v) for v in adj.values());dups=[{'endpoint_roots':list(e),'segment_count':n} for e,n in edge_mult.items() if n>1]
  exact_crossing_faces=sorted({sf for sf,bf in pairs if sf in faceset})
  owned_pairs=[s for s in pairs if s[0] in faceset]
  contour_ok=bool(len(comps)==1 and adj and all(len(v)==2 for v in adj.values()) and not collapsed and not dups and not degenerate_pairs and not point_touches and not nonsegment_pairs and not conflicts)
  per_component.append({'component_id':ci,'patch_face_count':int(len(faces)),'patch_face_sha256':sha(PATCHES/f'component-{ci:02d}-face-rows.npy'),'crossing_skin_face_count':len(exact_crossing_faces),'crossing_skin_face_rows':exact_crossing_faces,'exact_pair_count':len(owned_pairs),'segment_count':len(segs),'unique_canonical_endpoint_count':len({x for s in segs for x in (roots[s['endpoint_ids'][0]],roots[s['endpoint_ids'][1]])}),'endpoint_degree_histogram':dict(degrees),'endpoint_graph_component_count':len(comps),'endpoint_graph_component_sizes':[len(x) for x in comps],'collapsed_segment_count':len(collapsed),'duplicate_contour_edge_count':len(dups),'duplicate_edges':dups,'collapsed_segments':collapsed,'simple_closed_contour':contour_ok,'contour_edge_roots':[{ 'pair_index':s['pair_index'],'skin_face_row':s['skin_face_row'],'bone_face_row':s['bone_face_row'],'endpoint_roots':[roots[i] for i in s['endpoint_ids']]} for s in segs]})
 report.update({'status':'EXACT_CONTOUR_GRAPH_COMPLETE','exact_pair_count':len(pairs),'exact_intersection_pair_count_nonempty':exact_intersection_count,'crossing_skin_face_count':len(crossing_skin),'crossing_skin_face_rows':crossing_skin,'crossing_bone_face_count':len({bf for sf,bf in pairs}),'valid_contour_segment_count':len(pair_segments),'raw_endpoint_record_count':len(endpoint_rows),'canonical_endpoint_key_count':len(by_root),'degenerate_triangle_pair_count':len(degenerate_pairs),'degenerate_triangle_pair_witnesses':degenerate_pairs,'point_only_pair_count':len(point_touches),'point_only_pair_witnesses':point_touches,'nonsegment_intersection_pair_count':len(nonsegment_pairs),'nonsegment_pair_witnesses':nonsegment_pairs,'canonical_endpoint_conflict_count':len(conflicts),'canonical_endpoint_conflict_witnesses':conflicts,'canonicalization':'Union by exact shared skin vertex/edge parameter or exact emitted bone vertex/edge parameter; rational Float32-metre predicates; no distance tolerance or coordinate binning.','per_component_graphs':per_component,'all_components_simple_closed_contours':bool(len(per_component)==2 and all(x['simple_closed_contour'] for x in per_component)),'endpoint_records':[{'id':r['id'],'root':roots[r['id']],'pair_index':r['pair_index'],'skin_face_row':r['skin_face_row'],'bone_face_row':r['bone_face_row'],'ordinal':r['ordinal'],'local_m':[ratstr(x) for x in r['local']],'world_from_skin_barycentric_m':[ratstr(x) for x in r['world']],'skin_barycentrics':[ratstr(x) for x in r['skin_bary']],'bone_barycentrics':[ratstr(x) for x in r['bone_bary']],'skin_feature':keyjson(r['skin_feature']),'bone_feature':keyjson(r['bone_feature'])} for r in endpoint_rows],'elapsed_seconds':time.monotonic()-started})
 (OUT/'exact-contour-graph.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('endpoint_records','degenerate_triangle_pair_witnesses','point_only_pair_witnesses','nonsegment_pair_witnesses','canonical_endpoint_conflict_witnesses')},indent=2,allow_nan=False))
except Exception as e:
 report.update({'status':'EXACT_CONTOUR_GRAPH_FAILED','failure':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
 (OUT/'exact-contour-graph.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(report,indent=2,allow_nan=False))

