#!/usr/bin/env python3
import csv, hashlib, json, math, sys, time
from fractions import Fraction
from collections import Counter
from pathlib import Path
import numpy as np
E=Path(r'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=Path(r'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/ankle-parent-tet-refinement-026/generated-223-bilateral-source-oriented-cap-actual007-frame')
CAND=OUT/'candidate-222.npz'
INV=E/'registered-surrounding-tissue-native007-190/inventory.json'
BANK=E/'registered-surrounding-tissue-native007-190/registered-tissue-banks.npz'
FACECSV=E/'elbow-refinement-current-main-024-final080-candidate-source-frames-001/generated/faces.csv'
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import (
    _records, _prepare_surface_aabb, _query_surface_aabb,
    _audit_self_pairs_touching_faces, triangle_intersection_points,
    _allowed_shared_point
)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def canonical_edges(F):
 c=Counter()
 for tri in F:
  a,b,c0=map(int,tri)
  for x,y in ((a,b),(b,c0),(c0,a)):c[tuple(sorted((x,y)))]+=1
 return c
def parse_face_csv(p):
 out=[]
 with open(p,newline='') as f:
  r=csv.DictReader(f)
  for row in r:out.append([int(row[k]) for k in ('a','b','c')])
 return np.asarray(out,dtype=np.int64)
def approxpt(p):
 return [float(x) for x in p]
t0=time.monotonic()
pins={'candidate_sha256':sha(CAND),'inventory_sha256':sha(INV),'registered_bank_npz_sha256':sha(BANK),'baseline_d36_faces_sha256':sha(FACECSV)}
z=np.load(CAND,allow_pickle=False)
V64=z['outer_world_actual007_m']; F=z['surface_faces'].astype(np.int64)
kind=z['surface_face_kind'].astype(np.uint8); side=z['surface_face_source_side'].astype(np.uint8)
parent=z['surface_face_parent_surface_ids']; stable=z['surface_face_detailed_source_ids']; oldrow=z['surface_face_original_d36_row']
V=V64.astype(np.float32).astype(np.float64)
report={'schema':'numi.human.candidate223-actual007-f32-exact-surface-bone-audit.v1','pins':pins,'frame':'candidate 223 geometry expressed directly in the actual native007 registered world frame; vertices rounded to Float32 before exact predicates','status':'running'}
# Surface connectivity and preservation.
ec=canonical_edges(F); open_edges={e for e,n in ec.items() if n==1}; nonmanifold={e:n for e,n in ec.items() if n>2}
same_direction=0
dirs=Counter()
for tri in F:
 a,b,c=map(int,tri)
 for x,y in ((a,b),(b,c),(c,a)):dirs[tuple(sorted((x,y)))]+=1 if x<y else -1
same_direction=sum(1 for e,n in ec.items() if n==2 and dirs[e]!=0)
area2=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)
baseF=parse_face_csv(FACECSV); baseec=canonical_edges(baseF); base_open={e for e,n in baseec.items() if n==1}
oldids=np.flatnonzero((kind==0)&(oldrow>=0))
assert len(oldids)==4229
for i in oldids:
 if not np.array_equal(F[i],baseF[oldrow[i]]):raise RuntimeError(f'unmodified D36 face changed row {oldrow[i]}')
report['surface_topology']={'vertices':len(V),'faces':len(F),'kind_counts':{str(int(k)):int((kind==k).sum()) for k in np.unique(kind)},'face_side_counts':{str(int(s)):int((side==s).sum()) for s in np.unique(side)},'edge_incidence_histogram':{str(k):int(v) for k,v in Counter(ec.values()).items()},'open_boundary_edges':len(open_edges),'all_open_edges_are_original_d36_boundary':open_edges<=base_open,'original_d36_open_boundary_edges':len(base_open),'new_nonmanifold_edges':len(nonmanifold),'same_direction_two-face_edges':same_direction,'f32_zero_area_faces':int(np.count_nonzero(area2==0)),'min_f32_area2_m2':float(area2.min()),'unchanged_faces_bitwise_preserved':True}
if open_edges != base_open: report['surface_topology']['open_boundary_exact_match_original']=False
else: report['surface_topology']['open_boundary_exact_match_original']=True
if len(open_edges)!=45 or open_edges!=base_open or nonmanifold or same_direction or np.any(area2==0):
 raise RuntimeError('candidate topology changed, lost legacy boundary, or has degenerate/nonmanifold faces')
# Exact self-intersections involving all newly authored faces, queried against the complete whole-body surface.
records=_records([tuple(Fraction.from_float(float(x)) for x in tri) for tri in V],F.tolist())
changed=np.flatnonzero(kind!=0).tolist()
selfres=_audit_self_pairs_touching_faces(records,changed)
report['self_surface']={'queried_faces':len(changed),'complete_surface_faces':len(F),'exact_aabb_candidate_pairs':selfres['aabb_candidate_pairs'],'allowed_shared_vertex_or_edge_pairs':selfres['allowed_shared_vertex_or_edge_pairs'],'nonlocal_exact_intersection_count':selfres['count'],'first_pairs':selfres['triangle_pairs'][:40]}
if selfres['count']:
 report['status']='rejected-self-intersection'
 report['elapsed_s']=time.monotonic()-t0
 (OUT/'audit-223.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'stage':'self','count':selfres['count'],'elapsed_s':report['elapsed_s']}),flush=True)
 raise SystemExit(2)
print(json.dumps({'stage':'self','count':0,'aabb_pairs':selfres['aabb_candidate_pairs'],'seconds':time.monotonic()-t0}),flush=True)
# Exact candidate O against all spatially overlapping registered anatomy banks, with AABB face pruning.
inventory=json.loads(INV.read_text()); bankz=np.load(BANK,allow_pickle=False)
boneout={}
for s in (1,2):
 patchids=np.flatnonzero(side==s)
 patchrows=[records[i] for i in patchids]
 patchverts=np.unique(F[patchids])
 lo=V[patchverts].min(0)-1e-6; hi=V[patchverts].max(0)+1e-6
 metas=[]
 for i,m in enumerate(inventory['banks']):
  if m['kind']!='anatomy':continue
  mn=np.asarray(m['aabb_min_m'],float); mx=np.asarray(m['aabb_max_m'],float)
  if np.all(mx>=lo) and np.all(mn<=hi):metas.append((i,m))
 side_result={'candidate_side':int(s),'candidate_faces':len(patchids),'candidate_bbox_min_m':lo.tolist(),'candidate_bbox_max_m':hi.tolist(),'aabb_overlapping_anatomy_banks':len(metas),'anatomy_triangles_in_selected_banks':int(sum(m['triangles'] for _,m in metas)),'tested_aabb_pairs':0,'exact_intersection_pairs':0,'per_bank':[],'examples':[]}
 for i,m in metas:
  vkey=f"bank_{i:04d}_vertices"; fkey=f"bank_{i:04d}_faces"
  Bv=bankz[vkey].astype(np.float32).astype(np.float64); Bf=bankz[fkey].astype(np.int64)
  # Tighter face ROI against the changed side patch.
  tri=Bv[Bf]
  blo=tri.min(axis=1); bhi=tri.max(axis=1)
  candidate_lo=V[F[patchids]].min(axis=1)-1e-6
  candidate_hi=V[F[patchids]].max(axis=1)+1e-6
  qmask=np.all(candidate_hi>=np.asarray(m['aabb_min_m'])-1e-6,axis=1) if False else None
  # Use the helper's exact closed-box tree for candidate pair enumeration.
  br=_records([tuple(Fraction.from_float(float(x)) for x in t) for t in Bv],Bf.tolist())
  # avoid re-indexing global candidate IDs: retain original face IDs in record tuples
  candidates=0; hits=[]
  bidx=_prepare_surface_aabb(br)
  for sr in patchrows:
   # AABB-query one candidate triangle against this bank. The single row keeps original face id.
   for _,brrow in _query_surface_aabb([sr],bidx):
    candidates+=1
    pts=triangle_intersection_points(sr[0],brrow[0])
    if pts:
     hits.append({'surface_face':int(sr[3]),'surface_kind':int(kind[sr[3]]),'surface_parent_id':int(parent[sr[3]]),'surface_detailed_source_id':int(stable[sr[3]]),'bone_bank':m['key'],'bone_stable_id':int(m['stable_id']),'bone_body_indices':m['body_indices'],'bone_face':int(brrow[3]),'intersection_points_m':[approxpt(p) for p in pts[:4]]})
  side_result['tested_aabb_pairs']+=candidates
  side_result['exact_intersection_pairs']+=len(hits)
  if hits:
   side_result['per_bank'].append({'key':m['key'],'stable_id':m['stable_id'],'body_indices':m['body_indices'],'kind':m['kind'],'triangles':m['triangles'],'aabb_pairs':candidates,'intersections':len(hits),'examples':hits[:8]})
   side_result['examples'].extend(hits[:min(20,len(hits))])
  print(json.dumps({'stage':'bone','side':int(s),'bank':m['key'],'candidates':candidates,'hits':len(hits),'elapsed_s':time.monotonic()-t0}),flush=True)
 boneout[str(s)]=side_result
report['registered_anatomy_bone_intersections']=boneout
total=sum(v['exact_intersection_pairs'] for v in boneout.values())
report['status']='pass-exact-surface-and-bone' if total==0 else 'rejected-bone-intersections'
report['elapsed_s']=time.monotonic()-t0
(OUT/'audit-223.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'stage':'done','bone_intersections':total,'self_intersections':selfres['count'],'legacy_open_edges':len(open_edges),'elapsed_s':report['elapsed_s'],'report':str(OUT/'audit-223.json')}),flush=True)
sys.exit(0 if total==0 else 3)
