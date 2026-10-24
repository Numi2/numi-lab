#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,time,hashlib,sys
from fractions import Fraction
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import _records,_prepare_surface_aabb,_query_surface_aabb,triangle_intersection_points
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'bilateral-source-cut-plane-271'
P=E/'registered-surrounding-tissue-native007-190'
inv=json.loads((P/'inventory.json').read_text());bankpath=P/'registered-tissue-banks.npz';bank=np.load(bankpath,allow_pickle=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(bankpath)==inv['cache_sha256']
z=np.load(OUT/'candidate-271-max-cut-distance-bilateral-collar.npz')
X=z['outer_world_actual007_m'].astype(np.float32).astype(np.float64);F=z['surface_faces'].astype(np.int64)
kind=z['surface_face_kind'];d36=z['surface_face_original_d36_row']
face_csv=E/'elbow-refinement-current-main-024-final080-candidate-source-frames-001/generated/faces.csv'
with open(face_csv,newline='') as f: baseF=np.asarray([[int(r[k]) for k in ('a','b','c')] for r in __import__('csv').DictReader(f)],dtype=np.int64)
unchanged=np.zeros(len(F),dtype=bool)
for i in np.flatnonzero((kind==0)&(d36>=0)):
 unchanged[i]=np.array_equal(F[i],baseF[int(d36[i])])
changed=np.flatnonzero(~unchanged)
records=_records([tuple(Fraction.from_float(float(x)) for x in p) for p in X],F[changed].tolist())
used=X[np.unique(F[changed])];lo=used.min(0)-1e-6;hi=used.max(0)+1e-6
near=[b for b in inv['banks'] if b['kind']!='anatomy' and np.all(np.array(b['aabb_max_m'])>=lo) and np.all(np.array(b['aabb_min_m'])<=hi)]
indices=[]
for b in near:
 bv=bank[b['key']+'_vertices'].astype(np.float32).astype(np.float64);bf=bank[b['key']+'_faces'].astype(np.int64)
 br=_records([tuple(Fraction.from_float(float(x)) for x in p) for p in bv],bf.tolist())
 indices.append((b,_prepare_surface_aabb(br)))
report={'schema':'numi.human.candidate271-actual007-f32-nearby-nonbone-tissue-audit.v1','physical_admission':False,
 'scope':'changed registered outer triangles versus all spatially overlapping non-anatomy source-pose banks; exact triangle intersection predicate only',
 'candidate_sha256':sha(OUT/'candidate-271-max-cut-distance-bilateral-collar.npz'),
 'bone_self_audit_sha256':sha(OUT/'audit-271.json'),'bank_sha256':sha(bankpath),'inventory_sha256':sha(P/'inventory.json'),
 'changed_face_count':len(changed),'candidate_changed_bbox_min_m':lo.tolist(),'candidate_changed_bbox_max_m':hi.tolist(),
 'nearby_nonbone_bank_count':len(near),'roles':{}}
hits=[];queries=0;perbank={}
t0=time.monotonic()
for sr in records:
 for meta,idx in indices:
  for _,br in _query_surface_aabb([sr],idx):
   queries+=1;pts=triangle_intersection_points(sr[0],br[0])
   if pts:
    face=int(changed[int(sr[3])]);k=int(kind[face])
    item={'surface_face':face,'surface_kind':k,'surface_side':int(z['surface_face_source_side'][face]),
      'parent_surface_id':int(z['surface_face_parent_surface_ids'][face]),'detailed_source_stable_id':int(z['surface_face_detailed_source_ids'][face]),
      'bank':meta['key'],'bank_kind':meta['kind'],'bank_layer':meta.get('layer'),'bank_stable_id':meta['stable_id'],
      'body_indices':meta['body_indices'],'bank_face':int(br[3]),'points_m':[[float(v) for v in p] for p in pts]}
    hits.append(item);perbank[meta['key']]=perbank.get(meta['key'],0)+1
report['exact_nonbone_intersection_pairs']=len(hits);report['aabb_candidate_pairs']=queries
report['intersections_by_bank']=perbank;report['witnesses']=hits[:100]
report['zero_nonbone_intersections']=len(hits)==0;report['elapsed_s']=time.monotonic()-t0
(OUT/'tissue-audit-271.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'pass' if not hits else 'reject','changed_faces':len(changed),'nearby_banks':len(near),'aabb_pairs':queries,'hits':len(hits),'by_bank':perbank,'seconds':report['elapsed_s'],'report':str(OUT/'tissue-audit-271.json')},indent=2),flush=True)
sys.exit(0 if not hits else 3)
