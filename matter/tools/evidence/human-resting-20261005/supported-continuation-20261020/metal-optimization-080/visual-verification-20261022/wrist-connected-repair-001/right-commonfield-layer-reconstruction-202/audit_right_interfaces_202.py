#!/usr/bin/env python3
from pathlib import Path
import sys
from collections import Counter
from fractions import Fraction
import ast,sys,json,time,hashlib
import numpy as np
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import _records,_audit_self_pairs_touching_faces,_prepare_surface_aabb,_query_surface_aabb,triangle_intersection_points
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');BASE=E/'right-commonfield-layer-reconstruction-202';OUT=BASE/(sys.argv[1] if len(sys.argv)>1 else 'attempt-002')
def records(v,f):return _records([tuple(Fraction.from_float(float(x)) for x in p) for p in v],np.asarray(f,np.int64).tolist())
def cross(ar,br):
 idx=_prepare_surface_aabb(br);hits=[];queries=0
 for a in ar:
  for _,b in _query_surface_aabb([a],idx):
   queries+=1;pts=triangle_intersection_points(a[0],b[0])
   if pts:hits.append({'face_a':int(a[3]),'face_b':int(b[3]),'points':[[float(x) for x in p] for p in pts]})
 return {'aabb_candidates':queries,'intersection_pairs':len(hits),'witnesses':hits}
report={'physical_admission':False,'self':{},'cross':{}};rr={};start=time.monotonic()
for role in ['I','D']:
 z=np.load(OUT/f'right-{role}-interface-202.npz');v=z['vertices_world_f32_m'];f=z['faces'];rr[role]=records(v,f)
 q=_audit_self_pairs_touching_faces(rr[role],list(range(len(f))))
 report['self'][role]=q;print(json.dumps({'role':role,'self':q}),flush=True)
 # Save immediately, including failed exact witnesses.
 (OUT/'interface-exact-audit-202.json').write_text(json.dumps(report,indent=2)+'\n')
# Cross-field overlap and source outer skin are separate from self topology.
z=np.load(OUT/'right-commonfield-calibrated-202.npz');rr['O']=records(z['points_world_f32_m'],z['outer_faces'])
for a,b in [('I','D'),('O','I'),('O','D')]:
 q=cross(rr[a],rr[b]);report['cross'][a+'_'+b]=q;print(json.dumps({'pair':[a,b],'intersection_pairs':q['intersection_pairs']}),flush=True);(OUT/'interface-exact-audit-202.json').write_text(json.dumps(report,indent=2)+'\n')
report['wall_seconds']=time.monotonic()-start
(OUT/'interface-exact-audit-202.json').write_text(json.dumps(report,indent=2)+'\n')
