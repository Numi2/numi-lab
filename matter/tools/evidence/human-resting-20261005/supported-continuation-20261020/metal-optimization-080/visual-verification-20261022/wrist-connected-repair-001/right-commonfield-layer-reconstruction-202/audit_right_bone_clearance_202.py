#!/usr/bin/env python3
from pathlib import Path
import sys
from collections import Counter
from fractions import Fraction
import ast,sys,json,time,hashlib,math
import numpy as np
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import _records,_prepare_surface_aabb,_query_surface_aabb,triangle_intersection_points
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');A=E/'ankle-parent-tet-refinement-026';BASE=E/'right-commonfield-layer-reconstruction-202';OUT=BASE/(sys.argv[1] if len(sys.argv)>1 else 'attempt-002')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ns={'__name__':'frozen_bone_clearance','MAX_TET_VOL':1.5e-7}
for path in [A/'build_common_volume_126.py',A/'generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py']:
 tree=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),str(path),'exec'),ns)
ns.update(BANK=E/'registered-surrounding-tissue-native007-190/registered-tissue-banks.npz',INV=E/'registered-surrounding-tissue-native007-190/inventory.json',POSE=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json',sha=sha)
z=np.load(OUT/'right-commonfield-calibrated-202.npz');sources={'O':(z['points_world_f32_m'],z['outer_faces'])}
for role in ['I','D']:
 zz=np.load(OUT/f'right-{role}-interface-202.npz');sources[role]=(zz['vertices_world_f32_m'],zz['faces'])
banks=ns['classify_banks']('right',sources['O'][0][sources['O'][1]].reshape(-1,3))
def records(v,f):return _records([tuple(Fraction.from_float(float(x)) for x in p) for p in v],np.asarray(f,np.int64).tolist())
indices=[(m,V,F,_prepare_surface_aabb(records(V,F))) for m,V,F,_ in banks]
start=time.monotonic();report={'physical_admission':False,'roles':{}}
for role,(V,F) in sources.items():
 hits=[];queries=0
 for ar in records(V,F):
  for m,VB,FB,idx in indices:
   for _,br in _query_surface_aabb([ar],idx):
    queries+=1; pts=triangle_intersection_points(ar[0],br[0])
    if pts:hits.append({'surface_face':int(ar[3]),'bank':m['key'],'stable_id':m['stable_id'],'body_indices':m['body_indices'],'bank_face':int(br[3]),'points':[[float(x) for x in p] for p in pts]})
 report['roles'][role]={'aabb_queries':queries,'bone_crossing_pairs':len(hits),'witnesses':hits};print(json.dumps({'role':role,'bone_crossing_pairs':len(hits),'queries':queries}),flush=True)
 (OUT/'interface-bone-clearance-202.json').write_text(json.dumps(report,indent=2)+'\n')
# Whole-body ID group winding is independent from the triangle-pair audit.
groups={}
for m,V,F,_ in banks:groups.setdefault(tuple(sorted(m['body_indices'])),[]).append((V,F))
for role,(V,F) in sources.items():
 contained=[];used=np.unique(F)
 for body,items in groups.items():
  vv=[];ff=[];offset=0
  for v,f in items:vv.append(v);ff.append(f+offset);offset+=len(v)
  vv=np.vstack(vv);ff=np.vstack(ff);vv,iv=np.unique(vv,axis=0,return_inverse=True);ff=iv[ff]
  closed,top=ns['consistent_closed'](ff)
  if not closed:raise ValueError('closed cached anatomy union no longer closed')
  mask=np.all(V[used]>=vv.min(0),axis=1)&np.all(V[used]<=vv.max(0),axis=1);test=used[mask]
  if len(test):
   winding=ns['solid_angle_winding'](V[test],vv,ff);bad=test[np.abs(winding)>.5]
   contained.extend({'vertex':int(i),'body_indices':list(body)} for i in bad)
 report['roles'][role]['vertices_inside_bone']=contained
report['pass']=all(r['bone_crossing_pairs']==0 and not r['vertices_inside_bone'] for r in report['roles'].values());report['wall_seconds']=time.monotonic()-start
(OUT/'interface-bone-clearance-202.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'pass':report['pass'],'wall_seconds':report['wall_seconds']}),flush=True)
if not report['pass']:raise SystemExit('bone clearance failed')
