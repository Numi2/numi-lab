#!/usr/bin/env python3
"""Source-pose exact nearby tissue intersections, without interface reclassification."""
from pathlib import Path
import numpy as np,json,time,hashlib,sys
from fractions import Fraction
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import _records,_prepare_surface_aabb,_query_surface_aabb,triangle_intersection_points
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');S=E/'right-commonfield-layer-reconstruction-202/attempt-004';O=E/'right-all-tissue-clearance-209';O.mkdir(exist_ok=False);P=E/'registered-surrounding-tissue-native007-190';inv=json.loads((P/'inventory.json').read_text());bankpath=P/'registered-tissue-banks.npz';bank=np.load(bankpath);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(bankpath)==inv['cache_sha256'];z=np.load(S/'right-commonfield-calibrated-202.npz');X=z['points_world_f32_m'];F=z['outer_faces'];used=X[np.unique(F)];lo=used.min(0);hi=used.max(0);near=[b for b in inv['banks'] if b['kind']!='anatomy' and np.all(np.array(b['aabb_max_m'])>=lo) and np.all(np.array(b['aabb_min_m'])<=hi)]
def records(v,f):return _records([tuple(Fraction.from_float(float(x)) for x in p) for p in v],f.tolist())
indices=[(b,_prepare_surface_aabb(records(bank[b['key']+'_vertices'],bank[b['key']+'_faces']))) for b in near];report={'physical_admission':False,'scope':'actual007 registered source pose only; exact triangle intersections, not all containment or intended-interface certification','bank_sha256':sha(bankpath),'inventory_sha256':sha(P/'inventory.json'),'source_receipt_sha256':sha(S/'source-geometry-receipt-202.json'),'script_sha256':sha(__file__),'nearby_nonbone_bank_count':len(near),'roles':{}};start=time.monotonic()
for role in ['O','I','D']:
 if role!='O':zz=np.load(S/f'right-{role}-interface-202.npz');X=zz['vertices_world_f32_m'];F=zz['faces']
 hit=[];queries=0;bybank={}
 for a in records(X,F):
  for b,idx in indices:
   for _,c in _query_surface_aabb([a],idx):
    queries+=1;pts=triangle_intersection_points(a[0],c[0])
    if pts:
     key=b['key'];bybank[key]=bybank.get(key,0)+1
     hit.append({'surface_face':int(a[3]),'bank':key,'kind':b['kind'],'layer':b['layer'],'stable_id':b['stable_id'],'body_indices':b['body_indices'],'bank_face':int(c[3]),'points':[[float(y) for y in p] for p in pts]})
 report['roles'][role]={'unclassified_intersection_pairs':len(hit),'aabb_candidates':queries,'by_bank':bybank,'witnesses':hit};(O/'report-209.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'role':role,'unclassified_intersections':len(hit),'queries':queries,'banks':bybank}),flush=True)
report['wall_seconds']=time.monotonic()-start;report['zero_nonbone_intersections']=all(r['unclassified_intersection_pairs']==0 for r in report['roles'].values());(O/'report-209.json').write_text(json.dumps(report,indent=2)+'\n')
