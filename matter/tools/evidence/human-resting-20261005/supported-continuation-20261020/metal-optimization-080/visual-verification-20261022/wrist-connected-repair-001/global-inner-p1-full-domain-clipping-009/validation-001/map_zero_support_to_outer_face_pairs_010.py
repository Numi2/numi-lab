#!/usr/bin/env python3
"""Map validated zero-capacity P1 supports to physical O faces and exact Float32 test pairs."""
from __future__ import annotations
import hashlib,json,sys,time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
D=E/'global-inner-p1-full-domain-clipping-009'
V=D/'validation-001'
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.cardiac_cavity_intersections import triangle_intersection_points

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
  index_path=V/'zero-capacity-witness-index-010.npz'
  banks_path=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
  inventory_path=E/'registered-surrounding-tissue-native013-191/inventory.json'
  mesh_path=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
  started=time.monotonic()
  with np.load(index_path,allow_pickle=False) as z: ix={k:z[k] for k in z.files}
  inv=json.loads(inventory_path.read_text())['banks']
  with np.load(mesh_path,allow_pickle=False) as z:
    X=np.asarray(z['points_world_m'],np.float32)
    T=np.asarray(z['tetrahedra'],np.int64)
    O=np.asarray(z['physical_outer_faces'],np.int64)
  # Map physical O facets to adjacent tetrahedra by exact 3-node topology.
  outer_by_nodes=defaultdict(list)
  for of,tri in enumerate(O): outer_by_nodes[tuple(sorted(map(int,tri)))].append(int(of))
  tet_outer=defaultdict(list)
  for ti,tet in enumerate(T):
    ids=list(map(int,tet))
    for omit in range(4):
      tri=tuple(sorted(ids[:omit]+ids[omit+1:]))
      for of in outer_by_nodes.get(tri,()):tet_outer[int(ti)].append(int(of))
  # Unique zero (bank, local face, tet) keys are the source of pair candidates.
  witness_pair_tets=defaultdict(set)
  for b,lf,tet in zip(ix['key_bank_ids'],ix['key_local_face_ids'],ix['key_tet_ids']):
    for of in tet_outer.get(int(tet),()): witness_pair_tets[(int(b),int(lf),int(of))].add(int(tet))
  pairs=sorted(witness_pair_tets)
  bankdata=np.load(banks_path,allow_pickle=False); bank_cache={}
  exact_classes=[]; hit_offsets=[0]; hit_xyz=[]; witness_offsets=[0]; witness_tets=[]
  pair_rows=[]; aabb_candidates=0; degenerate=0
  for b,lf,of in pairs:
    bm=inv[b]
    if b not in bank_cache:
      bank_cache[b]=(np.asarray(bankdata[f"bank_{b:04d}_faces"],np.int64),
                     np.asarray(bankdata[f"bank_{b:04d}_vertices"],np.float32))
    bf,bv=bank_cache[b]
    atri=bv[bf[lf]]
    btri=X[O[of]]
    # Float32 AABB rejection, then exact rational triangle predicate on those packed values.
    if np.any(np.max(atri,axis=0)<np.min(btri,axis=0)) or np.any(np.max(btri,axis=0)<np.min(atri,axis=0)):
      cls='aabb_clear'; hits=[]
    else:
      aabb_candidates+=1
      af=[tuple(Fraction.from_float(float(x)) for x in row) for row in atri]
      bfq=[tuple(Fraction.from_float(float(x)) for x in row) for row in btri]
      # Reject exact zero-area triangles explicitly; don't call them clear.
      def sub(a,c):return tuple(a[k]-c[k] for k in range(3))
      def cross(a,c):return (a[1]*c[2]-a[2]*c[1],a[2]*c[0]-a[0]*c[2],a[0]*c[1]-a[1]*c[0])
      na=cross(sub(af[1],af[0]),sub(af[2],af[0])); nb=cross(sub(bfq[1],bfq[0]),sub(bfq[2],bfq[0]))
      if na==(0,0,0) or nb==(0,0,0):
        degenerate+=1; cls='degenerate_input';hits=[]
      else:
        hits=list(dict.fromkeys(triangle_intersection_points(af,bfq)))
        if not hits: cls='clear'
        elif len(hits)==1: cls='point_contact'
        else:
          parallel=cross(na,nb)==(0,0,0)
          coplanar=parallel and sum(na[k]*(bfq[0][k]-af[0][k]) for k in range(3))==0
          if coplanar:
            def on_edge(p,tri):
              for j in range(3):
                a,c=tri[j],tri[(j+1)%3]; d=sub(c,a); q=sub(p,a)
                if cross(q,d)==(0,0,0) and all(min(a[k],c[k])<=p[k]<=max(a[k],c[k]) for k in range(3)):
                  return True
              return False
            boundary=all(on_edge(p,af) and on_edge(p,bfq) for p in hits)
            cls='coplanar_edge_contact' if len(hits)==2 and boundary else 'coplanar_area_overlap'
          else: cls='transverse_segment'
    pair_rows.append((b,lf,int(bm['stable_id']),int(bm['layer']),of,len(witness_pair_tets[(b,lf,of)])))
    for p in hits:hit_xyz.append([float(x) for x in p])
    hit_offsets.append(len(hit_xyz))
    witness_tets.extend(sorted(witness_pair_tets[(b,lf,of)]));witness_offsets.append(len(witness_tets))
    exact_classes.append(cls)
  pair_rows=np.asarray(pair_rows,np.int64).reshape(-1,6)
  pair_path=V/'zero-support-physical-O-face-pairs-010.npz'
  tmp=pair_path.with_suffix('.tmp.npz')
  cls_names=['aabb_clear','clear','point_contact','transverse_segment','coplanar_edge_contact','coplanar_area_overlap','degenerate_input']
  cls_code=np.asarray([cls_names.index(x) for x in exact_classes],np.uint8)
  np.savez_compressed(tmp,
    bank_ids=pair_rows[:,0].astype(np.uint16) if len(pair_rows) else np.empty(0,np.uint16),
    bank_local_face_ids=pair_rows[:,1].astype(np.uint32) if len(pair_rows) else np.empty(0,np.uint32),
    bank_stable_ids=pair_rows[:,2].astype(np.uint32) if len(pair_rows) else np.empty(0,np.uint32),
    bank_layers=pair_rows[:,3].astype(np.uint8) if len(pair_rows) else np.empty(0,np.uint8),
    physical_O_face_ids=pair_rows[:,4].astype(np.uint32) if len(pair_rows) else np.empty(0,np.uint32),
    zero_support_tet_count=pair_rows[:,5].astype(np.uint32) if len(pair_rows) else np.empty(0,np.uint32),
    class_code=cls_code,class_names=np.asarray(cls_names),
    exact_hit_offsets=np.asarray(hit_offsets,np.int64),exact_hit_points_m=np.asarray(hit_xyz,np.float64).reshape(-1,3),
    witness_tet_offsets=np.asarray(witness_offsets,np.int64),witness_tet_ids=np.asarray(witness_tets,np.uint32))
  tmp.replace(pair_path)
  counts={k:int(sum(x==k for x in exact_classes)) for k in cls_names}
  bank_counts=[]
  for b,m in enumerate(inv):
    rowids=np.flatnonzero(pair_rows[:,0]==b) if len(pair_rows) else np.empty(0,np.int64)
    rows=pair_rows[rowids]
    row_classes=[exact_classes[int(i)] for i in rowids]
    bank_counts.append({'bank_id':b,'key':m.get('key'),'kind':m.get('kind'),'stable_id':m.get('stable_id'),'layer':m.get('layer'),
       'candidate_pairs':len(rows),'aabb_candidates':sum(x!='aabb_clear' for x in row_classes),
       'exact_hit_pairs':sum(x not in ('aabb_clear','clear','degenerate_input') for x in row_classes)})
  report={'schema':'numi.human.zero-support-to-physical-O-face-pairs.v1','status':'DIAGNOSTIC_EXACT_FLOAT32_PAIR_SCREEN',
    'source_index_path':str(index_path),'source_index_sha256':sha(index_path),'source_index_json_sha256':sha(index_path.with_suffix('.json')),
    'bank_payload_sha256':sha(banks_path),'inventory_sha256':sha(inventory_path),'mesh_sha256':sha(mesh_path),
    'input_scope':{'banks':len(inv),'zero_support_unique_bank_face_tet_keys':int(len(ix['key_bank_ids'])),'physical_O_faces':int(len(O)),'mesh_nodes':int(len(X)),'mesh_tets':int(len(T))},
    'mapping':{'unique_bank_face_to_physical_O_face_pairs':len(pairs),'pairs_with_zero_support_tet':len(pairs),'zero_support_bank_tet_keys':int(len(ix['key_bank_ids'])),'candidate_tets_per_pair_max':max((len(v) for v in witness_pair_tets.values()),default=0)},
    'exact_float32_predicate':{'coordinates':'registered 191 bank Float32 vertices and candidate P1 points rounded to Float32 before exact rational triangle test','aabb_candidates':aabb_candidates,'classes':counts,'degenerate_input_pairs':degenerate,
       'class_meanings':{'point_contact':'exact triangle intersection has one point','transverse_segment':'noncoplanar triangles intersect in a segment','coplanar_edge_contact':'two coplanar intersection points, each lying on the boundary of both triangles','coplanar_area_overlap':'coplanar intersection not limited to a common boundary segment','clear':'AABB overlap but exact triangle predicate is empty','aabb_clear':'Float32 AABBs disjoint; exact predicate not called','degenerate_input':'one packed triangle has zero exact area'}},
    'pair_npz':str(pair_path),'pair_npz_sha256':sha(pair_path),'pair_npz_bytes':pair_path.stat().st_size,'bank_summaries':bank_counts,
    'elapsed_seconds':time.monotonic()-started,'limitations':['pair generation only follows zero-capacity tet supports onto physical O facets that are actual faces of those same tetrahedra','this screen is diagnostic; it does not establish solver feasibility or physical/native admission']}
  report_path=V/'zero-support-physical-O-face-pairs-010.json';report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  report['report_path']=str(report_path);report['report_sha256']=sha(report_path)
  print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
