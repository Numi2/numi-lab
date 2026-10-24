#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=R/"global-skin-crossing-patches-012"
O=R/"global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz"
INV=R/"registered-surrounding-tissue-native013-191/inventory.json"
A=np.load(D/"connected-physical-O-patches-012.npz",allow_pickle=False)
x=np.load(O,allow_pickle=False)["points_world_m"]
banks=json.loads(INV.read_text())["banks"]
rows=[]
for pid in range(len(A["patch_face_offsets"])-1):
 f0,f1=map(int,A["patch_face_offsets"][pid:pid+2]);n0,n1=map(int,A["patch_node_offsets"][pid:pid+2]);p0,p1=map(int,A["patch_pair_offsets"][pid:pid+2])
 nodes=A["patch_node_ids"][n0:n1];pair_banks=A["pair_bank_ids"][p0:p1]
 counts={}
 for b in pair_banks:counts[int(b)]=counts.get(int(b),0)+1
 typed=[]
 for bi,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
  m=banks[bi]
  typed.append({"bank_id":bi,"key":m["key"],"stable_id":int(m["stable_id"]),"kind":m["kind"],"layer":int(m["layer"]),"body_indices":[int(v) for v in m["body_indices"]],"pair_rows":n})
 q=x[nodes]
 rows.append({"patch_id":pid,"O_face_count":f1-f0,"O_node_count":n1-n0,"canonical_pair_count":p1-p0,"O_AABB_min_m":q.min(0).tolist(),"O_AABB_max_m":q.max(0).tolist(),"typed_banks_by_pair_count":typed})
out={"schema":"numi.human.global-skin-crossing-patches.patch-bounds.v1","status":"extraction_label_and_bounds_only","scope":"Each label is a registered typed-bank key/stable ID with layer and body-index owner IDs; coordinates are original physical-O patch bounds.","patches":rows}
(D/"patch-bounds-summary-012.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(D/"patch-bounds-summary-012.json"),"patches":len(rows),"labels":[{"patch_id":r["patch_id"],"banks":[b["key"] for b in r["typed_banks_by_pair_count"]]} for r in rows]},indent=2))
