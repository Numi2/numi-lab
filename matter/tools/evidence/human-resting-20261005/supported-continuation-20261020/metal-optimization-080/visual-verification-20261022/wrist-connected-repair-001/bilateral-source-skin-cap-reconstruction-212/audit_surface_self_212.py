#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import hashlib,json,time,sys
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.cardiac_cavity_intersections import _records,_audit_self_pairs_touching_faces
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
C=E/"bilateral-source-skin-cap-reconstruction-212/candidate-surface-welded-unadmitted.npz"
O=E/"bilateral-source-skin-cap-reconstruction-212"
R=O/"exact-self-212.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(C,allow_pickle=False);X=z["outer_world_actual007_m"];F=z["surface_faces"].astype(np.int64);touch=np.unique(z["changed_face_ids"].astype(np.int64))
assert C.exists() and np.all((touch>=0)&(touch<len(F)))
def rec(v,f):
 return _records([tuple(Fraction.from_float(float(x)) for x in p) for p in v],f.tolist())
start=time.monotonic()
result=_audit_self_pairs_touching_faces(rec(X,F),touch.tolist())
out={"physical_admission":False,"scope":"212 changed cap/transition surface faces against full assembled physical outer surface; self stage only, stops before tissue/bone if any self pair","candidate_sha256":sha(C),"cleanup_report_sha256":sha(O/"report-212.json"),"script_sha256":sha(__file__),"changed_face_count":int(len(touch)),"full_face_count":int(len(F)),"self":result,"self_pass":int(result.get("count",-1))==0,"wall_seconds":time.monotonic()-start}
R.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
