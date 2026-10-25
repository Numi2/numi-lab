#!/usr/bin/env python3
"""Read-only global face-incidence and identity audit for run-001."""
from collections import Counter
import hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/"retessellation-001/volume-cavity-001/oid-cavity-replacement.npz"
CANDIDATE=HERE/"run-001/candidate-tetrahedra.npz"
OUTPUT=HERE/"run-001/topology-invariants.json"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def face_counts(tets):
    out=Counter()
    for tet in tets:
        a,b,c,d=map(int,tet)
        for f in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            out[tuple(sorted(f))]+=1
    return out

def main():
    src=np.load(SOURCE); cand=np.load(CANDIDATE)
    t=np.asarray(cand["candidate_tetrahedra"],dtype=np.int64)
    src_t=np.asarray(src["candidate_tetrahedra"],dtype=np.int64)
    counts=face_counts(t); base_counts=face_counts(src_t)
    boundary={f for f,n in counts.items() if n==1}
    base_boundary={f for f,n in base_counts.items() if n==1}
    cell_keys=[tuple(sorted(map(int,row))) for row in t]
    material_counts=Counter(map(int,cand["candidate_material_ids"]))
    report={
      "source_sha256":sha(SOURCE),"candidate_sha256":sha(CANDIDATE),
      "cells":len(t),"duplicate_cells":len(cell_keys)-len(set(cell_keys)),
      "max_face_incidence":max(counts.values()),
      "nonmanifold_faces":sum(n>2 for n in counts.values()),
      "boundary_face_count":len(boundary),
      "source_boundary_face_count":len(base_boundary),
      "boundary_face_set_identical":boundary==base_boundary,
      "reference_points_bitwise_identical":np.array_equal(src["reference_points_AP_m"],cand["reference_points_AP_m"]),
      "target_points_bitwise_identical":np.array_equal(src["target_points_AP_m"],cand["target_points_AP_m"]),
      "O_faces_bitwise_identical":np.array_equal(src["candidate_O_faces"],cand["candidate_O_faces"]),
      "I_faces_bitwise_identical":np.array_equal(src["candidate_I_faces"],cand["candidate_I_faces"]),
      "D_faces_bitwise_identical":np.array_equal(src["candidate_D_faces"],cand["candidate_D_faces"]),
      "material_ids_count":{str(k):v for k,v in material_counts.items()},
      "candidate_cell_uids_unique":len(np.unique(cand["candidate_cell_uids"]))==len(t),
    }
    OUTPUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
