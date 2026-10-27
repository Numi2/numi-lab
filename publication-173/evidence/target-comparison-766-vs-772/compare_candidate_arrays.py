#!/usr/bin/env python3
"""Compare only stored point and face arrays; no geometry audit or admission."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    with np.load(a.baseline,allow_pickle=False) as z: bp=z["points_world_m"].copy(); bf=z["source_O_faces"].copy()
    with np.load(a.candidate,allow_pickle=False) as z: cp=z["points_world_m"].copy(); cf=z["source_O_faces"].copy()
    if bp.shape!=cp.shape or bf.shape!=cf.shape or bp.dtype!=cp.dtype or bf.dtype!=cf.dtype:
        raise SystemExit("array shape/dtype mismatch")
    changed=np.flatnonzero(np.any(bf!=cf,axis=1))
    verts=np.unique(cf[changed].reshape(-1)) if changed.size else np.empty(0,dtype=np.int64)
    report={
      "classification":"array-level comparison only; no geometry audit, fit, or admission",
      "baseline_sha256":sha(a.baseline),"candidate_sha256":sha(a.candidate),
      "points_world_m":{"shape":list(cp.shape),"dtype":str(cp.dtype),"bitwise_equal":bp.tobytes()==cp.tobytes(),"changed_rows":int(np.count_nonzero(np.any(bp!=cp,axis=1)))},
      "source_O_faces":{"shape":list(cf.shape),"dtype":str(cf.dtype),"changed_rows":changed.astype(int).tolist(),"changed_row_count":int(changed.size),"involved_vertex_count":int(verts.size)},
      "changed_face_involved_vertices_world_m_bbox":{"min":cp[verts].min(axis=0).astype(float).tolist() if verts.size else None,"max":cp[verts].max(axis=0).astype(float).tolist() if verts.size else None},
    }
    Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
