import hashlib,json,sys,importlib.util
from pathlib import Path
import numpy as np
IN=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123/run-031-fixed-coordinate-edge-flips/candidate.npz")
OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123/run-032-fixed-edge-flips-exact-self")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PRED=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PIN_CAND="bae1b03ce4042d8c4cde2e5124af9a65618b131f3bff351411b387dbb6a8cf42"
PIN_PRED="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pairpath=OUT/"surface-self-pairs.npy"; reportpath=OUT/"summary.json"
 if pairpath.exists() or reportpath.exists(): raise FileExistsError("run-032 outputs exist")
 if sha(IN)!=PIN_CAND or sha(PRED)!=PIN_PRED: raise RuntimeError("input pin mismatch")
 z=np.load(IN,allow_pickle=False);X=np.asarray(z["points_world_m"],dtype=np.float32);F=np.asarray(z["source_O_faces"],dtype=np.int32)
 if X.shape!=(8776,3) or F.shape!=(17507,3) or not np.isfinite(X).all(): raise RuntimeError("unexpected candidate geometry")
 sys.path.insert(0,str(E/"incremental-exact-audit-owner-253/src"))
 spec=importlib.util.spec_from_file_location("numilab_human._connected132",PRED)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 rec=mod.float32_surface_records(X,F,source_seam_groups=())
 result=mod._audit_self_pairs_touching_faces(rec,set(range(len(F))))
 pairs=np.asarray(result["triangle_pairs"],dtype=np.int64).reshape(-1,2)
 with pairpath.open("xb") as f: np.save(f,pairs,allow_pickle=False)
 report={"status":"EXACT_FLOAT32_WHOLE_SURFACE_SELF_AUDIT","candidate":str(IN),"candidate_sha256":PIN_CAND,"predicate":str(PRED),"predicate_sha256":PIN_PRED,"points":len(X),"faces":len(F),"exact_PLC_surface_pairs":len(pairs),"complete_surface_audit":bool(result["complete_surface_audit"]),"pairs_path":str(pairpath),"pairs_sha256":sha(pairpath),"physical_steps":0,"geometry_admitted":False}
 with reportpath.open("x") as f: json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
