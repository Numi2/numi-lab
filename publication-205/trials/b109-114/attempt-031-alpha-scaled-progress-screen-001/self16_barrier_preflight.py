from pathlib import Path
import hashlib,json,importlib.util,sys
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-031-alpha-scaled-progress-screen-001")
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz")
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-198-stage-001/scripts/owners/self_separation_026.py")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with np.load(BASE,allow_pickle=False) as z:
  x=np.asarray(z["points_world_m"],np.float32);f=np.asarray(z["source_O_faces"],np.int64)
 pairs=np.load(D/"audit-130/surface-self-pairs.npy")
 spec=importlib.util.spec_from_file_location("self16_owner",OWNER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 rows,records=m.build_pair_rows(x.astype(np.float64)*1000.,f,pairs,margin_fraction=.25)
 out={"schema":"numi.human.b109-114.alpha0625-self16-baseline-barrier-preflight.v1","base_candidate_sha256":sha(BASE),"candidate_sha256":sha(D/"candidate-segment-partial-alpha-0p0625.npz"),"exact_self_pair_file_sha256":sha(D/"audit-130/surface-self-pairs.npy"),"owner_path":str(OWNER),"owner_sha256":sha(OWNER),"pair_count":len(pairs),"pairs":pairs.astype(int).tolist(),"row_count":len(rows),"records":records,"status":"PASS","construction_only":True,"physical_steps":0}
 p=D/"self16-barrier-preflight.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(p),"sha256":sha(p),"pairs":len(pairs),"rows":len(rows),"baseline_gap_min_mm":min(r["closest_distance_mm"] for r in records if r["closest_distance_mm"] is not None),"baseline_gap_max_mm":max(r["closest_distance_mm"] for r in records if r["closest_distance_mm"] is not None),"record_kinds":{k:sum(r["kind"]==k for r in records) for k in sorted(set(r["kind"] for r in records))}},indent=2))
if __name__=="__main__":main()

