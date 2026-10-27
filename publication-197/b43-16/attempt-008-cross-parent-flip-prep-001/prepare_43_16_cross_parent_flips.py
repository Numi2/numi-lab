#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

S = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
BASE_DIR = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
CAND = BASE_DIR / "candidate.npz"
PARENT = S / "fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz"
AUDIT = BASE_DIR / "audit-159/report.json"
SELF_SUMMARY = BASE_DIR / "audit-130/summary.json"
SELF_PAIRS = BASE_DIR / "audit-130/surface-self-pairs.npy"
PREFLIGHT = S / "per-bank-workers-20261027/b43-16/attempt-006-bone-aware-centroid-fan-prep-001/selective-preflight.json"
OWNER = Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py")
GUARD = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/refresh_guards_multi_parent_546.py")
COMPOSER = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-quality-patch-538/compose_surviving_flips_538.py")
OUT = Path(__file__).resolve().parent / "run-001"
PINS = {
    "candidate": "7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af",
    "immediate_parent": "beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35",
    "audit159": "c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e",
    "self_summary": "0ec99fa47245355d60858b05e1240b0e562de229c13721ea11659f6f4acdf5cf",
    "self_pairs": "55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de",
    "owner402": "f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764",
    "guard546": "090e368e693bc6f7d12c403812c1b691dc8de6d3e409cfae8d869f910aae076f",
    "composer538": "9dc78be3a52e2607771c49b11b30e175099ae0c6295ccecf504cdf9c9519cd52",
}
EXCLUDED = [19309,19310,19311,19312,19352,19353,19354,19355,19417,19418,19419,19420,19870,19871,19872,19873]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def loadmod(name: str, path: Path):
    sp = importlib.util.spec_from_file_location(name, path)
    if sp is None or sp.loader is None: raise RuntimeError(f"cannot load {path}")
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m

def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}

def area(points, tri):
    p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
    return float(.5*np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0])))

def min_angle(points, tri):
    p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
    e=[np.linalg.norm(p[1]-p[0]),np.linalg.norm(p[2]-p[1]),np.linalg.norm(p[0]-p[2])]
    if min(e)<=0: return 0.0
    a=np.arccos(np.clip(np.dot(p[1]-p[0],p[2]-p[0])/(e[0]*e[2]),-1,1))
    b=np.arccos(np.clip(np.dot(p[0]-p[1],p[2]-p[1])/(e[0]*e[1]),-1,1))
    return float(np.rad2deg(min(a,b,np.pi-a-b)))

def aspect(points, tri):
    p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
    e=[np.dot(p[1]-p[0],p[1]-p[0]),np.dot(p[2]-p[1],p[2]-p[1]),np.dot(p[0]-p[2],p[0]-p[2])]
    a=area(points,tri)
    return float(max(e)/(4*np.sqrt(3)*a)) if a>0 else float("inf")

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    required={"candidate":CAND,"immediate_parent":PARENT,"audit159":AUDIT,"self_summary":SELF_SUMMARY,"self_pairs":SELF_PAIRS,"owner402":OWNER,"guard546":GUARD,"composer538":COMPOSER}
    actual={k:sha(p) for k,p in required.items()}
    if actual!=PINS: raise ValueError(f"pinned input mismatch: {actual}")
    c=load_npz(CAND); parent=load_npz(PARENT); audit=json.loads(AUDIT.read_text()); self_summary=json.loads(SELF_SUMMARY.read_text())
    preflight=json.loads(PREFLIGHT.read_text()); self_pairs=np.load(SELF_PAIRS,allow_pickle=False)
    if self_summary.get("candidate_sha256")!=PINS["candidate"] or self_summary.get("complete_surface_audit") is not True or self_summary.get("exact_PLC_surface_pairs")!=0:
        raise ValueError("stored exact self receipt does not pin current candidate with zero pairs")
    if self_pairs.shape!=(0,2) or self_pairs.dtype!=np.int64: raise ValueError("stored self-pair table is not the pinned empty [0,2] array")
    b=loadmod("two_parent_patch_flips_402_for_b43_16_attempt008", OWNER)
    g=loadmod("refresh_guards_multi_parent_546_for_b43_16_attempt008", GUARD)
    auditrow=next((x for x in audit["banks"] if x.get("body")==43 and x.get("stable_bone_id")==16),None)
    if auditrow is None: raise ValueError("pinned audit is missing body 43 / stable bone 16")
    pairs=np.asarray(auditrow["result"]["triangle_pairs"],dtype=np.int64).reshape(-1,2)
    pair_faces=set(map(int,pairs[:,0]))
    if not set(EXCLUDED)<=pair_faces: raise ValueError("one or more 16 excluded faces are not in exact target witness pairs")
    faces=np.asarray(c["source_O_faces"]); details=np.asarray(c["parent_fine_detail_face_row"]); kinds=np.asarray(c["face_kind"])
    detail_rows=sorted(set(map(int,details[EXCLUDED])))
    if detail_rows!=[8265,8269] or any(int(kinds[f])!=2 for f in EXCLUDED): raise ValueError("excluded faces do not match expected registered fine-detail material domain")
    b.validate_candidate(c)
    baseline_ancestry=None; baseline_guard_error=None
    try: baseline_ancestry=g.validate_ancestry(c,parent,audit)
    except Exception as e: baseline_guard_error=f"{type(e).__name__}: {e}"
    out,owner_report=b.propose_patch_flips(c,detail_rows,EXCLUDED,max_flips=4,max_rounds=1,min_angle_gain_deg=0.5,min_normal_dot=0.0)
    operations=owner_report["operations"]
    if any(not (set(op["face_rows"]) & set(EXCLUDED)) for op in operations): raise ValueError("an operation is not incident to a targeted excluded face")
    self_face_set=set(map(int,self_pairs.ravel()))
    self_conflicts=[op for op in operations if set(op["face_rows"]) & self_face_set]
    if self_conflicts: raise ValueError(f"proposal touches stored self-pair faces: {self_conflicts}")
    shape=[]
    x=c["points_world_m"]
    if x.dtype!=np.float32: raise ValueError("current points are not emitted Float32")
    shape_ok=True
    for op in operations:
        fr=list(map(int,op["face_rows"]))
        old_area=sum(area(x,faces[r]) for r in fr)
        new_metrics=[]
        for r in fr:
            tri=out["source_O_faces"][r]; a=area(x,tri); ang=min_angle(x,tri); asp=aspect(x,tri)
            new_metrics.append({"face_row":r,"triangle":list(map(int,tri)),"area_m2":a,"min_angle_deg":ang,"aspect":asp,"area_fraction_of_old_two_face_patch":a/old_area if old_area>0 else 0.0})
        gate=all(m["area_m2"]>0 and m["min_angle_deg"]>=0.5 and m["aspect"]<=50 and m["area_fraction_of_old_two_face_patch"]>=0.1 for m in new_metrics)
        shape_ok &= gate
        shape.append({"operation_face_rows":fr,"targeted_seed_faces":sorted(set(fr)&set(EXCLUDED)),"new_faces":new_metrics,"pair_min_angle_gain_deg":op["angle_gain_deg"],"all_new_faces_pass_min_angle_0p5_aspect_50_area_fraction_0p1":gate})
    result={
      "schema":"numi.human.b43-16.cross-parent-flip-prep.v1",
      "status": ("BASELINE_ANCESTRY_GUARD_BLOCKED" if baseline_guard_error else ("NO_ELIGIBLE_SHAPE_SAFE_FLIP" if not operations or not shape_ok else "PRIVATE_TOPOLOGY_CANDIDATE_PREPARED_PENDING_EXACT_AUDITS")),
      "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0,
      "candidate_input_path":str(CAND),"candidate_input_sha256":PINS["candidate"],
      "immediate_parent_path":str(PARENT),"immediate_parent_sha256":PINS["immediate_parent"],
      "pinned_target_audit_sha256":PINS["audit159"],"pinned_self_receipt_sha256":PINS["self_summary"],"pinned_self_pairs_sha256":PINS["self_pairs"],
      "owner402_sha256":PINS["owner402"],"refresh_guard546_sha256":PINS["guard546"],"composer538_sha256":PINS["composer538"],
      "target_body":43,"target_stable_bone_id":16,"target_exact_pair_count":int(len(pairs)),"target_unique_crossing_faces":len(pair_faces),
      "excluded_intrinsically_skinny_crossing_faces":EXCLUDED,"excluded_count":len(EXCLUDED),"excluded_all_in_pinned_exact_pairs":True,
      "source_detail_rows":detail_rows,"crossing_face_kind":2,
      "proposal_limits":{"max_flips":4,"max_rounds":1,"min_angle_gain_degrees":0.5,"max_output_operations":4},
      "owner_report":owner_report,"proposal_shape":shape,
      "shape_gate":{"min_angle_degrees":0.5,"max_aspect":50,"min_area_fraction_of_old_two_face_patch":0.1,"passes":bool(operations and shape_ok)},
      "points_bitwise_unchanged":bool(np.array_equal(out["points_world_m"],c["points_world_m"])) and out["points_world_m"].dtype==np.float32,
      "baseline_refresh_ancestry_receipt":baseline_ancestry,"baseline_refresh_ancestry_error":baseline_guard_error,
      "stored_exact_self_pairs":int(len(self_pairs)),"stored_self_pair_conflict_count":len(self_conflicts),
      "exact_bone_check":"NOT_RUN; audit159 is the unchanged baseline witness set only",
      "exact_self_check":"NOT_RUN; only the pinned baseline self receipt was read; 538 stored-witness exclusion rule applied",
      "source_material_transfer":"Owner 402 unions source/detail/subface and immediate-parent CSR on each flipped pair; it requires equal face_kind, source-detail and source-subface, preserves outer boundary and detail seams, leaves points fixed, writes multi-parent scalar/barycentric sentinels where needed, and byte-preserves every other array. This represents categorical face_kind and registered ancestry only; it does not recompute/transfer continuous per-face material, volume, or layer fields.",
      "limitations":["no updated exact bone or self predicates were run","child pair-count changes would not establish clearance","no physical or geometry admission"],
    }
    OUT.mkdir(parents=True)
    if operations and shape_ok and not baseline_guard_error:
        guard_after=g.validate_ancestry(out,parent,audit)
        if not np.array_equal(out["source_O_faces"],out["area_reference_037_source_O_faces"]): raise ValueError("037 topology not synchronized")
        result["post_flip_ancestry_receipt_baseline_witness_domain_only"]=guard_after
        result["exact_self_and_all_bank_audits_after_topology_change"]="NOT_RUN"
        cp=OUT/"candidate.npz";np.savez_compressed(cp,**out);result["candidate_output_path"]=str(cp);result["candidate_output_sha256"]=sha(cp)
    else:
        result["post_flip_ancestry_receipt_baseline_witness_domain_only"]=("BLOCKED_BY_BASELINE_ANCESTRY_GUARD" if baseline_guard_error else "NOT_RUN_NO_ACCEPTED_SHAPE_SAFE_FLIP")
        result["candidate_output_path"]=None
    rp=OUT/"report.json";rp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    result["report_path"]=str(rp);result["report_sha256"]=sha(rp)
    print(json.dumps({k:v for k,v in result.items() if k not in ("owner_report",)},indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__": main()
