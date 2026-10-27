"""Truthful compatibility report for frozen 408 source-hemisphere selector.

The 597 joint direction vectors remain unchanged. This copy only routes them
through 408's legacy selector branch, which selects proposal_direction and
checks unit normals plus the source hemisphere. It does not claim the vectors
were nearest-direction optima.
"""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np

MODE = "source-current-outward-with-source-fold-exceptions"
ROUTE = "outward-nearest"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict): raise ValueError(f"expected JSON object: {path}")
    return value

def build_compatibility_report(report, arrays, records, recovery, *,
        report_path, npz_path, records_path, recovery_path, adapter_path):
    report_path,npz_path,records_path,recovery_path,adapter_path=map(Path,(report_path,npz_path,records_path,recovery_path,adapter_path))
    pins=recovery.get("artifact_pins",{})
    for path in (report_path,npz_path,records_path):
        if pins.get(str(path))!=sha(path): raise ValueError(f"recovery receipt pin mismatch: {path}")
    if recovery.get("status")!="PASS_CERTIFIED_DIRECTION_TARGETS_ONLY":
        raise ValueError("597 recovery verification did not pass")
    if recovery.get("candidate_sha256")!=report.get("inputs",{}).get("candidate_sha256"):
        raise ValueError("recovery/report candidate mismatch")
    if recovery.get("target_count")!=report.get("crossing_pairs"):
        raise ValueError("recovery/report count mismatch")
    if report.get("direction_mode")!=MODE or report.get("matched_159_complete") is not True or report.get("banks")!=185:
        raise ValueError("source 597 report mode or exact bank coverage mismatch")
    n=int(report.get("crossing_pairs",-1))
    disk_records=[json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if disk_records!=records:
        raise ValueError("parsed target records differ from pinned JSONL")
    if len(records)!=n or len(arrays.get("face_rows",()))!=n:
        raise ValueError("597 arrays/records do not cover every reported pair")
    required=("face_rows","bank_body","bank_stable_bone_id","bank_triangle_stable_id",
              "proposal_direction_world_unit","source_surface_normal_world_unit",
              "source_normal_valid","target_exactly_outside_all185")
    if any(k not in arrays for k in required): raise ValueError("597 NPZ missing selector or identity field")
    directions=np.asarray(arrays["proposal_direction_world_unit"],dtype=np.float64)
    normals=np.asarray(arrays["source_surface_normal_world_unit"],dtype=np.float64)
    valid=np.asarray(arrays["source_normal_valid"],dtype=bool)
    if directions.shape!=(n,3) or normals.shape!=(n,3) or valid.shape!=(n,) or not valid.all():
        raise ValueError("source-normal selector rows are incomplete")
    if not np.isfinite(directions).all() or not np.isfinite(normals).all():
        raise ValueError("nonfinite target direction/normal")
    dn=np.linalg.norm(directions,axis=1); nn=np.linalg.norm(normals,axis=1)
    if np.max(np.abs(dn-1),initial=0)>2e-10 or np.max(np.abs(nn-1),initial=0)>2e-10:
        raise ValueError("unit direction/normal contract failed")
    dots=np.einsum("ij,ij->i",directions,normals)
    if float(dots.min(initial=1))<0.05-1e-10:
        raise ValueError("597 proposal leaves its certified source hemisphere floor")
    if not np.asarray(arrays["target_exactly_outside_all185"],dtype=bool).all():
        raise ValueError("597 contains an endpoint not certified outside all185")
    joint=0; fallback=0; fallback_faces=set()
    for i,row in enumerate(records):
        if int(row.get("pair_index",-1))!=int(np.asarray(arrays.get("pair_index",np.arange(n)))[i]):
            raise ValueError("target-record order does not match NPZ pair index")
        if not np.allclose(row.get("proposal_direction_world_unit"),directions[i],rtol=0,atol=2e-14):
            raise ValueError("target-record proposal differs from NPZ")
        if not np.allclose(row.get("source_surface_normal_world_unit"),normals[i],rtol=0,atol=2e-14):
            raise ValueError("target-record source normal differs from NPZ")
        source=str(row.get("direction_source",""))
        if source.startswith("545 current/source selector") or source.startswith("angular projection toward nearest bank-boundary vector within source/current cosine caps 0.05"):
            joint+=1
        elif source.startswith("pinned 500 source-only direction"):
            fallback+=1; fallback_faces.add(int(row["face_row"]))
        else:
            raise ValueError("unclassified 597 direction source")
    expected_faces=set(map(int,report.get("source_only_fold_exception_face_rows",[])))
    if joint!=int(report.get("joint_outward_direction_count",-1)) or fallback!=int(report.get("source_only_fold_exception_count",-1)):
        raise ValueError("597 joint/fallback row counts disagree with report")
    if fallback_faces!=expected_faces:
        raise ValueError("597 source-only exception face set disagrees with rows")
    if (joint+fallback!=n or recovery.get("joint_source_current_targets")!=joint
            or recovery.get("source_only_exception_targets")!=fallback):
        raise ValueError("597 joint/fallback accounting does not cover all goals")
    result=copy.deepcopy(report)
    result["direction_mode"]=ROUTE
    result["fit_direction_compatibility"]={
        "schema":"numi.human.408-direction-selector-compatibility.598.v1",
        "selector_route":ROUTE,
        "selector_route_meaning":"frozen 408 branch selects proposal_direction_world_unit and checks registered source-hemisphere consistency only",
        "operative_direction_mode":MODE,
        "operative_direction_method":report.get("direction_method"),
        "nearest_optimality_claim":False,
        "original_report_path":str(report_path.resolve()),
        "original_report_sha256":sha(report_path),
        "original_npz_sha256":sha(npz_path),
        "original_records_sha256":sha(records_path),
        "recovery_verification_path":str(recovery_path.resolve()),
        "recovery_verification_sha256":sha(recovery_path),
        "adapter_path":str(adapter_path.resolve()),
        "adapter_sha256":sha(adapter_path),
        "source_normal_rows":n,"source_normal_valid_rows":int(valid.sum()),
        "min_direction_unit_norm":float(dn.min()),"max_direction_unit_norm":float(dn.max()),
        "min_source_normal_unit_norm":float(nn.min()),"max_source_normal_unit_norm":float(nn.max()),
        "min_proposal_source_dot":float(dots.min()),"negative_source_dot_rows":int((dots<0).sum()),
        "joint_direction_rows":joint,"source_only_fold_exception_rows":fallback,
        "source_only_fold_exception_face_rows":sorted(fallback_faces),
        "targets_npz_sha256":sha(npz_path),"native_admission":False,
        "accepted_physical_steps":0,
    }
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--npz",type=Path,required=True)
    p.add_argument("--records",type=Path,required=True)
    p.add_argument("--recovery",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    if not a.report.is_file() or not a.npz.is_file() or not a.records.is_file() or not a.recovery.is_file():
        raise FileNotFoundError("missing 597 source artifact")
    report=read(a.report); recovery=read(a.recovery)
    with np.load(a.npz,allow_pickle=False) as z: arrays={k:np.array(z[k],copy=True) for k in z.files}
    records=[json.loads(x) for x in a.records.read_text().splitlines() if x.strip()]
    result=build_compatibility_report(report,arrays,records,recovery,
        report_path=a.report,npz_path=a.npz,records_path=a.records,
        recovery_path=a.recovery,adapter_path=Path(__file__))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"compat_report":str(a.out),"compat_report_sha256":sha(a.out),
        "rows":len(records),"joint":result["fit_direction_compatibility"]["joint_direction_rows"],
        "source_only":result["fit_direction_compatibility"]["source_only_fold_exception_rows"]},sort_keys=True),flush=True)

if __name__=="__main__": main()
