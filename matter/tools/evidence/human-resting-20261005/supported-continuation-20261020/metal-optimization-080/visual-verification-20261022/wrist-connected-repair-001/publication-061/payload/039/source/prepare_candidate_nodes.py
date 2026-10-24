#!/usr/bin/env python3
"""Hash-gated candidate position export in the existing initial-FEM-node CSV schema."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_hash(path, expected, label):
    if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{label} expected SHA-256 must be 64 lowercase hex chars")
    actual=sha(path)
    if actual!=expected: raise ValueError(f"{label} SHA mismatch: {actual}")
    return actual

def validate_candidate_report(report):
    # Support only the explicit all-checks report or the pinned D37 fit-report
    # schema. The D37 optimizer's broader 1.25e-5 active-set target is not
    # substituted for the established 1e-5 positive-cell construction gate.
    checks=report.get("checks")
    if isinstance(checks,dict):
        if checks.get("allcell_positive_q1e5_pass") is not True:
            raise ValueError("candidate fails all-cell positive normalized-quality gate")
        if checks.get("bone_plane_pass") is not True or checks.get("layer_order_pass") is not True:
            raise ValueError("candidate fails a basic bone-plane or layer-order gate")
        quality=checks.get("candidate_float32_allcell")
        if not isinstance(quality,dict): raise ValueError("candidate report lacks Float32 all-cell metrics")
        min_q=float(quality.get("min_normalized_q",float("nan")))
        min_det=float(quality.get("min_det_m3",float("nan")))
        nonpositive=int(quality.get("nonpositive_tets",-1))
        below=int(quality.get("below_q1e5",-1))
    elif report.get("schema")=="numi.human.d37-componentwise-bone-clearance-fit.v1":
        final=report.get("final")
        if not isinstance(final,dict): raise ValueError("D37 fit report lacks final metrics")
        quality=final.get("all_cell_metrics_float32_emitted")
        plane=final.get("bone_plane")
        layer=final.get("layer")
        if not all(isinstance(v,dict) for v in (quality,plane,layer)):
            raise ValueError("D37 fit report lacks quality/plane/layer gates")
        if plane.get("pass_with_declared_tolerance") is not True or int(plane.get("violations_tolerance_2e-7m",-1))!=0:
            raise ValueError("D37 candidate fails declared bone-plane tolerance")
        if layer.get("pass_with_20nm_emission_tolerance") is not True or int(layer.get("violations_skin",-1))!=0 or int(layer.get("violations_fat",-1))!=0:
            raise ValueError("D37 candidate fails construction layer-order floors")
        min_q=float(quality.get("min_q_float32",float("nan")))
        min_det=float(quality.get("min_det_float32_m3",float("nan")))
        nonpositive=int(quality.get("nonpositive_float32",-1))
        below=int(quality.get("below_q1e5_float32",-1))
    else:
        raise ValueError("candidate report schema is not an admitted source of basic geometry gates")
    if nonpositive!=0 or below!=0:
        raise ValueError("candidate Float32 all-cell quality counts fail")
    if not math.isfinite(min_q) or min_q<1.e-5:
        raise ValueError("candidate Float32 minimum normalized quality is below 1e-5")
    if not math.isfinite(min_det) or min_det<=0:
        raise ValueError("candidate Float32 cells include a nonpositive determinant")
    return {"min_normalized_q":min_q,"min_det_m3":min_det,"nonpositive_tets":nonpositive,"below_q1e5":below}

def prepare(candidate_path,candidate_sha,report_path,report_sha,baseline_path,baseline_sha,out_path):
    cand=Path(candidate_path); rep=Path(report_path); base=Path(baseline_path); out=Path(out_path)
    for p in (cand,rep,base):
        if not p.is_file() or p.is_symlink(): raise ValueError(f"required immutable input absent or symlink: {p}")
    if out.exists() or out.is_symlink() or out.with_suffix(out.suffix+".manifest.json").exists():
        raise ValueError("candidate export outputs must be new paths")
    cand_sha=require_hash(cand,candidate_sha,"candidate NPZ")
    rep_sha=require_hash(rep,report_sha,"candidate report")
    base_sha=require_hash(base,baseline_sha,"baseline node CSV")
    report=json.loads(rep.read_text())
    quality=validate_candidate_report(report)
    if report.get("output_sha256") not in (None,cand_sha):
        raise ValueError("candidate report output SHA does not match candidate NPZ")
    with np.load(cand,allow_pickle=False) as z:
        if "global_node_ids" not in z.files or "points_world_f32" not in z.files:
            raise ValueError("candidate NPZ lacks global_node_ids/points_world_f32")
        gids=np.asarray(z["global_node_ids"])
        points=np.asarray(z["points_world_f32"])
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("candidate points must be an Nx3 Float32 array")
    if gids.dtype.kind not in "iu" or gids.ndim!=1 or len(gids)!=len(points):
        raise ValueError("candidate global node IDs have invalid shape/type")
    if not np.isfinite(points).all(): raise ValueError("candidate positions contain nonfinite values")
    with base.open(newline="") as f:
        reader=csv.DictReader(f)
        expected=("global_node","layer","row","x_m","y_m","z_m","mass_kg")
        if tuple(reader.fieldnames or ())!=expected: raise ValueError("baseline node CSV header mismatch")
        rows=list(reader)
    if len(rows)!=7233 or points.shape!=(7233,3):
        raise ValueError("candidate/baseline must both contain exactly 7,233 volume nodes")
    baseline_ids=np.asarray([int(r["global_node"]) for r in rows],dtype=np.int64)
    if len(np.unique(baseline_ids))!=len(baseline_ids) or not np.array_equal(gids.astype(np.int64),baseline_ids):
        raise ValueError("candidate global node IDs do not exactly match baseline local-node order")
    output_lines=["global_node,layer,row,x_m,y_m,z_m,mass_kg"]
    for i,(r,p) in enumerate(zip(rows,points)):
        vals=[float(v) for v in p]
        if not all(math.isfinite(v) for v in vals): raise ValueError(f"nonfinite candidate node {i}")
        output_lines.append(",".join((r["global_node"],r["layer"],r["row"],
            *(format(float(np.float32(v)),".9g") for v in vals),r["mass_kg"])))
    payload=("\n".join(output_lines)+"\n").encode()
    manifest={
      "schema":"numi.human.d37-candidate-node-audit-override.v1",
      "scope":"candidate positions only for source-only per-bone all-bank node containment; never changes FEM state",
      "candidate_npz":{"path":str(cand),"sha256":cand_sha,"point_field":"points_world_f32","node_id_field":"global_node_ids"},
      "candidate_report":{"path":str(rep),"sha256":rep_sha,"basic_gate":"Float32 all-cell quality >=1e-5, positive determinants, bone planes and layer order pass"},
      "baseline_node_csv":{"path":str(base),"sha256":base_sha,"rows":len(rows),"node_order":"exact global_node_ids equality"},
      "positions":{"dtype":"Float32","rows":len(rows),"max_abs_m":float(np.max(np.abs(points))),"sha256_f32_c_order":hashlib.sha256(points.tobytes(order="C")).hexdigest()},
      "output_csv":{"path":str(out),"sha256":hashlib.sha256(payload).hexdigest(),"schema":"existing initial-skin-fem-nodes.csv header; baseline global/layer/row/mass retained"},
      "admission":{"physical":False,"native":False,"claim":"per-node closed-solid containment only; not triangle/tetrahedron intersection clearance"}
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as f: f.write(payload)
    manifest_path=out.with_suffix(out.suffix+".manifest.json")
    with manifest_path.open("x") as f: json.dump(manifest,f,indent=2,sort_keys=True); f.write("\n")
    return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-npz",required=True)
    ap.add_argument("--candidate-sha256",required=True)
    ap.add_argument("--candidate-report",required=True)
    ap.add_argument("--candidate-report-sha256",required=True)
    ap.add_argument("--baseline-node-csv",required=True)
    ap.add_argument("--baseline-node-csv-sha256",required=True)
    ap.add_argument("--output-csv",required=True)
    a=ap.parse_args()
    try: print(json.dumps(prepare(a.candidate_npz,a.candidate_sha256,a.candidate_report,
        a.candidate_report_sha256,a.baseline_node_csv,a.baseline_node_csv_sha256,a.output_csv),
        indent=2,sort_keys=True))
    except Exception as e: print(f"candidate override refused: {e}",file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
