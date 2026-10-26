from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np, os

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT=D/"D/D/anchor-compatible-finite-triangle-513"
VENV=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
launch=json.loads((D/"D/area-blocker-fit-506/run-001-launch/argv.json").read_text())
def val(flag): return Path(launch[launch.index(flag)+1])
base=val("--base-candidate")
audit=val("--audit159-490")
runner=load("fit458_preflight_508",D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py")
sat=load("sat466_preflight_508",D/"D/D/triangle-sat-rows-466/triangle_sat_rows_466.py")
helper=load("helper508_preflight",ROOT/"anchor_finite_triangle_513.py")
os.environ["NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458"]=sha(base)
os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458"]=sha(audit)
cross=runner.prepare_crossing_inputs(base,audit)
anchors=np.load(base,allow_pickle=False)["typed_coarse_anchor_output_rows_018"].astype(np.int64)
support=np.unique(cross["faces"].reshape(-1))
baseline=sat.build_fixed_bone_outward_plane_rows(
 cross["points_mm"],cross["faces"],support,anchors,cross["pairs"],
 cross["bone_triangles_mm"],total_weight=1.0)
selected={(int(p["face_row"]),int(p["body"]),int(p["bone_id"]),int(p["triangle_id"]))for p in cross["pairs"]}
result=helper.build_anchor_compatible_rows(
 cross["points_mm"],cross["faces"],support,anchors,cross["pairs"],
 cross["bone_triangles_mm"],sat=sat,selected_pair_ids=selected,total_weight=1.0)
if result["diagnostics"]["anchored_positive_lower_count"] != 0:
 raise SystemExit("FAIL: positive anchored corner lower bounds remain")
receipt={
 "schema":"numi.anchor_compatible_finite_triangle_513.matched_row_preflight.v1",
 "candidate_path":str(base),"candidate_sha256":sha(base),
 "audit159_path":str(audit),"audit159_sha256":sha(audit),
 "pair_count":len(cross["pairs"]),
 "baseline_anchored_positive_count":baseline["diagnostics"]["anchored_positive_lower_count"],
 "selected_pair_count":len(selected),
 "selected_identities":[list(x) for x in sorted(selected)],
 "anchor_nodes":[30,194],
 "output_anchor_positive_count":result["diagnostics"]["anchored_positive_lower_count"],
 "output_lower_row_count":result["diagnostics"]["lower_bound_row_count"],
 "selected_axes":result["diagnostics"]["anchor_compatible_axes"],
 "actual_anchor_count":len(anchors), "all_pairs_selected":True,
 "helper_sha256":sha(ROOT/"anchor_finite_triangle_513.py"),
 "sat466_sha256":sha(D/"D/D/triangle-sat-rows-466/triangle_sat_rows_466.py"),
 "construction_only":True,"global_clearance_certificate":False
}
out=ROOT/"matched-row-preflight-001.json"
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:receipt[k] for k in ("pair_count","baseline_anchored_positive_count","selected_pair_count","output_anchor_positive_count","output_lower_row_count")},indent=2))
print("receipt",out,"sha256",sha(out))

