#!/usr/bin/env python3
"""Verify publication-061 payload hashes and bounded result summaries."""
from pathlib import Path
import gzip, hashlib, json
ROOT=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
idx=json.loads((ROOT/"payload-files-061.json").read_text())
assert idx["schema"]=="numi.human.publication.payload-files.v1" and len(idx["files"])==idx["count"]
for item in idx["files"]:
 p=ROOT/item["path"]
 assert p.is_file() and not p.is_symlink(),item["path"]
 assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],item["path"]
assert sum(x["bytes"] for x in idx["files"])==idx["total_bytes"]
compressed=ROOT/"payload/039/run-002/exact-emitted.triangles.csv.gz"
h=hashlib.sha256(); n=0
with gzip.open(compressed,"rb") as f:
 for block in iter(lambda:f.read(1<<20),b""): h.update(block); n+=len(block)
assert n==146247212 and h.hexdigest()=="0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac"
report=json.loads((ROOT/"payload/039/run-002/export-report-final.json").read_text())
assert (report["triangles"],report["banks"],report["nodes"],report["body_transforms"])==(469136,185,7233,157)
assert report["physical_qualification"] is False and report["native_simulation_steps"]==0
assert report["app_source_sha256"]=="72b0726f3c6190f36e706dcb83f3ecb4b5c4f3370c8abacabf57e9b298b1c762"
assert report["app_binary_sha256"]=="dc463fb9c341448e0a7d4accd52ba392c4bd757c8226541de0890462bd04b6ab"
old=json.loads((ROOT/"payload/039/run-002/export-report-as-run.json").read_text())
assert old["source_app_sha256"]==report["app_binary_sha256"]
prov=json.loads((ROOT/"payload/039/run-002/export-report-provenance-correction.json").read_text())
assert prov["as_run_field_name"]=="source_app_sha256" and prov["as_run_field_value"]==report["app_binary_sha256"]
assert prov["app_source_sha256"]==report["app_source_sha256"] and prov["correction_changes_execution_or_measurements"] is False
route=json.loads((ROOT/"payload/041/route-report.json").read_text())
adapter=json.loads((ROOT/"payload/041/root-independent-route-adapter-041.json").read_text())
assert route["status"]=="EXACT_NATIVE_GEOMETRY_ROUTE_WITNESSES_NOT_FULL_TISSUE_CLEARANCE"
assert (route["counts"]["proper_hits"],route["counts"]["hit_segment_bank_pairs"],route["columns_with_hits"],route["construction_planes"],route["uncertainties"])==(241,121,121,121,0)
assert adapter["status"]=="PASS_ADAPTER_EXACT_PLANES_AND_LAYER_EXPANSION"
assert (adapter["planes"],adapter["constraints"])==(121,125)
assert adapter["min_exterior_start_distance_m"]>0
assert json.loads((ROOT/"payload/041/construction-plane-adapter-receipt.json").read_text())["status"]=="derived_constraint_sidecar_not_physical_acceptance"
outer=json.loads((ROOT/"payload/041/baseline-outer-inside-local-planes.json").read_text())
assert outer["status"]=="12_BASELINE_OUTER_INSIDE_LOCAL_EXIT_CONSTRAINTS" and outer["count"]==12
assert outer["max_native_frame_residual_m"]<4e-18
j=json.loads((ROOT/"payload/062/material-specific-reference-J-062.json").read_text())
i=json.loads((ROOT/"payload/062/root-independent-material-J-owner-bounds-062.json").read_text())
v=j["results"]["candidate_037_002"]
count=sum(v[m]["below_minimum_J_count"]+v[m]["above_maximum_J_count"] for m in ("skin","fat"))
assert count==162 and i["total_violations"]==162
assert i["materials"]["skin"]["below"]==26 and i["materials"]["skin"]["above"]==3
assert i["materials"]["fat"]["below"]==92 and i["materials"]["fat"]["above"]==41
q=json.loads((ROOT/"payload/062/root-review-037-component-attempt-002.json").read_text())
pins=json.loads((ROOT/"source-input-pins-061.json").read_text())
assert abs(pins["results"]["062"]["existing_positive_q_floor"]["minimum_q"]-1.2160859598096454e-5)<1e-18
assert q["status"]=="PASS_EXISTING_POSITIVE_GEOMETRY_FLOOR_ONLY"
assert q["nonpositive_cells"]==0 and q["below_q_1e_minus_5"]==0 and q["below_construction_target_q_1_25e_minus_5"]==24
print(f"PASS payload_files={idx['count']} payload_bytes={idx['total_bytes']} restored_triangles_csv_bytes={n} route_hits=241 route_constraints=125 baseline_O_planes=12 J_violations=162")
