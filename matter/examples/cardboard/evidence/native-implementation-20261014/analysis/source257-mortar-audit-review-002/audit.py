#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
SRC = Q / "snapshots/source-mortar-first-uncertified-audit-257-006"
BASE = Q / "snapshots/source-freebody-scene-observer-252-002"
PKG = Q / "analysis/source-mortar-first-uncertified-audit-257-006"
BUILD = Q / "build-source257-mortar-audit-007"
CAP = Q / "native/freebody-newton7-full-private-trial-601/diagnostics/capture-coupled-trial-capture"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

frozen_path = SRC / "frozen-source.json"
base_frozen_path = BASE / "frozen-source.json"
frozen = json.loads(frozen_path.read_text())
base_frozen = json.loads(base_frozen_path.read_text())
source_map = frozen["source_sha256"]
base_map = base_frozen["source_sha256"]
overlay = json.loads((PKG / "source-overlay.json").read_text())
contract = json.loads((PKG / "cpu_contract.json").read_text())
build_path = BUILD / "source-build.json"
build = json.loads(build_path.read_text())
changed = sorted(rel for rel in set(source_map) | set(base_map)
                 if source_map.get(rel) != base_map.get(rel))
actual_source_mismatches = [
    rel for rel, expected in source_map.items()
    if not (SRC / rel).is_file() or sha(SRC / rel) != expected
]
artifact_mismatches = []
for path, expected in build["artifacts_sha256"].items():
    p = Path(path)
    if not p.is_file() or sha(p) != expected:
        artifact_mismatches.append(path)
runtime = (SRC / "matter/src/runtime.mm").read_text()
metal = (SRC / "matter/src/metal/contact.metalinc").read_text()
header = (SRC / "matter/include/numi/matter/shared.h").read_text()
push_i = runtime.find('push("actual_mortar_complement_first_uncertified_query"')
capture_snip = runtime[max(0, push_i - 500):push_i + 300]
kernel_start = metal.find('kernel void nm_contact_narrowphase_deformable')
clear_i = metal.find('mortarAudit[0] = {};', kernel_start)
early_return_i = metal.find('if (slot >= candidateCounts[environment]) return;', kernel_start)
store_guard_i = metal.find('audit[0].attempt.x != 0u')
all_cert_i = metal.find('allCertified = allCertified && certified;', kernel_start)
# Verify previous 257-005 build is retained and record its inventory discrepancy.
oldsrc = Q / "snapshots/source-mortar-first-uncertified-audit-257-005"
oldbuild = Q / "build-source257-mortar-audit-006/source-build.json"
oldmap = json.loads((oldsrc / "frozen-source.json").read_text())["source_sha256"]
omitted = sorted(set(base_map) - set(oldmap))
# Recompute a geometry-only receipt from the already captured native 601 payload.
cap_json = CAP / "capture.coupled.json"
cap_bin = CAP / "capture.coupled.bin"
cj = json.loads(cap_json.read_text())
payload_sha_ok = sha(cap_bin) == cj["payload"]["sha256"]
payload = cap_bin.read_bytes()
def array(name, phase):
    a = next(x for x in cj["arrays"] if x["name"] == name and x["phase"] == phase)
    return a, payload[a["offset"]:a["offset"] + a["byte_count"]]
def vec3(data, off):
    return struct.unpack_from("<3f", data, off)
def sub(a, b):
    return tuple(x-y for x,y in zip(a,b))
def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    return math.sqrt(sum(x*x for x in a))
def mapped(points, weights):
    return tuple(sum(weights[i]*points[i][k] for i in range(3)) for k in range(3))
geometry = {}
for phase, prim_name, node_name, face_name, label in [
    ("candidate_before_private_contact_rebuild", "surface_primitives_accepted_before_rebuild",
     "fem_accepted_nodes", "cohesive_faces_accepted", "start"),
    ("private_contact_rebuild_before_free_body_terms", "surface_primitives_private_trial",
     "fem_armijo_candidate_nodes", "fem_armijo_cohesive_faces_trial", "finish"),
]:
    _, candidates = array("deformable_candidates_accepted_before_rebuild" if label == "start"
                          else "deformable_candidates_private_trial", phase)
    candidate78 = struct.unpack_from("<4I", candidates, 78*16)
    _, prims = array(prim_name, phase)
    node_ids_a = struct.unpack_from("<3I", prims, 16*64)
    node_ids_b = struct.unpack_from("<3I", prims, 99*64)
    _, nodes = array(node_name, phase)
    _, face = array(face_name, phase)
    pa = [vec3(nodes, n*80) for n in node_ids_a]
    pb = [vec3(nodes, n*80) for n in node_ids_b]
    wa = [struct.unpack_from("<3f", face, 224+i*16) for i in range(3)]
    wb = [struct.unpack_from("<3f", face, 272+i*16) for i in range(3)]
    ma = [mapped(pa,w) for w in wa]
    mb = [mapped(pb,w) for w in wb]
    area_a = 0.5*norm(cross(sub(pa[1],pa[0]),sub(pa[2],pa[0])))
    area_b = 0.5*norm(cross(sub(pb[1],pb[0]),sub(pb[2],pb[0])))
    patch_a = 0.5*norm(cross(sub(ma[1],ma[0]),sub(ma[2],ma[0])))
    patch_b = 0.5*norm(cross(sub(mb[1],mb[0]),sub(mb[2],mb[0])))
    dists = [norm(sub(mb[i],ma[i])) for i in range(3)]
    geometry[label] = {
       "candidate_slot_78": candidate78,
       "primitive16_nodes": node_ids_a,
       "primitive99_nodes": node_ids_b,
       "parent_triangle_area_m2": [area_a, area_b],
       "mapped_patch_area_m2": [patch_a, patch_b],
       "paired_map_vertex_distances_m": dists,
       "all_maps_same_bits": wa == wb,
    }
checks = {
    "complete_836_entry_inventory": len(source_map) == len(base_map) == 836,
    "candidate_source_files_match_frozen_map": not actual_source_mismatches,
    "candidate_delta_is_exactly_three_files": len(changed) == 3 and changed == sorted(overlay["changed_files"]),
    "patch_hash_matches_overlay": sha(PKG / "overlay.patch") == overlay["patch_sha256"],
    "candidate_frozen_hash_matches_overlay": sha(frozen_path) == overlay["candidate_frozen_source_sha256"],
    "base_frozen_hash_matches_overlay": sha(base_frozen_path) == overlay["base_frozen_source_sha256"],
    "all_omitted_base_entries_restored_in_candidate": len(omitted) == 5 and all(source_map[x] == base_map[x] for x in omitted),
    "cpu_contract_passed": contract.get("passed") is True,
    "record_abi_384_bytes": overlay.get("record_size_bytes") == 384 and "sizeof(NMMortarComplementAuditGPU) == 384" in header,
    "no_new_environment_keys": overlay.get("new_environment_keys") == [],
    "only_reserved_3_2_selector_extension": "f[7] == 3u && f[8] == 2u" in runtime and "f[7] < 3u && f[8] < 2u" in runtime,
    "capture_emitted_only_post_rebuild": "if (postRebuild) {" in capture_snip,
    "selected_candidate_record_clear_precedes_slot_early_return": clear_i >= kernel_start and clear_i < early_return_i,
    "first_writer_guard_present": store_guard_i >= 0,
    "production_certification_accumulation_preserved": all_cert_i >= 0,
    "source_build_map_matches_frozen_map": build.get("source_sha256") == source_map,
    "source_build_success": build.get("success") is True,
    "artifacts_match_build_record": not artifact_mismatches,
    "601_payload_hash_matches_capture": payload_sha_ok,
    "601_candidate_pair_is_16_99_slot78": all(tuple(g["candidate_slot_78"]) == (16,99,78,1) for g in geometry.values()),
    "601_paired_maps_have_positive_area_and_no_zero_corresponding_gap": all(
       g["all_maps_same_bits"] and min(g["mapped_patch_area_m2"]) > 0 and min(g["paired_map_vertex_distances_m"]) > 0
       for g in geometry.values()),
}
report = {
 "schema":"numi.cardboard.source257-mortar-audit-review.v2",
 "candidate":"source-mortar-first-uncertified-audit-257-006",
 "superseded_candidate":"source-mortar-first-uncertified-audit-257-005",
 "source_frozen_sha256":sha(frozen_path),
 "source_file_count":len(source_map),
 "source_mismatches":actual_source_mismatches,
 "base_frozen_sha256":sha(base_frozen_path),
 "changed_files":changed,
 "257_005_omitted_base_entries":omitted,
 "overlay_sha256":sha(PKG/"source-overlay.json"),
 "patch_sha256":sha(PKG/"overlay.patch"),
 "cpu_contract_sha256":sha(PKG/"cpu_contract.json"),
 "build_record":str(build_path),
 "build_record_sha256":sha(build_path),
 "build_success":build.get("success"),
 "build_target":build.get("targets"),
 "build_artifact_sha256":build.get("artifacts_sha256"),
 "artifact_mismatches":artifact_mismatches,
 "601_capture_json_sha256":sha(cap_json),
 "601_capture_bin_sha256":sha(cap_bin),
 "601_capture_payload_sha256":cj["payload"]["sha256"],
 "601_source_physics_fingerprint":cj["source_physics_fingerprint"],
 "601_device_program_fingerprint":cj["device_program_fingerprint"],
 "601_geometry":geometry,
 "checks":checks,
 "all_checks_pass":all(checks.values()),
 "gpu_execution":False,
 "qualification":"diagnostic source/build evidence only; exact mortar query outcome is pending root native replay; no physical/contact claim",
}
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd()/"report.json"
out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"all_checks_pass":report["all_checks_pass"],"checks":checks,"changed_files":changed,"geometry":geometry},indent=2))
if not report["all_checks_pass"]: sys.exit(1)
