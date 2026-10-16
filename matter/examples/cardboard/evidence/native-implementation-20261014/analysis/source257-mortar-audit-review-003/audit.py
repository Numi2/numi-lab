#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import struct,sys
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
SRC=Q/"snapshots/source-mortar-first-uncertified-audit-257-006"
BASE=Q/"snapshots/source-freebody-scene-observer-252-002"
PKG=Q/"analysis/source-mortar-first-uncertified-audit-257-006"
BUILD=Q/"build-source257-mortar-audit-007"
CAP=Q/"native/freebody-newton7-full-private-trial-601/diagnostics/capture-coupled-trial-capture"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((SRC/"frozen-source.json").read_text()); bf=json.loads((BASE/"frozen-source.json").read_text())
sm=f["source_sha256"]; bm=bf["source_sha256"]
ov=json.loads((PKG/"source-overlay.json").read_text()); cc=json.loads((PKG/"cpu_contract.json").read_text())
bp=BUILD/"source-build.json"; br=json.loads(bp.read_text()); bsm=br["source_sha256"]
delta=sorted(k for k in set(sm)|set(bm) if sm.get(k)!=bm.get(k))
missing_src=[k for k,v in sm.items() if not (SRC/k).is_file() or sha(SRC/k)!=v]
build_only=sorted(set(bsm)-set(sm)); build_omitted=sorted(set(sm)-set(bsm))
build_bad=[k for k,v in bsm.items() if sm.get(k)!=v]
arts_bad=[p for p,h in br["artifacts_sha256"].items() if not Path(p).is_file() or sha(Path(p))!=h]
rt=(SRC/"matter/src/runtime.mm").read_text(); mt=(SRC/"matter/src/metal/contact.metalinc").read_text(); hd=(SRC/"matter/include/numi/matter/shared.h").read_text()
push=rt.find('push("actual_mortar_complement_first_uncertified_query"')
capture_snip=rt[max(0,push-500):push+300]
kernel=mt.find("kernel void nm_contact_narrowphase_deformable")
clear=mt.find("mortarAudit[0] = {};",kernel)
slot_return=mt.find("if (slot >= candidateCounts[environment]) return;",kernel)
# The helper is defined before the kernel; test production aggregation directly.
checks={
 "snapshot_map_has_836_entries_and_matches_all_bytes":len(sm)==836 and not missing_src,
 "complete_map_delta_is_exactly_intended_three_files":delta==sorted(ov["changed_files"]) and len(delta)==3,
 "patch_and_frozen_binding_match":sha(PKG/"overlay.patch")==ov["patch_sha256"] and sha(SRC/"frozen-source.json")==ov["candidate_frozen_source_sha256"] and sha(BASE/"frozen-source.json")==ov["base_frozen_source_sha256"],
 "target_build_succeeded":br.get("success") is True,
 "build_map_is_exact_selected_subset":len(bsm)==831 and not build_only and build_omitted==[
  "apps/NumiHumanRestingIntercostalRibAttachment.hpp",
  "apps/NumiHumanRestingSourceAllocation.hpp",
  "apps/NumiHumanRestingVisual.hpp",
  "tools/test_intercostal_rib_motion_attachment.mm",
  "tools/test_resting_referenced_source_attachments.cpp"],
 "all_build_inputs_match_snapshot":not build_bad and all(x in bsm for x in ov["changed_files"]),
 "all_artifacts_match_build_receipt":not arts_bad,
 "384_byte_ABI_bound":ov["record_size_bytes"]==384 and "sizeof(NMMortarComplementAuditGPU) == 384" in hd,
 "no_new_environment_keys":ov["new_environment_keys"]==[],
 "only_reserved_selector_3_2_added": "f[7] == 3u && f[8] == 2u" in rt and "f[7] < 3u && f[8] < 2u" in rt,
 "capture_array_post_rebuild_only":"if (postRebuild) {" in capture_snip,
 "selected_candidate_initializes_record_before_early_return":0<=clear<slot_return,
 "first_writer_guard_present":"audit[0].attempt.x != 0u" in mt,
 "certification_accumulation_present":"allCertified = allCertified && certified;" in mt,
 "native601_payload_hash_valid":None,
}
capj=CAP/"capture.coupled.json"; capb=CAP/"capture.coupled.bin"
cj=json.loads(capj.read_text()); blob=capb.read_bytes()
checks["native601_payload_hash_valid"]=sha(capb)==cj["payload"]["sha256"]
def arr(name,phase):
 a=next(x for x in cj["arrays"] if x["name"]==name and x["phase"]==phase)
 return blob[a["offset"]:a["offset"]+a["byte_count"]]
def xyz(d,off):return struct.unpack_from("<3f",d,off)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(a):return math.sqrt(sum(v*v for v in a))
def mp(p,w):return tuple(sum(w[i]*p[i][k] for i in range(3)) for k in range(3))
geom={}
for phase,can_name,prim_name,node_name,face_name,label in [
 ("candidate_before_private_contact_rebuild","deformable_candidates_accepted_before_rebuild","surface_primitives_accepted_before_rebuild","fem_accepted_nodes","cohesive_faces_accepted","start"),
 ("private_contact_rebuild_before_free_body_terms","deformable_candidates_private_trial","surface_primitives_private_trial","fem_armijo_candidate_nodes","fem_armijo_cohesive_faces_trial","finish")]:
 can=arr(can_name,phase); c78=struct.unpack_from("<4I",can,78*16)
 prim=arr(prim_name,phase); idsA=struct.unpack_from("<3I",prim,16*64);idsB=struct.unpack_from("<3I",prim,99*64)
 nodes=arr(node_name,phase); pA=[xyz(nodes,n*80) for n in idsA];pB=[xyz(nodes,n*80) for n in idsB]
 face=arr(face_name,phase);wA=[struct.unpack_from("<3f",face,224+i*16) for i in range(3)];wB=[struct.unpack_from("<3f",face,272+i*16) for i in range(3)]
 mA=[mp(pA,w) for w in wA];mB=[mp(pB,w) for w in wB]
 area=lambda p: .5*norm(cross(sub(p[1],p[0]),sub(p[2],p[0])))
 geom[label]={"candidate_slot78":c78,"nodes16":idsA,"nodes99":idsB,
  "parent_areas_m2":[area(pA),area(pB)],"mapped_patch_areas_m2":[area(mA),area(mB)],
  "paired_vertex_distances_m":[norm(sub(mB[i],mA[i])) for i in range(3)],
  "maps_bitwise_equal":wA==wB}
checks["native601_pair_slot_identity_valid"]=all(tuple(x["candidate_slot78"])==(16,99,78,1) for x in geom.values())
checks["native601_maps_nondegenerate_and_separated_at_sampled_endpoints"]=all(
 x["maps_bitwise_equal"] and min(x["mapped_patch_areas_m2"])>0 and min(x["paired_vertex_distances_m"])>0
 for x in geom.values())
report={
 "schema":"numi.cardboard.source257-mortar-audit-review.v3",
 "candidate":"source-mortar-first-uncertified-audit-257-006",
 "superseded_incomplete_inventory_candidate":"source-mortar-first-uncertified-audit-257-005",
 "source_frozen_sha256":sha(SRC/"frozen-source.json"),"base_frozen_sha256":sha(BASE/"frozen-source.json"),
 "source_count":len(sm),"changed_files":delta,"source_mismatches":missing_src,
 "source_overlay_sha256":sha(PKG/"source-overlay.json"),"patch_sha256":sha(PKG/"overlay.patch"),"cpu_contract_sha256":sha(PKG/"cpu_contract.json"),
 "build_record":str(bp),"build_record_sha256":sha(bp),"build_success":br.get("success"),"build_targets":br.get("targets"),
 "build_input_count":len(bsm),"unchanged_snapshot_files_not_in_selected_target_build":build_omitted,
 "build_artifact_sha256":br.get("artifacts_sha256"),"artifact_mismatches":arts_bad,
 "native601_capture_json_sha256":sha(capj),"native601_payload_sha256":sha(capb),"native601_declared_payload_sha256":cj["payload"]["sha256"],
 "native601_source_fingerprint":cj["source_physics_fingerprint"],"native601_program_fingerprint":cj["device_program_fingerprint"],
 "native601_pair_geometry":geom,
 "checks":checks,"all_checks_pass":all(checks.values()),
 "scope":"diagnostic-only compile/source audit plus independent endpoint geometry summary from the existing 601 capture",
 "limits":["No query result from the new mortar trace yet.","Mapped paired vertices are not a complete closest-point/contact proof; gaps are near the expected contact scale.","No contact feasibility, full solver, or physical qualification is claimed.","No GPU was executed by this review."],
}
out=Path(sys.argv[1]);out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\\n")
print(json.dumps({"all_checks_pass":report["all_checks_pass"],"checks":checks,"source_count":len(sm),"build_input_count":len(bsm),"geom":geom},indent=2))
if not report["all_checks_pass"]:sys.exit(1)
