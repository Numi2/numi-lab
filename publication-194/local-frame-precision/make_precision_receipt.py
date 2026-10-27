import hashlib, json
from pathlib import Path
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
OUT=S/"per-bank-workers-20261027/common-sub-float32-root-local-frame-001"
OWNER=Path("/Users/numi/Work/unpaired-skin-fat-mass-ownership-070")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def pin(p):
    if not p.is_file(): raise FileNotFoundError(p)
    return {"path":str(p),"sha256":sha(p)}
report=json.loads((OUT/"local_frame_precision_report.json").read_text())
source_files=[
    OWNER/"apps/NumiHumanCoarseTissuePreparation.hpp",
    OWNER/"include/metalrobo/numi_human_resting_visual_gpu.h",
    OWNER/"src/metal/NumiHumanRestingSupportGeometry.metalinc",
    OWNER/"include/metalrobo/NumiHumanSurfaceFrameRebase.hpp",
    Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py"),
]
outputs=[
    OUT/"local_frame_precision_probe_stdlib.py",
    OUT/"local_frame_precision_probe_stdlib.log",
    OUT/"finalize_local_frame_report.py",
    OUT/"finalize_local_frame_report.log",
    OUT/"local_frame_precision_report.json",
    OUT/"local_frame_precision_assertions.py",
    OUT/"local_frame_precision_assertions.log",
    OUT/"make_precision_receipt.py",
]
input_files=[
    S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz",
    BANKDIR/"exact-emitted.triangles.csv",
    BANKDIR/"exact-emitted.nodes.csv",
    BANKDIR/"exact-emitted.bodies.csv",
    BANKDIR/"all-bank-containment.csv.banks.csv",
    S/"per-bank-workers-20261027/b43-16/attempt-005-exact-endpoint-identity/receipt.json",
    S/"per-bank-workers-20261027/b43-16/attempt-005-exact-endpoint-identity/facewise-failure-witness.json",
    S/"per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/receipt.json",
    S/"per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/facewise-failure-witness.json",
]
receipt={
    "schema":"numi.human.common-sub-float32-root-local-frame-receipt.v1",
    "status":"LOCAL_BODY_FRAME_PRESERVES_ROOT_DISTINCTION_BUT_WORLD_EMISSION_IS_NOT_SAFE",
    "scope":"Two exact incident skin/bone triangle pairs from banks 43/16 and 65/99; no patch candidate, no geometry mutation, no all-bank or full-body audit.",
    "candidate_and_source_geometry_pins":[pin(p) for p in input_files],
    "owner_code_pins":[pin(p) for p in source_files],
    "prototype_and_test_artifacts":[pin(p) for p in outputs],
    "results":[{
        "bank":r["bank"],"skin_face":r["skin_face"],"bone_triangle":r["bone_triangle"],
        "source_vertices":r["skin_source_vertex_ids"],
        "exact_root_count":r["reconstructed_exact_local_root_count"],
        "exact_segment_length_m":r["reconstructed_exact_local_segment_length_m"],
        "body_local_float32_root_count":r["body_local_float32_reintersection_root_count"],
        "body_local_float32_segment_length_m":r["body_local_float32_reintersection_segment_length_m"],
        "body_local_roots_distinct":r["body_local_float32_roots_distinct"],
        "rounded_local_root_incidence":r["rounded_local_roots_exactly_on_quantized_skin_and_bone_triangles"],
        "world_float32_roots_distinct_after_round_trip":r["body_local_root_round_trip_world_f32_distinct"],
        "world_float32_delta_from_pinned_root_m":r["world_f32_root_delta_from_pinned_exact_root_m"],
        "source_skin_vertex_world_float32_bit_changes":r["body_local_source_vertex_changed_world_f32_component_count"],
        "source_skin_vertex_max_round_trip_world_error_m":r["body_local_f32_source_vertex_round_trip_world_max_norm_delta_m"],
    } for r in report["results"]],
    "accepted_owner_abi_limits":{
        "registered_surface_source":"V3 is double in RegisteredSurface, but accepted shell FEM conditioning rounds world FEM node positions componentwise to Float32; source/owner hashes are pinned above.",
        "local_metal_path":"The existing MRHumanRestingInfluence ABI stores per-body local Float32 point influences with body ID and compensated world-position reconstruction for support point queries. It has no skin triangle connectivity or per-patch origin.",
        "candidate_format":"The pinned accepted candidate is points_world_m Float32 plus source_O_faces. Matched-query validation requires exact parity of emitted world-coordinate Float32 bits. No public format was changed.",
    },
    "conclusion":report["conclusion"],
    "attempt_history":[
        {"path":str(OUT/"local_frame_precision_prototype.py"),"sha256":sha(OUT/"local_frame_precision_prototype.py"),"outcome":"initial NumPy probe could not start because the configured remote Python has no NumPy; retained unchanged as failed evidence"},
        {"path":str(OUT/"run.log"),"sha256":sha(OUT/"run.log"),"outcome":"ModuleNotFoundError: numpy"},
        {"path":str(OUT/"local_frame_precision_probe_stdlib.py"),"outcome":"successful standard-library exact Fraction probe and local assertions; it uses pinned source data and does not import NumPy"},
    ],
    "geometry_changed":False,"candidate_written":False,"public_format_changed":False,
    "full_or_all_bank_audit_run":False,"native_admission":False,"physical_support_claim":False,"accepted_physical_steps":0,
}
(OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
