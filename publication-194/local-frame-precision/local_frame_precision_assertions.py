import json, math
from pathlib import Path
p=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/common-sub-float32-root-local-frame-001/local_frame_precision_report.json")
d=json.loads(p.read_text())
assert d["candidate_sha256"]=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
assert len(d["results"])==2
r43,r65=d["results"]
for r in (r43,r65):
    assert r["reconstructed_exact_local_root_count"]==2
    assert r["body_local_float32_reintersection_root_count"]==2
    assert r["body_local_float32_roots_distinct"] is True
    assert r["body_local_source_vertex_changed_world_f32_component_count"]==0
    assert all(not x["skin"] and not x["bone"] for x in r["rounded_local_roots_exactly_on_quantized_skin_and_bone_triangles"])
    rel=abs(r["body_local_float32_reintersection_segment_length_m"]/r["retained_exact_local_segment_length_m"]-1)
    assert rel<0.0025
assert r43["body_local_root_round_trip_world_f32_distinct"] is True
assert math.isclose(r43["world_f32_max_delta_from_pinned_exact_root_m"],2.9802322387695312e-8,rel_tol=0,abs_tol=1e-20)
assert r65["body_local_root_round_trip_world_f32_distinct"] is False
assert r65["world_f32_max_delta_from_pinned_exact_root_m"]==0
assert d["geometry_changed"] is False and d["candidate_written"] is False
assert d["full_or_all_bank_audit_run"] is False and d["native_admission"] is False
assert not (p.parent/"patch-candidate.npz").exists()
print("LOCAL_PRECISION_ASSERTIONS_PASS cases=2 local_roots_distinct=2 exact_local_incidence_after_f32=0/4 source_vertex_world_bits_changed=0/18 world_43_one_ulp_shift=PASS world_65_recollapse=PASS geometry_written=NO")
