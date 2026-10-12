from __future__ import annotations
import hashlib, json, math, pickle
from pathlib import Path
import numpy as np

report_path = Path("/Users/numi/numi-human-fds132-m6-1350/attempt-092-late-witness-arithmetic-001/report.json")
candidate_path = Path("/Users/numi/numi-human-fds132-m6-1350/attempt-085-interface-refinement-2mm-001/result-001/candidate.pkl")
out_path = Path(__file__).with_name("corrected-unit-receipt.json")
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
report=json.loads(report_path.read_text())
candidate=pickle.load(candidate_path.open("rb"))
witnesses=[]
for w in report["witness_faces"]:
    sid=int(w["stable_id"]); fi=int(w["row_face_index"])
    surf=candidate["surfaces"][sid]
    tri_ids=np.asarray(surf["faces"][fi],dtype=np.int64)
    tri_exact=[surf["vertices_exact_m"][int(i)] for i in tri_ids]
    p=np.array([[float(q) for q in v] for v in tri_exact],dtype=np.float64)
    edge_m=np.array([np.linalg.norm(p[(i+1)%3]-p[(i+2)%3]) for i in range(3)],dtype=np.float64)
    cross=np.cross(p[1]-p[0],p[2]-p[0])
    twice_area_m2=float(np.linalg.norm(cross))
    if not twice_area_m2 or np.any(edge_m==0): raise RuntimeError(f"degenerate source face {sid}:{fi}")
    altitude_m=twice_area_m2/edge_m
    old=np.asarray(w["source_altitude_um"],dtype=np.float64)
    if len(old)!=3: raise RuntimeError("bad old altitude arity")
    witnesses.append({
      "stable_id":sid,"row_face_index":fi,"global_vertex_ids":list(map(int,w["gids"])),
      "coordinate_source":"candidate.pkl surfaces[stable_id].vertices_exact_m and faces; exact Fraction coordinates converted to binary64 only for Euclidean norm",
      "minimum_altitude_m":float(altitude_m.min()),
      "minimum_altitude_um":float(altitude_m.min()*1e6),
      "altitudes_m_by_vertex":altitude_m.tolist(),
      "altitudes_um_by_vertex":(altitude_m*1e6).tolist(),
      "edge_lengths_m_opposite_each_vertex":edge_m.tolist(),
      "edge_lengths_mm_opposite_each_vertex":(edge_m*1e3).tolist(),
      "old_report_field_name":"source_altitude_um",
      "old_report_values":old.tolist(),
      "old_field_values_match_recomputed_m_as_unordered_multiset_within_abs_1e-15":bool(np.allclose(np.sort(old),np.sort(altitude_m),rtol=0,atol=1e-15)),
      "old_field_corner_order_matches_face_vertex_order":bool(np.allclose(old,altitude_m,rtol=0,atol=1e-15)),
      "corrected_factor_from_old_field_to_um":1e6,
      "corrected_altitudes_um_from_old_field_unordered":np.sort(old*1e6).tolist(),
      "recomputed_altitudes_um_sorted":np.sort(altitude_m*1e6).tolist(),
      "matches_coordinate_recomputed_um_as_unordered_multiset_within_abs_1e-8":bool(np.allclose(np.sort(old*1e6),np.sort(altitude_m*1e6),rtol=0,atol=1e-8)),
    })
# Dimensional fixture: 1 mm x 1 mm right isosceles triangle.
fixture=np.array([[0.,0.,0.],[1e-3,0.,0.],[0.,1e-3,0.]],dtype=np.float64)
fixture_edges=np.array([np.linalg.norm(fixture[(i+1)%3]-fixture[(i+2)%3]) for i in range(3)])
fixture_area2=float(np.linalg.norm(np.cross(fixture[1]-fixture[0],fixture[2]-fixture[0])))
fixture_alt_um=fixture_area2/fixture_edges*1e6
expected=1000.0/math.sqrt(2.0)
assert math.isclose(float(fixture_alt_um.min()),expected,rel_tol=0,abs_tol=1e-12)
assert all(x["old_field_values_match_recomputed_m_as_unordered_multiset_within_abs_1e-15"] for x in witnesses)
assert all(x["matches_coordinate_recomputed_um_as_unordered_multiset_within_abs_1e-8"] for x in witnesses)
receipt={
 "schema":"numi.human.forearm-092-corrected-altitude-units.v1",
 "status":"completed_interpretation_correction",
 "scope":"Correct units for source altitude values in the immutable local arithmetic report 092 by recomputation from exact source coordinates in candidate 085. This does not alter geometry, rerun poses, or establish a source-conditioning fix for the late intersections.",
 "input_pins":{str(report_path):sha(report_path),str(candidate_path):sha(candidate_path)},
 "old_report_sha256":sha(report_path),
 "unit_finding":{"old_field":"witness_faces[].source_altitude_um",
  "corner_order_caveat":"Do not attach the legacy vector entries to face vertices by index; the corrected receipt reports face-vertex-aligned coordinate results and compares legacy values as an unordered multiset.","actual_unit":"metres","correct_unit":"micrometres","conversion_to_um":1e6,"basis":"independent coordinate recomputation from candidate surfaces exact Fraction vertices; each old-value multiset matches the resulting metre values, though the old per-corner ordering is not the face-vertex order"},
 "witnesses":witnesses,
 "dimensional_sanity_fixture":{"name":"right_isosceles_triangle_1mm_legs","coordinates_m":fixture.tolist(),"minimum_altitude_um":float(fixture_alt_um.min()),"expected_minimum_altitude_um":"1000/sqrt(2)","expected_numeric_um":expected,"absolute_error_um":abs(float(fixture_alt_um.min())-expected),"passed":True},
 "qualification":{"candidate085_source_geometry_recomputed":True,"late_pose_rerun":False,"native_gpu_capture":False,"claim_oldlate_repaired":False,"all_reported_altitudes_coordinate_recomputed":True}
}
out_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"receipt":str(out_path),"old_report_sha256":receipt["old_report_sha256"],"witnesses":[{"stable_id":w["stable_id"],"face":w["row_face_index"],"min_um":w["minimum_altitude_um"],"old_was_m":w["old_field_values_match_recomputed_m_as_unordered_multiset_within_abs_1e-15"]} for w in witnesses],"fixture_min_um":fixture_alt_um.min()},indent=2))
