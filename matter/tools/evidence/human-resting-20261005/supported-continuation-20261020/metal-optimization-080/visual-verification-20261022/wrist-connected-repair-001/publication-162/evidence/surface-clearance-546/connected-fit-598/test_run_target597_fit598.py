import hashlib, importlib.util
from pathlib import Path
import numpy as np
import pytest

P = Path(__file__).with_name("run_target597_fit598.py")
spec = importlib.util.spec_from_file_location("fit598_tested", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def good(tmp_path, n=2):
    paths = []
    pins = {}
    for name, data in (("r", b"report"), ("n", b"npz"), ("j", b"records")):
        path = tmp_path / name
        path.write_bytes(data)
        paths.append(path)
        pins[str(path)] = hashlib.sha256(data).hexdigest()
    report = {
        "direction_mode": m.MODE,
        "inputs": {"candidate_sha256": "a"*64, "surface_report_sha256": "b"*64},
        "source_current_direction_extension": {"all_rows": n, "passing_rows": n,
            "candidate_sha256": "a"*64, "surface_report_sha256": "b"*64},
        "crossing_pairs": n, "banks": 185, "matched_159_complete": True,
        "matched_surface_bank_triangle_pairs": n, "passing_targets": n,
        "nearest_boundary_certified_interval_targets": n,
        "inside_samples_certified_for_implicated_bank": n,
        "joint_outward_direction_count": n-1, "source_only_fold_exception_count": 1,
    }
    arrays = {"face_rows": np.arange(n), "bank_body": np.ones(n),
        "bank_stable_bone_id": np.arange(n), "bank_triangle_stable_id": np.arange(n),
        "target_world_f32_m": np.ones((n,3),np.float32),
        "proposal_direction_world_unit": np.ones((n,3)),
        "target_exactly_outside_all185": np.ones(n,bool)}
    receipt = {"status":"PASS_CERTIFIED_DIRECTION_TARGETS_ONLY",
        "artifact_pins":pins, "candidate_sha256":"a"*64, "state_sha256":"c"*64,
        "target_count":n, "exact_endpoint_bank_count":185}
    return report, arrays, receipt, paths

def check(r,a,receipt,paths):
    return m.validate_targets(r,a,receipt,candidate_sha="a"*64,surface_sha="b"*64,
        report_path=paths[0],npz_path=paths[1],records_path=paths[2])

def test_full_coverage_and_explicit_exception(tmp_path):
    r,a,v,paths=good(tmp_path)
    got=check(r,a,v,paths)
    assert got["endpoint_banks"] == 185
    assert got["source_only_fold_exceptions"] == 1

@pytest.mark.parametrize("field,value",[
    ("matched_159_complete",False),("passing_targets",1),
    ("source_only_fold_exception_count",-1),("direction_mode","source-outward")])
def test_rejects_incomplete_or_mislabeled_report(tmp_path,field,value):
    r,a,v,paths=good(tmp_path);r[field]=value
    with pytest.raises(ValueError):check(r,a,v,paths)

def test_rejects_unoutside_target_and_missing_array(tmp_path):
    r,a,v,paths=good(tmp_path);a["target_exactly_outside_all185"][0]=False
    with pytest.raises(ValueError):check(r,a,v,paths)
    r,a,v,paths=good(tmp_path);del a["bank_body"]
    with pytest.raises(ValueError):check(r,a,v,paths)
