import importlib.util
from pathlib import Path
import json
SCRIPT=Path(__file__).with_name("probe_alternative_43_16_axes641_v3.py")
spec=importlib.util.spec_from_file_location("probe641v3",SCRIPT)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_complete_representative_report_roundtrips_atomically(tmp_path):
    payload={"schema":"probe-report","axis_inventory":{"19024:43:16:133":{"axis_count":22,"rhs_mm":[0.01,-0.02,0.03]}},"group_results":[{"group":"face_19024","status":"FEASIBLE_LINEAR_HARD_SYSTEM","solver_seconds":0.4,"component_ids":[0],"minimum_all_hard_residual_mm":-1e-9}],"native_admission":False,"accepted_physical_steps":0}
    dest=tmp_path/"complete.json"
    mod._atomic_json(dest,payload)
    assert json.loads(dest.read_text())==payload
    assert not dest.with_name(dest.name+".tmp").exists()

def test_identity_key_is_stable_and_distinct():
    assert mod._identity_key([19024,43,16,133])=="19024:43:16:133"
    assert mod._identity_key((19024,43,16,140))!="19024:43:16:133"
