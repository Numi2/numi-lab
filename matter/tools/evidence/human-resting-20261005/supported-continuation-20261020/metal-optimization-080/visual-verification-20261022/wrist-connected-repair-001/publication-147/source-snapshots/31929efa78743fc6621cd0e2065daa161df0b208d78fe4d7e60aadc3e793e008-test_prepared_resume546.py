import json
from pathlib import Path
import tempfile
import unittest
from continue_surface_round546 import sha, validate_prepared_baseline

class PreparedResumeTests(unittest.TestCase):
    def fixture(self, d):
        state = {r+"_sha256": r+"-hash" for r in ("candidate","point_summary","surface_report")}
        state_path=d/"state.json";state_path.write_text(json.dumps(state))
        prep=d/"prepare";prep.mkdir()
        (prep/"baseline.json").write_text(json.dumps(state))
        (prep/"constraint-context.json").write_text(json.dumps({"candidate_sha256":state["candidate_sha256"]}))
        (prep/"prepared.json").write_text(json.dumps({"status":"FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED","state":str(state_path),"native_admission":False,"accepted_physical_steps":0}))
        return prep,state_path

    def test_matching_preparation_passes(self):
        with tempfile.TemporaryDirectory() as td:
            prep,state=self.fixture(Path(td));validate_prepared_baseline(prep,state)

    def test_changed_baseline_role_fails(self):
        with tempfile.TemporaryDirectory() as td:
            prep,state=self.fixture(Path(td));v=json.loads((prep/"baseline.json").read_text());v["surface_report_sha256"]="changed"
            (prep/"baseline.json").write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,"role mismatch"):validate_prepared_baseline(prep,state)

    def test_native_scope_and_changed_context_fail(self):
        with tempfile.TemporaryDirectory() as td:
            prep,state=self.fixture(Path(td));p=prep/"prepared.json";v=json.loads(p.read_text());v["native_admission"]=True;p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,"scope"):validate_prepared_baseline(prep,state)
            v["native_admission"]=False;p.write_text(json.dumps(v))
            (prep/"constraint-context.json").write_text(json.dumps({"candidate_sha256":"changed"}))
            with self.assertRaisesRegex(ValueError,"candidate mismatch"):validate_prepared_baseline(prep,state)

if __name__=="__main__":unittest.main()
