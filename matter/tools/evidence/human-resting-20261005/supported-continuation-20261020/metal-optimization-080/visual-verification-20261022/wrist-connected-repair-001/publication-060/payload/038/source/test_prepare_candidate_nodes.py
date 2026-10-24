import importlib.util,json,unittest
from pathlib import Path
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
SCRIPT=ROOT/"bone-clearance-fit-038-candidate-query/prepare_candidate_nodes.py"
SPEC=importlib.util.spec_from_file_location("candidate_override_038",SCRIPT)
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)
class CandidateOverrideTests(unittest.TestCase):
    def test_rejects_034_before_writing_override(self):
        report=json.loads((ROOT/"bone-clearance-fit-034/fit-report.json").read_text())
        with self.assertRaisesRegex(ValueError,"all-cell"):
            MOD.validate_candidate_report(report)
    def test_rejects_missing_quality_metrics(self):
        with self.assertRaises(ValueError):
            MOD.validate_candidate_report({"checks":{"allcell_positive_q1e5_pass":True}})
    def test_accepts_exact_d37_fit_report_schema(self):
        report=json.loads((ROOT/"bone-clearance-fit-037-componentwise-035affine-002/fit-report.json").read_text())
        metrics=MOD.validate_candidate_report(report)
        self.assertGreaterEqual(metrics["min_normalized_q"],1e-5)
        self.assertGreater(metrics["min_det_m3"],0)
        self.assertEqual(metrics["nonpositive_tets"],0)

    def test_rejects_d37_report_when_layer_gate_fails(self):
        report=json.loads((ROOT/"bone-clearance-fit-037-componentwise-035affine-002/fit-report.json").read_text())
        report["final"]["layer"]["pass_with_20nm_emission_tolerance"]=False
        with self.assertRaisesRegex(ValueError,"layer-order"):
            MOD.validate_candidate_report(report)

    def test_refuses_bad_hash_before_output(self):
        p=ROOT/"bone-clearance-fit-034/candidate.npz"
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            MOD.require_hash(p,"0"*64,"candidate NPZ")
if __name__=="__main__": unittest.main()
