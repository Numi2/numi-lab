import unittest
from human_skin_fat_bone_audit import summarize_surface_crossing_report as aggregate_report

def row(body,bone,pairs):
    return {"body":body,"stable_bone_id":bone,"result":{"count":len(pairs),"triangle_pairs":pairs}}

class AggregateTests(unittest.TestCase):
    def fixture(self,n=185):
        banks=[row(i,i+1,[]) for i in range(n)]
        banks[0]["result"]={"count":2,"triangle_pairs":[[1,7],[2,8]]}
        return {"complete":True,"bank_count":n,"expected_bank_count":n,"banks":banks}
    def test_sums_explicit_counts_and_crosschecks_pair_arrays(self):
        r=aggregate_report(self.fixture())
        self.assertEqual(r["total_crossing_pairs"],2)
        self.assertEqual(r["nonzero_bank_count"],1)
    def test_missing_count_never_defaults_to_zero(self):
        r=self.fixture(); del r["banks"][3]["result"]["count"]
        with self.assertRaises(ValueError): aggregate_report(r)
    def test_null_result_requires_explicit_aabb_exclusion(self):
        r=self.fixture(); r["banks"][4].update(result=None,skin_aabb_active=0,bank_aabb_active=0,skin_faces=10,bank_faces=8)
        summary=aggregate_report(r)
        self.assertEqual(summary["per_bank"][4]["count"],0)
        self.assertEqual(summary["per_bank"][4]["count_source"],"aabb_disjoint_no_narrowphase")
    def test_null_result_without_aabb_proof_is_rejected(self):
        r=self.fixture(); r["banks"][4]["result"]=None
        with self.assertRaises(ValueError): aggregate_report(r)
    def test_mismatched_array_and_count_rejected(self):
        r=self.fixture(); r["banks"][2]["result"]["count"]=1
        with self.assertRaises(ValueError): aggregate_report(r)
    def test_incomplete_bank_set_rejected(self):
        r=self.fixture(); r["banks"].pop()
        with self.assertRaises(ValueError): aggregate_report(r)
    def test_incomplete_report_rejected(self):
        r=self.fixture(); r["complete"]=False
        with self.assertRaises(ValueError): aggregate_report(r)



def test_duplicate_typed_bank_cannot_fill_missing_inventory():
    r=AggregateTests().fixture()
    r["banks"][1]["body"]=r["banks"][0]["body"]
    r["banks"][1]["stable_bone_id"]=r["banks"][0]["stable_bone_id"]
    with __import__("pytest").raises(ValueError,match="duplicate"):
        aggregate_report(r)

def test_expected_bank_inventory_is_checked():
    r=AggregateTests().fixture()
    keys=[(i,i+1)for i in range(185)]
    assert aggregate_report(r,expected_bank_keys=keys)["total_crossing_pairs"]==2
    keys[-1]=(999,999)
    with __import__("pytest").raises(ValueError,match="inventory"):
        aggregate_report(r,expected_bank_keys=keys)

def test_boolean_is_not_an_anatomical_identity():
    r=AggregateTests().fixture();r["banks"][0]["body"]=False
    with __import__("pytest").raises(ValueError,match="identity"):
        aggregate_report(r)

def test_actual_run027_report_has_2317_crossings_not_zero():
    from pathlib import Path
    import json
    r=json.loads((Path(__file__).parent/"fixtures/human_skin_fat_surface_counts_027.json").read_text())
    summary=aggregate_report(r)
    assert summary["total_crossing_pairs"]==2317
    assert summary["nonzero_bank_count"]==33
