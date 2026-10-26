import unittest
from verify_connected_construction_522 import receipt_passes
class Gates(unittest.TestCase):
    def setUp(self):
        self.proof=dict(status="ROOT_INDEPENDENT_CONSTRUCTION_PASS",candidate_sha256="abc")
        self.row=dict(label="selected",predicate_pass=True,candidate_sha256="abc")
        self.edge=dict(candidate_checks=[self.row])
    def test_pass(self):
        self.assertTrue(receipt_passes(self.proof,self.edge,"abc",[0,0],True))
    def test_zero_exit_code_is_not_pass(self):
        self.row["predicate_pass"]=False
        self.assertFalse(receipt_passes(self.proof,self.edge,"abc",[0,0],True))
    def test_wrong_candidate(self):
        self.assertFalse(receipt_passes(self.proof,self.edge,"def",[0,0],True))
    def test_missing_receipt(self):
        self.assertFalse(receipt_passes({},self.edge,"abc",[0,0],True))
    def test_changed_input(self):
        self.assertFalse(receipt_passes(self.proof,self.edge,"abc",[0,0],False))
if __name__=="__main__":unittest.main()
