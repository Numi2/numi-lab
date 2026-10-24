import unittest
from human_skin_contact_history import accumulate_resolved_contacts as advance

class ContactHistoryTests(unittest.TestCase):
    def test_two_updates_preserve_resolved_and_add_new(self):
        history={"rest": {(1,2)}, "flex": set()}
        first=advance(history, {"rest": {(3,4)}, "flex": {(5,6)}},
                      {"rest": {(7,8)}, "flex": set()})
        second=advance(first, {"rest": {(7,8)}, "flex": set()},
                       {"rest": set(), "flex": {(9,10)}})
        self.assertEqual(second, {"rest": {(1,2),(3,4),(7,8)}, "flex": {(5,6)}})
        self.assertEqual(history, {"rest": {(1,2)}, "flex": set()})
    def test_recurrence_never_erases_guard(self):
        first=advance({"rest": set()}, {"rest": {(1,2)}}, {"rest": set()})
        second=advance(first, {"rest": set()}, {"rest": {(1,2)}})
        self.assertEqual(second["rest"], {(1,2)})
        self.assertIsNot(first["rest"], second["rest"])
    def test_empty_domains(self):
        self.assertEqual(advance({}, {}, {}), {})
    def test_missing_pose_fails(self):
        with self.assertRaises(ValueError):
            advance({"rest": set()}, {"rest": set()}, {})
if __name__=="__main__": unittest.main()
