import importlib.util,sys,tempfile,unittest
from pathlib import Path
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
SCRIPT=D/"refresh-491/run_refresh491.py"
def load():
 s=importlib.util.spec_from_file_location("refresh491",SCRIPT);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
class Refresh491PlanTests(unittest.TestCase):
 def test_matched_476_template_rebuilds_required_hash_flags_and_previous_guard_binding(self):
  m=load()
  current=D/"outward-plane-backtrack-461/run-001/backtrack-373/selected-result.json"
  previous=D/"constraint-project-backtrack-456/run-001/backtrack-373/selected-result.json"
  prep=D/"connected-continuation-prep-457/run-001"
  guard=D/"fixed-bone-guard-refresh-457/run-001"
  fails=[D/"component-backtrack-479/run-001/backtrack-373",D/"soft-target-backtrack-482/run-002/backtrack-373"]
  with tempfile.TemporaryDirectory() as td:
   plan=m.build_plan(current,previous,prep,guard,Path(td)/"out",fails,False)
   import json
   out=Path(td)/"out"; d=json.loads((out/"depth-argv.json").read_text()); p=json.loads((out/"prep-argv.json").read_text()); g=json.loads((out/"guards-argv.json").read_text())
   for flag in ("--expected-candidate-sha256","--expected-point-summary-sha256","--expected-surface-report-sha256","--expected-query-receipt-sha256","--expected-owner-sha256"): self.assertEqual(d.count(flag),1)
   for flag in ("--candidate-sha256","--audit130-sha256","--audit159-sha256","--prior-seed-report-sha256","--prior-seed-pairs-sha256","--failed-trials-manifest-sha256","--point-depth-npz-sha256","--point-depth-report-sha256"): self.assertEqual(p.count(flag),1)
   self.assertNotIn("/tmp/numi-human-geometry-slot.py",d+g+p)
   def arg(a,k): return a[a.index(k)+1]
   self.assertEqual(arg(g,"--previous-candidate"),str(m.selected(previous)["candidate"]))
   self.assertNotEqual(arg(g,"--previous-candidate"),arg(g,"--candidate"))
   self.assertGreater(plan["failed_trials_included"],0)
   manifest=json.loads((out/"failed-trials.json").read_text())
   self.assertTrue(all(x["exact_self_pair_count"]>0 for x in manifest["trials"]))
   self.assertEqual(plan["outputs"]["connected_prep"].split("/")[-1],"run-001")
if __name__=="__main__": unittest.main()
