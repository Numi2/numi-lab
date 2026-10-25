import importlib.util,tempfile,json,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('r',Path(__file__).with_name('run_backtrack_with_resume_249.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ResumeTests(unittest.TestCase):
 def fixture(self,d,counts,status='EXACT_AUDIT_BUDGET_REACHED'):
  for i,c in enumerate(counts):
   p=d/f'pass-{i+1:03d}';p.mkdir();(p/'result.json').write_text(json.dumps({'exact_self_pairs':c}))
  p=d/f'pass-{len(counts)+1:03d}';p.mkdir();(p/'result.json').write_text(json.dumps({'status':status}));(p/'node-step-scales.npy').write_bytes(b'fixture')
 def test_small_decreasing_budget_tail_resumes_once(self):
  with tempfile.TemporaryDirectory()as t:
   d=Path(t);self.fixture(d,[100,4]);self.assertIsNotNone(m.continuation_scale_path(d))
 def test_large_stalled_or_nonbudget_tail_does_not_resume(self):
  for counts,status in [([100,20],'EXACT_AUDIT_BUDGET_REACHED'),([4,4],'EXACT_AUDIT_BUDGET_REACHED'),([100,4],'NO_COMBINED_PROGRESS')]:
   with tempfile.TemporaryDirectory()as t:
    d=Path(t);self.fixture(d,counts,status);self.assertIsNone(m.continuation_scale_path(d))
if __name__=='__main__':unittest.main()
