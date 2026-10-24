"""Regression checks for captured-payload identity admission."""
from pathlib import Path
import tempfile,unittest
from audit_registered_skin_surface import sha,validate_launch_payload
class IdentityTests(unittest.TestCase):
 def test_changed_payload_at_same_path_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'skin';p.write_bytes(b'before');launch={'argv':['app','--skin-payload',str(p)],'input_hashes':{str(p):sha(p)}}
   validate_launch_payload(p,launch);p.write_bytes(b'after')
   with self.assertRaisesRegex(ValueError,'hash pin'):validate_launch_payload(p,launch)
 def test_same_bytes_at_different_unlaunched_path_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a';q=Path(d)/'b';p.write_bytes(b'asset');q.write_bytes(p.read_bytes())
   launch={'argv':['app','--skin-payload',str(p)],'input_hashes':{str(q):sha(q)}}
   with self.assertRaisesRegex(ValueError,'path differs'):validate_launch_payload(q,launch)
if __name__=='__main__':unittest.main()

