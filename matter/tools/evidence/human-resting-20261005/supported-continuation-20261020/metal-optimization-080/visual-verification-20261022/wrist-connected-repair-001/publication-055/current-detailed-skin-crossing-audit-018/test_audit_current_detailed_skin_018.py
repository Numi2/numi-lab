import tempfile,unittest
from pathlib import Path
from audit_current_detailed_skin_018 import load_native_positions
class NativePositionTests(unittest.TestCase):
 def parse(self,text,n):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'skin.csv';p.write_text(text);return load_native_positions(p,n)
 def test_exact_f32_native_rows(self):
  x=self.parse('vertex,wx,wy,wz\n0,0.5,-1,0.25\n1,1,2,3\n',2)
  self.assertEqual(x.dtype.name,'float32');self.assertEqual(x.shape,(2,3))
 def test_wrong_vertex_order_rejected(self):
  with self.assertRaisesRegex(ValueError,'order'):
   self.parse('vertex,wx,wy,wz\n1,0.5,-1,0.25\n0,1,2,3\n',2)
 def test_binary64_only_coordinates_rejected(self):
  with self.assertRaisesRegex(ValueError,'Float32'):
   self.parse('vertex,wx,wy,wz\n0,0.1,-1,0.25\n1,1,2,3\n',2)
if __name__=='__main__':unittest.main()
