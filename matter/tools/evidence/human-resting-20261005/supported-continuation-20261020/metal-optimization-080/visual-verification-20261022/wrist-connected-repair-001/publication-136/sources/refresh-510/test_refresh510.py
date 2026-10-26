from pathlib import Path
import importlib.util,sys,unittest
P=Path(__file__).with_name("run_refresh510.py")
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def test_preparation_only_and_pins(self):
  a=load("adapter510_test",P);b=load("base491_test",a.BASE)
  for p,h in a.PINS.items():self.assertEqual(a.sha(p),h)
  old={n:b.owner_command(b.TEMPLATES/(n+"-argv.json"),o)for n,o in [("depth",b.DEPTH),("guards",b.GUARDS),("prep",b.PREP)]}
  a.install(b)
  for n,o in [("depth",b.DEPTH),("guards",b.GUARDS),("prep",b.PREP)]:
   actual=b.owner_command(b.TEMPLATES/(n+"-argv.json"),o)
   expected=list(old[n])
   if n=="prep":expected[1]=str(a.ADAPTER)
   self.assertEqual(actual,expected)
  for p in [a.ADAPTER,a.SOURCE,P]:self.assertIn(p,b.EXTRA_SOURCES)
 def test_unknown_owner_rejected(self):
  a=load("adapter510_negative",P);b=load("base491_negative",a.BASE);a.install(b)
  with self.assertRaises(ValueError):b.owner_command(b.TEMPLATES/"prep-argv.json",Path("/no/owner.py"))
if __name__=="__main__":unittest.main()
