import unittest,sys
from pathlib import Path
O=Path(__file__).parent
sys.path[:0]=[str(O/'src'),str(O/'tests')]
import test_cardiac_cavity_intersections as tests
def flatten(suite):
 for item in suite:
  if isinstance(item,unittest.TestSuite):yield from flatten(item)
  else:yield item
cases=[c for c in flatten(unittest.defaultTestLoader.loadTestsFromModule(tests))
       if not isinstance(c,tests.SourceGeometryTests)]
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))
raise SystemExit(not result.wasSuccessful())
