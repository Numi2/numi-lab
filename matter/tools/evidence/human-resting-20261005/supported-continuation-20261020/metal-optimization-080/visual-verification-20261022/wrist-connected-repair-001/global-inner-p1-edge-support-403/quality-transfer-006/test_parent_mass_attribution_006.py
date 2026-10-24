from pathlib import Path
import ast,unittest,numpy as np
SRC=Path(__file__).with_name('build_conservative_transfer_006.py')
tree=ast.parse(SRC.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='attributed_child_mass')
ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(SRC),'exec'),ns)
transfer=ns['attributed_child_mass']
class ParentMassAttributionTests(unittest.TestCase):
 def test_source_columns_conserve_mass_while_child_rows_need_not_sum_one(self):
  result=transfer([0,1,3,4],[0,0,1,1],[.25,.75,.4,.6],[2.,3.])
  np.testing.assert_allclose(result,[.5,2.7,1.8],rtol=0,atol=1e-15)
  self.assertAlmostEqual(result.sum(),5.)
 def test_composed_attribution_matches_sequential_transfer(self):
  old=transfer([0,2,4],[0,1,1,2],[1.,.2,.8,1.],[2.,3.,5.])
  result=transfer([0,1,3,4],[0,0,1,1],[.25,.75,.4,.6],old)
  composed=transfer([0,2,5,7],[0,1,0,1,2,1,2],[.25,.05,.75,.47,.4,.48,.6],[2.,3.,5.])
  np.testing.assert_allclose(result,composed,rtol=0,atol=1e-14)
  self.assertAlmostEqual(result.sum(),10.)
 def test_invalid_maps_fail_closed(self):
  for off,parents,weights in [([1,2],[0],[1.]),([0,2],[0],[1.]),([0,1],[2],[1.]),([0,1],[0],[-1.]),([0,1],[.5],[1.]),([0.,1.],[0],[1.])]:
   with self.assertRaises(ValueError):transfer(off,parents,weights,[1.])
 def test_empty_output_and_empty_child_rows(self):
  np.testing.assert_array_equal(transfer([0],[],[],[1.]),np.empty(0))
  np.testing.assert_array_equal(transfer([0,0,1],[0],[1.],[2.]),[0.,2.])
if __name__=='__main__':unittest.main()

