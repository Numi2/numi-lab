import importlib.util
import unittest
import numpy as np
from pathlib import Path

SRC=Path('/Users/numi/Work/numi-human-connected-surface-069/digit-source-diagnosis-110/source_path_batch_110_v2.py')
spec=importlib.util.spec_from_file_location('source_path_batch_110_v2',SRC)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class SourcePathBatchTests(unittest.TestCase):
    def test_path_midpoint_uses_arclength_not_vertex_midpoint(self):
        pts=np.array([[0.,0,0],[3.,0,0],[3.,4,0]])
        got=mod.path_midpoint(pts)
        self.assertEqual(got['edge_index'],1)
        self.assertAlmostEqual(got['edge_fraction'],.125)
        np.testing.assert_allclose(got['point_m'],[3,.5,0])

    def test_shortest_path_tie_is_deterministic_and_reported(self):
        g={1:[(2,1.),(3,1.)],2:[(1,1.),(4,1.)],3:[(1,1.),(4,1.)],4:[(2,1.),(3,1.)]}
        p,r=mod.shortest_path_counted(g,1,4)
        self.assertEqual(p,[1,2,4])
        self.assertEqual(r['shortest_path_count_capped_2'],2)

    def test_layers_transport_existing_offsets_from_curved_O_midpoint(self):
        R=np.array([[0.,0,0],[2.,0,0],[0.,0,1],[2.,0,1],[0.,1,0],[2.,1,0]])
        C=R.astype(np.float32)
        Onew=np.array([1.,0.2,0.])
        Cnew=np.array([1.,0.3,0.])
        layers=mod.construct_midpoint_layers(Onew,Cnew,R,C,(0,1),.5,column_count=2)
        np.testing.assert_allclose(layers['O']['proposed_reference_position_m'],Onew)
        np.testing.assert_allclose(layers['I']['interpolated_offset_reference_m'],[0,0,1])
        np.testing.assert_allclose(layers['D']['interpolated_offset_reference_m'],[0,1,0])
        np.testing.assert_allclose(layers['I']['proposed_reference_position_m'],Onew+[0,0,1])
        np.testing.assert_allclose(layers['D']['proposed_current037_position_m_f64'],Cnew+[0,1,0])

    def test_layer_builder_rejects_out_of_range_fraction_and_rows(self):
        R=np.zeros((6,3)); C=R.astype(np.float32)
        with self.assertRaises(ValueError): mod.construct_midpoint_layers([0,0,0],[0,0,0],R,C,(0,1),1.01,2)
        with self.assertRaises(ValueError): mod.construct_midpoint_layers([0,0,0],[0,0,0],R,C,(1,2),.5,2)

if __name__=='__main__': unittest.main()
