import unittest
import numpy as np
from human_skin_p1_cut_conditioning import select_conditioned_level


class CutConditioningTests(unittest.TestCase):
    def test_single_edge_finds_balanced_cut(self):
        r=select_conditioned_level([0.,1.],np.array([[0,1]]),.49,.1)
        self.assertAlmostEqual(r["selected_level"],.5)
        self.assertAlmostEqual(r["minimum_edge_fraction"],.5)
        self.assertFalse(r["physical_admission"])

    def test_avoids_near_source_vertex_and_respects_window(self):
        p=np.array([0.,.5000001,1.])
        e=np.array([[0,1],[1,2],[0,2]])
        r=select_conditioned_level(p,e,.5,.01)
        self.assertGreater(r["minimum_edge_fraction"],.019)
        self.assertLessEqual(abs(r["actual_shift"]),.01)
        l=r["selected_level"];a=p[e].min(1);b=p[e].max(1);m=(a<l)&(b>l)
        self.assertAlmostEqual(r["minimum_edge_fraction"],np.minimum((l-a[m])/(b[m]-a[m]),(b[m]-l)/(b[m]-a[m])).min())

    def test_scale_offset_and_edge_order(self):
        p=np.array([0.,.1,.46,.8,1.]);e=np.array([[0,2],[1,3],[2,4],[0,4]])
        a=select_conditioned_level(p,e,.48,.04)
        b=select_conditioned_level(p*20+3,e[::-1],.48*20+3,.04*20)
        self.assertAlmostEqual(a["selected_level"]*20+3,b["selected_level"])
        self.assertAlmostEqual(a["minimum_edge_fraction"],b["minimum_edge_fraction"])

    def test_constant_edges_do_not_create_cuts(self):
        with self.assertRaisesRegex(ValueError,"no nonempty"):
            select_conditioned_level([1.,1.],np.array([[0,1]]),1,.1)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,1.],np.array([[0,2]]),.5,.1)
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,1.],np.array([[0,1]]),.5,0)
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,np.nan],np.array([[0,1]]),.5,.1)


if __name__=="__main__":unittest.main()

