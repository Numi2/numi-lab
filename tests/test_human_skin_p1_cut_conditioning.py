import unittest
import numpy as np
from human_skin_p1_cut_conditioning import select_conditioned_level, condition_nodal_cut_clearance


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

    def test_monotone_conditioning_resolves_near_both_endpoints(self):
        p=np.array([0.,.500000001,.499999999,1.])
        e=np.array([[0,1],[2,3],[0,2],[1,3]])
        out,r=condition_nodal_cut_clearance(p,e,[.5],minimum_fraction=.001,
            maximum_raise=.002,fixed_nodes=np.array([True,False,False,True]))
        self.assertTrue(r["conditioning_complete"])
        self.assertGreaterEqual(r["minimum_crossing_edge_fraction"],.001)
        self.assertTrue(np.all(out>=p))
        np.testing.assert_array_equal(out[[0,3]],p[[0,3]])

    def test_monotone_changes_preserve_all_lower_bound_obstacles(self):
        p=np.array([0.,.500000001,.499999999,1.])
        e=np.array([[0,1],[2,3],[0,2],[1,3]])
        out,r=condition_nodal_cut_clearance(p,e,[.5],minimum_fraction=.001,
            maximum_raise=.002,fixed_nodes=np.array([True,False,False,True]))
        bary=np.random.default_rng(91).dirichlet(np.ones(4),size=2000)
        self.assertTrue(np.all(bary@out>=bary@p))
        self.assertFalse(r["physical_admission"])

    def test_fixed_or_raise_budget_failure_remains_explicit(self):
        p=np.array([0.,.500000001,1.]);e=np.array([[0,1]])
        for fixed,cap in [(np.ones(3,dtype=bool),.01),
                          (np.zeros(3,dtype=bool),1e-10)]:
            out,r=condition_nodal_cut_clearance(p,e,[.5],minimum_fraction=.001,
                maximum_raise=cap,fixed_nodes=fixed)
            self.assertFalse(r["conditioning_complete"])
            self.assertEqual(r["unresolved_edges"],1)

    def test_empty_edges_and_constant_fields(self):
        for edges in [np.empty((0,2),dtype=int),np.array([[0,1]])]:
            out,r=condition_nodal_cut_clearance([1.,1.],edges,[.5],
                minimum_fraction=.01,maximum_raise=.1,fixed_nodes=np.zeros(2,dtype=bool))
            np.testing.assert_array_equal(out,[1.,1.])
            self.assertTrue(r["conditioning_complete"]);self.assertEqual(r["changed_nodes"],0)

    def test_multiple_levels_have_no_silent_budget_escape(self):
        p=np.array([0.,.20000001,.19999999,.50000001,.49999999,1.])
        e=np.array([[i,j] for i in range(6) for j in range(i+1,6)])
        out,r=condition_nodal_cut_clearance(p,e,[.2,.5],minimum_fraction=.001,
            maximum_raise=.004,fixed_nodes=np.array([True,False,False,False,False,True]))
        self.assertTrue(r["conditioning_complete"])
        self.assertLessEqual(r["maximum_actual_raise"],.004)
        np.testing.assert_array_equal(out[[0,5]],p[[0,5]])

    def test_retained_hand_float32_microcut_regression(self):
        # Exact parent tet 242790, source grid nodes
        # [1862061,1879377,1879525,1879526] from retained hand trial049.
        # Construction conditioning is bounded; this is not anatomical admission.
        from human_skin_patch_geometry import extract_p1_tetrahedral_interface
        x=np.array([[.029777138446644827,-.005191282868392572,-.01073849445581436],
                    [.031027138446644828,-.005191282868392572,-.01073849445581436],
                    [.031027138446644828,-.0039412828683925705,-.01073849445581436],
                    [.031027138446644828,-.0039412828683925705,-.008988494455814358]])
        t=np.array([[0,1,2,3]])
        p=np.array([-.00433978158980608,-.004117205273360014,
                    -.003975813277065754,2.3283064365386963e-10])
        with self.assertRaisesRegex(ValueError,"degenerates or reverses after Float32"):
            extract_p1_tetrahedral_interface(x,t,p,0.)
        conditioned,report=condition_nodal_cut_clearance(
            p,np.array([[i,j] for i in range(4) for j in range(i+1,4)]),[0.],
            minimum_fraction=.005,maximum_raise=2e-5,
            fixed_nodes=np.zeros(4,dtype=bool))
        self.assertFalse(report["physical_admission"])
        self.assertTrue(np.all(conditioned>=p))
        self.assertLessEqual(float(np.max(conditioned-p)),2e-5)
        v,f,edge,alpha=extract_p1_tetrahedral_interface(x,t,conditioned,0.)
        self.assertEqual(v.shape,(3,3));self.assertEqual(f.shape,(1,3))
        reference=(1-alpha[:,None])*x[edge[:,0]]+alpha[:,None]*x[edge[:,1]]
        a=reference[f[0]];b=v.astype(np.float64)[f[0]]
        self.assertGreater(float(np.dot(np.cross(a[1]-a[0],a[2]-a[0]),
                                       np.cross(b[1]-b[0],b[2]-b[0]))),0.)

    def test_nodal_invalid_domains_rejected(self):
        p=[0.,1.];e=np.array([[0,1]]);f=np.array([True,False])
        bad=[dict(minimum_fraction=0),dict(maximum_raise=0),dict(fixed_nodes=[1,0]),
             dict(levels=[.5,.4]),dict(edges=np.array([[0,2]])),dict(iterations=0)]
        defaults=dict(values=p,edges=e,levels=[.5],minimum_fraction=.001,
                      maximum_raise=.01,fixed_nodes=f)
        for change in bad:
            with self.subTest(change=change),self.assertRaises(ValueError):
                condition_nodal_cut_clearance(**dict(defaults,**change))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,1.],np.array([[0,2]]),.5,.1)
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,1.],np.array([[0,1]]),.5,0)
        with self.assertRaises(ValueError):
            select_conditioned_level([0.,np.nan],np.array([[0,1]]),.5,.1)


if __name__=="__main__":unittest.main()

