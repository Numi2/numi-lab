import unittest
import numpy as np
from human_skin_p1_material_volume import split_material_bands,material_role_faces,exterior_source_face_rows,check_material_volume_targets
from human_skin_material_boundary import certify_material_boundaries


def volume(x,t):
    v=x[t].astype(float);e=v[:,1:]-v[:,:1]
    return np.einsum("ij,ij->i",e[:,0],np.cross(e[:,1],e[:,2]))/6


class MaterialVolumeTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float32)
        self.t=np.array([[0,1,2,3],[0,2,1,4]],np.int64)

    def certify(self,r):
        roles=material_role_faces(r["tetrahedra"],r["material_ids"])
        return certify_material_boundaries(r["points_world_m"],r["tetrahedra"],
                    r["material_ids"],**roles,minimum_quality=1e-12)

    def test_analytical_linear_band_volumes(self):
        r=split_material_bands(self.x,self.t,self.x[:,0],.25,.75)
        vv=volume(r["points_world_m"],r["tetrahedra"])
        self.assertAlmostEqual(vv[r["material_ids"]==1].sum(),(1-.75**3)/3,places=7)
        self.assertAlmostEqual(vv.sum(),(1-.25**3)/3,places=7)
        self.assertTrue(self.certify(r)["boundary_incidence_pass"])

    def test_shared_crossings_and_interface_facets(self):
        r=split_material_bands(self.x,self.t,self.x[:,0],.25,2)
        np.testing.assert_allclose(volume(r["points_world_m"],r["tetrahedra"]).sum(),1/3)
        r2=self.certify(r)
        self.assertGreater(len(r2["interface_faces"]),0)
        np.testing.assert_array_equal(np.unique(r["material_ids"]),[1,2])
        # Every original edge/level has exactly one emitted node identity.
        keys=list(zip(map(tuple,r["source_node_ids"]),r["source_field_values"]))
        self.assertEqual(len(keys),len(set(keys)))

    def test_parent_edge_ancestry_replays_positions(self):
        p=np.array([-.4,.8,.5,.2,1.2])
        r=split_material_bands(self.x,self.t,p,.25,.7)
        a=r["source_edge_alpha"];ij=r["source_node_ids"]
        replay=(1-a[:,None])*self.x[ij[:,0]]+a[:,None]*self.x[ij[:,1]]
        np.testing.assert_array_equal(replay.astype(np.float32),r["points_world_m"])
        np.testing.assert_allclose((1-a)*p[ij[:,0]]+a*p[ij[:,1]],r["source_field_values"],atol=1e-15)
        self.assertEqual(set(r["source_tetrahedron_rows"]),{0,1})
        self.certify(r)

    def test_exact_level_vertex_keeps_source_identity(self):
        p=np.array([.25,1,0,0,0])
        r=split_material_bands(self.x,self.t,p,.25,2)
        self.assertEqual(sum((r["source_node_ids"]==[0,0]).all(axis=1)),1)
        self.certify(r)

    def test_constant_fields_and_empty_volume(self):
        r=split_material_bands(self.x,self.t,np.full(5,.1),.25,2)
        np.testing.assert_array_equal(r["material_ids"],[1,1])
        self.assertAlmostEqual(volume(r["points_world_m"],r["tetrahedra"]).sum(),1/3)
        with self.assertRaisesRegex(ValueError,"no positive"):
            split_material_bands(self.x,self.t,np.full(5,3.),.25,2)

    def test_random_fields_full_volume_and_conforming_interface(self):
        rng=np.random.default_rng(119)
        for k in range(30):
            p=rng.uniform(-1,1,5);level=float(np.median(p))
            with self.subTest(case=k):
                r=split_material_bands(self.x,self.t,p,level,2.)
                vv=volume(r["points_world_m"],r["tetrahedra"])
                self.assertTrue((vv>0).all())
                self.assertAlmostEqual(vv.sum(),1/3,places=7)
                self.certify(r)

    def test_constant_exact_levels_own_volume_once(self):
        for value,material in ((.25,1),(2.,2)):
            with self.subTest(value=value):
                r=split_material_bands(self.x,self.t,np.full(5,value),.25,2.)
                np.testing.assert_array_equal(r["material_ids"],[material,material])
                self.assertAlmostEqual(volume(r["points_world_m"],r["tetrahedra"]).sum(),1/3)

    def test_source_boundary_ancestry_distinguishes_isosurfaces(self):
        source=material_role_faces(self.t,np.ones(2,np.uint32))["outer_faces"]
        r=split_material_bands(self.x,self.t,self.x[:,0],.25,2.)
        roles=material_role_faces(r["tetrahedra"],r["material_ids"])
        parents=exterior_source_face_rows(r,roles,source)
        self.assertTrue((parents["outer_faces"]>=0).all())
        self.assertTrue((parents["deep_faces"]>=0).all())
        self.assertTrue((parents["interface_faces"]==-1).all())

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError,"ordered"):
            split_material_bands(self.x,self.t,np.zeros(5),1,1)
        with self.assertRaisesRegex(ValueError,"positive"):
            split_material_bands(self.x,self.t[:,[0,2,1,3]],np.zeros(5),.2,.3)
        with self.assertRaisesRegex(ValueError,"finite source"):
            split_material_bands(self.x,self.t,[0,1,2,np.nan,4],.2,.3)
        with self.assertRaisesRegex(ValueError,"integer"):
            split_material_bands(self.x,self.t.astype(float),np.zeros(5),.2,.3)

    def test_flat_minimum_makes_small_target_unattainable(self):
        p=np.zeros(5)
        r=check_material_volume_targets(self.x,self.t,p,{"skin":.01,"all":1/3,"none":0.})
        self.assertFalse(r["necessary_conditions_pass"])
        self.assertEqual([v["target"] for v in r["failures"]],["skin"])
        self.assertEqual(r["constant_field_tetrahedra"],2)
        np.testing.assert_allclose(r["guaranteed_volume_gaps"][0]["unattainable_open_interval_m3"],[0,1/3])

    def test_interior_jump_is_also_detected(self):
        # Disconnected tetrahedra with constant values 0, 1 and 2.
        x=np.vstack([self.x[:4]+[3*i,0,0] for i in range(3)])
        t=np.arange(12).reshape(-1,4)
        p=np.repeat([0.,1.,2.],4)
        r=check_material_volume_targets(x,t,p,{"inside_middle_jump":.25,"at_lower":1/6,"at_upper":1/3})
        self.assertEqual([v["target"] for v in r["failures"]],["inside_middle_jump"])
        self.assertEqual(r["failures"][0]["field_value"],1.)

    def test_nonconstant_field_and_outside_domain(self):
        r=check_material_volume_targets(self.x,self.t,self.x[:,0],{"skin":.01,"too_large":1.})
        self.assertEqual(r["constant_field_tetrahedra"],0)
        self.assertEqual(r["guaranteed_volume_gaps"],[])
        self.assertEqual([v["target"] for v in r["failures"]],["too_large"])

    def test_nearly_constant_is_not_flat(self):
        p=np.array([0.,1e-20,0.,0.,0.])
        r=check_material_volume_targets(self.x,self.t,p,{"skin":.01})
        self.assertTrue(r["necessary_conditions_pass"])
        self.assertEqual(r["constant_field_tetrahedra"],0)

    def test_target_check_rejects_invalid_geometry_and_targets(self):
        for target in ({"skin":float("nan")},{"skin":-1.},{"skin":True},{}):
            with self.subTest(target=target),self.assertRaises(ValueError):
                check_material_volume_targets(self.x,self.t,np.zeros(5),target)
        with self.assertRaisesRegex(ValueError,"positive"):
            check_material_volume_targets(self.x,self.t[:,[0,2,1,3]],np.zeros(5),{"skin":.1})
        with self.assertRaisesRegex(ValueError,"finite source"):
            check_material_volume_targets(self.x,self.t,[0,0,0,0,float("nan")],{"skin":.1})


if __name__=="__main__":unittest.main()

