import unittest
import numpy as np
from reference_fit_layer_counterfactual import map_source_anchors, shift_all_layers, tet_metrics, N, SENTINEL

class ReferenceFitLayerTests(unittest.TestCase):
    def test_direct_and_barycentric_source_anchor_mapping(self):
        src=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        faces=np.array([[0,1,2],[0,1,3]])
        ids=np.array([2,SENTINEL]+[SENTINEL]*(N-2),dtype=np.int64)
        fids=np.array([0,0]+[0]*(N-2),dtype=np.int64)
        w=np.zeros((N,3)); w[:,0]=.2; w[:,1]=.3; w[:,2]=.5
        R=np.eye(3); t=np.array([2.,3.,4.])
        targets,direct,_=map_source_anchors(src,faces,ids,fids,w,R,t)
        np.testing.assert_allclose(targets[0],[2,4,4])
        np.testing.assert_allclose(targets[1],[2.3,3.5,4])
        self.assertEqual(int(direct.sum()),1)

    def test_same_column_delta_preserves_layer_thickness(self):
        points=np.zeros((3*N,3),dtype=float).reshape(3,N,3)
        points[0,:,0]=0.; points[1,:,0]=2.; points[2,:,0]=5.
        delta=np.arange(N,dtype=float)[:,None]*np.array([[1e-5,0.,0.]])
        result=shift_all_layers(points.reshape(3*N,3),delta).reshape(3,N,3)
        np.testing.assert_allclose(result[1]-result[0],points[1]-points[0],atol=1e-14)
        np.testing.assert_allclose(result[2]-result[1],points[2]-points[1],atol=1e-14)
        np.testing.assert_allclose(result-points,np.broadcast_to(delta[None,:,:],result.shape),atol=1e-14)

    def test_regular_tet_has_unit_mean_ratio_and_positive_det(self):
        pts=np.array([[0.,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],
                      [.5,np.sqrt(3)/6,np.sqrt(2/3)]])
        det,vol,q=tet_metrics(pts,np.array([[0,1,2,3]]))
        self.assertGreater(det[0],0)
        self.assertAlmostEqual(float(vol[0]),1/(6*np.sqrt(2)),places=14)
        self.assertAlmostEqual(float(q[0]),1.0,places=14)

if __name__=="__main__":
    unittest.main()
