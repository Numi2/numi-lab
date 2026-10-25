import unittest
import numpy as np
from scipy import sparse
import human_skin_fat_connected_repair as repair

class InputValidation037Tests(unittest.TestCase):
    def setUp(self):
        self.X=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        self.R=self.X.astype(np.float64)
        self.T=np.asarray([[0,1,2,3]],dtype=np.int64)
        self.F=np.asarray([0],dtype=np.int64)
        self.C=sparse.csr_matrix(([1.0],([0],[0])),shape=(1,12))
        self.target=np.asarray([0.01])
    def call(self,**kw):
        args=dict(points=self.X,reference=self.R,tetrahedra=self.T,free_nodes=self.F,contact_matrix=self.C,contact_targets=self.target)
        args.update(kw)
        return repair.coupled_clearance_step(**args)
    def test_empty_tet_domain_fails_closed_in_both_owners(self):
        with self.assertRaisesRegex(ValueError,'nonempty'):
            repair.cell_values_gradients(self.X,self.R,np.empty((0,4),dtype=np.int64))
        with self.assertRaisesRegex(ValueError,'nonempty'):
            self.call(tetrahedra=np.empty((0,4),dtype=np.int64))
    def test_float_and_repeated_tet_ids_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError,'integer'):
            self.call(tetrahedra=self.T.astype(float))
        with self.assertRaisesRegex(ValueError,'repeat'):
            self.call(tetrahedra=np.asarray([[0,1,1,3]],dtype=np.int64))
    def test_empty_or_float_free_node_ids_fail_closed(self):
        for f in (np.empty(0,dtype=np.int64),np.asarray([0.0])):
            with self.subTest(free_nodes=f),self.assertRaises(ValueError):self.call(free_nodes=f)
    def test_empty_or_zero_contact_domain_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'empty contact'):
            self.call(contact_matrix=sparse.csr_matrix((0,12)),contact_targets=np.empty(0))
        with self.assertRaisesRegex(ValueError,'empty contact'):
            self.call(contact_matrix=sparse.csr_matrix((1,12)),contact_targets=np.asarray([0.0]))
    def test_nonfinite_and_reversed_physical_bounds_fail_closed(self):
        cases=({'J_bounds':(np.nan,2.5)},{'J_bounds':(2.5,0.35)},{'q_min':np.inf},{'step_limit_mm':np.nan},{'time_limit_s':np.inf},{'max_incremental_gradient':np.nan})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'bound'):
                self.call(**kwargs)
    def test_bad_layer_rows_and_targets_fail_closed(self):
        L=sparse.csr_matrix(np.asarray([[np.nan]+[0.0]*11]))
        with self.assertRaisesRegex(ValueError,'layer constraint'):
            self.call(lower_linear_matrix=L,lower_linear_targets=np.asarray([0.0]))
        L=sparse.csr_matrix(([1.0],([0],[0])),shape=(1,12))
        with self.assertRaisesRegex(ValueError,'layer constraint'):
            self.call(lower_linear_matrix=L,lower_linear_targets=np.asarray([np.inf]))
        with self.assertRaisesRegex(ValueError,'layer constraint'):
            self.call(lower_linear_matrix=sparse.csr_matrix((0,12)),lower_linear_targets=np.empty(0))
    def test_sparse_jacobian_rejects_empty_or_noninteger_ids(self):
        G=np.ones((1,4,3))
        with self.assertRaisesRegex(ValueError,'nonempty integer'):
            repair.sparse_cell_jacobian(G,np.empty((0,4),dtype=np.int64),self.F,4)
        with self.assertRaisesRegex(ValueError,'integer'):
            repair.sparse_cell_jacobian(G,self.T,np.asarray([0.0]),4)
        with self.assertRaisesRegex(ValueError,'nonempty integer'):
            repair.sparse_cell_jacobian(G,self.T,np.empty(0,dtype=np.int64),4)

if __name__=='__main__':unittest.main()
