#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=ROOT/'E/connected-skin-mechanical-transfer-013'
BUILDER=BASE/'build_boundary_refinement_013_v2.py'
CAND=BASE/'boundary-refinement-002/candidate.npz'
REPORT=BASE/'boundary-refinement-002/report.json'
EXPECTED_SHA='1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1'

spec=importlib.util.spec_from_file_location('builder013',BUILDER)
builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
spec2=importlib.util.spec_from_file_location('helper403',builder.HELPER403)
helper=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(helper)

class BoundaryRefinement013Tests(unittest.TestCase):
    def test_sequential_boundary_templates_match_tets(self):
        X=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        T=np.asarray([[0,1,2,3]],dtype=np.int64)
        oldB=helper.oriented_boundary_faces(T)
        for edges in [[(0,1)],[(0,1),(0,2)],[(0,1),(0,2),(1,2)]]:
            x,t,parent,e=helper.split_edges_globally(X,T,edges)
            midpoint={tuple(map(int,edge)):len(X)+i for i,edge in enumerate(e)}
            expected=builder.sequential_boundary_children(oldB,e,midpoint)
            actual=helper.oriented_boundary_faces(t)
            self.assertEqual(set(map(builder.canonical,actual)),
                             set(builder.canonical(tri) for _,tris in expected for tri in tris))
            self.assertTrue(np.array_equal(x[:len(X)],X))
            self.assertTrue(np.all(builder.dets_f64(x,t)>0))
            self.assertTrue(np.all(builder.dets_f32(x,t)>0))

    def test_parent_volume_shares_conserve_csr_columns(self):
        old_offsets=np.asarray([0,2,3],np.int64)
        old_rows=np.asarray([4,9,4],np.int64)
        old_weights=np.asarray([0.3,0.7,1.0],np.float64)
        child_parent=np.asarray([0,0,1,1,1],np.int64)
        child_vol=np.asarray([0.2,0.8,0.2,0.3,0.5],np.float64)
        sums=np.bincount(child_parent,weights=child_vol,minlength=2)
        off,rows,w=builder.remap_csr_volume_fraction(old_offsets,old_rows,old_weights,child_parent,child_vol,sums)
        old=builder.column_sums(old_offsets,old_rows,old_weights)
        new=builder.column_sums(off,rows,w)
        n=max(len(old),len(new)); old=np.pad(old,(0,n-len(old))); new=np.pad(new,(0,n-len(new)))
        self.assertLess(float(np.max(np.abs(old-new))),1e-15)
        self.assertEqual(len(off),len(child_parent)+1)

    def test_midpoint_source_ancestry_is_affine(self):
        ptr=np.asarray([0,1,2],np.int64)
        inds=np.asarray([101,205],np.int64)
        weights=np.asarray([1.0,1.0],np.float64)
        outptr,outinds,outweights,sums=builder.midpoint_source_csr(ptr,inds,weights,np.asarray([[0,1]],np.int64))
        self.assertEqual(outptr.tolist(),[0,1,2,4])
        self.assertEqual(outinds[-2:].tolist(),[101,205])
        self.assertTrue(np.allclose(outweights[-2:],[0.5,0.5],rtol=0,atol=0))
        self.assertTrue(np.allclose(sums,[1.0],rtol=0,atol=1e-15))

    def test_emitted_shell_and_parent_barycentric_map(self):
        self.assertEqual(builder.sha256(CAND),EXPECTED_SHA)
        with np.load(CAND,allow_pickle=False) as c, np.load(ROOT/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz',allow_pickle=False) as a:
            X=c['points_world_m']; T=c['tetrahedra']
            shell=c['physical_shell_faces']
            self.assertEqual(len(helper.oriented_boundary_faces(T)),len(shell))
            self.assertEqual(set(map(builder.canonical,helper.oriented_boundary_faces(T))),set(map(builder.canonical,shell)))
            faces=c['physical_outer_faces']; parents=c['physical_O_parent_face_rows']; bary=c['physical_O_parent_barycentric']
            self.assertEqual(bary.shape,(len(faces),3,3))
            self.assertEqual(parents.shape,(len(faces),))
            self.assertTrue(np.allclose(bary.sum(axis=2),1.0,rtol=0,atol=1e-15))
            source=a['physical_outer_faces'][parents]
            predicted=np.einsum('fij,fjk->fik',bary,X[source].astype(np.float64))
            actual=X[faces].astype(np.float64)
            self.assertLess(float(np.max(np.abs(predicted-actual))),1e-7)
            self.assertEqual(len(c['physical_O_parent_detailed_source_face_ids']),len(faces))
            self.assertEqual(len(c['midpoint_edge_endpoints']),857)

    def test_emitted_transfer_columns_and_lumped_volume(self):
        with np.load(CAND,allow_pickle=False) as c, np.load(ROOT/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz',allow_pickle=False) as a:
            for name in ('005','014'):
                old=builder.column_sums(a[f'transfer_new_to_{name}_offsets'],a[f'transfer_new_to_{name}_parent_rows'],a[f'transfer_new_to_{name}_parent_fractions'])
                new=builder.column_sums(c[f'transfer_new_to_{name}_offsets'],c[f'transfer_new_to_{name}_parent_rows'],c[f'transfer_new_to_{name}_parent_fractions'])
                n=max(len(old),len(new)); old=np.pad(old,(0,n-len(old))); new=np.pad(new,(0,n-len(new)))
                self.assertLess(float(np.max(np.abs(old-new))),1e-12)
            X=c['points_world_m']; T=c['tetrahedra']
            vol=np.abs(builder.dets_f64(X,T))/6
            self.assertTrue(np.all(vol>0))
            cost=np.zeros(len(X),dtype=np.float64); np.add.at(cost,T.ravel(),np.repeat(vol/4,4))
            self.assertTrue(np.allclose(cost,c['lumped_volume_cost_m3'],rtol=1e-14,atol=1e-15))
            self.assertLess(abs(float(cost.sum()-vol.sum())),1e-14)

if __name__=='__main__': unittest.main(verbosity=2)
