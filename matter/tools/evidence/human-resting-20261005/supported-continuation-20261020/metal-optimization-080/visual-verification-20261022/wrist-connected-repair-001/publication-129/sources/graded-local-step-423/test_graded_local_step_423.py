import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("graded_local_step_423",HERE/"graded_local_step_423.py")
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)

def make_context(path):
    faces=np.array([[0,1,2],[2,3,4],[4,5,6],[7,8,9]],np.int64)
    pts=np.array([[0,0,0],[.001,0,0],[0,.001,0],[.001,.001,0],[.002,.001,0],
                  [.002,.002,0],[.003,.002,0],[.01,0,0],[.011,0,0],[.01,.001,0]],np.float64)
    delta=np.zeros_like(pts);delta[:,0]=.001
    edges=owner._edges_from_faces(faces)
    lengths=np.linalg.norm(pts[edges[:,0]]-pts[edges[:,1]],axis=1)
    amps=np.maximum(np.linalg.norm(delta[edges[:,0]],axis=1),np.linalg.norm(delta[edges[:,1]],axis=1))
    budgets=np.minimum(1,.1*lengths/amps)
    anchors=np.array([9],np.int64)
    np.savez(path,faces=faces,edges=edges,edge_scale_budget=budgets,points_baseline_m=pts,
             proposal_displacement_m=delta,anchors=anchors)
    return faces,anchors

class GradedLocalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"ctx.npz"
        self.faces,self.anchors=make_context(self.path)
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def call_helper(self,scales=None,bad=(0,),pairs=()):
        env={"NUMI_GRADED_LOCAL_STEP_CONTEXT_423":str(self.path),
             "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423":self.digest,
             "NUMI_GRADED_LOCAL_STEP_KAPPA_423":"0.1"}
        with patch.dict(os.environ,env,clear=False):
            return owner.reduce_witness_node_scales(
                self.faces,np.ones(10) if scales is None else scales,
                np.asarray(bad,dtype=np.int64),np.asarray(pairs,dtype=np.int64).reshape(-1,2),
                anchors=self.anchors,zero_below=1/128)

    def test_local_half_step_is_graded_over_adjacent_edges(self):
        out,report=self.call_helper()
        self.assertEqual(report["status"],"GRADED_LOCAL_WITNESS_SCALES_ONLY")
        np.testing.assert_allclose(out[:3],.5)
        self.assertLessEqual(out[3],.6+1e-12)
        self.assertLessEqual(out[5],.8+1e-12)
        self.assertEqual(out[9],0.0)
        self.assertLessEqual(report["max_edge_budget_excess_nonanchor_edges"],2e-12)

    def test_self_pair_witness_reduces_both_faces_and_grades_neighbors(self):
        out,report=self.call_helper(bad=(),pairs=((0,2),))
        np.testing.assert_allclose(out[[0,1,2,4,5,6]],.5)
        self.assertLessEqual(out[3],.6+1e-12)
        self.assertEqual(out[9],0.0)

    def test_disconnected_nonincident_region_is_bitwise_unchanged(self):
        out,_=self.call_helper()
        np.testing.assert_array_equal(out[[7,8]],np.ones(2,dtype=np.float64))

    def test_context_hash_topology_and_anchor_mismatch_fail_closed(self):
        env={"NUMI_GRADED_LOCAL_STEP_CONTEXT_423":str(self.path),
             "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423":"0"*64,
             "NUMI_GRADED_LOCAL_STEP_KAPPA_423":"0.1"}
        with patch.dict(os.environ,env,clear=False):
            with self.assertRaisesRegex(ValueError,"path/hash"):
                owner.reduce_witness_node_scales(self.faces,np.ones(10),[0],[],anchors=self.anchors)
        env["NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423"]=self.digest
        with patch.dict(os.environ,env,clear=False):
            with self.assertRaisesRegex(ValueError,"topology"):
                owner.reduce_witness_node_scales(self.faces[::-1].copy(),np.ones(10),[0],[],anchors=self.anchors)
            with self.assertRaisesRegex(ValueError,"anchors"):
                owner.reduce_witness_node_scales(self.faces,np.ones(10),[0],[],anchors=np.array([],np.int64))

    def test_budget_witness_and_context_validation(self):
        for kappa in ("nan","0","2"):
            with patch.dict(os.environ,{"NUMI_GRADED_LOCAL_STEP_CONTEXT_423":str(self.path),
                 "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423":self.digest,
                 "NUMI_GRADED_LOCAL_STEP_KAPPA_423":kappa},clear=False):
                with self.assertRaisesRegex(ValueError,"kappa"):
                    owner.reduce_witness_node_scales(self.faces,np.ones(10),[0],[],anchors=self.anchors)
        badpath=Path(self.temp.name)/"bad.npz"
        z=np.load(self.path)
        arrays={k:z[k] for k in z.files};arrays["edge_scale_budget"]=arrays["edge_scale_budget"]*.9
        np.savez(badpath,**arrays)
        env={"NUMI_GRADED_LOCAL_STEP_CONTEXT_423":str(badpath),
             "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423":hashlib.sha256(badpath.read_bytes()).hexdigest(),
             "NUMI_GRADED_LOCAL_STEP_KAPPA_423":"0.1"}
        with patch.dict(os.environ,env,clear=False):
            with self.assertRaisesRegex(ValueError,"budgets do not match"):
                owner.reduce_witness_node_scales(self.faces,np.ones(10),[0],[],anchors=self.anchors)
        with patch.dict(os.environ,{"NUMI_GRADED_LOCAL_STEP_CONTEXT_423":str(self.path),
             "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423":self.digest,
             "NUMI_GRADED_LOCAL_STEP_KAPPA_423":"0.1"},clear=False):
            with self.assertRaisesRegex(ValueError,"exact failure witnesses"):
                owner.reduce_witness_node_scales(self.faces,np.ones(10),[],[],anchors=self.anchors)

if __name__=="__main__":unittest.main()
