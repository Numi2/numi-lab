import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parent
TOOLS=ROOT/'canonical/matter/tools'
sys.path.insert(0,str(TOOLS))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface

def test_positive_tet_detailed_fold_gate_backtracks():
    O=np.array([
        [0.,0.,0.],
        [.001,0.,0.],
        [0.,.001,0.],
        [0.,0.,.001],
        [.0003,.0003,.001],
    ],dtype=np.float32)
    T=np.array([[0,1,2,3]],dtype=np.int64)
    detail=O[[0,1,4]].copy()
    W=sparse.csr_matrix((np.ones(3),(np.arange(3),np.array([0,1,4]))),shape=(3,5))
    mapped=MappedSkinSurface(O,detail,np.array([[0,1,2]],dtype=np.int64),W,
                             area_bounds=(.90,1.10),normal_cosine_min=.5)
    assert mapped.passes(O)
    C=sparse.csr_matrix(([-1.],([0],[14])),shape=(1,15))
    target=np.array([-.0005])
    def fake_linprog(c, A_ub, b_ub, bounds, method, options):
        x=np.zeros(len(c),dtype=float)
        x[14]=-2.0  # mm: flips the mapped detail face, while FEM tet is unaffected
        return SimpleNamespace(success=True,status=0,message='fixture',x=x)
    original=repair.linprog
    repair.linprog=fake_linprog
    try:
        Y,report=repair.coupled_clearance_step(
            O,O,T,np.arange(5,dtype=np.int64),C,target,
            step_limit_mm=2.,q_min=1e-8,J_bounds=(.1,2.5),time_limit_s=1.,
            max_incremental_gradient=None,construction_origin=O,
            cumulative_displacement_weight=.01,embedded_surface=mapped,
            linear_method='highs-ipm')
    finally:
        repair.linprog=original
    assert report['lp_success']
    assert report['step_applied']
    assert report['mapped_skin_geometry_pass']
    assert report['components'][0]['applied']
    assert 0. < report['components'][0]['line_fraction'] < 1.
    assert Y[4,2] > 0. and Y[4,2] < O[4,2]
    j,_,q,_=repair.cell_values_gradients(Y,O,T)
    assert j.min()>0 and q.min()>0
    bad=O.copy();bad[4,2]=-.001
    assert not mapped.passes(bad)
    assert mapped.passes(Y)
