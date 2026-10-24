import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neo_hookean_fit_support import (
    MAX_ABS_PIOLA_PA, evaluate_neo_hookean_batch, prepare_reference_tetrahedra,
)


def _reference():
    return np.array([[[0.,0.,0.],[.012,0.,0.],[0.,.011,0.],[0.,0.,.013]]], dtype=np.float64)


def _move_columns(X, corner_columns, direction_by_column, step):
    Y=X.copy()
    for c, direction in direction_by_column.items():
        Y[corner_columns == c] += step*np.asarray(direction)
    return Y


def test_energy_and_signed_piola_jacobians_match_random_directional_differences():
    R=np.concatenate((_reference(), _reference()+np.array([.03,.01,-.02])),axis=0)
    cache=prepare_reference_tetrahedra(R)
    rng=np.random.default_rng(4219)
    X=R+rng.normal(size=R.shape)*2.e-5
    columns=np.array([[2,2,3,4],[2,5,6,7]],np.int64)
    result=evaluate_neo_hookean_batch(X,cache,np.array([1,2]),columns)
    assert result["valid"].all()
    assert result["energy_jacobian_j_per_m"].shape==(2,4,3)
    assert result["piola_component_jacobian_pa_per_m"].shape==(2,9,4,3)
    assert result["stress_constraint_jacobian_pa_per_m"].shape==(2,18,4,3)
    eps=2.e-8
    directions={2:np.array([.3,-.5,.2]),3:np.array([-.1,.4,.7]),4:np.array([.6,.2,-.2]),5:np.array([.1,.1,-.4]),6:np.array([-.3,.7,.1]),7:np.array([.2,-.2,.5])}
    Rp=_move_columns(X,columns,directions,+eps)
    Rm=_move_columns(X,columns,directions,-eps)
    plus=evaluate_neo_hookean_batch(Rp,cache,np.array([1,2]),columns)
    minus=evaluate_neo_hookean_batch(Rm,cache,np.array([1,2]),columns)
    pred_energy=0.
    pred_piola=np.zeros((2,9))
    for t in range(2):
        for slot,c in enumerate(result["jacobian_column_ids"][t]):
            if c<0: continue
            d=directions[int(c)]
            pred_energy+=np.dot(result["energy_jacobian_j_per_m"][t,slot],d)
            pred_piola[t]+=result["piola_component_jacobian_pa_per_m"][t,:,slot,:]@d
    fd_energy=(plus["energy_joule"].sum()-minus["energy_joule"].sum())/(2*eps)
    fd_piola=(plus["piola_components_pa"]-minus["piola_components_pa"])/(2*eps)
    assert np.isclose(pred_energy,fd_energy,rtol=2.e-6,atol=1.e-12)
    assert np.allclose(pred_piola,fd_piola,rtol=2.e-6,atol=1.e-3)
    # Two sign branches per component are differentiable signed inequalities.
    assert np.allclose(result["stress_constraint_jacobian_pa_per_m"][:,0::2], result["piola_component_jacobian_pa_per_m"])
    assert np.allclose(result["stress_constraint_jacobian_pa_per_m"][:,1::2], -result["piola_component_jacobian_pa_per_m"])
    assert np.all(result["stress_constraint_residual_pa"] < 0)
    assert MAX_ABS_PIOLA_PA==5.e6


def test_rigid_rotation_has_zero_neo_hookean_energy_and_stress():
    R=_reference();cache=prepare_reference_tetrahedra(R)
    axis=np.array([.3,-.4,.5]);axis/=np.linalg.norm(axis);theta=.71
    K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    Q=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
    X=R@Q.T+np.array([.04,-.02,.08])
    out=evaluate_neo_hookean_batch(X,cache,np.array([1]),np.array([[0,1,2,3]]))
    assert out["valid"].all()
    np.testing.assert_allclose(out["J"],[1.],rtol=2e-15,atol=2e-15)
    assert abs(out["energy_density_pa"][0])<1.e-8
    assert np.max(np.abs(out["piola_pa"]))<1.e-8


def test_shared_column_corner_contributions_sum_and_rigid_translation_cancels():
    R=_reference();cache=prepare_reference_tetrahedra(R)
    X=R.copy();X[0,1,2]+=1.e-4;X[0,2,0]-=8.e-5
    # Corners 0 and 1 share a single DOF: its Jacobian is their exact sum.
    C=np.array([[8,8,9,10]],np.int64)
    out=evaluate_neo_hookean_batch(X,cache,np.array([2]),C)
    grads=cache.shape_gradients_per_m[0]
    P=out["piola_pa"][0]
    expected=cache.volume_m3[0]*P@(grads[0]+grads[1])
    slot=int(np.flatnonzero(out["jacobian_column_ids"][0]==8)[0])
    assert np.allclose(out["energy_jacobian_j_per_m"][0,slot],expected,rtol=1.e-13,atol=1.e-13)
    # One translation shared by all four corners is rigid and changes neither F nor P.
    all_shared=evaluate_neo_hookean_batch(X,cache,np.array([2]),np.array([[5,5,5,5]],np.int64))
    assert np.max(np.abs(all_shared["energy_jacobian_j_per_m"]))<1.e-10
    assert np.max(np.abs(all_shared["piola_component_jacobian_pa_per_m"]))<1.e-7


def test_invalid_j_is_explicit_finite_and_not_optimizer_nan():
    R=_reference();cache=prepare_reference_tetrahedra(R)
    X=R[:,[0,2,1,3],:]
    out=evaluate_neo_hookean_batch(X,cache,np.array([1]),np.array([[0,1,2,3]]))
    assert out["J"][0]<0 and not out["valid"][0]
    for key in ("energy_density_pa","energy_joule","piola_pa","energy_jacobian_j_per_m",
                "piola_component_jacobian_pa_per_m","stress_constraint_residual_pa",
                "stress_constraint_jacobian_pa_per_m"):
        assert np.isfinite(out[key]).all()
    assert np.all(out["energy_jacobian_j_per_m"]==0)
    assert not out["stress_constraint_feasible"].any()


def test_reference_cache_is_content_keyed_and_orientation_fail_closed():
    R=_reference();a=prepare_reference_tetrahedra(R);b=prepare_reference_tetrahedra(R.copy())
    assert a is b and a.sha256==b.sha256
    with np.testing.assert_raises(ValueError):
        prepare_reference_tetrahedra(R[:,[0,2,1,3],:])
    degenerate=R.copy();degenerate[0,3]=degenerate[0,2]
    with np.testing.assert_raises(ValueError):
        prepare_reference_tetrahedra(degenerate)
