import numpy as np,pytest
from angular_joint_direction597 import nearest_joint_direction597 as select
def test_identity():
 v=np.array([0.,0.,1.]);d,p=select(v,v,v);assert np.array_equal(d,v)
def test_both_active():
 s=np.array([1.,0.,0.]);c=np.array([0.,1.,0.]);v=-(s+c)/np.sqrt(2)
 d,p=select(s,c,v);assert p["active_case"]=="two_caps";assert np.allclose(d[:2],[.05,.05],rtol=0,atol=1e-12)
def test_opposed_caps():
 with pytest.raises(ValueError):select(np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([0.,1.,0.]))
def test_parallel_reverse():
 s=np.array([1.,0.,0.]);d,p=select(s,s,-s);assert abs(d@s-.05)<1e-12
def test_random_directions_dominate_dense_feasible_samples():
 rng=np.random.default_rng(597);sample=rng.normal(size=(20000,3));sample/=np.linalg.norm(sample,axis=1)[:,None]
 for _ in range(40):
  a=rng.normal(size=(3,3));a/=np.linalg.norm(a,axis=1)[:,None];s,c,v=a
  if s@c<-.99:continue
  d,p=select(s,c,v);assert min(d@s,d@c)>=.05
  feasible=sample[(sample@s>=.05)&(sample@c>=.05)]
  if len(feasible):assert d@v>=np.max(feasible@v)-2e-12
def test_invalid_fails():
 with pytest.raises(ValueError):select(np.array([2.,0.,0.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))

