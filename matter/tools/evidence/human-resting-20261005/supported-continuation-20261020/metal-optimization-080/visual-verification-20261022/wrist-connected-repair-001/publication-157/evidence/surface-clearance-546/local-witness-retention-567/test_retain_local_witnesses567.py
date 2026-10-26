import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('l567',Path(__file__).with_name('retain_local_witnesses567.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_disjoint_region_target_kept():
 b=np.array([[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]],np.float32)
 f=np.array([[0,1,2],[3,4,5]],np.int64);t=b.copy();t[2,1]=-1;t[5,1]=1.2
 delta=t.astype(float)-b;alpha=np.ones(6);mutable=np.any(delta!=0,axis=1)
 for _ in range(4):
  x=(b+alpha[:,None]*delta).astype(np.float32);bad,_,_=m.area_witnesses(x,b,b.astype(float),f)
  if not len(bad):break
  alpha,_=m.damp(alpha,f[bad].ravel(),mutable)
 assert alpha[2]==.25 and alpha[5]==1
 assert np.array_equal(x[[0,1,3,4,5]],t[[0,1,3,4,5]])
def test_nonfinite_rejected():
 b=np.eye(3,dtype=np.float32);b[1,1]=np.nan
 with pytest.raises(ValueError):m.area_witnesses(b,b,b,np.array([[0,1,2]]))
def test_static_failure_is_not_silently_accepted():
 with pytest.raises(ValueError):m.damp(np.ones(3),[1],np.zeros(3,bool))
def test_witness_domain_is_checked():
 with pytest.raises(ValueError):m.damp(np.ones(3),[3],np.ones(3,bool))

