import numpy as np
import pytest
from run_point_batch545 import point_weights
def test_absolute_point_deficit_and_surface_mass():
 h=np.array([.1,2.,.3,.4]);w=np.array([.25]*4);m=[{"kind":"point"}]*2+[{"kind":"surface"}]*2
 y=point_weights(h,w,m)
 assert y[:2].sum()==pytest.approx(.99)
 assert y[2:].sum()==pytest.approx(.01)
 assert y[0]/h[0]==pytest.approx(y[1]/h[1])
 assert np.array_equal(w,np.full(4,.25))
def test_no_missing_or_invalid_point_rows():
 with pytest.raises(ValueError):point_weights(np.array([0.]),np.ones(1),[{"kind":"point"}])
 with pytest.raises(ValueError):point_weights(np.ones(1),np.ones(1),[{"kind":"surface"}])
