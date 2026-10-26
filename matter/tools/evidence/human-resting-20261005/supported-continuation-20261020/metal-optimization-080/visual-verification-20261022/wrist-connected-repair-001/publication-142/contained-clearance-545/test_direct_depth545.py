import numpy as np
import pytest
from build_direct_depth_targets545 import direct_depth_unit_direction
def test_opposed_source_ray_does_not_override_current_depth():
 n=np.array([1.,0,0]);w=-n
 d,r=direct_depth_unit_direction(w,n)
 assert np.array_equal(d,n) and r['source_ray_hemisphere_required'] is False
 assert r['selected_source_hemisphere_cosine']==-1
 assert r['selected_depth_cosine']==1
 with pytest.raises(ValueError):direct_depth_unit_direction(w,n*2)
 with pytest.raises(ValueError):direct_depth_unit_direction(w,np.array([np.nan,0,0]))
