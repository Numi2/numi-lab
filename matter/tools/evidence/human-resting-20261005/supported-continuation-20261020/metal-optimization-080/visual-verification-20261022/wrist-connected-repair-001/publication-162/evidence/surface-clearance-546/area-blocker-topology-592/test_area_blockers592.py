import numpy as np
from search_area_blockers592 import admissible592
def test_neutral_bone_counts_require_strict_area_progress():
 assert admissible592(np.array([0,2]),np.array([0,2]),2,1)
 assert not admissible592(np.array([0,2]),np.array([0,2]),2,2)
def test_per_bank_regression_rejected_even_if_total_falls():
 assert not admissible592(np.array([0,5]),np.array([1,2]),2,0)
def test_bone_reduction_alone_does_not_admit_quality_preparation():
 assert not admissible592(np.array([1,2]),np.array([0,1]),2,2)
