import importlib.util
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).with_name('correct_emitted_directions606_v2.py')
spec=importlib.util.spec_from_file_location('correct_emitted_directions606_v2',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_actual_float32_displacement_is_measured_separately_from_nominal():
 base=np.zeros((1,3),dtype=np.float32);target=np.array([[1.0,-.001,0]],dtype=np.float32)
 source=np.array([[.05,(1-.05**2)**.5,0]],dtype=np.float64);depth=np.array([[1,0,0]],dtype=np.float64)
 nominal=np.array([[1,0,0]],dtype=np.float64)
 r=m.emitted_direction_audit(base,target,nominal,source,depth)
 assert r['nominal_proposal_direction']['source_dot']['minimum']==pytest.approx(.05)
 assert r['emitted_float32_displacement_direction']['source_dot']['minimum']<.05
 assert r['emitted_float32_displacement_direction']['source_dot']['count_below_0_05']==1
 assert r['emitted_float32_displacement_direction']['source_dot']['count_nonpositive']==0
 assert r['emitted_float32_displacement_direction']['actual_0_05_floor_claimed'] is False
def test_actual_nonpositive_source_direction_rejects():
 base=np.zeros((1,3),dtype=np.float32);target=np.array([[1,0,0]],dtype=np.float32)
 source=np.array([[-1,0,0]],dtype=np.float64);depth=np.array([[1,0,0]],dtype=np.float64);nominal=np.array([[1,0,0]],dtype=np.float64)
 with pytest.raises(ValueError,match='not positive'): m.emitted_direction_audit(base,target,nominal,source,depth)
def test_report_and_float32_target_row_must_match():
 report={'targets':[{'query_row':1,'target_world_f32_m':[1,0,0],'start_world_f32_m':[0,0,0]}]}
 z={'query_rows':np.array([1]),'target_world_f32_m':np.array([[1,0,0]],dtype=np.float32)}
 m.validate_report_target_emission(report,z,np.array([[0,0,0],[0,0,0]],dtype=np.float32))
 report['targets'][0]['target_world_f32_m']=[2,0,0]
 with pytest.raises(ValueError,match='target mismatch'): m.validate_report_target_emission(report,z,np.zeros((2,3),dtype=np.float32))
