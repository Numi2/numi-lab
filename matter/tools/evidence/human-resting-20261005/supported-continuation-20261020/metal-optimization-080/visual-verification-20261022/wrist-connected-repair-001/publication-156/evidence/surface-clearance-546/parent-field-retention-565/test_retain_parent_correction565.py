from pathlib import Path
import importlib.util,numpy as np,pytest
s=importlib.util.spec_from_file_location('p565',Path(__file__).with_name('retain_parent_correction565.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def base():
 return {'points_world_m':np.zeros((3,3),np.float32),'source_prolongation_indptr':np.arange(4,dtype=np.int64),'source_prolongation_node_rows':np.arange(3,dtype=np.int64),'source_prolongation_weights':np.ones(3),'typed_coarse_anchor_output_rows_018':np.array([0]),'faces':np.array([[0,1,2]])}
def test_transfer_preserves_payload_and_anchor():
 b=base();s=base();t=base();t['points_world_m'][1]=.001
 o,r=m.transfer(b,s,t);assert np.array_equal(o['points_world_m'],t['points_world_m']);assert o['faces']is b['faces'];assert r['changed_nodes']==1
def test_anchor_motion_rejected():
 b=base();s=base();t=base();t['points_world_m'][0]=.001
 with pytest.raises(ValueError,match='anchors'):m.transfer(b,s,t)
def test_wrong_material_identity_rejected():
 b=base();s=base();t=base();b['source_prolongation_node_rows'][1]=100
 with pytest.raises(ValueError,match='registered'):m.transfer(b,s,t)
def test_provenance_change_rejected():
 b=base();s=base();t=base();t['source_prolongation_weights'][1]=.5
 with pytest.raises(ValueError,match='provenance'):m.transfer(b,s,t)
