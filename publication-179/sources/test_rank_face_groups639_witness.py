import importlib.util
from pathlib import Path
import numpy as np
SCRIPT=Path(__file__).with_name("rank_face_groups639_witness.py")
spec=importlib.util.spec_from_file_location("rank639w",SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_target_delta_records_are_sorted_unique_node_xyz():
    row1=([1,2,3,4],np.array([11,12,13]),None,None,None,None)
    row2=([1,2,3,5],np.array([12,14,11]),None,None,None,None)
    scope={11:0,12:1,13:2,14:3}
    d=np.arange(12,dtype=np.float64)
    nodes,values=mod._target_node_delta_records(d,scope,[row1,row2])
    assert nodes==[11,12,13,14]
    assert values==[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0],[9.0,10.0,11.0]]
