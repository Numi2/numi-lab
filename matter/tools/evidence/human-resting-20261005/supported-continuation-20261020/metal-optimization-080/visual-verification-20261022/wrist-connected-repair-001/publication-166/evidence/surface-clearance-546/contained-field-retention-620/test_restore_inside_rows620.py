import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/contained-field-retention-620/restore_inside_rows620.py")
s=importlib.util.spec_from_file_location("restore620",SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def arrays():
    p=np.zeros((4,3),np.float32);p[:,0]=np.arange(4)
    return {"points_world_m":p,"typed_coarse_anchor_output_rows_018":np.array([0],np.int64),
            "source_O_faces":np.array([[0,1,2]],np.int64),
            "source_prolongation_weights":np.array([1.,.5],np.float64)}

def test_restore_only_contained_row_preserves_outside_and_payload():
    b=arrays();t={k:v.copy() for k,v in b.items()};t["points_world_m"][1,1]=.01;t["points_world_m"][2,2]=.002
    out,steps,outside=m.restore_rows(b,t,np.array([2],np.int64))
    assert np.array_equal(out["points_world_m"][2],b["points_world_m"][2])
    assert np.array_equal(out["points_world_m"][outside],t["points_world_m"][outside])
    assert m.same(out["source_O_faces"],t["source_O_faces"])
    assert m.same(out["typed_coarse_anchor_output_rows_018"],t["typed_coarse_anchor_output_rows_018"])
    assert np.isclose(steps[0],2.,rtol=0,atol=2e-7)

def test_anchor_witness_rejected():
    b=arrays();t={k:v.copy() for k,v in b.items()};t["points_world_m"][0,0]=1.
    with pytest.raises(ValueError,match="anchor"):
        m.restore_rows(b,t,np.array([0],np.int64))

def test_nonpoint_mutation_rejected():
    b=arrays();t={k:v.copy() for k,v in b.items()};t["points_world_m"][2,0]=9.
    t["source_O_faces"][0,0]=3
    with pytest.raises(ValueError,match="nonpoint"):
        m.restore_rows(b,t,np.array([2],np.int64))

def test_coordinate_identical_witness_rejected():
    b=arrays();t={k:v.copy() for k,v in b.items()}
    with pytest.raises(ValueError,match="coordinate-identical"):
        m.restore_rows(b,t,np.array([2],np.int64))

def test_witness_rows_are_candidate_node_ids_and_unique_count_checked():
    report={"complete_surface_audit":True,"inside_node_bank_pairs":2,"inside_nodes":2,
       "inside_witnesses":[{"classification":"inside","output_node":2},
                           {"classification":"inside","output_node":3}]}
    assert m.contained_rows(report,4).tolist()==[2,3]
    report["inside_witnesses"][1]["output_node"]=9
    with pytest.raises(ValueError,match="output row"):
        m.contained_rows(report,4)

def test_mixed_identical_and_changed_witness_rows_rejected():
    b=arrays();t={k:v.copy() for k,v in b.items()}
    t["points_world_m"][3,2]=.002
    with pytest.raises(ValueError,match="coordinate-identical"):
        m.restore_rows(b,t,np.array([2,3],np.int64))

def test_mixed_identical_and_changed_witness_rows_rejected():
    b=arrays();t={k:v.copy() for k,v in b.items()}
    t["points_world_m"][3,2]=.002
    with pytest.raises(ValueError,match="coordinate-identical"):
        m.restore_rows(b,t,np.array([2,3],np.int64))
