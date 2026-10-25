import copy
import json
from pathlib import Path
import pytest
from human_skin_fat_bone_audit import assess_point_clearance_construction_progress as assess

def report(rows):
    return dict(points=100, faces=200, complete_surface_audit=True,
                exact_PLC_surface_pairs=0, inside_node_bank_pairs=len(rows),
                inside_nodes=len({r[0]for r in rows}),
                inside_witnesses=[dict(query_row=str(i),bank_body="1",bank_stable_id="2",
                   classification="inside",nearest_distance_m=str(d))for i,d in rows])

def test_shallow_new_pair_can_reduce_count_and_penetration():
    r=assess(report([(0,.003),(1,.002),(2,.001)]), report([(0,.001),(3,.00001)]))
    assert r["construction_progress"] and r["introduced_typed_pairs"]==[[3,1,2]]
    assert r["geometry_admitted"] is False and r["accepted_physical_steps"]==0

def test_zero_is_construction_progress_not_admission():
    r=assess(report([(0,.001)]),report([]))
    assert r["construction_progress"] and not r["geometry_admitted"]

def test_count_reduction_cannot_conceal_deeper_penetration():
    assert not assess(report([(0,.001),(1,.001)]),report([(2,.0011)]))["construction_progress"]

def test_squared_depth_cannot_increase_even_if_maximum_drops():
    assert not assess(report([(0,.003),(1,0),(2,0),(3,0)]),report([(0,.002),(1,.002),(2,.002)]))["construction_progress"]

def test_same_count_is_not_progress():
    assert not assess(report([(0,.001)]),report([(0,.0005)]))["construction_progress"]

@pytest.mark.parametrize("change",[
    lambda r:r.pop("inside_node_bank_pairs"),
    lambda r:r.update(inside_node_bank_pairs=True),
    lambda r:r.update(inside_nodes=0),
    lambda r:r.update(complete_surface_audit=False),
    lambda r:r.update(exact_PLC_surface_pairs=1),
    lambda r:r.update(exact_PLC_surface_pairs=False),
    lambda r:r["inside_witnesses"].append(copy.deepcopy(r["inside_witnesses"][0])),
    lambda r:r["inside_witnesses"][0].update(nearest_distance_m="nan"),
    lambda r:r["inside_witnesses"][0].update(nearest_distance_m="-1"),
    lambda r:r["inside_witnesses"][0].update(nearest_distance_m=True),
    lambda r:r["inside_witnesses"][0].update(nearest_distance_m="1e200"),
    lambda r:r["inside_witnesses"][0].update(query_row=True),
    lambda r:r["inside_witnesses"][0].update(query_row="100"),
    lambda r:r["inside_witnesses"][0].update(classification="outside"),
])
def test_incomplete_or_malformed_witnesses_rejected(change):
    a=report([(0,.001),(1,.001)]);b=report([(0,.0005)])
    change(b)
    with pytest.raises(ValueError):assess(a,b)

def test_different_topology_rejected():
    a=report([(0,.001),(1,.001)]); b=report([(0,.0005)]);b["faces"]+=1
    with pytest.raises(ValueError,match="topology"):assess(a,b)

def test_actual_refined_trial_improves_despite_two_shallow_new_pairs():
    r=json.loads((Path(__file__).parent/"fixtures/human_skin_point_merit_032.json").read_text())
    q=assess(r["before"],r["after"])
    assert q["construction_progress"]
    assert (q["before_inside_pairs"],q["after_inside_pairs"])==(1419,1242)
    assert len(q["introduced_typed_pairs"])==2
    assert max(q["introduced_depths_m"])<15e-6
