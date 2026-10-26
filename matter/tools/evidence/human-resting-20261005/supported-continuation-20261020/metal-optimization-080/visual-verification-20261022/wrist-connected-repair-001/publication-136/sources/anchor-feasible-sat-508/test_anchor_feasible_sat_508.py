import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SAT_PATH = HERE.parent / "triangle-sat-rows-466" / "triangle_sat_rows_466.py"
HELPER_PATH = HERE / "anchor_feasible_sat_508.py"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sat = load("sat466_test_for_508", SAT_PATH)
helper = load("anchor_sat508_test", HELPER_PATH)

def test_anchor_compatible_selection_uses_feasible_standard_sat_axis():
    skin = np.array([
        [3.67978319100942, -0.06091887648126931, -0.4992564483290908],
        [1.4548824671029015, 0.3072820791286886, -0.11160839877939792],
        [1.3159088197567672, 0.7947769978145437, -0.017214200890901175],
    ])
    bone = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    selected = helper.anchor_compatible_plane(skin, bone, [0], sat)
    assert selected["corner_lower_mm"][0] <= 0
    assert selected["axis_outward_dot"] >= 0
    assert selected["anchor_compatible_axis_count"] > 0
    original = sat.triangle_sat_plane_for_pair(skin, bone)
    assert selected["axis_source"] == "edge_cross"
    assert selected["axis_index"] != original["axis_index"]
    assert original["corner_lower_mm"][0] > 0

def test_anchor_inside_bone_support_fails_closed():
    bone = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    skin = np.array([[.2,.2,-.1],[2.4,.2,.1],[.2,2.4,.1]])
    with pytest.raises(ValueError, match="no outward-hemisphere SAT axis"):
        helper.anchor_compatible_plane(skin, bone, [0], sat)

def test_no_anchor_matches_frozen_466_choice():
    bone = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    skin = np.array([[2.4,.2,.4],[2.3,1.1,.2],[1.9,1.4,.1]])
    got = helper.anchor_compatible_plane(skin, bone, [], sat)
    want = sat.triangle_sat_plane_for_pair(skin, bone)
    assert got["axis_source"] == want["axis_source"]
    assert got["axis_index"] == want["axis_index"]
    assert got["axis_sign"] == want["axis_sign"]
    assert np.allclose(got["corner_lower_mm"], want["corner_lower_mm"], rtol=0, atol=1e-14)



def test_rows_keep_all_three_corners_and_zero_anchor_deficit():
    skin = np.array([
        [3.67978319100942, -0.06091887648126931, -0.4992564483290908],
        [1.4548824671029015, 0.3072820791286886, -0.11160839877939792],
        [1.3159088197567672, 0.7947769978145437, -0.017214200890901175],
    ])
    bone = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    pairs = [{"face_row":0,"body":1,"bone_id":2,"triangle_id":3}]
    result = helper.build_anchor_compatible_rows(
        skin, np.array([[0,1,2]]), np.array([0,1,2]), np.array([0]),
        pairs, {(1,2,3):bone}, sat=sat,
        selected_pair_ids={(0,1,2,3)}, total_weight=1.0)
    assert result["diagnostics"]["lower_bound_row_count"] == 3
    assert result["diagnostics"]["anchored_positive_lower_count"] == 0
    assert result["diagnostics"]["anchor_compatible_pair_count"] == 1
    assert result["G"].shape == (3,9)
    assert result["G"].getrow(0).nnz == 0
    assert result["diagnostics"]["anchor_compatible_axes"][0]["axis_source"] == "edge_cross"

def test_unselected_row_assembly_matches_frozen_466_baseline():
    skin = np.array([[2.4,.2,.4],[2.3,1.1,.2],[1.9,1.4,.1]])
    bone = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    pairs = [{"face_row":0,"body":1,"bone_id":2,"triangle_id":3}]
    args = (skin, np.array([[0,1,2]]), np.array([0,1,2]), np.array([],dtype=np.int64))
    got = helper.build_anchor_compatible_rows(
        *args, pairs, {(1,2,3):bone}, sat=sat, selected_pair_ids=set())
    want = sat.build_fixed_bone_outward_plane_rows(
        *args, pairs, {(1,2,3):bone}, separating_axis_pairs=set())
    for field in ("G","lower","C","h","weights"):
        a, b = got[field], want[field]
        if hasattr(a, "toarray"):
            assert np.array_equal(a.toarray(), b.toarray())
        else:
            assert np.array_equal(a, b)
