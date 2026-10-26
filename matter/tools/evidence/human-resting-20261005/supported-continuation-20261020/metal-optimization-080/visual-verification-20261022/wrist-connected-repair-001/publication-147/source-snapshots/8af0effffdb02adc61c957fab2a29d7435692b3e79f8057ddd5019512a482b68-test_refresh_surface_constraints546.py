from pathlib import Path
import importlib.util
import numpy as np

SOURCE = Path(__file__).with_name("refresh_surface_constraints546.py")
spec = importlib.util.spec_from_file_location("refresh_surface_constraints546_tested", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_seed_pairs_map_only_when_both_unique_triangles_survive():
    source = np.asarray([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    destination = np.asarray([[4, 3, 2], [1, 2, 0]], dtype=np.int64)
    result = mod.map_seed_pairs_partial(source, destination, np.asarray([[0, 1], [0, 2]], dtype=np.int64))
    assert result["mapped_pairs"].tolist() == [[0, 1]]
    assert result["source_status_code"].tolist() == [0, 1]
    payload = result["payload"]
    assert payload["dropped_source_pair_indices"].tolist() == [1]
    assert payload["dropped_source_pairs"].tolist() == [[0, 2]]
    assert "mapped_face_rows" not in payload
    assert np.all(payload["mapped_pairs_by_source_row"] >= 0)


def test_duplicate_destination_face_is_dropped_as_ambiguous():
    source = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    destination = np.asarray([[2, 1, 0], [1, 0, 2], [3, 4, 5]], dtype=np.int64)
    result = mod.map_seed_pairs_partial(source, destination, np.asarray([[0, 1]], dtype=np.int64))
    assert len(result["mapped_pairs"]) == 0
    assert result["source_status_code"].tolist() == [2]
    assert result["payload"]["dropped_status_code"].tolist() == [2]


def test_duplicate_source_face_is_not_treated_as_unique_identity():
    source = np.asarray([[0, 1, 2], [2, 1, 0], [3, 4, 5]], dtype=np.int64)
    destination = np.asarray([[0, 1, 2], [3, 5, 4]], dtype=np.int64)
    result = mod.map_seed_pairs_partial(source, destination, np.asarray([[0, 2]], dtype=np.int64))
    assert len(result["mapped_pairs"]) == 0
    assert result["source_status_code"].tolist() == [2]


def test_prior_surface_pairs_only_become_guards_when_cleared_and_mappable():
    source = np.asarray([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    destination = np.asarray([[3, 4, 2], [2, 0, 1]])
    prior = {(7, 10, 0, 100), (7, 10, 1, 101), (7, 10, 2, 102)}
    current = {(7, 10, 0, 101)}
    result = mod.remap_prior_surface_pairs(source, destination, prior, current)
    assert result["source_status_code"].tolist() == [0, 3, 1]
    assert result["newly_cleared_source_pairs"] == ((7, 10, 0, 100),)
    assert result["newly_cleared_current_pairs"] == ((7, 10, 1, 100),)
    payload = result["payload"]
    assert payload["dropped_source_pairs"].tolist() == [[7, 10, 2, 102]]
    assert payload["mapped_current_pairs"].tolist() == [[7, 10, 1, 100], [7, 10, 0, 101]]


def test_prior_pair_ambiguous_triangle_is_logged_not_guarded():
    source = np.asarray([[0, 1, 2]], dtype=np.int64)
    destination = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    result = mod.remap_prior_surface_pairs(source, destination, {(7, 10, 0, 100)}, set())
    assert result["source_status_code"].tolist() == [2]
    assert result["newly_cleared_source_pairs"] == ()
    assert result["payload"]["dropped_source_pairs"].tolist() == [[7, 10, 0, 100]]
