import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("batch611", HERE / "run_harmonic_batch611.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_fixed_center_adapter_replaces_only_center_generation():
    source = "before\n  neighbors=defaultdict(set)\n  old_centers()\n  centers=sorted(patches)[:a.max_centers];used_nodes=set();round_ops=0\n  after\n"
    result = M.adapt_search_source(source)
    assert "centers=[tuple(map(int,rows)) for rows in FIXED_CENTER_ROWS]" in result
    assert "old_centers" not in result
    assert "  after\n" in result


def test_fixed_center_adapter_fails_closed_on_changed_or_ambiguous_source():
    with pytest.raises(ValueError):
        M.adapt_search_source("no pinned block")
    with pytest.raises(ValueError):
        M.adapt_search_source("  neighbors=defaultdict(set)\n;used_nodes=set();round_ops=0\n" * 2)


def seed_fixture():
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]], dtype=np.int64)
    return {
        "schema": "numi.human.harmonic_batch611_seed_patches.v1",
        "status": "INERT_VALIDATED_FACE_SEED_LIST_ONLY",
        "patch_count": 17, "covered_pair_incidences": 703, "unique_skin_faces": 224,
        "patches": [{"crossing_face_rows": [0, 1], "face_count": 2, "face_kind": 2,
                     "validated_as_single_oriented_disk": True, "boundary_vertex_count": 4,
                     "used_vertex_rows": [0, 1, 2, 3]}] * 17,
    }


def test_seed_domain_checks_patch_rows_and_face_kind():
    seed = seed_fixture()
    # Exercise domain checks independently of the final pinned 17-patch aggregate.
    p = seed["patches"][0]
    with pytest.raises(ValueError, match="incidence/unique-face"):
        M.fixed_rows_from_seed(seed, np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64))
    p["face_kind"] = 1
    with pytest.raises(ValueError, match="skin-disk"):
        M.fixed_rows_from_seed(seed, np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64))
