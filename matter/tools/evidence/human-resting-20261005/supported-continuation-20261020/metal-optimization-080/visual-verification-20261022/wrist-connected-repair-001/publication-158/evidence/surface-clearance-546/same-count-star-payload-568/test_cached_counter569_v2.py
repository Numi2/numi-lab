from types import SimpleNamespace
import numpy as np
import pytest

from cached_counter569_v2 import CachedCounter


class FakeExact:
    def __init__(self, pairs):
        self.pairs = pairs

    def _audit_pair(self, skin_records, bank_records, same_surface=False):
        return {
            "triangle_pairs": self.pairs,
            "count": len(self.pairs),
            "aabb_candidate_pairs": 1,
            "allowed_shared_vertex_or_edge_pairs": 0,
        }


class FakeOriginal:
    def __init__(self, pairs):
        self.points_f32 = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        tri = np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        self.banks = [SimpleNamespace(
            body=1, key=(1, 7), triangles_local_m=tri,
            stable_triangle_ids=np.asarray([11], dtype=np.int64),
        )]
        self.exact = FakeExact(pairs)

    def _local_points(self, bank):
        return self.points_f32.astype(np.float64)

    def records(self, triangles, ids, exact):
        return {"triangles": np.asarray(triangles), "ids": np.asarray(ids)}


def count_with(pairs):
    owner = FakeOriginal(pairs)
    counter = CachedCounter(owner)
    return owner, counter.count_faces(np.asarray([5], dtype=np.int64),
                                      np.asarray([[0, 1, 2]], dtype=np.int64))


def test_v2_matches_typed_pair_domain_and_counts():
    _, result = count_with([[5, 11]])
    assert result[(1, 7)]["count"] == 1
    assert result[(1, 7)]["triangle_pairs"] == [(5, 11)]


@pytest.mark.parametrize("bad", [[[6, 11]], [[5, 12]], [(5, 11)]])
def test_v2_rejects_untyped_or_out_of_domain_pairs(bad):
    with pytest.raises(ValueError, match="typed input domain"):
        count_with(bad)


def test_v2_rejects_incomplete_typed_bank_coverage():
    owner = FakeOriginal([[5, 11]])
    counter = CachedCounter(owner)
    tri = np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    owner.banks.append(SimpleNamespace(
        body=2, key=(2, 9), triangles_local_m=tri,
        stable_triangle_ids=np.asarray([19], dtype=np.int64),
    ))
    with pytest.raises(ValueError, match="cover every loaded typed bank"):
        counter.count_faces(np.asarray([5], dtype=np.int64),
                            np.asarray([[0, 1, 2]], dtype=np.int64))

