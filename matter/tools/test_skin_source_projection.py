#!/usr/bin/env python3
"""Focused tests for fail-closed source-C ancestry selection."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np
from skin_source_projection import select_source_embedding

FIXTURE = Path(__file__).resolve().parent / "testdata/skin_source_projection_row8221.json"


class SourceEmbeddingTransactionTests(unittest.TestCase):
    def test_rejected_far_projection_preserves_actual_row_8221_ancestry(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture["schema"], "numi.human.source-projection-row-fixture.v1")
        self.assertEqual(fixture["provenance"]["coarse_row"], 8221)
        for key, value in fixture["provenance"].items():
            if key.endswith("_sha256"):
                self.assertRegex(value, r"^[0-9a-f]{64}$")

        prior_record = fixture["prior"]
        current_record = fixture["current"]
        proposal_record = fixture["rejected_proposal"]
        prior = (prior_record["face"], tuple(prior_record["barycentric"]))
        current = (current_record["face"], tuple(current_record["barycentric"]))
        proposal = (proposal_record["face"], tuple(proposal_record["barycentric"]))
        self.assertEqual(prior, current)
        self.assertEqual(prior[0], 4683)
        self.assertEqual(proposal[0], 16993)

        outer = np.asarray(fixture["outer_registered_world_m"], dtype=np.float64)
        residual = np.asarray(fixture["source_embedding_residual_vector_m"], dtype=np.float64)
        def c_point(embedding, record):
            face, bary = embedding
            self.assertEqual(face, record["face"])
            source_triangle = np.asarray(record["source_triangle_registered_world_m"], dtype=np.float64)
            return np.asarray(bary, dtype=np.float64) @ source_triangle

        prior_c = c_point(prior, prior_record)
        proposal_c = c_point(proposal, proposal_record)
        self.assertAlmostEqual(float(np.linalg.norm(outer-prior_c))*1000, 0.6484697392, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(outer-proposal_c))*1000, 75.4721669535, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(residual))*1000, 0.6484697392, places=5)
        np.testing.assert_allclose(prior_c+residual, outer, rtol=0, atol=2e-6)

        def valid(embedding):
            if embedding is None:
                return False
            face, bary = embedding
            b = np.asarray(bary, dtype=float)
            return (face in (prior[0], proposal[0]) and b.shape == (3,) and np.isfinite(b).all()
                    and b.min() >= -1e-8 and b.max() <= 1+1e-8 and abs(float(b.sum())-1) <= 1e-8)
        selected, action = select_source_embedding(
            proposal, False, prior, valid,
            lambda: (_ for _ in ()).throw(AssertionError("valid prior must be retained")),
        )
        self.assertEqual(selected, prior)
        self.assertEqual(action, "retained_prior")
        np.testing.assert_allclose(c_point(selected, prior_record)+residual, outer, rtol=0, atol=2e-6)

    def test_accepts_only_valid_accepted_proposal_and_resolves_missing_prior(self):
        prior = (4, (0.25, 0.5, 0.25))
        proposal = (9, (0.2, 0.3, 0.5))
        valid = lambda item: item is not None and item[0] in (4, 9, 12) and min(item[1]) >= 0 and max(item[1]) <= 1 and abs(sum(item[1])-1) < 1e-12
        selected, action = select_source_embedding(proposal, True, prior, valid, lambda: None)
        self.assertEqual((selected, action), (proposal, "accepted_proposal"))
        invalid = (9, (1.2, -0.2, 0.0))
        selected, action = select_source_embedding(invalid, True, prior, valid, lambda: None)
        self.assertEqual((selected, action), (prior, "retained_prior"))
        fallback = (12, (1.0, 0.0, 0.0))
        selected, action = select_source_embedding(proposal, False, None, valid, lambda: fallback)
        self.assertEqual((selected, action), (fallback, "resolved_anchor"))
        with self.assertRaisesRegex(ValueError, "invalid source ancestry"):
            select_source_embedding(proposal, False, None, lambda _: False, lambda: proposal)


if __name__ == "__main__":
    unittest.main()
