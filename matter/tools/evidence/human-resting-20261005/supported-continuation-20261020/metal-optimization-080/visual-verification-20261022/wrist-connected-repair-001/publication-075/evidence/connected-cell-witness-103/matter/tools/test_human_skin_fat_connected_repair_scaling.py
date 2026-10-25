import numpy as np
from scipy import sparse

import human_skin_fat_connected_repair as repair


def test_incremental_gradient_uses_si_reference_and_mm_step_scaling():
    # A 100 mm displacement of one vertex on this 1 m tetra has a maximum
    # componentwise reference displacement gradient of exactly 0.1. It must be
    # feasible under the 0.12 hard cap; treating LP millimetres as metres would
    # instead constrain it to roughly 0.12 mm.
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    reference = points.astype(np.float64)
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    free_nodes = np.asarray([0], dtype=np.int64)
    contact = np.zeros((1, 12), dtype=np.float64)
    contact[0, 0] = 1.0
    targets = contact @ points.astype(np.float64).ravel() + 0.1

    result, report = repair.coupled_clearance_step(
        points,
        reference,
        tetrahedra,
        free_nodes,
        sparse.csr_matrix(contact),
        targets,
        step_limit_mm=150.0,
        q_min=1e-5,
        J_bounds=(0.35, 2.5),
        time_limit_s=5.0,
        max_incremental_gradient=0.12,
    )

    assert report["lp_success"], report
    assert report["step_applied"], report
    np.testing.assert_array_equal(result[1:].view(np.uint32), points[1:].view(np.uint32))
    displacement = result.astype(np.float64) - reference
    shape_gradients = np.asarray([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    gradient = sum(np.outer(displacement[i], shape_gradients[i]) for i in range(4))
    np.testing.assert_allclose(np.max(np.abs(gradient)), 0.1, atol=2e-7, rtol=0)
    assert report["maximum_step_mm"] > 99.0, report
    assert report["emitted_q_min"] >= 1e-5
    assert report["emitted_J_min"] >= 0.35
    assert report["emitted_J_max"] <= 2.5


if __name__ == "__main__":
    test_incremental_gradient_uses_si_reference_and_mm_step_scaling()
    print("PASS test_incremental_gradient_uses_si_reference_and_mm_step_scaling")
