Exact preparation derivatives for signed-prism barriers
======================================================

The existing local layer fitting solver can supply these analytic derivatives to SLSQP instead of numerically perturbing every free coordinate. The physical equations, native signed-prism admission and accepted-state runtime are unchanged. This code runs only during CPU mesh preparation on SSH Numi.

The helper differentiates the signed tetrahedron determinant divided by the cube of its longest edge, using the native sorted-global-index prism decomposition and authored face parity. It includes Euclidean skin/fat widths, total depth caps and inward orientation constraints. At a longest-edge tie it selects the first maximum edge deterministically; this is a piecewise derivative, not a smooth guarantee at ties.

The retained deterministic check covers 24 randomized cases, all six face permutations, fixed and free point mappings, and compares every derivative with central differences. It does not certify final geometry, physical skin/fat layers, native dynamics, or hand function. Candidate layers still require those independent gates.

Run on Numi with the retained preparation virtual environment: `python test_hand_layer_barrier_jacobian.py --output derivative-test-report.json`.
