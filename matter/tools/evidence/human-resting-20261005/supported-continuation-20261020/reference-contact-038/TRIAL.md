# Reference-displacement contact precision 038

Preserve the scene, mass, reference materials, forces, physical timestep, solver budgets, collision floor and strict Armijo certificates. Reuse existing authoritative FEM reference-displacement state for all-reference-FEM contact geometry in a common local reference frame. Contact generation, its derivative, exact trial energy and CCD must use the same geometry. Legacy and mixed coordinate modes retain the current path. Add no physical state, replay layout or CPU physics path.

The preceding analytic slope037 candidate is retained as tested-candidate.patch, with its failed continuation and exact root2/tick3/slot2624 witness. Its pure arithmetic checks passed but native continuation regressed to two accepted steps. The tick2 diagnostic was pre-failure and must not be used as the terminal failure witness.

Before continuation: independent FP64 reference-position/trial/witness controls plus all existing preparation, boundary, geometry, sorter, material and vascular checks, then an integrated two-step injected rejection/retry. Stop at the first failed gate. Native continuation requests 32 steps at125us; capture root2/tick3/contact2624 only as a historical target, requiring actual source identity before comparing feature outcomes.

This is a bounded numerical test, not supported equilibrium, complete ligament coverage, broad anatomical clearance, or physiological qualification.
