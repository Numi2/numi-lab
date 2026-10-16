# Source207: free-BOX true-residual certification

This isolated source snapshot extends the opt-in same-operator true-residual certification path to one bounded layout: one environment, one static explicit FEM object, and one non-articulated dynamic six-DOF BOX with FEM/BOX contact. Existing FEM-only observer/trace scopes retain their old admission. All other generalized, MPM, mixed-field, support, Human, vascular, articulated, and multi-environment cases remain rejected.

The certificate path reuses the production FEM element, cohesive, node gather/contact, free-rigid mass/inertia, and primal rigid-contact kernels. The true-residual measurement and optional restart replacement now include the six scalar rows after the FEM mechanical/field backing blocks. The forcing and nonlinear thresholds are unchanged.

CPU/source contract tests are not GPU qualification. The coupled BOX native checker now dispatches the production contact/operator kernels and the true-residual measurement kernel, checks both contact cross-blocks, every rigid tail row, restart replacement, wrong-sign/omitted-tail discriminators, and a zero-residual latch. It reports zero physical transactions. Parent-owned GPU execution is still required before claiming native qualification.

The frozen source map contains 819 files, inherited from source205 with four changed source paths. The source overlay manifest binds all four before/after hashes and the base frozen-manifest digest.
