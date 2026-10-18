# Aortic wall force-budget sensitivity control

This is a short wall-only numerical experiment, not a full anatomical artery or physiology qualification. It uses NHA stable ID 9 / source body 7, an 87-edge section from the registered BodyParts3D external vessel surface, and an explicitly inferred annular wall proxy. The mesh is translated to a wall-local frame; the inverse translation is recorded. The world has one 216-node, 576-tetrahedron FEM object, 12 fixed proximal nodes, no vascular compartments or cavities, no fluid, and no Human rigid attachments.

The helper header retains older loaded-reference and capped-reservoir builder paths for traceability; the accepted probe does not call them. They are unqualified experimental code and are not a production integration.

The fixed model uses identity/unprestressed FEM reference, pressure 11223.5 Pa, timestep 0.25 ms, 12 Newton iterations, 64 FGMRES iterations, density 1066 kg/m^3, shear modulus 130 kPa, Lame lambda 1.3 MPa, and wall thickness 1.2 mm. These are explicit reference parameters, not person-specific measurements. Internal strain energy is not exposed by the snapshot API. Kinetic energy is calculated from accepted snapshot nodal masses and velocities; external work is accumulated against the same frozen nodal load. External work becomes negative in the last steps; this is a dynamic response, not equilibrium.

The historical strict parameter 1e-7 remains unchanged in strict-gate-baseline-receipts.json. The sensitivity runs set only the local diagnostic WorldSource mixedSolver.relativeResidual. Its value is dt * external nodal force L2 * fraction, with fractions 0.0005 and 0.00025. Measured force L2 is 5.33658730281 N; the corresponding impulse-norm budgets are 6.67073412851e-7 and 3.33536706425e-7 N*s. This is a local wall-only numerical budget experiment, not a global threshold change or a general dimensionless relative-error claim.

Both GPU cases accepted eight consecutive steps (2 ms simulated). Exact per-node float32 positions and velocities are retained under runs/ and compared in force-budget-trajectory-comparison.json. At the final step the maximum node-position difference is 7.431 micrometres, RMS 2.197 micrometres; maximum velocity difference across the trace is 0.00657 m/s. Cumulative frozen-load work is 0.0908801 J and 0.0908822 J; final FEM kinetic energy is 0.00270744 J and 0.00270811 J.

Limits: the control does not establish loaded anatomical equilibrium, patent lumen behavior, vascular cavity pressure feedback, coupling to CVSim, sustained artery mechanics, or clinical validity. Prior strict failures remain in the evidence.

## Reproduction

On the pinned numi host, build src/wall_probe.mm using the exact commands and dependency hashes in source-pins.json. The test-source loader is required because the host has no rebuilt installed metallib. Run scripts/run_budget_case.py once with NUMI_AORTIC_WALL_FORCE_BUDGET_FRACTION=0.0005 and once with 0.00025. The runner uses the exclusive native wrapper, records source/runtime/shader hashes, and saves accepted FP32 node position/velocity snapshots. The comparison receipt is force-budget-trajectory-comparison.json. tests/force_budget_reference.cpp recomputes the source-derived pressure-load force scale; it is a CPU reference calculation, not a solver pass test.
