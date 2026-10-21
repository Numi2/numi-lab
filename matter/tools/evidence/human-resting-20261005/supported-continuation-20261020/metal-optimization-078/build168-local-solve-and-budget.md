# Build 168: local-solve, Newton-budget, and telemetry controls

All native executions used SSH Numi, a frozen executable/library/Metal source closure, the supported initial pose, full gravity, real support, active coupled physics, and the existing residual and physical acceptance gates. These are one-root initialization experiments (125 microseconds simulated), not settled or sustained performance qualifications.

| Trial | Body wall seconds | Comparison |
| --- | ---: | --- |
| Existing refined graph LU | 9.694047 | Reference |
| 8-sweep Jacobi with certified LU fallback | 11.939205 | 1% audit passes, max difference 0.936865%; slower |
| 4-sweep Jacobi with certified LU fallback | 11.017791 | 1% audit fails, max difference 2.802398%; slower |
| 12 Newton iterations | 7.000909 | Physical root accepted; 1% audit fails, max difference 4.98306% |
| 8 Newton iterations | 5.795879 | Physical root accepted; 1% audit fails, max difference 5.48165% |
| Work counters disabled | 9.708940 | All 17 metrics, 14,202 node records, and 54,133 tetrahedron records exactly equal to reference |

No faster trial qualifies as an equivalent replacement. The Jacobi path remains opt-in and retains the original pivoted/refined LU whenever its local backward-error certificate fails. The native compact-graph fixture passes under Metal API validation, exercising both Jacobi acceptance and fallback. Actual integrated fallback counters are added for the next build so the failed performance experiment determines the next implementation change.

The Newton-budget experiments do not relax final physical certification, but their endpoint differences exceed the existing comparison gate. No budget reduction is promoted. The telemetry control is a single pair and provides no evidence that observational counters dominate runtime.

Exact invocations, process times, audit comparisons, failed build logs, and the frozen source closure are retained in build168-local-solve-and-budget-metadata.tar.gz. Its independently reopened per-member SHA-256 receipt is beside it. The full Human goal and 0.1x end-to-end requirement remain unfulfilled.
