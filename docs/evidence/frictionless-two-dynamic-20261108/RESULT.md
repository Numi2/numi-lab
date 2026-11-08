# Two dynamic bodies: frictionless response screen

The isolated zero-friction block candidate passed the preregistered one-step, two-dynamic-body fixture, while the existing shader accepted substantial incorrect contact response. This is new evidence beyond the earlier static- and kinematic-plane checks. It is not a production promotion: [the previous mixed rod–rigid test](../frictionless-active-mixed-20261108/RESULT.md) still fails its normal-residual gate, and full rollback and force qualification remain open.

A 1 kg tilted articulated box contacts a 2 kg dynamic scene box. The two bodies start 0.1 mm apart with no gravity, friction, damping, restitution, or CCD. Two tilts, two relative normal velocities, three tangential velocities and two solver budgets give 24 cases per arm. All 48 cases generated two contacts, were accepted and reproduced exactly in the fixture's repeated exposed state.

| Measurement | Existing shader | Isolated candidate |
| --- | ---: | ---: |
| Closing cases with positive normal impulse | 8/12 | 12/12 |
| Nonclosing cases with spurious velocity change above 1e-6 | 4/12 | 0/12 |
| Cases exceeding the independent 0.002 m/s normal-residual screen | 16/24 | 0/24 |
| Largest independently reconstructed normal residual | 1.03521 m/s | 0.0000491 m/s |
| Largest total linear-momentum error | 3.06e-8 N s | 1.79e-8 N s |
| Largest difference between reconstructed and solver normal residual | 4.99e-8 m/s | 7.61e-9 m/s |

The candidate preserves each body's tangential velocity and, across matched tangential-speed arms, produces identical normal and angular velocities and normal impulses. Its largest reconstructed residual is 4.91e-5 m/s at two iterations and 1.51e-8 m/s at 32. The baseline has eight residual failures at each budget; increasing its fixed iteration count does not repair this failure. Equal-and-opposite linear momentum transfer alone is insufficient: the baseline conserves momentum in these cases while producing an incorrect response.

`analyze.py` calculates point velocities from each body's final linear/angular velocity and its initial centre and the captured shared witness, then independently forms each recorded normal complementarity equation from its evaluated target, regularization, and impulse. The computed residual agrees with the reported residual within the preregistered 2e-5 m/s screen. That agreement and conservation support the fixture's measurement but do not prove whole-trajectory clearance, angular momentum conservation, physically calibrated force, other contact manifolds, or complete owner/cache restoration. The matched response maximum combines velocity components and impulse and is used only as the preregistered per-component numerical screen, not as a quantity with one physical unit. All per-case values and failure counts are retained in `analysis.json`.

Both shaders use the current integrated native library and the same linked AIR dependencies; the candidate and its build recipe are retained in the adjacent [active-contact evidence](../frictionless-active-contact-20261108/RESULT.md). This fixture's preregistered `PLAN.md`, generator, native source, raw results, analysis and pretrial identities are included. `commands.txt` records execution; the executable and linked binary artifacts remain on `ssh numi`. The selected solver is discrete and CCD is disabled. No performance or general collision-safety claim is made. Next address the already documented mixed-island convergence and acceptance problem and qualify exact restoration before reconsidering any production change.
