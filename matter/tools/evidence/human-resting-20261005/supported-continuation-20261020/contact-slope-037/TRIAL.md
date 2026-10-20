# Analytic contact slope 037

Keep the current registered body, mass debits, material laws, physical
timestep, solver budgets, and all force/accepted-state owners. Change only
the IPC potential directional derivative in deformable line search: contract
the existing product-rule gradient with the already calculated analytic
nodal derivative (dt times correction; zero for fixed/not-due FEM nodes).
The old contraction used the rounded alpha-one endpoint chord, which is
not the derivative of the Newton path.

The exact rounded endpoints still own every trial energy and CCD check.
Keep strict Armijo, minimum separation and all physical certificates.
Do not claim this change resolves a rounded-endpoint plateau.

Before native continuation, require the original geometry checks plus
independent potential finite-difference slope controls, the production
sorter reference, and consolidated two-step injected rejection/rollback.
Then request 32 steps at 125 microseconds in the same integrated scene.
The root24/tick0/contact2624 selector is a historical diagnostic target;
after a slope change its slot need not denote the same feature. Require
recorded source identities before claiming a matching-contact diagnosis.
Baseline036 reproduced the preceding failure but its tick1 selector missed
the failing contact; it provides no per-feature failure diagnosis.

A pass of four milliseconds is numerical continuation only. It does not
qualify supported equilibrium, broad anatomical clearance, sustained
circulation, all ligament coverage, or a physiological resting interval.
