# Reacting-body rollback fixture: impulse accounting diagnostic

The rollback fixture is a dynamic one-kilogram prismatic body contacting a
mixed FEM pad, with zero rigid gravity and applied effort. It is a legacy
contact-law diagnostic, not the requested area-law midpoint comparison.
At the original `1e-4` legacy residual tolerance, the reported contact
impulse on the rigid body was 9.65069e-5 N s, while its rigid momentum changed
by 7.39344e-5 N s. The scalar normal impulse, vector z impulse and reduced
reaction agreed with one another. The difference is about 23.4% of the
reported impulse; it does not meet a force or momentum-accuracy gate. The
reported nonlinear certificate was 4.57814e-5, comparable in magnitude to
the discrepancy, but this does not prove its cause. Torque samples are
reported in the raw log without an independent moment-balance certificate.

Keeping geometry, material, physical timestep, contact law and iteration
budgets unchanged while tightening the legacy scalar tolerance to `1e-7`
rejected the very first step with Matter status 10 (nonlinear solver failure).
Its diagnostics were `(1.34402e-7, 0.0123025, 1.87503e-5, 1.01434e-7)`.
This observation does not establish whether the cause is an inaccurate
operator, line search, scaling, or another stopping issue. It is not a
force-qualified response, and no iteration-budget increase or tolerance
relaxation is used to make it pass.

The numerical comparison, prior failure, and instrumented repetition are
retained in the numbered logs. The four scoped rollback regressions still
pass with the unchanged default tolerance. Resolving this diagnostic and
qualifying the actual frictionless reference-area midpoint transaction remain
separate work.
