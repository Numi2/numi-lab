# Earlier hip balance test

The late negative hip correction delayed termination slightly but did not prevent tilt exceeding 0.35 radians. Test whether earlier bounded correction, beginning at interval 50 once both contact-ramp counters reach 12, can prevent the drift instead of reacting late. Use the same combined x=17.765 round fixture and front-toe target; increase pitch and pitch-rate coefficients from -0.20/-0.025 to -0.50/-0.08 and clamp the added hip target at 0.16 rad. All other state, policy and geometry remain the same. These are position targets only, not body-force injection; this is a reconstructed forced-landing development fixture.

Run 600 intervals, compare with retained baseline and late negative correction over matched first-episode steps, and require no reset/termination plus at least final 250 intervals satisfying every existing substep/contact/balance gate. Audit realized joint targets and limits if the test passes. Preserve failure.
