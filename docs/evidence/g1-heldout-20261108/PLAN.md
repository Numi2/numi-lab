# G1 untouched-seed evaluation

Registered before evaluation. Test the three retained 200-update candidate
deployment actors on two seeds not used by this training study or selector:
4081022 and 4081023. These seeds are fresh relative to the documented pilot
and this study; the entire historical G1 seed inventory is not established.
Keep the same staged revision 3a439b63, ground, g1-legs-locomotion, eight
environments, 1000 steps, four physics substeps, four velocity iterations,
two final iterations, and one control step per submission. Each of the six
trials uses the candidate actor, not an incumbent or altered action stream.

Prediction: at least five of the six runs will have no clean-horizon
environment. This is a falsifiable prediction based on the failed selector
screen, not a qualification criterion. Record physics errors, termination
counts/reasons, all per-environment final progress and clean-horizon rates.
The previously declared skill screen requires zero physics failures, all
environments clean through the full horizon, and each final displacement at
least 1 m. Preserve every run, error, input hash and policy; use a 60-second
timeout per trial. Task GPU work is paused during each pair, while CPU and
system activity are uncontrolled.

The simulator observations are idealized. These evaluations cannot establish
real-robot performance, contact-force accuracy or general deployment validity.
No experimental CCD or contact option is enabled.
