# Stronger stance-pitch feedback: failed release

Increasing the stance pitch gain from 0.80 to 1.20, with the same target limit and slew, reduced the longest strict released-foot clearance streak from 47 intervals in the matched control to zero. The candidate root fell below 0.1 m at one-based interval 383 versus 402 for control. Its released left toe reached the accepted-state 12 rad/s threshold at zero-based tick 341, before the fall. The stance right toe reached that threshold only after the fall. The accepted-state audit does not bound substep speed or effort.

Control q/v exactly reproduced all 450 prior combined-candidate states; the two lanes shared an exact 172-state prefix. The pitch target reconstruction passed with no mismatch; all 32 substeps were present and no native or observer error or nonfoot support was recorded. Contact/state epoch alignment remains unverified. The initial local audit used a Python without numpy, and a separate optional audit was called without its required directory argument; those errors are retained, and the affected audits were rerun with the bundled numpy runtime and correct argument. These are analysis execution errors, not physical successes.

This constructed stationary, open-toe perch probe did not test walking, regrip, landing or biological fidelity. It fails the registered 100-interval criterion; do not adopt the stronger gain.
