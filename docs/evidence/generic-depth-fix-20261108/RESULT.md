# Result: isolated depth-validity candidate

The pinned comparison ran from 2026-11-07 23:46:11.772990 to 23:46:11.854790 UTC and exited 0; binary and shader checks passed before and after. The exact original failing probe executable was used, with only the candidate shader replaced.

The pre-existing synthetic baseline passed. All eight sensor-invalid samples now remained at far depth in the unsegmented, instance-selected and direct-winner cases: 0/8 false near samples in each, versus 3/8 in each original run. Complete values and the original failure are retained.

This is an isolated kernel candidate, not a product source change or an end-to-end rollout result. Large images, nonzero edge flicker, object-track validity, additional camera/renderer configurations, observation-fingerprint compatibility, policy behavior and failed-physics transaction semantics remain open. The production CCD and contact experiments are unaffected.

