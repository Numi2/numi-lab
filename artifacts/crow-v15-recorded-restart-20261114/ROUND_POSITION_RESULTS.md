# Round-perch position screen results

These are development fixtures at a reconstructed 48-second recorded state, with the landing phase forced and the rigid capsule moved close to the bird. They are not original-course autonomous landings. The capsule has 25 mm radius and a 695 mm axial half-length. The unchanged control at x=17.745 m and positions x=17.755 and 17.765 m were evaluated with the same state, material, policy tensors and full-substep contact observer. Each first episode alone counts; the host automatically resets following termination.

| Capsule center x (m) | First tilt termination (step) | Longest continuous bilateral support and opposing-grip streak | Result |
| --- | ---: | ---: | --- |
| 17.745 | 154 | 23 steps (46–68) | Fail |
| 17.755 | 174 | 22 steps (46–67) | Fail |
| 17.765 | 216 | 0 | Fail |

One step is 0.02 s. All three tipped, and none met the predeclared 250-step end-of-run grip and stability criterion. At x=17.765, right-foot opposing-force criteria failed for 213 of 215 audited intervals; the longer survival is not a qualified grip. The traces indicate the front contacts unload as forward tilt increases. Raw replay, reset audit, all-substep contacts and per-step analysis are retained alongside these results. No claim of general collision robustness follows from these trials.
