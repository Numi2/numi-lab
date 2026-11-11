# Longer-stance knee swing: failed support gate

The candidate used the larger knee command from the previous negative screen, a 0.80-second cycle rather than 0.50 seconds, and a narrower smooth knee-lift pulse. The v15 body, foot materials, frozen flight controller and starts were unchanged. Both 300-step teacher-off native runs completed successfully on the previously used eight-lane development seeds. Logged pre-walk q/v, action and contact-substep summaries matched through step 199 in every lane.

All 16 lanes increased mean bilateral knee span by at least 15%, and all moved forward in aggregate without reported non-foot contacts. However, no-foot-normal-contact physics substeps rose from 33 to 2,744 of 25,600 in the walking windows. The earlier simple larger-swing test had 2,112; neither result meets the predeclared support gate. This candidate is rejected and is not incorporated into the main gait. No-foot-contact substeps are not automatically falls; per-lane traces and foot-clearance observations are preserved.

This screen does not establish a stable walk, takeoff, navigation, landing, unseen-layout performance or anatomical realism. `comparison.json`, `body-comparison.json`, analyses and raw compressed recordings support the result; `feedback.patch` and the archived compiled library identify the candidate.
