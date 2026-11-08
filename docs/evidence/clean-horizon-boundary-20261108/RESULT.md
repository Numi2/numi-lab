# Clean horizon native regression, 2026-11-08

Status: narrow reporting repair; no physics, performance, learned skill, or deployment qualification. Run on Apple M6 with a synthetic Franka fixture while the independent CPU contact reference was active.

The old metric reports 1/0/0 for 255/256/257 steps of a 256-step episode. The 256 timeout occurs at the requested boundary, so 1/1/0 is the intended result. The initial two-environment retry also revealed an uncounted host reset at step 220: the second environment had no termination but had not completed one continuous horizon. The revised metric counts a sole non-physics-error timeout at the final requested step as clean, and excludes any later-than-initial host reset. Termination counts are unchanged.

The retained post-fix native matrix passes 12 combinations of 1/2 environments, chunk 1/8, and 255/256/257 steps, with zero failed environment steps. Four reset-edge cases and two multi-repeat cases also pass. A separate artificial contact-termination fixture gives reason 3 and clean rate 0 for one and eight steps, including a final-step failure. The self-contained generator and regression script pass all 20 checks with both zero actions and the earlier trained-smoke incumbent policy. The latter exercises the compiled-policy path but does not establish skill. Raw outputs and input hashes are included.

Preserved failures: the first chunk-8 post-run tried an unsupported state trace and stopped before simulation; the first two-environment expectation incorrectly predicted both environments would timeout; and an attempted height-termination fixture was rejected because Franka has a fixed root. The amended plans and outputs record these. The correction was validated at the integration base revision recorded in reset-inputs; it has not itself established default-branch release integration.

Scheduled resets still occur during selection. The metric now exposes interruptions, but evaluating many environments across a whole uninterrupted episode may require a separate scheduling change.
