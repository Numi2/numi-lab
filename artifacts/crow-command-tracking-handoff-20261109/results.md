# Matched flight-only controller continuity result

With only the flight-bench blend kept active after waypoint five, the previously failing lane 15 survived all 1600 steps and passed its original sampled straight-flight screen. All 80 lanes passed their assigned screen; there were no physical failures or failed environment steps, and climb-minus-descent median height remained 0.39989 m. The matched sample comparison is recorded in comparison.json. This supports the handoff explanation for the matched crash; it does not establish robustness beyond these starts, continuously safe flight, realistic landing, or navigation.

The controller still uses a teacher-only accepted-state feedback path and has not been frozen or deployed. The hard waypoint handoff in the actual task remains a separate integration issue.
