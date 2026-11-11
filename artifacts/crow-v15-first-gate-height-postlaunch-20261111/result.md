# Post-launch first-gate height diagnostic

The stage-specific fixed 0.85 m height command starts after 8 seconds. All eight first episodes reached the first waypoint, versus zero under the original actor; accepted q/v, policy outputs and contacts matched the original run exactly through observer step 400 across all eight starts. This is stronger evidence that the airborne height command contributes to the original gate miss, but it remains a fixed diagnostic command, not trained navigation or a final controller. Two starts failed by hitting slalom A; six remained incomplete at 1600 steps. Both first observed obstacle contacts for failed starts were left wing against slalom A, sampled at control endpoints rather than every substep. No full-route or landing success is claimed.

The original host and policies are archived in the walk-launch controls runtime; the corrected replay writer is archived in the replay-writer roundtrip runtime. Frozen flight source remains unchanged.
