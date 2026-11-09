# Follow-up: constrain the exploratory endpoint to the current reflection map

The unrestricted CPU correction lowered the independently recomputed residual
below the unchanged limit, but 875 velocity components at its repeated
candidate were not produced by the three closest FP32 stage-velocity values
tested. This does not establish a mathematical impossibility. Repeat the same
independent iteration but choose each endpoint velocity from the result of the
current FP32 reflection `v1 = 2*vm - v0`, testing the nearest stage float and
two neighboring floats on each side. Select the output with the smallest
frozen-force component residual. Recompute positions and forces as before.

Predict this restriction makes the residual exceed the same limit. Reject the
prediction if the corrected full-step residual is at or below the limit while
the kinematic and clearance checks pass. Retain all iterations and hash both
executable and input. This is still only a nearby finite search, and satisfying
these CPU equations does not prove that the native stage, its convergence
certificate, or any contacting transaction can reach that state. Do not modify
Matter runtime behavior or acceptance limits in this experiment.
