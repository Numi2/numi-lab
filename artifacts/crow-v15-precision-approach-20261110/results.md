# Precision approach result: failed landing gate

The guided first episode reached all five waypoint triggers at step 2787, with no reported physical or solver failure, but had zero perch-foot contact samples by its 5000-step timeout. No supported landing or grip occurred. Tightening the descent gate and changing the forward command did not settle the body; selected samples after step 3400 show forward speeds in both directions of approximately 0.18 m/s and substantial horizontal oscillation. At the episode end, the body remained airborne with a 1.278 m height target.

Although the invocation requested 7000 steps, the runtime reset at its 5000-step horizon and began another episode. The analysis stops at the first reset. This result is a used-layout privileged-guide exploration, not a held-out or autonomous evaluation. It does not qualify precision hover or the toe-release sequence.
