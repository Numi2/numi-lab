# Wider stance-pitch correction: delayed fall, failed release

Increasing only the stance pitch-correction cap from +/-0.10 to +/-0.14 rad left the longest qualifying clearance streak at 44 intervals, below the required 100, but delayed the first root height below 0.1 m from one-based step 391 to 401. Candidate stance-toe speed first reached the accepted-state 12 rad/s threshold after the fall; its peak before the fall was 2.16 rad/s. This is a limited development result, not stable support.

The control reproduced all 450 prior candidate states, with a 296-state exact paired prefix. Pitch, lift and leg command reconstruction passed, accepted hip positions and targets remained within configured bounds, and all 32 contact substeps were present without native/observer errors or nonfoot support. The binary matched its prebuild. Substep toe speed and actuator effort were not established. The generic geometry audit's `postreturn` label is inapplicable because this test holds the foot released and never commands return; contact/state epochs remain independently unverified.

This constructed stationary perch test does not establish walking, regrip, landing or biological validity. Raw control and candidate failures are retained.
