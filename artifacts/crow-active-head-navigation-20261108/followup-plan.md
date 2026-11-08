# Follow-up admission after native visual-policy requirement

The initial zero-actions command was rejected before physics because device
visual observations require a native policy. Repeat the same eight-step test
with a fresh deterministic zero-output policy initialized for the exact
runtime-reported v15 world/task/observation/action fingerprints. Compile-time
checks independently report 816 actor inputs, 106 critic inputs and 18 actions.
Verify all 18 accepted actions are exactly zero in each retained replay frame,
all observations are finite, and there are no failed steps.

The old feeding policy was rejected with `incompatible_contract`. The initial
summary incorrectly labeled that rejection false because its string matcher did
not include that error name; the raw stderr and original summary are preserved.
This is an admission test only, not learned or held-out navigation.
