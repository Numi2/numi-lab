# Ten-step mixed-reset follow-up

The eight-step study at commit 06a05dcc passed all 3,072 comparisons, but
the reset environment could not yet publish a nonzero post-reset value at
delay 3: its first such value would appear at step 9. This separately
registered follow-up extends the otherwise identical fixture to ten control
steps with new seed 4081014. It uses the same corrected historical owner,
four separately compiled customized tasks with delay components 0–3 and
maximum action delay two, eight environments, 20 ms control steps, four
physics substeps and 4+2 contact iterations.

Environment 1 is reset at step 5. Across all depths, steps, environments,
and twelve observed leg actions, require each actor-frame value to equal the
supplied action from `step - 1 - delay`, or zero if that action does not
exist or predates the reset. In particular, delay 3 for the reset environment
must publish the step-5 command at step 9 and the step-6 command at step 10.
All seven other environments must retain their histories. This is 3,840
comparisons with absolute tolerance 1e-6. Require successful physics and no
task termination.

Run the single native process once with a 120-second bound after GPU
coordination. Retain its exact command, per-depth task fingerprints, input
formula, 3,840 rows, exit, UTC start/end, source and binary/library/shader
identities. No retry or threshold modification. The compiled probe SHA-256 is
`7267c035cb9e33a05b82557a7c5701eb3414908205d789cf2b7524e222126330`.
The customized tasks do not evaluate or rebind the old PolicyPack. This does
not qualify actor-current, critic, sensor history after physics rejection,
physical realism or policy performance.
