# Fresh-seed policy pack and replay qualification

The research plan in `G1_TILT_DIAGNOSTIC_20261107.md` specifies the question,
controls, environment and physical scope. This run tests only its measurement
and exact-replay prerequisites, not the proposed action intervention.

- Fresh seed 4081010; eight environments, 1,000 control steps, 20 ms control
  period, chunk one, four physics substeps, 4+2 contact iterations, normal
  automatic resets, and no periodic resets. Trace environment zero. Training
  seed independence remains unknown.
- One closed-loop policy run; one same-seed policy run with rollout-pack
  capture; one unmodified action-stream replay if earlier checks pass.
- Run in that order with a 120-second timeout per native rollout. No retry,
  no parallel GPU work, no perturbations. A failed run or mismatched gate stops
  the sequence and is retained.
- Enforce the historical recorded executable, library, metallib and policy
  hashes, historical owner revision and source patch/file hashes, and matching
  exact world/task/observation/action contracts before the native runs.
- Compile the policy against the real historical task manifest on the CPU:
  program fingerprint 17940125103817627925. The separate data-dependent
  `policy_rollout_fingerprint` is not the program fingerprint.
- Capturing the rollout must preserve env 0 trace through its first terminal
  state, and all eight first terminal steps/tilts and failure counts.
- The pack extractor must match the compiled fingerprint and policy revision.
  The unmodified action-stream replay must match the *entire* env 0 trace and
  all eight first terminal steps/tilts and failure counts exactly. Any mismatch
  or physics failure invalidates the intervention; it is not silently relaxed.
- The experiment retains per-arm command, stdout, stderr, exit, hashes, source
  provenance, parity reports, serialized pack and action stream. The host-only
  policy artifact and historical source remain dependencies; if outputs are
  too large for Git, preserve their hashes and location explicitly.

The selected env 0 trace contains configuration values only; exact replay
qualification for other environments is limited to their reported terminal and
failure fields. The replay does not establish policy competence, sensor
realism, physical correctness, or contact-force accuracy. It also does not
assert that recording and replay are bitwise equivalent before measuring them.
