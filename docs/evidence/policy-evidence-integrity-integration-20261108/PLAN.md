# Generic policy promotion: incomplete physical evidence

Before changing selector code, test a WorldPack incumbent and a higher-reward
candidate with (a) a recorded termination per environment but a missing
termination total and (b) a missing failed-step counter. Both are incomplete
and should preserve the incumbent. Test current `main` and the integration
branch. Then repair the generic WorldPack/URDF comparison so incomplete,
malformed or mutually inconsistent physical evidence never advances a policy,
while valid clean evidence can still advance and genuine failure evidence
cannot be offset by reward. Exercise actual selector CLI artifact selection,
not only the helper. Preserve native timeout and known success controls; do
not classify unknown termination reasons as success.

Native `task_rollout.swift` evidence is the schema reference. Qualification
includes retained real Franka evidence if available and existing targeted
selector tests. Do not reinterpret this as trained skill, contact-force
validation or deployment qualification. Record all failures and fixture
updates, and keep G1 work out of scope.

Native follow-up: use the previously compiled relocated task-rollout binary,
its matching metallib, the retained synthetic Franka WorldPack and typed packs,
and the two retained policy packs. Run the modified source selector for one
16-step environment in a fresh output directory and seed 4081019. Confirm both
native evidence records satisfy the physical-counter contract, the published
deployment matches the selected existing pack, inputs remain byte-identical,
and a repeat with the same contracts reproduces its decision and physical
counters. This is an old pinned evaluator with the new Python selector, not a
freshly built combined release, trained skill, force or timing qualification.

Native follow-up amendment: both first runs rejected before simulation because
the previously relocated evaluator predates `--no-periodic-resets`. Preserve
those logs and the incumbent deployments. Repeat the same contract with the
reconciliation build task-rollout binary, its matching dynamic library and
metallib; hash them and all unchanged inputs before execution. No flag is
removed and no fixture or policy is replaced.

Final-source follow-up: the selector was subsequently reformatted after the
first matching-evaluator series. Re-run the same two native trials with the
final source hash, unchanged evaluator and fixture, in fresh output directories.

Integration follow-up: repeat the same native Franka workflow twice using the
resolved integration-branch selector source and the same pinned native build.
Hash resolved source and inputs before running. Rerun integration selector
and deployment tests, retaining the previously known G1 assertion failure.
