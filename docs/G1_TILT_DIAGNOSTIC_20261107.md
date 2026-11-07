# G1 first-episode tilt diagnosis and bounded intervention

The existing no-scheduled-reset traces do not identify the mechanical cause of
any fall. They establish the following orientation history before each sampled
environment's first terminal state. Tilt was independently reconstructed from
the root quaternion and agreed with recorded terminal tilt within 1e-5 rad.
Steps are 20 ms control steps; only one environment per arm was traced.

| Arm and traced environment | First > 0.5 rad | Start of final continuous > 0.5 rad interval | First > 1 rad | First terminal |
| --- | ---: | ---: | ---: | ---: |
| Policy 4081007, env 2 | 472 | 472 | 494 | 497 |
| Policy 4081008, env 0 | 138 | 148 | 164 | 174 |
| Zero action 4081007, env 2 | 47 | 47 | 59 | 63 |

The second policy trace crosses 0.5 rad and returns below it before its final
continuous excursion. Its first five consecutive samples above 0.5 rad end at
step 142; this short excursion alone is not proof that the subsequent fall was
inevitable. In both policy traces root height decreases before terminal, but
configuration-only sampling omits action, velocity, contact force, and sensor
history. Neither a prior crossing nor a post-step configuration proves which
component initiated the tilt. Other first episodes were not traced. The data
and analysis script, including source hashes, are in
`docs/evidence/g1-tilt-diagnostic-20261107/`; the retained trace and terminal
records are in the adjacent `g1-episode-local-20261107` directory.

## Preregistered fresh-seed diagnostic: action perturbation

This is a bounded mechanism probe, not a policy improvement or deployment
qualification. Seed 4081010 is new to this analysis; training-seed independence
is unknown. Preserve the historical source owner, exact pack and all four
policy/executor semantic fingerprints from `G1_EPISODE_LOCAL_RESULTS_20261107.md`.
Keep eight environments, 1,000 steps, chunk 1, the same task, ground, physics,
automatic task resets, and no scheduled resets. Each native process has a
120-second timeout; record every exit and timeout, with no retries. Retain all
raw outputs and commands, source patch, SHA-256 of pack, executable, native
library and Metal library. This plan is unexecuted and reserves no GPU slot.

1. Run the original closed-loop policy for seed 4081010 with env 0 state trace
   and `--rollout-pack`, on the original fingerprint-matched historical owner.
   Require no physics failure and retain all 1,000 steps and termination data.
   Verify that enabling pack recording does not alter the traced first-episode
   evolution by comparing against a separate same-seed closed-loop run without
   recording; retain both if they differ and stop the intervention.
2. Independently compile the same PolicyPack against the exact compiled task to
   obtain its *policy program fingerprint*, and compare it to the rollout-pack
   fingerprint. The `policy_rollout_fingerprint` printed in the rollout JSON
   hashes collected data and is **not** that program fingerprint. Run
   `extract_actions.cpp` against the policy and rollout packs using the compiled
   program fingerprint. It converts latents using the Metal affine FMA and clamp
   rule, and rejects contract, fingerprint, revision or dimension mismatches.
   Hardware math parity is not assumed: retain the generated action file/hash.
3. Replay that unmodified stream through the same task and physics with
   `--action-stream` in place of `--policy-pack`. Require exactly matching
   traced state rows for env 0, first terminal step and tilt for all eight
   environments, and equal physical failure counts for the whole 1,000-step
   baseline. Otherwise stop and retain the discrepancy. This test distinguishes
   open-loop action reconstruction and runtime behavior from actual policy
   feedback; it is a gate, not assumed from the local transform test.
4. Find the first five consecutive post-step states of env 0 whose tilt exceeds
   0.5 rad in its first episode. If none exists, or the baseline terminal falls
   within the next ten control steps, report no eligible intervention. Otherwise
   use `make_interventions.py` to produce two action streams: on the ten *next*
   control steps only, add +0.1 or -0.1 respectively to action index 5, the
   `left_ankle_roll_joint` in this task's joint order. Require both modified
   action values stay in [-1,1], and reject both arms without clipping if either
   fails. All other actions and environments retain the captured baseline
   stream. The extractor additionally rejects a policy clip below 1.
5. Run each intervention once under the same 120-second limit. Compare env 0
   orientation and height at the end of the ten-step pulse, and first terminal
   step/tilt. If an intervention terminates before the pulse finishes, report
   the termination rather than imputing a pulse-end state. Retain physics
   failures and all traces. Report signed differences for each arm against the
   exact-replay baseline; do not select the favorable arm as a successful
   policy or claim causality beyond the applied action's observed effect.

These arms are **open-loop after the perturbation**: the policy does not respond
to changed observations, and baseline actions after divergence are retained.
Any later auto-reset trajectory therefore has no interpretation as closed-loop
policy performance. An effect on tilt in this fixture would show sensitivity
to the chosen action at the trigger, not establish the root cause of the
original failure. Contact and velocity instrumentation would still be needed
for a fuller causal account. The test cannot be run until a GPU slot is
coordinated and the pack-recording and unmodified replay gates pass.

CPU preparation checks: the extractor compiled against the historical native
library and passed its affine/clamp self-test. The intervention generator was
tested with synthetic 29-action streams and the retained 4081008 trace: trigger
and step indexing, exactly ten modified entries per sign, and rejection of
out-of-range values before writing either arm passed. These checks do not prove
that a real Metal rollout pack can be extracted or replayed; no fresh-seed GPU
or intervention run has occurred.
