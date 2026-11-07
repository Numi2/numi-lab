# American crow: native feeding task and learner controls

macmini M4 Pro, 7 November 2026. This is a grounded primitive curriculum for
the user's lifelong-intelligence goal. It is not a completed lifelong ecology
or a qualified learned animal controller.

The v12 feeding task connects 97 actor/critic inputs and 18 actions to the
native resource ledger. Four inputs expose nutritional crop fullness, new
intake, and local intake-volume contact/availability. No remote food position
or eye images enter the actor. The standing curriculum holds other controls;
the three physical head/jaw controls remain active. Nutritional payloads are
intake-volume tracers, not resolved food grains, swallowing or body-mass gain.

Exactly 45 mg in a patch can produce at most 4.5 utility per lifetime.
Five native controls each use eight lanes and 448 steps / 8.96 s. All canonical
feeding lanes consume 45 mg in three bites. Head-yaw misses and held-closed
controls give zero food utility. Chunks 8 and 64 give identical ledger state.
Task resets preserve depletion and cannot manufacture a closing-gesture bite.
All physical controls have zero physical terminations and failed device steps.
The C/Swift/Metal connected action-stream controls independently agree on
reward and typed intake outcomes, within F32 rounding of 4.5 per lane.

Task ABI 47 binds the native state-program fingerprint into PolicyPack task
identity. Resource ABI 3 binds gesture/reset semantics. Changing food geometry,
quantity, bite size, capacity or thresholds changes that identity; a mismatched
runtime is rejected. TaskPack roundtrip preserves it in a tagged optional
extension. Legacy TaskPack bytes remain unchanged when the identity is zero.
Resource-only checkpoints remain separate from complete life checkpoints.

The first admitted learning pipeline used default 0.2 standard-deviation
exploration. Its reported updates had zero food reward. Wider unit-standard-
deviation exploration produced food reward, but its first resume correctly
rejected a mismatched exploration configuration. After the harness preserved
the same configuration, two PPO updates and a resumed update passed: 196,608
accepted samples, revisions 1 to 3 to 4, zero device failures. Final reported
mean food utility was 0.0003052 per transition in the initial process and
0.0040665 in the resumed process, with maximum about 2. This is not evidence
of improvement: resume creates an independent fresh finite world and restores
the learner only. The learning rate was fixed at 0.0001. Whole-process speed
was approximately 18,260 and 19,391 transitions/s; MLX peak was about 458 MB.

The three-update deterministic deployment screen consumed ZERO food on all
three development seeds. It had no physical terminations or device failures.
The candidate is unqualified. Positive stochastic training reward and revision
publication do not establish a learned feeding skill. Both ordinary and
residual stochastic PolicyPacks replay exactly across chunks 1 and 8 under
the current ABI. Qualification seeds/layouts remain unused.

All ten trial directories retain prospective plans, raw JSON, receipts,
failures, authored action streams and small native checkpoint files. The
inventory hashes larger retained macmini policy/learner/rollout payloads.
Admission 01 rejected inconsistent command limits. ABI-46 instrument trials
are retained separately from the final ABI-47 controls; no old policy was
silently rebound. Six native CTests and sixteen Python crow tests passed.
Some Python tests are synthetic qualification fixtures, not crow results.

Pending: learned primitive reliability, audited head eyes and visible food,
free locomotion/search, physical manipulation/caching, water and physiology,
complete world/RNG/sensory/memory restore, recurrent memory, social individuals
and communication, and behavioral qualification of the lifelong environment.
