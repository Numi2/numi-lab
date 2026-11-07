# G1 episode-local follow-up

The no-scheduled-reset experiment completed all three preregistered arms on
Apple M6. Each arm ran 8 environments for 1,000 control steps with no
reported physical solver failures and no requested periodic resets. The
fingerprint-matched retained policy did not demonstrate consistent standing
or walking through the full horizon: all eight environments in the first seed
terminated before step 1,000, and seven of eight did so in the second. The
remaining environment in the second seed reached the normal 1,000-step
timeout.

| Arm | First terminal steps | First episodes reaching timeout | Mean first-episode final displacement |
| --- | --- | ---: | ---: |
| Policy, seed 4081007 | 148–776, median 257 | 0/8 | +1.922 m |
| Policy, seed 4081008 | 174–1,000, median 397.5 | 1/8 | +1.128 m |
| Zero action, seed 4081007 | all at 63 | 0/8 | +0.749 m |

Displacement is measured from the first accepted post-step sample to the
first terminal step; it omits the first step and includes any motion while
falling. The zero-action result illustrates why positive displacement alone
is not a competence or walking certificate. For the three traced
environments, the retained row data independently reproduced the reported
terminal displacement within 1e-7 m. No physical push was applied.

Across all episodes within the 1,000-step runs, reason counts for policy
seed 4081007 were 18 tilt and one height termination; seed 4081008 had 13
tilt terminations and one normal timeout. Zero actions had 120 tilt
terminations. Multiple episodes are possible after automatic task reset.
These totals do not identify a cause for each first-episode fall.

The historical JSON field called clean_horizon_environment_count also counts
normal timeouts as terminations, so it reports 0/8 for the second policy
seed despite one normal timeout at step 1,000. The raw first-terminal step,
tilt, and aggregate reason counts establish the distinction above; no
per-environment terminal reason was recorded in this run. The current
experimental telemetry now emits explicit first-terminal reasons and timeout
counts for later runs. This additional code was not part of the recorded
historical binary.

The two seeds are distinct from the retained selection seed, but complete
training-seed history is unknown. Observations are idealized proprioceptive
and command values, not calibrated hardware sensors. These are bounded
simulated outcomes, not deployment, learned-policy promotion or disturbance
recovery qualification.

The exact owner was commit 791fb586934f8222fcf8269c411c17eb2a774087 plus the
retained owner.patch and EpisodeForwardEvidence.swift in the evidence folder.
The host-only compiler probe confirmed the policy's four world, task,
observation and action fingerprints matched before execution. The plan,
per-arm arguments, exits, stdout/stderr and one full state trace per arm are
retained alongside this report. The launcher enforced the preregistered
120-second per-arm timeout; no timeout or retry occurred. The rollout
executable SHA-256 was
571a7c461b698dfa0e2d4df48ef6e8a28515d4432c18fcea677b1a3fa5f9921f;
the linked native library was
80663b5c54b80798c7cfcfeb67dfb344de6992e2ecc9a48db531ecdaedd2209a;
the Metal library was
32d9ed14031ab83f87dfcf8281b1d3c65c6315a02e04ce15ab2728772ef35fab.
