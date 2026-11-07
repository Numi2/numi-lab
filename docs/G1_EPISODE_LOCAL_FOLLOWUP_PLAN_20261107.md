# G1 episode-local follow-up plan

Status: prepared, not executed. This plan is separate from the already
completed fixed-seed runs and does not alter their retained evidence.

Question: with the same fingerprint-matched retained G1 PolicyPack, when do
the first tilt terminations occur and what displacement occurs before that
environment's first termination, without periodic forced resets?

Use the reconstructed 791fb586 owner plus its retained pre-motion patch.
Port the checked episode-local telemetry change to that isolated historical
executor and retain the additional source patch and binary/library/metallib
hashes. Verify the four PolicyPack contract fingerprints again before GPU
submission. Do not relax the policy compiler or change the policy.

Arms: policy seeds 4081007 and 4081008, plus zero actions at 4081007.
Each arm uses the same eight environments, 1,000 control steps, one repeat,
one control step per submission, 4 physics substeps, 4 velocity iterations,
2 final velocity iterations, ground and ordinary physical termination rules.
Use --no-scheduled-resets and --state-trace; trace environment 2 for the
first seed and environment 0 for the second. Record every arm's stdout,
stderr, exit status and state trace. Limit each process to 120 seconds,
recording a timeout as a failure and performing no automatic retry. Metal
pipeline warmup is included in that limit.

Report physics-failed steps, reason and step of the first termination in
each environment, tilt at that termination, and the number of environments
without termination. Report first-episode peak and final forward displacement
from the first accepted post-step position through the first terminal step.
The first-step displacement is not captured. Values after automatic reset
cannot enter that first-episode record. Missing or rejected samples produce
unavailable evidence, not inferred zero displacement. Raw per-step traces
must remain available to independently examine the selected environment.

The current telemetry implementation enables this progress record only with
trace, one-step submissions, one repeat and scheduled resets disabled. This
is deliberately narrower than the general rollout API: multi-step chunks
do not expose every intermediate final configuration. The legacy scalar
zero placeholders remain in the JSON for existing readers when availability
is false; they are not measurements and must not be used as evidence.
A new uninterrupted-horizon indicator distinguishes a no-termination count
from a run that was periodically reset.

The goal is diagnosis, not promotion. Even successful completion would use
idealized proprioceptive measurements and an authored zero-push task, with
only two fixed seeds; it would not establish hardware performance or
disturbance recovery. If exact fingerprints cannot be reproduced, or the
executor reports a physical failure, retain the result and do not run a
modified policy or silently rebind the contract.
