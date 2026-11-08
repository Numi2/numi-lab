# Object tracker: immediate encode rejection and retry

## Scope and acceptance

A real invalid-body-count failure in the native object tracker must not advance the host sensor cadence or replace the accepted render metadata. A subsequent valid callback must use the next sensor sequence and retain the existing moving-track and depth-history checks. Exercise both the materialized renderer and the graph-only direct-winner path. This is an encode-callback guarantee, not a physical-step transaction guarantee. Plans were written before their respective measurements and are retained here.

## Result

On Apple M6 with two Franka environments, the unmodified baseline completed a no-fault first frame with sequence 0. In the fault case the initial rejected callback left metadata unchanged but the first successful frame used sequence 1. The stronger native regression first commits sequence 0, rejects a body-count mismatch in an unsubmitted command, and retries: the baseline incorrectly produced sequence 2 and failed; the candidate produced sequence 1 and passed the existing moving-track and held-depth checks. The candidate also completed the direct-winner graph-only first frame and retry with finite normalized depth after a rejected callback. The graph-only test confirms path viability but does not independently inspect its sequence.

The candidate keeps host cadence and direct-render uniforms provisional until the callback completes and checks its resources before rendering. The four native CTests for tracker retry, masked-depth validity, visual observation fingerprint, and task program all passed. The first direct-path fixture run failed because the probe removed its temporary visual pack before compiling its second renderer; the failed output is retained. Moving that cleanup after both renderer compilations allowed the same intended test to run successfully.

## Limits

The tracker still publishes its host cadence after a successful encode callback, before the owning MetalWorld command is submitted or physics is accepted. GPU histories and per-environment acceptance remain unqualified if a later operation fails, the command fails, or physics is rejected. No rollback, deployment performance, learning result, or force accuracy is claimed. Preserve the separate owner-lifecycle design requirement; this repair does not implement it. Experimental collision advancement and augmented-Lagrangian contact settings are unchanged.

Baseline source was main `bdce5880f`; hashes, outputs and the local source/build identities are recorded alongside this result. The failed direct setup and the uncorrected baseline are evidence, not physics failures. No timing comparison was attempted.
