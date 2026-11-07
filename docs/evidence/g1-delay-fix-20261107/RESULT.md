# Repaired native mixed-reset result

The preregistered historical-port probe passed on Apple M6: all 768 sampled
delay-one observations matched their known input values exactly at the
published Float32 precision. Every reported physics status succeeded and no
task termination occurred. For the reset environment, observations at
steps 5 and 6 were zero, then steps 7 and 8 returned the newly supplied
post-reset actions. All seven other environments retained their expected
histories throughout.

The native command ran from 2026-11-07 22:48:04.186244 UTC to
22:48:04.477310 UTC, below its 120-second bound. The complete per-sample
output, exit, command, UTC timing, executable/library/shader and source
hashes are retained here. The observation fingerprint was
9355055053806751635; world, task and action fingerprints matched the historical
owner. The independently compiled original PolicyPack was rejected because
its observation fingerprint was 15900562835394482329. No policy was rebound.

This qualifies delay one and the tested reset sequence in the ported historical
owner only. It does not qualify other delay depths, rejected-physics sensor
transactions, actual contact force, the full current sparse checkout, or
behavior of a policy trained with corrected observations. In particular,
the earlier first-episode falls remain unexplained.
