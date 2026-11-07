# Preregistered generic depth-validity candidate

This research-only shader candidate changes the depth-history kernel: materialized pixels and edge neighbors require MR_VISUAL_VALIDITY_DEPTH, and the direct-winner path clears its selected flag when sensor measurement fails. The unchanged, already executed probe from 8e836afd is used byte-for-byte, with only the candidate metallib changed. Its baseline and all three invalid-depth cases must pass; retain all pixel values and failures. Run once with a 120-second cap in an exclusive GPU slot, with UTC timestamps and pre/post input hashes.

The candidate does not change source used by the product and is not a complete promotion proposal. In particular it has not established rendered-scene behavior, all renderer configurations, intentional edge-flicker behavior, large planes, object-track validity, observation fingerprint compatibility, mixed-environment transactions, or policy accuracy. The geometric-only direct-winner path with sensor sampling explicitly disabled retains its existing semantics. Nonzero edge flicker can intentionally fill nonselected pixels and is outside this no-flicker test.

