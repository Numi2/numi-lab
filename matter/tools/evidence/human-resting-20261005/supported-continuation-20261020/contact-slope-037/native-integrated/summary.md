# Native contact-slope 037 result

The run exited 1 after 35.499342000 s wall time. It accepted 2 rows through 0.25 ms of a requested 32 x 125 us (4 ms) horizon, then failed at step 2 with Matter code 6, object 19, failing index 2624, and 3 microsteps. Diagnostics were [6.346143521795966e-08, -8.35808158794471e-11, 1, 0]; runtime owner was passive_tissue_runtime.

Compared with native-sort baseline 036, which accepted 24 rows through 3 ms of a 25-step request, 037 is a regression in accepted horizon. Both report the same Matter code/object/index tuple, but 037 stops earlier and has a different microstep count and diagnostic vector; this recurrence does not establish an identical failure mechanism. Strict physical acceptance settings and normalized arguments are unchanged. Only run-specific paths, diagnostic trace/timing capture settings, and requested step count differ.

This is a failed short numerical continuation, not a completed 4 ms horizon, steady-support result, or physiological qualification. The exact invocation, provenance, all 13 rehashed input identities, accepted-prefix CSV, surface audit, failure receipt, and selected logs are retained here. The 297 MB output pack, large JSON diagnostics, and full raw stdout/stderr remain in the original run directory and are hash-identified rather than copied. Replay analysis is retained separately.
