# Authenticated refined-child source-only preflight

Canonical detail maps can now be loaded in an explicit preparation-only, source-only preflight over the authenticated 002→530→006 child route. Seed CSV parsing and canonical CSV parsing remain separate. The new permission is false by default; the runtime identity gate remains enforced without it. No physical bound is relaxed.

A guarded census evaluates all 56,054 detailed rows with the native child indices and the same barycentric, extrapolation, Float32 sum and reconstruction helpers as production preparation. The census is permitted only with preparation-only + source-only + preflight + canonical input + an explicit census request. The normal preparation still returns its original strict failure afterward.

On current006 (SHA 888eb124…cbbede63), native source-only replay took 8.455 seconds. It found 31,913 rows above the 1.5 absolute-weight limit, 29 above the 50 mm extrapolation limit, 224 failed Float32 weight sums, and 9,602 failed 0.2 µm reconstruction. No label, condition or barycentric construction failures occurred. These categories overlap. All 31,913 weight failures contain negative weights; maximum absolute weight is 172.439. This establishes a geometric embedding problem rather than a child/parent namespace mismatch.

A separate read-only same-label search examines every eligible child for each row. Of 31,916 currently failing rows, 14,214 have an alternative passing the same numerical bounds; 17,702 have none. The search does not validate representable boundary faces, immutable-parent adjacency paths or canonical stable-face ownership. No alternative has been installed or admitted.

The three-file source patch was reviewed against fb1fe73. The focused assertion executable was independently compiled with -UNDEBUG and passed, including default runtime rejection of refined canonical input. The native census binary was built and run on SSH Numi. Exact commands, binary/input hashes, stdout/stderr, all per-row census data and the feasibility classifications are retained here.

No Runtime was created, no physical step was accepted, and no native motion was qualified. The current coarse geometry remains unadmitted. Source assets, immutable mass ledgers, force ownership and unrelated dirty work are unchanged.
