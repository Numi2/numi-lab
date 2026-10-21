# Build199: batched flat-column solver command copies

All builds and runs used SSH Numi. Copies of source prefix/tail/CGS2 command segments are now batched once per restart cycle, retaining command barriers, cache identity/forcing checks, fallback and mandatory true-residual certification. The option remains explicit and default off.

Validation: 73,430 native checks and 28 source contracts passed. The contract was updated to follow the relocated batch block; its initial failure and a wrong native-check executable path are retained as harness failures. The first compatible-pair harness referenced a nonexistent control directory; the existing completed candidate was retained and paired on rerun. No physical failure was waived.

The API-validation-on flat-only comparison and all three API-validation-off compatible combinations match all 17 endpoint metrics and all 68,335 FEM node/tet records exactly. Combined options: flat-column ICB, completed graph, zero-action and cycle suppression; certified-empty restart reuse remains off under its existing admission restriction.

Three post-first-root body throughput ratios: 1.027727, 1.024805, 1.024391. Median 1.024805.

Whole-process ratios (include startup): 1.009416, 1.011465, 1.010133. Body and process timings are retained separately in build199-results.json.

These runs simulate only 0.375 ms from initialization. They do not establish settlement, sustained coupled throughput or native-presentation performance. The 0.1x requirement remains unmet. No larger-timestep or reduction candidate is qualified by these command-execution parity results.
