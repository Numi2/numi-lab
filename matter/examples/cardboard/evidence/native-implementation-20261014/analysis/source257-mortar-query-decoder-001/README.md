# Source257 mortar query CPU replay

This is an independent CPU-only decoder and geometry replay for the first uncertified mortar-complement VT query recorded by native605 and the smallest-alpha native610 run. It does not alter source257, production state, solver tolerances, or a GPU run.

The 384-byte records are read directly from the native capture payloads and checked against both per-record and whole-payload SHA-256 values. For query kinds 3/4, captured start[6] and finish[6] arrays are already the selected left/right mortar-subtriangle CCD vertices: makeMortarComplementAudit is called with subtriangleEndpoints=true. Stored barycentric arrays are parent-map metadata. Applying them to the captured coordinates again would double-map the subtriangle.

The independent geometry reference promotes each captured binary32 endpoint and effective-thickness scalar to exact rational values. It evaluates exact nearest-point squared distances at t=0 and t=1 and compares each against the exact square of the captured production effective thickness. Both records have strict opposite signs, so continuity of the linearly interpolated point and triangle gives at least one finite-thickness contact along each captured query path. The binary64 distance/bisection values are descriptive; the exact squared-distance sign test is the endpoint bracket evidence.

| Native run | Query | Native CA result | Independent endpoint bracket |
|---|---|---|---|
| 605 | right subtriangle vertex 2 vs left cell 3 (pair 16/99, slot 78), t=0.6481599212 | hit=0, certified=0, separation=1.0001717783e-5 m; 9.09e-13 m above effective band | start 1.0384943506e-5 m, finish 9.7936891251e-6 m, band 1.0001716873e-5 m |
| 610 | right subtriangle vertex 1 vs left cell 2 (pair 16/99, slot 78), t=0.8289262652 | hit=0, certified=0, separation=1.0001921510e-5 m; 1.455e-11 m above effective band | start 1.0398644968e-5 m, finish 9.9199936116e-6 m, band 1.0001906958e-5 m |

The source-like float32 CPU shadow at each captured native time reproduces the recorded separation bits. Its full 16-step recurrence reproduces 605's time, while the 610 accumulated time differs despite its final separation matching the native output. Therefore the shadow is not claimed as Metal recurrence parity. The native result itself is uncertified, not a certified clear result.

This evidence supports near-band conservative-advance exhaustion on paths that enter the collision band. It does not prove the complete production CCD algorithm, the first time of contact under Metal rounding, or the whole candidate's admissibility. A certified hit can produce a contact response; downstream residual, geometry, and final-certificate checks remain authoritative. Nothing in this package permits an uncertified query to be treated as clear or relaxes a contact threshold.

Run:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/source257-mortar-query-decoder-001/decode_mortar_query.py --self-test

The self-test covers parser layout/hash rejection and synthetic closest-point arithmetic. Native inputs are read from native/mortar-inline-observer-on-605 and ...-610; exact paths and hashes are in report.json. Source/build/artifact hashes are bound there and in manifest.json.

Earlier source/build hash messages e722d116... and 68d8eac... were stale draft values. The verified final source-build record and checker are the 2529a502... and d256c748... values recorded in the manifest. The old source257-mortar-audit-review-003 file is preserved; its literal escaped newline makes it invalid strict JSON, so this package is a separate valid binding.
