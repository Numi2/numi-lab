# Fixed-bed full-body capture audit for run 1198

This is a read-only CPU geometry audit. It does not launch a native simulation. It reuses the pinned 1172/1193 exact NHSKIN-to-target and NHSKIN self-intersection pipeline, reports the 17 ocular target surfaces separately without exempting them, and adds exact NHSKIN-to-fixed-bed triangle intersections plus sampled per-skin-vertex signed gap queries. The fixed bed is rebuilt from the frozen 1196 manifest using the production Float32 FMA node coordinates and triangle ordering.

The accepted native run must be closed before audit. The declared capture schedule must match the native invocation environment and prelaunch declaration exactly. The reader also verifies the run declaration's 300 immutable asset hashes, accepted pack receipts/timestamps, runtime verification, and source-stability evidence before scanning. Audit output directories must be new.

The terminal-only command below is an early diagnostic. Its result is explicitly partial and cannot establish the full schedule. If it is clean, run the full declared schedule in a separate fresh output directory.

```sh
PYTHON=/Users/n/numi-human-prep-venv-20261005/bin/python3.13
AUDIT=/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/audit_fixed_bed_full_body_captures_1198.py
RUN=/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/native-run
OUT=/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/skin-bed-audit-1198-attempt002-terminal
NHA_SHA=1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241
STEPS=0,4991,5375,5759,6111,6495,7743,10000

# Preflight first, after native closure:
"$PYTHON" "$AUDIT" --run "$RUN" --out "$OUT" --nha-sha256 "$NHA_SHA" --capture-steps "$STEPS"

# Terminal accepted pose only; explicitly partial:
"$PYTHON" "$AUDIT" --run "$RUN" --out "$OUT" --nha-sha256 "$NHA_SHA" --capture-steps "$STEPS" --scan --scan-only-step 10000

# Only after the terminal result is reviewed, scan all eight captures:
"$PYTHON" "$AUDIT" --run "$RUN" --out /Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/skin-bed-audit-1198-attempt002-full --nha-sha256 "$NHA_SHA" --capture-steps "$STEPS" --scan
```

For a later 310-second run, copy the exact accepted-step list, NHA path and hash, and native-run directory from that run's own immutable declaration and invocation. Do not reuse the 1198 values.

## Pins and parser admission

- Audit driver: audit_fixed_bed_full_body_captures_1198.py, SHA-256 fc04a1bf331aa8adc5b1f8309843129bcbfd5d446f1f4488f38e11432782a785.
- 1198 attempt-002 prelaunch declaration SHA-256 e9b711ea0490f19b5351cde1b20f80b1a6e1f83ba8068a6a4e1664614e8b667a. It binds the source Visual header SHA-256 51af221fa4039b3dc313d9deec4e0a34850f71f791868c60254d68e34b113b99, native bed source SHA-256 a1fe30a211dad9b54975b9aaa4ce0551d7003bf6f48af46a2a0732f271d8aef4, and bed Metal include SHA-256 47ee93f2c41d6741d45f974fa335ba4b2f1d00ba24a3cb6313ef626738d9c700.
- Bed manifest: resting-supine-scene-contoured-5mm.manifest.json, SHA-256 189ffb56c3b3b6de3426627068b17e049134dca8efa3551fb4c37c0c92fb5c2c. It binds the 1187 NHSKIN (b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b), the existing support payload (bcfece8e5da553b98694b724644234407fa4c38383b1e18d3c24d7caefa28927), and the 1191 initial source capture.
- CPU native bed-admission parser attempt: parser-admission-attempt001/execution.json, SHA-256 04bd490e008b0387bf5d18f22b3cf87bc31b102d5d6468c4aea56dc4e262e349. The fixture-only invocation passed 18 checks; the invocation with the exact frozen 1196 manifest, NHSKIN hash, and support payload hash passed 21 checks. Binary SHA-256 c28d74765770e518a883ab1c04251c440cebf157e52bd21fc716dd5e3c8be03c; test source SHA-256 1a103a5027cb4e70153cdd9f259feb2794b38820feb550a3ec25fb225646625c. Logs and exact argv are retained beside the execution record.

The audit reports bed intersections and signed gaps separately from NHSKIN-to-target/self findings. Exact bed contact/intersection is not, by itself, treated as an automatic failure; interpret it against the declared unilateral support-contact owner. The current signed-gap query samples every referenced NHSKIN vertex against the production-selected bed facet. Triangle-to-triangle intersections are exact; the point-gap field is not a continuous triangle-interior minimum. This is a geometry snapshot audit at the declared accepted captures, not continuous-time clearance or physiological qualification.

1198 attempt 001 stopped before accepted steps because the bed shading tangent was invalid. Its failure is retained separately. This audit is bound only to the corrected attempt-002 declaration and must not be run against the failed attempt.



## Attempt-002 terminal diagnostic

The preflight and terminal-only scan completed against the closed native run at accepted step 10000 (20.000000949949026 s). Summary: skin-bed-audit-1198-attempt002-terminal-scan/summary.json, SHA-256 ad3307b61da3c576cfaf94dd182f85f57cfd2f8ec3b56e2efc597fa272f4b1ca.

At this terminal pose, the pinned exact audit found 3,727 NHSKIN-to-target triangle intersections across 26 of 859 target surfaces; none were ocular. NHSKIN self-intersections and invalid triangles were both zero, with complete pair coverage. Per-target and exact witness files are retained in that output directory. This is a positive anatomy clearance finding requiring review; no all-capture result is claimed.

The fixed bed produced 430 exact NHSKIN-to-bed triangle intersection pairs. The sampled point-to-facet query covered all 54,663 referenced skin vertices, with 10 below their selected facet and minimum signed point gap -7.608556188642979e-6 m; no points were outside the finite grid and neither mesh had degenerate triangles. Bed contacts are reported separately and are not automatically classified as failures. The point-gap statistic does not measure the continuous triangle-interior minimum. Bed result: step-10000.bed-result.json, SHA-256 5e0cdc5f1b14eea1216e2122e7dc7cb7504193c7312fc898c7d58fa89dc1626e.

The eight-capture scan is deliberately not started while this terminal target-intersection finding is being reviewed. The full-schedule command above remains ready and must use a fresh output path if authorized.
