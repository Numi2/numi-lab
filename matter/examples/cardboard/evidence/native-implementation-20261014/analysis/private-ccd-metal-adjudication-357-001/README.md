# Native EE CCD adjudication 357

This report independently checks native attempt 357 against the sealed diagnostic package 007 and applies the existing Armijo trace-health adjudicator to related runs 355, 358, and 359. It does not change or rerun physics.

The 007 package manifest and all 22 listed files match their sealed byte counts and SHA-256 values. The native result is strict JSON; its source, generated Metal, build, input-case, payload, and metallib hashes all match the package. The native receipt has exit code 0, unchanged inputs, and zero physical transactions. The four packed inputs join in order to captures 333/trial 20 and 334/trial 21, root 1201 / Newton 12, exact edge endpoints `[13,14]` and `[28,32]`; each 144-byte case record matches its source manifest hex words and the 576-byte packed payload.

For both captured cases, unchanged production and the extracted budget-16 clone produce identical four-float CCD outputs bit-for-bit. Both remain `uncertified_fail_closed` after all 16 iterations. The budget-17 clones each return `certified_miss` with no hit after 17 iterations. This identifies a one-iteration CCD budget boundary for these exact rays. The final budget-16 safe advancement remains inside the normalized sweep (approximately 0.999548); from the next sample, the safe step passes the sweep endpoint (approximately 1.004880), consistent with the source rule that certifies a miss after the conservative step exceeds `1 + 8*kEpsilon`. The result does not export a branch code, so the report records this as source-ordered interpretation of the observed trace, not additional production telemetry.

The diagnostic is not an acceptance test: the 17-step clone does not replace the production 16-step function, and neither result evaluates Armijo, candidate feasibility, transaction acceptance, or physical validity.

The related 355/358/359 traces each contain 32 ordered records and have first solver-fatal status at Newton 7. Their complete raw Armijo records and source input hash maps match. The nine non-log transaction-state files in each run also match byte-for-byte. At Newton 7 the base residual is a valid predecision center, but the line search fails. Run 358's selected private trial separately fails contact rebuild (status 6, failing index 283; diagnostics `[92,127,0,0]`). Run 359's center capture is at that first-fatal iteration, before the rejected line-search outcome; it is not an accepted center/trial pair. Later residual rows are failure-latched and are not valid center measurements. All three native transaction receipts have exit code 1; the failure is retained.

## Reproduce

```sh
python3.10 adjudicate.py
```

The script uses the sealed `private-trace-health-001/trace_health.py`; its exact SHA is recorded in `report.json` and `package-manifest.json`. It strict-parses every JSON input, checks hashes and case joins, preserves all 32 Armijo records per trace, and fails if parity or source bindings change.
