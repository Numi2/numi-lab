# Source 104 versus source 103 first-fatal parity audit

This CPU-only auditor compares completed source104 run 362 with source103 run 359. It requires the exact three-file diagnostic source overlay and binds each run to its own successful build receipt. The differing executable, dylib, and metallib hashes are admitted only when they exactly match the corresponding `artifacts_sha256` entries in that receipt.

The completed comparison is in `source104-vs-source103-report.json`. It reports exact parity for the selected coupled center payload (145 array hashes), all 32 Armijo records and nonstructured log lines, transaction JSON, and all diagnostic snapshot payload bytes except the serialized `deviceProgramFingerprint` u64. Source fingerprint and every remaining snapshot byte are exact. The two rebuilt device fingerprints match the two run-specific build receipts. 362 and 359 both exited 1; these are matching failed-prefix diagnostics, not a convergence result.

The schema-3 health fields are checked against pretrial status and acceptance. The run had **zero VT marker deltas**: pair 92/127 failure at Newton 7 retained `diagnostics.w = 0` in both runs. The source104 VT marker branch was therefore not exercised by this trial; no claim is made that it was verified by 362. An unexpected code-6 marker remains permitted only for the documented pair 92/127 and vertex 0..2.

## Reproduce

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 test_audit.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit.py \
  --baseline-run /Users/home/cardboard-implementation-20261014/native/first-fatal-center-359 \
  --candidate-run /Users/home/cardboard-implementation-20261014/native/health-and-vt-marker-firstfatal-362 \
  --output /Users/home/cardboard-implementation-20261014/analysis/source104-native-parity-362-audit-002/source104-vs-source103-report.json
```

Tests cover exact numeric comparisons, scoped marker admission, health consistency, allowed serialized fingerprint-only changes, rejection of altered source/payload bytes, and unstructured-log changes. The report preserves the post-fatal and physical-qualification limits.

## Pair-CCD diagnostic package lineage

Package002's verifier contract passed its 10 synthetic parser tests, but native run364 failed before dispatch because the host requested an unqualified Metal symbol. The metallib exports `numi_matter_metal::nm_cardboard_source103_pair_ccd_diagnostic`. Package003 uses that exact qualified symbol and is compile/CPU-verified; root-owned native dispatch is still required. See `/Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-diagnostic-003/manifest.json`. Packages001,002 and failed run364 remain preserved.
