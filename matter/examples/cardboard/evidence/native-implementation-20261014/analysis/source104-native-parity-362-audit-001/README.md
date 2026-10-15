# Source 104 versus source 103 first-fatal parity audit

This package prepares a fail-closed CPU comparison of the forthcoming source 104 run 362 with source 103 center capture 359. It does not run the native executable and does not assert parity before the candidate receipt/capture exists.

The reviewed source delta is exactly three selected files from the sealed source manifests:

- `matter/src/metal/contact.metalinc`: at the existing uncertified vertex-triangle return, diagnostics.w changes from zero to `3 + direction` (marker 3 or 4). The status, pair IDs, vertex index, CCD arithmetic, and return path are unchanged. The reviewed consumer audit says w is observational at this producer and the x/y pair check is unchanged.
- `matter/src/metal/fem.metalinc`: adds a read-only kernel to copy solver status at Armijo entry.
- `matter/src/runtime.mm`: adds the bounded base-status snapshot and schema-3 classifications. It changes the interpretation labels for the raw residual metric, but not the solver, trial, tolerance, acceptance, or history update.

The auditor requires exact equality for the coupled center capture payload, all 145 typed array layouts and hashes, each accepted-state/history binary in the diagnostic-state tree, transaction metadata, and all pre-existing numerical fields in the schema-1 Armijo trace. It permits only run-specific executable fingerprint/payload-path metadata, the reviewed schema 1-to-3 health fields (which it validates internally), and a code-6 `diagnostics.w` transition from 0 to 3 or 4 for the existing pair 92/127 and vertex 0..2. Source physics fingerprint, selection identity, material, protocol arguments, input hashes outside the reviewed overlay/build receipt, numeric residuals, line-search results, and all other metadata remain exact requirements.

A source104 run that starts from a different candidate, changes an ordinary numerical value, omits the selected root/iteration, differs in any binary/history record, changes the pair/vertex, or emits a marker outside code-6 trial status fails closed. The accepted-state and history binaries are compared byte-for-byte. The report will still label a matching run a failed-prefix diagnostic comparison, not convergence or physical qualification; post-fatal values remain unqualified.

## CPU preparation and tests

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 test_audit.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit.py \
  --baseline-run /Users/home/cardboard-implementation-20261014/native/first-fatal-center-359 \
  --candidate-run /Users/home/cardboard-implementation-20261014/native/health-and-vt-marker-firstfatal-362 \
  --output /Users/home/cardboard-implementation-20261014/analysis/source104-native-parity-362-audit-001/source104-vs-source103-report.json
```

The synthetic tests cover exact numeric equality, the scoped VT marker, schema-3 health consistency, and rejection of altered metrics/invalid marker locations. `audit.py` also verifies the complete source inventory against the two frozen source manifests and the run prediction/build receipts before comparing output.

At package preparation time, run 362 had no receipt or capture payload, so the parity verdict is pending. The first execution of the comparison command should occur only after the root-owned native run has completed and written its receipt.

## Recheck of CCD package 002 for runner 364

The source103 pair CCD verifier is internally consistent with its host output contract: it recomputes production/clone equality from raw float32 words, comparing records 0..2 in full and record 3.x only; it validates the fifth clone row as the speed-bound diagnostic, checks case identity, trace length/terminal and source/capture/input bindings, and requires the exact sealed metallib path and hash. The synthetic native-result parser tests passed 10/10. This supports the 364 runner only if it uses the package-002 executable and metallib and preserves the exact prepared input/result contract. It is a standalone CCD diagnostic with zero physical transactions, not a full-step parity result.
