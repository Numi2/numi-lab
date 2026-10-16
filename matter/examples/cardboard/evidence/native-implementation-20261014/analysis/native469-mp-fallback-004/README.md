# Explicit MP fallback for source166003 FP64-failed rows

This package adds an optional, independent arbitrary-precision reference for the 17 material point/state rows that failed the source166003 FP64 return-map reader. It does not change the original FP64 reports, native kernels, tolerances, or any physical result. The original failures remain present in every generated report.

## Reproduction

Run the exact bound reader with the qualified Python and a fresh output directory:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-004/run_mp_fallback.py \
  --output /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-004/run-006
```

Run the six package checks with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v \
  /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-004/test_mp_fallback.py
```

The reader verifies the source166003 regression arrays and manifests, all 787 selected source-file hashes for frozen source157, source metadata, and the exact native469 capture, prediction, receipt, and stderr hashes before evaluating the rows. See `run-006/report.json` and `manifest.json` for the concrete bindings. `run-001` through `run-005` are preserved; they are historical iterations, not the final results. In particular, run001 used an incorrect starch shear parameter binding and is superseded. Run005 did not yet compute the native-stress-versus-reference-at-captured-state attribution. Nothing in an older run is silently rewritten.

## Result

The final run covers center plus four captured perturbation states: 4 center failures, 6 s1 failures, 3 s2 failures, 3 s3 failures, and 1 s4 failure. Each row is solved independently at 70, 90, and 110 decimal digits from the exact captured deformation gradient, authored material parameters, and accepted old history. The seed is the accepted history plus the elastic trial stress; no native projected state is used as a seed. The model uses the authored 15-state Hill/tensile-hardening residual and strict residual-decreasing line search.

All 17 rows pass the unchanged MP reference gates: scaled residual at most `2e-12`, dynamic-scaled correction at most `2e-11`, and maximum 70/90/110 root spread below `1e-12`. The maximum observed root spread is `1.15e-70`; MP implicit-tangent 90-vs-110 relative L2 difference is `4.82e-92`. The corrected FP64 implicit derivative evaluated at the same rounded MP root differs from the MP derivative by at most `6.22e-15` relative L2. These compare two implementations at a common MP root; they do not turn the original FP64 return-map outcomes into passes.

At the captured center state, the native cached tangent differs from the MP implicit tangent by at most `5.04e-8` relative L2. Captured center stress differs from the independent source166 stress evaluated at that same captured state by at most `4.29e-8` relative L2. The MP/native branch-predicate comparison nevertheless has 11 mismatches out of 17, concentrated in the sign of very small yield residuals; the report retains raw values rather than forcing branch agreement.

One side-state mismatch must remain visible: for `s1`, tet1317, the captured native stress differs from source166 FP64 stress evaluated at that same captured state by `7.0054e-2` relative L2 (maximum component error about `1.2492e5 Pa`). The captured native state also differs from the independent MP root by `0.5617` relative L2, and the source166 residual evaluated at that captured state is `4.3656e-6`, above the strict reference residual gate. The report preserves the captured projection diagnostics/status words. This is an unresolved state/stress/reference discrepancy, not an MP fallback pass for native behavior.

All five output stress/tangent arrays are finite at 1920 rows after substituting MP IFT tangents/stress only for the exact FP64-failed rows. They are explicitly mixed-reference arrays: the other rows remain source166003 FP64 results. Side-state MP tangents are independent CPU references; native cached tangent comparison is available only at the captured center state. Finiteness of these arrays does not establish branchwise finite-difference accuracy, complete native tangent coverage, or assembled element/global action accuracy.

## Scope and limits

- This is an optional high-precision reference fallback, not a production projection change.
- FP64 failure records and gates are retained unchanged; no row is omitted.
- Native projection, complete native tangent, full assembled action, and physical/calibration qualification all remain false in the report.
- No finite-difference noise-floor conclusion or global operator comparison is made here.
- The s1/tet1317 state/stress discrepancy and all branch mismatches remain open for separate diagnosis.

## Validation

The exact Python 3.10.12 interpreter compiled both Python files with `py_compile`; all 6 unit tests passed. `validation/py_compile.log` and `validation/unittest.log` retain the commands and outputs. No GPU, native run, production build, source edit, or qualification-gate change was performed for this package.
