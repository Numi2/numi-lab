# Source-bound per-material MP fallback for native469

This package repairs the material-parameter binding in the earlier `native469-mp-fallback-004` reference-only analysis. It does not edit source, native artifacts, FP64 results, or production behavior.

The older 004 script selected the medium parameter block `[28:56]` once and used it for every paper tet (`run_mp_fallback.py:348-349`). That is incorrect for liner tets. The 004 output and its conclusions are preserved at `analysis/native469-mp-fallback-004`; in particular, its reported s1/tet1317 stress discrepancy is invalid as a constitutive finding because the reference used the wrong grade's parameters.

The corrected script first binds all 1,920 capture records to the compiled material descriptors, then cross-checks the regression archive's global tet material index against the capture's material index/name/state count/parameter count. It uses the captured descriptor's parameter offset for every liner or medium row. The same mapping is used for the 17 MP fallback rows and for successful FP64 tangent rows. The starch block remains independently bound from its captured descriptor.

## Reproduction

The source166003 arrays, source/reference files, source snapshot, and native469 capture/prediction/receipt/stderr are hash-checked by the runner. Use the qualified Python 3.10.12 interpreter and a new output directory:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-005/run_mp_fallback.py \
  --output /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-005/run-002

/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v \
  /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-005/test_mp_fallback.py
```

Commands and output are retained in `validation/final-receipts.json` and its command logs, as well as the earlier `validation/receipts.json`, `validation/py_compile.log`, and `validation/unittest.log`. The first test attempt had one harness-only failure: the test looked up `successful_row_material_checks`, while the report field was named `successful_row_material_binding_checks`. That test selector was corrected. After adding the full center-success-row preservation assertion, the final replay is `run-002`; earlier `run-001` remains as a retained intermediate output. The final suite passes 9/9.

## Results

The bound input is the source166003 regression469 capture: 1,728 paper tets (1,152 liner and 576 medium) plus 192 starch tets. Both paper parameter vectors are 28 words, but they are distinct; the runner records the captured offsets and checks every tet identity before evaluation.

All 17 original FP64 projection failures are preserved and independently solved at 70, 90, and 110 decimal digits. All pass the existing strict MP reference gates: scaled residual `<=2e-12`, dynamically scaled correction `<=2e-11`, and 70/90/110 root spread `<1e-12`. In final run-002, the maximum final residual is `1.6802e-12`, maximum final correction is `1.7801e-16`, and maximum root spread is `1.1471e-70`. These are high-precision reference results; the original FP64 failures remain failures.

The material-binding regression for the originally reported s1/tet1317 mismatch uses the exact captured native projected state and the same captured deformation gradient. This tet is material index 0, the liner, whose parameter offset is 0. Native stress versus source166 stress at that captured state is `5.4500e-9` relative L2 with the liner parameters. Applying the medium block instead produces `7.0054e-2` relative L2 (maximum component difference about `1.2492e5 Pa`). This identifies the prior 7% result as a fallback-analysis parameter-selection error, not evidence of a production stress discrepancy.

For successful FP64 rows, the package proves that all 1,724 valid center paper-tet `reference_C` tensors are bitwise preserved in the mixed output. It also checks one representative liner and medium tangent in each of the five states (center and s1-s4); the mixed arrays exactly match direct source166 tangent recomputation at those 10 rows. Substituting the opposite grade's parameters changes those tangent samples by 12.27% to 13.43% relative L2, so these checks catch accidental use of one shared paper parameter block.

All five mixed reference stress/tangent arrays are finite over 1,920 tets. The report still records 11 native/MP branch-predicate mismatches. Native projection, complete native tangent, full assembled action, all-state FP64 projection, and physical/calibration qualification remain false. No global finite-difference noise-floor claim is made; side-state tangents are CPU references, and native cached tangent comparison is only available at the captured center state.

## Bound inputs and retained history

`manifest.json` binds the new runner/tests/output and exact source166003 regression arrays/report/manifest, reference implementation, source snapshot metadata and 787-file source hash map, native469 capture/prediction/receipt/stderr, plus the superseded 004 script and run-006 report/arrays. The original 004 files are not overwritten. The generated `run-002/report.json` carries the 17 original failure reasons, exact per-row material identity and offsets, MP precision histories, state/tangent/stress comparisons, and false qualification axes.
