# BCT source-bound pre-run harness overlay

This overlay prepares and verifies an explicit BCT calibration case before native execution. It extends the frozen source identified by `base/frozen-source.json` (recorded parent revision `35bfb518c542a306f563cc364ef8496837068b9a`). The frozen-source manifest's file hashes are authoritative; its `base_snapshot` value is retained as supplied by the snapshot producer.

The prepared case binding ties one candidate and one condition to the exact calibration-plan bytes. Before a shipping run starts, the harness checks the candidate's parameter values against distinct command-line flags (or source-hashed JSON inputs), validates the protocol ID used by the evaluator, checks the numerical-capacity qualification and its mesh/time report hashes, and copies/hashes those inputs into the run. It checks the copied bytes against the pre-run identity. After a successful process, it checks the executed manifest parameters. A failed check is recorded in the raw receipt and produces a nonzero harness result. The BCT packager independently recomputes the pre-run identity and post-run checks, then writes a new derived package. It leaves the raw native result, manifest, prediction, and receipt untouched.

Protocol identity is `shipping_bct_without_offset_first_local_peak_v1`, shared with the evaluator and BCT producer. Unbound `--shipping-box` runs remain ineligible for calibration packaging. The protocol/candidate binding is metadata plumbing; it does not qualify material properties or physical box capacity.

## Use

1. Prepare `parameter-bindings.json`, mapping every candidate parameter to a unique executed-manifest JSON pointer and `argv_flag`. Use `input_json_pointer` when the command-line value names a JSON parameter file.
2. Prepare the exact case binding before launch:

   ```sh
   python3 matter/tools/bct_capacity_history_producer.py prepare PLAN.json CANDIDATE_ID CONDITION_ID \
     --parameter-bindings parameter-bindings.json --output prepared-case-binding.json
   ```

3. Run the native harness. Keep `--shipping-box`, the v2 numerical-capacity qualification and its bound reports, and every candidate's exact CLI/file mapping in the command:

   ```sh
   python3 matter/tools/cardboard_source_bound_run.py \
     --case RUN_DIR --question 'BCT forward run' --prediction 'predeclared candidate' \
     --source-snapshot SOURCE_ROOT --runtime-dir BUILD_DIR \
     --ensemble-plan PLAN.json --case-binding prepared-case-binding.json \
     --build-provenance BUILD_RECEIPT.json --timeout 1800 -- \
     BUILD_DIR/numi-matter-cardboard-probe --shipping-box \
     --shipping-score-depth-ratio 0.25 \
     --shipping-capacity-qualification shipping-capacity-qualification.v2 \
     --output RUN_DIR/output
   ```

4. Only after a successful raw receipt, create a separate derived package:

   ```sh
   python3 matter/tools/bct_capacity_history_producer.py finalize RUN_DIR PLAN.json prepared-case-binding.json \
     --source-data-catalog matter/examples/cardboard/source-data/catalog.json \
     --source-data-dir matter/examples/cardboard/source-data \
     --output PLAN_DIR/native_histories/CANDIDATE_ID/CONDITION_ID
   ```

The harness rejects duplicate or missing parameter mappings, candidate/argv mismatches, altered source inputs, and an executed manifest that does not contain the declared parameter values. The derived package retains the plan, prepared binding, exact pre-run identity, raw receipt hash, and post-run validation record.

## Validation and limits

CPU-only checks in `checks/` passed: 7 end-to-end harness tests, 4 producer tests, 27 evaluator tests, Python bytecode compilation, and exact replay of `base/probe.patch`. The end-to-end fixture uses synthetic files and a fake executable; it verifies binding, failure retention, immutable raw receipts, packaging, and evaluator admission wiring. It is not a native solver run and makes no physical, calibration, or shipping-capacity claim. Native/GPU execution was not run for this overlay. No CMake or production shared-checkout files were changed.

`source-overlay.json` binds the overlay's changed files, dependencies, test logs, and patch hashes. `harness.patch` contains the new harness/producer changes and tests; `base/probe.patch` is the separately retained source-077 capacity-binding producer patch.
