# BCT capacity-binding producer overlay

This external overlay adds the missing native-result producer for the strict BCT capacity artifact and a post-run packager for a complete source-bound native history. It is based on frozen snapshot076 and does not modify the checkout or sealed validation-admission overlay001.

`probe.patch` adds `shipping_capacity_binding` to `box-native-result.json` only after the native probe reports the v2 numerical-capacity gate passed, the nominal and half-speed branches are complete, and the source-bound qualification, curve, checkpoint, sidecar, and branch-ledger hashes still match. The binding is derived from the frozen probe inputs and actual output files; it does not contain candidate/condition identity and does not hash its own result file.

The separate `patched/bct_capacity_history_producer.py` has two stages. `prepare` seals a plan candidate, BCT condition, and explicit parameter-to-executed-manifest JSON-pointer map before a native run. `finalize` verifies that plan and case-binding files were included in the run's `prediction.json` input hashes, verifies the unchanged native result and the named Matter/rigid checkpoint pair against `shipping-rigid-accepted-checkpoint.json`, checks the executed parameter values, and writes a new ensemble history package. It never rewrites the raw run. The new package repeats the exact capacity-binding object in its history, manifest, receipt, and certificate, and hashes the original native result, receipt, executable, solver source-input map, compiled package, reports, curve, checkpoint, rigid sidecar, and pair metadata.

Example commands:

```sh
python3 patched/bct_capacity_history_producer.py prepare \
  /path/to/ensemble-plan.json candidate-id condition-id \
  --parameter-bindings /path/to/parameter-bindings.json \
  --output /path/to/pre-run-case-binding.json

python3 patched/bct_capacity_history_producer.py finalize \
  /path/to/native-run \
  /path/to/ensemble-plan.json \
  /path/to/pre-run-case-binding.json \
  --source-data-catalog /path/to/source-data/catalog.json \
  --source-data-dir /path/to/source-data \
  --output /path/to/ensemble/native_histories/candidate-id/condition-id
```

The `parameter-bindings.json` object must contain one mapping per candidate parameter, for example:

```json
{
  "score_depth_ratio": {
    "kind": "manifest_json_pointer",
    "pointer": "/shipping_score_command/selected_depth_ratio_of_caliper"
  }
}
```

The native run harness must record the plan and prepared case-binding files as hashed external inputs in `prediction.json`; the probe command must also use the same capacity-qualification input. The finalizer checks that the qualification path and output path match the sealed prediction. If the harness cannot record those external inputs, or if candidate values cannot be mapped exactly to the raw executed manifest, finalization rejects the run. Existing raw runs without that pre-run identity cannot be retroactively converted into candidate evidence.

The emitted artifact is only numerical BCT capacity evidence for the stated native configuration. It does not establish exact recipe calibration, same-grade material validation, physical-box validation, or shipping qualification. The fixture tests are synthetic contract tests, not solver measurements. No GPU or native execution was performed for this overlay.

Validation receipts are under `checks/`. The wrapper's four tests include successful synthetic packaging without raw-run mutation, parameter mismatch rejection, absent pre-run binding rejection, and checkpoint-pair mismatch rejection. The copied evaluator and its 27-test suite are byte-identical to sealed overlay001; all 27 tests pass here as well.
