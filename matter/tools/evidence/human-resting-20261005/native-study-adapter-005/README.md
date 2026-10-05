# Native paired-study plan adapter 005

This increment adds `prepare-native` to the existing `resting_intervention_study.py` adapter. It writes a draft `numi.science.plan.v2` plus its model, parser calibration, and exact identity manifest. It does not register a study or execute an arm.

The draft defaults are 160,000 accepted steps at 2 ms (320 s total), a 10 s initial exclusion, respiratory-drive scale 0.5 on `[60,100)` s, and 30 s windows: pre `[30,60)`, dose `[70,100)`, and late recovery `[290,320)`. That leaves 310 s after the initial exclusion. The primary endpoint is the treatment-minus-control difference of each arm's dose-window PaCO2 minus its pre-dose PaCO2. The existing executable alveolar-ventilation predictor declares a broad 0 to +40 mmHg sensitivity range from a 40 mmHg reference under its fixed-production and effective-ventilation assumptions; this is not a statistical interval or clinical prediction.

The plan records secondary late-recovery equivalence limits against the unchanged control: absolute PaCO2 difference no greater than 1 mmHg, inspiratory minute ventilation relative difference no greater than 10%, and absolute PaO2 difference no greater than 5 mmHg. The comparison interval is generated from the configured analysis-window length. These are predeclared numerical study tolerances, not clinical cutoffs and not guaranteed outcomes. The plan validator binds the same unit, Apple device, vascular world, step count, timestep, Dense45/Brain-control flags, and common asset identity across arms, while requiring distinct owner-reported coupled-program fingerprints. Anatomy status remains an honest observation/limitation field; anatomical qualification is an independent measured gate and is not encoded as a required failure for this physiology study.

The frozen source identity is a JSON map with exact revision and diff SHA-256 for all three owners, `numi-lab`, `numilab-human`, and `numi-brain`. The separate compiled-source hash inventory binds the exact source files; both inventory files are listed and hashed in the plan artifacts.

On 2026-10-05, the focused Python suite passed on the SSH Mac mini (macOS 26.6, Python 3.9.6): 11 tests in 0.030 s. The test builds an ephemeral plan from temporary test-only identities and verifies it with the existing v2 plan validator; the temporary directory is removed by the test. The retained test output is `../native-study-adapter-005-tests.log`. Adapter SHA-256: `3b868581d0ee7fe453fe32d35db40b66c4d9c4c0e2f15be622f7c97749439cba`; test SHA-256: `31bc7ebe23a4d02ee0d67d9796fefe09ba4ca2d03713d5ff776ee4e4f89e090c`.

## Final freeze command template

Run this only after the integrated scene, source snapshot, assets, executable, libraries, and per-arm program identities are final and frozen. Replace each uppercase variable with the actual recorded value; the builder verifies every source and consumed-asset digest before writing a draft.

```sh
python3 /Users/n/numi-human-resting-lab-20261005/matter/tools/resting_intervention_study.py prepare-native \
  --directory /Users/n/numi-human-resting-evidence-20261005/native-drive-final/plan-draft \
  --invocation /Users/n/numi-human-resting-evidence-20261005/native-drive-final/baseline-invocation.json \
  --source-hashes /Users/n/numi-human-resting-evidence-20261005/native-drive-final/source-hashes.json \
  --source-revisions /Users/n/numi-human-resting-evidence-20261005/native-drive-final/source-revisions.json \
  --parser-fixture /Users/n/numi-human-resting-evidence-20261005/native-breath-001/resting-coupled.csv \
  --world-fingerprint ACTUAL_OWNER_WORLD_FINGERPRINT \
  --control-program-fingerprint ACTUAL_CONTROL_PROGRAM_FINGERPRINT \
  --treatment-program-fingerprint ACTUAL_TREATMENT_PROGRAM_FINGERPRINT \
  --device 'Apple M4 Pro' --steps 160000 --dt 0.002 \
  --start-s 60 --end-s 100 --scale 0.5 --window-s 30
```

`source-revisions.json` must contain all three source owners, for example:

```json
{
  "numi-lab": {"revision": "<40-character commit>", "diff_sha256": "<64-character SHA-256>"},
  "numilab-human": {"revision": "<40-character commit>", "diff_sha256": "<64-character SHA-256>"},
  "numi-brain": {"revision": "<40-character commit>", "diff_sha256": "<64-character SHA-256>"}
}
```

The program fingerprints must come from the owner for the exact frozen baseline and intervention inputs. The current owner prints them during scene execution; it does not yet expose a non-advancing identity-only report. Until the final source/assets are ready and actual owner identities are available, no real plan, registration, or trial has been created. The parser fixture calibration covers only CSV parsing/window arithmetic, not the gas model, anatomy, or clinical validity.
