# Final integrated study readiness package v015

v015 is a CPU-only readiness update for a caller-selected final anatomy and respiration configuration. It preserves the frozen science plan owner and does not change the native physics runtime, viewer, study protocol, margins, or final assets.

The preparation path accepts hash-pinned NHA, source receipt, composed native receipt, and respiration configuration inputs. It verifies the v015 scene assembly and the actual native invocation against those exact caller-selected inputs. It does not assume the old 924 NHA, 907 skin, or 761 respiration file is the final candidate.

The 20-second preflight comparison against retained run 931 reports all 60 measured trace-field differences and all eight captured 86-body pose differences. It requires the exact accepted 10,000-root horizon, q0/COM8 cadence, finite differences, verified loaded runtime, and the no-advance terminal capture proof. It does not require output parity with 931. It separately compares the invocation-bound circulation/respiration configurations, contact-driving skin/support/soft-tissue/thorax geometry, relevant runtime scalar settings, and relevant NUMI_/DYLD_ environment values by their recorded bytes/values. If those inputs are identical to 931, changed traces or poses warrant regression investigation. If an input differs, the differences remain measurable facts to interpret against that declared input; finite deltas do not establish physiology or anatomy acceptance.

The captured respiration reference in run 931 is the 761 configuration at SHA-256 c518926bf47fba945cef52bb952ed6c559d604508988082c5b641b720bda503d. The run invocation is pinned at 1842eb1f8dba4f83b0ff4e86d491e1936a386eaba91ae06fb5acf84400123175. v015 compares final inputs with those actual invocation-bound files instead of inferring equality from a reused parameter name.

## Ownership and scope

- Science plan preparation uses the separately committed Lab owner at d81e236663e0d3427956a6253ac44900409c145a.
- Frozen Lab014 remains the physical/runtime owner and science CLI. The readiness package does not edit it.
- Viewer018 and terminal accepted-state capture017 are independently pinned as presentation/capture layers; they do not imply new physics qualification.
- A completed direct-native identity-only probe contains the future [60, 100, 0.5] program but simulates only 20 seconds. It verifies identity, not dose response.
- A final paired 310-second study, physiological interpretation, numerical verification, and anatomical clearance are separate steps and are not performed by this package.

## Required final inputs

Before using the command shape below, the owner must supply and verify:

1. A fresh non-fixture v015 scene assembly and completed 20-second native preflight for the final skin, NHA, composed receipt, and owner-selected respiration configuration.
2. The exact NHA path/hash; source receipt path/hash; composed native receipt path/hash; and respiration configuration path/hash. Both receipts must bind the selected NHA.
3. A measured comparison JSON tied to that preflight invocation and trace, with the required q0/COM8 cadence and terminal proof.
4. A successful v015 direct-native identity-probe child of the selected control run, with the exact selected assets/configuration and treatment fingerprint matching the independently predicted uint64 identity.
5. Final 936 assembly, owner source/build pins, runtime verification, and current native input hashes. These are validated by the preparation path; they are not replaceable by the historical 927/924 fixture.

The command is intentionally not executable until every placeholder value below is replaced with a real, absolute, hash-verified final path. It creates an unregistered plan draft only:

~~~sh
/Applications/Xcode-26.6.0.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 \
  /Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-917/package-v015/prepare_final_plan.py \
  --scene-preflight-dir FINAL_20S_NATIVE_RUN_DIR \
  --nha FINAL_NHA_PATH --nha-sha FINAL_NHA_SHA256 \
  --base-anatomy-receipt FINAL_SOURCE_RECEIPT_PATH \
  --base-anatomy-receipt-sha FINAL_SOURCE_RECEIPT_SHA256 \
  --native-anatomy-receipt FINAL_COMPOSED_RECEIPT_PATH \
  --native-anatomy-receipt-sha FINAL_COMPOSED_RECEIPT_SHA256 \
  --respiration-config FINAL_RESPIRATION_CONFIG_PATH \
  --respiration-config-sha FINAL_RESPIRATION_CONFIG_SHA256 \
  --preflight-comparison FINAL_V015_VS_931_COMPARISON_PATH \
  --preflight-comparison-sha FINAL_V015_VS_931_COMPARISON_SHA256 \
  --draft-dir FRESH_UNREGISTERED_DRAFT_DIR
~~~

Do not run any printed registration, native study, analysis, or verification command as part of readiness preparation. A successful preparation is still only a plan draft; it is not a registered or completed study.
