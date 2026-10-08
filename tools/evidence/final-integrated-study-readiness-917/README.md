# Final integrated resting-study readiness (917), revision 5

This directory prepares a corrected 310 s paired native Human study. It does not register or run the study. The preparation script is fail-closed; no source tree, scene asset, previous evidence, or native process is changed by plan generation.

The frozen Lab science owner and CLI remain Lab014 (d550d8a). The native display/capture executable is separately pinned to terminal-capture source 017 (b091d7d, merged main ancestor efde8e7). Build pins, source pins, focused native tests, binary, respiratory metallib, and the frozen014 physical library are all explicit. The 017 change publishes the exact final accepted body/respiratory state without advancing another physical/controller step. It is capture correctness evidence, not a new runtime-physics qualification. Each registered arm also verifies its own native log's actually loaded MetalRobo image with the frozen Human loaded_metal_runtime helper, bound to the parent-verified path and SHA-256.

## Pending inputs and preflight

The only unresolved paths are the corrected control preflight directory and its exact anatomy receipt. Replace both placeholders in the commands below. The control preflight must contain a completed owner invocation, owner run metadata, and native log. It must report 10,000 accepted roots at requested 2 ms (nominal 20 s), unassisted matched physiology/body clock, verified loaded Metal runtime, unchanged sources, and exactly matching invocation/run-metadata asset, argv, and environment records. The anatomy receipt must hash the exact NHANATOMY payload used by the invocation.

The preflight uses the 017 native binary, corrected scene and anatomy receipt, and the admitted observer settings: full-q integration audit off and COM/momentum diagnostics at 8-root cadence. A 914 diagnostic invocation is not admissible because it used q audit on and COM cadence 1. The captured runtime timestep is Float32 0.0020000000949949026 s. Requested horizon and analysis windows use nominal 0.002 s; receipts preserve actual accepted time as step_id multiplied by Float32(dt). The terminal state is exact accepted N=155000 (nominal 310.000 s; its timestamp is approximately 310.000014724 s). This representation offset is about 14.724 microseconds and is reported rather than hidden.

The treatment program-identity probe is a short identity check only. Its intervention tuple is [60, 100) s at scale 0.5, so a 20 s probe does not exercise the intervention. The native parser accepts a finite interval whose start is nonnegative, end is greater than start, and scale is in [0, 2]; the runtime applies the factor only inside that interval. The readiness-owned direct-native recorder retains invocation, run metadata, native log, and before/after source hashes. Its native-observed fingerprint must match the separately predicted fingerprint. Offline FNV replay is a consistency check only.

## Prepare and review

From this readiness directory, first print the treatment-probe command without running it:

    /Applications/Xcode-26.6.0.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 \
      ./prepare_final_plan.py \
      --scene-preflight-dir <FINAL_CORRECTED_CONTROL_PREFLIGHT_DIR> \
      --anatomy-receipt <FINAL_CORRECTED_ANATOMY_RECEIPT.json> \
      --print-treatment-probe-command

Run the printed record_treatment_program_probe.py command with --execute only when the native slot is available. Then rerun the preparation command without --print-treatment-probe-command. It will write a science-v2 draft and print registration/run commands; it does not register or launch native code. Review the draft and its hashes before using those commands.

The registered arms use separate, hash-pinned launch templates derived from the completed 20 s owner invocation. Each template sets only NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS to that arm's predeclared frame IDs, replacing the corrected 20 s preflight list when present. The original preflight value is recorded in the template lineage and checked against the compiled 017 initial/submission/terminal cadence; all other argv, scene assets, source identity, and environment values remain unchanged. The preflight itself retains its initial-cycle geometry captures. The template is not presented as an owner run receipt: the parent preflight's owner run metadata and native log remain the evidence for the 20 s run. The science runner supplies the registered 155000-step horizon when it builds each final native command.

## Paired intervention and geometry schedule

Each arm runs 155,000 accepted roots at requested 2 ms (nominal 310 s). Control drive remains unchanged; treatment multiplies delivered respiratory excitation by 0.5 on [60, 100) s. The registered primary remains the PaCO2 difference-in-differences: treatment-minus-control change from [30, 60) s to [70, 100) s. The predeclared 0–40 mmHg sensitivity envelope is not a probability interval or clinical prediction. The predeclared late-recovery comparison is [280, 310) s with numerical margins of 1 mmHg PaCO2, 5 mmHg PaO2, and 10% inspiratory ventilation; those are study decision limits, not clinical standards. Do not tune ranges or margins after outcomes.

The eight-frame cap is used as follows. The five shared interior accepted-step IDs are 47519, 49151, 51903, 54047, and 55647 (nominally 95.038, 98.302, 103.806, 108.094, and 111.294 s). Control adds 152191 and 154143 (nominally 304.382 and 308.286 s); treatment adds 152447 and 154367 (nominally 304.894 and 308.734 s). Each arm then captures true accepted terminal N=155000. All seven interior IDs are ordinary 32-root presentation endpoints. N−1 is not labeled terminal. These are prior-867-informed exploratory phase selections, not fitted to final outcomes. The corrected 20 s preflight provides early initialization/cycle coverage; the eight-frame limit leaves the historical global rib-volume minimum near 65.168 s unsampled. Sparse geometry captures do not prove anatomy through all intervening states.

## Run and analyze

The generated command list follows the frozen Numi Lab science-v2 owner workflow:

1. Register the reviewed plan once.
2. Run baseline once and treatment once, sequentially.
3. Run the registered numi science analyze and numi science verify.
4. Run analyze_final_pair.py for supplemental cycle, reference-range, and support reporting.

The registered science analysis remains the primary-result authority. The supplemental analyzer refuses an active/incomplete trial and requires the registered exit receipt, exact accepted terminal count, native terminal record, owner observation, and every frame pack/receipt to match their hashes. It reports complete-breath ledger intervals with both measured airflow signs and lung-volume excursions, cardiac counter intervals with positive LV stroke and aortic/pulmonary ejection, dose direction, baseline-adjusted recovery, explicit generic-range outliers, support drift, and cumulative-balance high-watermarks. Cumulative oxygen/CO2 error fields are running maxima of total inventory residuals; they are not summed. Sparse endpoint airflow is not treated as an exact integrated volume. Instantaneous aortic and pulmonary artery samples are not compared sample-by-sample to mean-pressure intervals.

## Physiological reference context

Generic clinical ranges and posture-specific cohort summaries are kept separate. Per-arm reports preserve the AACN generic PaO2 band of 80–100 mmHg, PaCO2 35–45 mmHg, generic saturation and mean-pressure comparisons, and every observed outlier. MedlinePlus ABG guidance gives PaO2 75–100 mmHg, PaCO2 35–45 mmHg, and saturation 95–100%. MedlinePlus lists 12–18 breaths/min for the average healthy resting adult and notes individual variation. None of these comparisons diagnoses a person or qualifies simulated physiology.

Kovacs et al.'s review covers 1,187 individuals across 47 studies; the supine mPAP estimate is from a posture-specific subset of 882 and is 14.0 +/- 3.3 mmHg. The 882 is not the review-wide total. Mendes et al. reports male supine quiet-breathing cohort means +/- SD of RR 16.15 +/- 4.72/min, VT 0.58 +/- 0.28 L, and VE 8.32 +/- 2.78 L/min. Cohort summaries are group variation, not individual clinical intervals. Mean MAP/mPAP ranges are compared only to labeled complete-cycle mean proxies, never individual instantaneous pressure samples.

References:

- [MedlinePlus, Arterial Blood Gas (ABG) Test](https://medlineplus.gov/lab-tests/arterial-blood-gas-abg-test/)
- [MedlinePlus, Vital Signs](https://medlineplus.gov/ency/article/002341.htm)
- [Kovacs et al., Pulmonary arterial pressure during rest and exercise in healthy subjects: a systematic review](https://doi.org/10.1183/09031936.00145608)
- [Mendes et al., Influence of posture, sex, and age on breathing pattern and chest wall motion in healthy subjects](https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/)
- [AACN, Normal Ranges](https://aacn.s3-us-west-2.amazonaws.com/Courses/ecco/course-resources/resources/common-resources/Normal_Ranges.pdf)

## Evidence limits and retained checks

The retained 930 comparison checks that an explicit accepted terminal snapshot matches a following accepted-state presentation for the tested body/respiratory and surface fields. The 931 20 s regression confirms seven ordinary captures plus terminal N=10000 with unchanged physiology and exact terminal state. These bounded checks establish publication behavior only; they are not endurance, physiological, or whole-body anatomy qualification. The 932 q0/COM8 check confirms the terminal observer remains available under the final reduced diagnostic cadence. Its three capture packs (steps 0, 63, and terminal 64) are byte-identical to 930. The 8-row trace has 50 endpoint fields matching the 64-row q1/COM1 trace; ten diagnostics are exact min/max aggregates over each eight-root window (minimum for contact gap, maximum for the other nine). Whole-trace CSV identity is not claimed. This is a bounded terminal-publication check, not endurance or anatomy qualification. The pinned report is `native-terminal-production-review-932/verification.json` (SHA-256 `453e8f69bdad311d1d273d5049d17b886b609780bfbe3614700e82c424e782f9`).

Cardiac interface localization 911 finds 11,568 current source-neutral RA/RV intersections near the common-map tricuspid leaflet projection, with a localized source-to-current RV residual up to 0.75 mm. It does not establish a 3D leaflet surface or valve-plane/orifice ownership. Reduced-order CVSim chambers and valves remain the sole functional and blood owner. This is a bounded source-interface limitation, not a cardiac or whole-body clearance pass: [exact localization report](../native-cardiac-interface-localization-911/final-localization-report.json).

A failed or killed arm stays failed evidence even when its sampled trace looks close to target. Do not rerun an arm inside the registered pair, relax a gate, or analyze an active prefix as complete. A completed deterministic pair and numerical conservation checks do not establish individual physiology, clinical validity, static equilibrium, or whole-body anatomy qualification.

## Readiness checks

The CPU-only schedule/template tests cover the exact per-arm IDs/times, cadence endpoints, true-terminal handling, rejection of N−1, validation of the preflight capture list, authorized replacement of only that list, and rejection of unapproved argv/environment/asset changes. A compatibility test checks that frozen-owner scene invocation and observation serialization bind the derived template file, while parent preflight lineage remains separately checked. The direct-native identity recorder has separate failure and environment tests. Frozen Lab014, Human003, and Brain source identities remain pinned independently of the 017 viewer/capture source. No final corrected-scene preflight, registered arm, final study registration, or GPU run is launched by this package.
