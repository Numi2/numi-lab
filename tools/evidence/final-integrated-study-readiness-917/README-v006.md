# Final integrated resting-study readiness (917), revision 6

This directory prepares a corrected 310 s paired native Human study. It does not register or run the study. The preparation script is fail-closed; no source tree, scene asset, previous evidence, or native process is changed by plan generation.

The frozen Lab science owner and CLI remain Lab014 (d550d8a); the frozen science-plan source revisions remain Lab d550d8a, Human b354949, and Brain a1cf721. These are separate from the source that generated the corrected scene. Readiness v006 requires and binds the adjacent 936 non-fixture assembly report, including its Human 935 and Lab d550 source revisions, candidate skin and manifest, composition receipts, dynamic NHA, scene assets, and exact native invocation. The terminal-capture executable is separately pinned to source 017 (b091d7d, merged main ancestor efde8e7). Build pins, source pins, focused native tests, binary, respiratory metallib, and the frozen014 physical library are all explicit. The 017 change publishes the exact final accepted body/respiratory state without advancing another physical/controller step. It is capture correctness evidence, not a new runtime-physics qualification. Each registered arm also verifies its own native log's actually loaded MetalRobo image with the frozen Human loaded_metal_runtime helper, bound to the parent-verified path and SHA-256.

## Pending inputs and preflight

The script requires the caller to provide the corrected 936 scene's completed native-run directory and exact owner-composed anatomy receipt. The native-run directory must be a child of the 936 output directory, beside its assembly-preflight.json; the receipt must be <936-output>/anatomy/resting-anatomy-receipt.json. Revision 6 has exercised this contract against the completed 927-skin/current-924-anatomy scene documented below. It verifies the adjacent assembly report is a successful non-fixture corrected-skin assembly, binds its candidate NHSKIN and registration manifest, requires its composed receipt to be the receipt used by the native invocation, and follows the report's exact NHA path/SHA without assuming an old 924 payload hash. It rejects the retained 907 legacy fixture. The control run must report 10,000 accepted roots at requested 2 ms (nominal 20 s), unassisted matched physiology/body clock, verified loaded Metal runtime, unchanged sources, and exactly matching invocation/run-metadata asset, argv, and environment records.

The preflight uses the 017 native binary, corrected scene and anatomy receipt, and the admitted observer settings: full-q integration audit off and COM/momentum diagnostics at 8-root cadence. A 914 diagnostic invocation is not admissible because it used q audit on and COM cadence 1. The captured runtime timestep is Float32 0.0020000000949949026 s. Requested horizon and analysis windows use nominal 0.002 s; receipts preserve actual accepted time as step_id multiplied by Float32(dt). The terminal state is exact accepted N=155000 (nominal 310.000 s; its timestamp is approximately 310.000014724 s). This representation offset is about 14.724 microseconds and is reported rather than hidden.

The treatment program-identity probe is a short identity check only. Its intervention tuple is [60, 100) s at scale 0.5, so a 20 s probe does not exercise the intervention. The native parser accepts a finite interval whose start is nonnegative, end is greater than start, and scale is in [0, 2]; the runtime applies the factor only inside that interval. The readiness-owned direct-native recorder retains invocation, run metadata, native log, and before/after source hashes. Its native-observed fingerprint must match the separately predicted fingerprint. Offline FNV replay is a consistency check only.

## Prepare and review

From this readiness directory, first print the treatment-probe command without running it:

    /Applications/Xcode-26.6.0.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 \
      ./prepare_final_plan-v006.py \
      --scene-preflight-dir <FINAL_936_SCENE_OUTPUT>/native-run \
      --anatomy-receipt <FINAL_936_SCENE_OUTPUT>/anatomy/resting-anatomy-receipt.json \
      --print-treatment-probe-command

The accepted scene output is prepared through the existing 936 owner route. With the final corrected candidate NHSKIN and its registration manifest, use the exact command documented in ../final-native-scene-preflight-936/README-v011.md; review its assembly-preflight.json, then run its generated launch-command.sh when the native slot is available. For readiness, pass <936-output>/native-run and <936-output>/anatomy/resting-anatomy-receipt.json as shown above. The 936 v011 builder currently composes skin onto its reported base anatomy; v006 binds the resulting dynamic payload and receipt from that report rather than hardcoding their hashes.

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

## Current 936 CPU validation

Revision 6 was checked against the completed 20 s owner run at final-native-scene-preflight-936/skin-927-current-lung-924. The adjacent assembly report is bound by SHA-256 a15fd3a56d0077f7f3d53b5707ad4f299959a5e442f0b290b0a7a282fef22c29; the native invocation is 85e1f81f2576a7ac89e454311424e4d217f56ef9f45f8485a4c119cda83b9b6b, and the composed anatomy receipt is e2f818888bf292120992072d5c02b14ba9b4a078820ccf8514b86351305d7b56. The scene uses the caller-pinned 927 NHSKIN and its registration manifest with the dynamic 924 NHA/receipt. The checker verified the assembly/source identities, invocation assets, owner metadata, exit code 0, native terminal count of 10,000 at requested 2 ms, and loaded Metal runtime path/hash. Final scene/support files are runtime assets: 936 binds them through its scene fields and native asset map, separately from its preparation-input source hash map.

The captured treatment-probe preview is marked planned_not_run. Its recorder command was printed for review but not executed. No treatment identity probe, 310 s pair, science-v2 draft, registration, or endurance run was created by this validation. This verifies readiness of the supplied 20 s scene input only.

## Readiness checks

The CPU-only schedule/template tests cover the exact per-arm IDs/times, cadence endpoints, true-terminal handling, rejection of N−1, validation of the preflight capture list, authorized replacement of only that list, and rejection of unapproved argv/environment/asset changes. A compatibility test checks that frozen-owner scene invocation and observation serialization bind the derived template file, while parent preflight lineage remains separately checked. The direct-native identity recorder has separate failure and environment tests. New v006 scene-lineage tests accept a corrected candidate with a non-924 NHA and exact dynamic receipt, and reject a missing assembly report, the 907 fixture, a mismatched receipt, or an invocation asset mismatch. Frozen Lab014, Human003, and Brain source identities remain pinned independently of the 017 viewer/capture source. No corrected final 936 preflight, registered arm, final study registration, or GPU run is launched by this package. The 936 assembly report is mandatory because a bare owner invocation/receipt alone cannot distinguish the corrected candidate scene from the retained legacy 907 fixture.
