# Registered 60 s respiratory-drive diagnostic

This is an exploratory paired diagnostic for the reduced Matter/CVSim21 circulation, respiratory compartments, and NumiBrain chemoreflex instrument. It does not include a qualified whole-body anatomical scene and is not the five-minute body acceptance run. It is now stopped with one recorded failed control attempt and no treatment attempt.

Each arm is 30,000 accepted 2 ms steps (60 s nominal). In the treatment arm only, the existing Brain-delivered diaphragm/intercostal excitation is multiplied by 0.5 on `[15,35)` s. The paired primary outcome is the treatment-minus-control difference in each arm's PaCO₂ change from the 10–15 s pre-dose window to the 30–35 s dose window. The adapter also records inspiratory minute ventilation, lung volume, PaO₂, SaO₂, blood/O₂/CO₂ accounting, cardiac cycles, and 55–60 s recovery values. No response direction is imposed as a validity condition.

The executable model registers a broad steady-state algebraic sensitivity: at fixed CO₂ production, if effective alveolar ventilation lies from 0.5 to 1.0 of baseline, PaCO₂ change from a 40 mmHg reference lies from 0 to +40 mmHg. This is conditional, not a finite-time or closed-loop prediction interval. The pilot measures the response rather than assuming the controller cannot compensate.

The v2 notebook registration hash is `63b153342f9cba904a1b81916bedb9ebef222db89d30054a7ebf10419a2fd400`. The complete remote notebook is `/Users/n/numi-human-resting-evidence-20261005/intervention-study-002/study`; it retains the failed control receipt and stop record. The frozen registration plan, predictor, calibration, build identity, and build receipt are copied next to this README.

The dedicated Mini build used source clone revision `106243df31c3ce8fa8d0860fbcf7baec4e28e73a` plus the recorded dirty source snapshot (diff SHA-256 `5148ffc20290ab9ecae50f96c03c6629115777d09405d71c571c1b0ef0e8dfea`). It was configured at `/Users/n/numi-human-resting-study-build-20261005` and built with target `numi-human-resting`; the binary embeds that private build's metallib paths. The physical device is Apple M4 Pro, and the source-world fingerprint is `11218263775426672239`.

```text
runner SHA-256: 462a089285cf2b05b022650937375ef91151a2ecba1a984220b71e34f36c814f
NumiMatter.metallib SHA-256: 8c5f0c6078f7f385b3cce3b08b816df95721528cc797da27e134c14717e03a93
HumanRespiration.metallib SHA-256: f1fe31119fb00f310d3f64d42b6a6afa6c8e7183971e88e64b21c1b9a2eb72d3
CVSim21 input SHA-256: eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818
respiratory parameters SHA-256: 27100bf8941fcd623bba1ea87474f6fe6a4b1ab4f89b3e6ec0df5e4f8d9551f3
build receipt SHA-256: dda6836c89da6fb1187e05a5a453a9017a649a8ccefdb86f5cb63a758b2859ad
registration record SHA-256: 63929da1fd2ef550ad1fa423f57a5272cb3a0ae3fed12558a0271a39dbca4ccd
plan file SHA-256: 4345eada2fc382cffb4acef4c015156008cbf76aaef659c13026a5ad59a80e87
```

The 64-step transaction check on this exact dedicated binary passed: a rejected candidate left circulation, clock, respiratory state, and Brain controller history unchanged, and the accepted replay was bitwise identical. The parser calibration passed known CSV column/time/PaCO₂/cycle values, window arithmetic, monotonic-time validation, and nonfinite-value rejection; it does not calibrate physiology. The native transaction log is retained here.

The attempted control completed 30,000 steps (60.00000285 s; 217.8706 s wall; 213.7733 s GPU; RTF 0.275393) and emitted a raw CSV. The adapter correctly refused to emit an observation because the plan mistakenly carried the world fingerprint from a 1 ms transaction test (`11218263775426672239`) while the registered study ran at 2 ms and the runner reported `6020581806073115747`. This is a source-identity/protocol failure after execution, not a valid control measurement; the v2 notebook records the failed trial and was explicitly stopped before treatment. Its trace is retained without using it for physiological inference. A corrected preregistration uses the measured 2 ms fingerprint in study-003.

Two setup failures are retained separately. The first prepare attempt correctly rejected an old shared-build HumanRespiration library that no longer matched its frozen binary; see `../intervention-study-001-prereg/README.md`. The first notebook registration attempt rejected an unbound absolute loaded-library path. No simulation ran during either failed setup; the final plan includes and snapshots those library paths and then registered successfully. The rejected draft plan is stored in `failed-unbound-artifact/`.
