# Object-track frame and depth validity

## Measured failure and correction

The materialized object-track reducer previously accepted any nonzero pixel-validity mask. In the native two-environment Franka fixture, sensor depth dropout was set to 100%. A separately rendered control retained one and two selected geometry pixels respectively and zero valid-depth pixels; the unmodified tracker nevertheless reported confidence 1 for environment 0 and failed the preregistered no-track gate. The valid-depth control remained visible. The baseline stopped on the first failed environment, so its native output does not establish the old confidence for environment 1.

The reducer now requires both FRAME and DEPTH validity before a pixel can contribute to a centroid. On Apple M6, the same native dropout fixture reports confidence 0 and zero position/velocity histories for both environments, with both masked-depth history planes entirely far. The real valid-depth control and independent image/observation comparison continue to pass. A separate synthetic Metal kernel test covers both environments with valid plus missing-depth pixels, fully missing depth, missing frame, and valid plus missing-frame pixels. The old kernel showed 24, 32, 32 and 24 output-component mismatches respectively; the candidate showed none. Values include both actor observations and compact history, not independent scene counts.

The persistent synthetic sequence valid → missing depth → valid passes confidence histories [1,1] → [1,0] → [0,1], with no velocity inferred through the missing measurement. This sequence was run on the candidate only. It does not establish held-frame behavior, actual sensor latency or physical-step rollback.

The observation fingerprint changes for every actor program containing object tracks, because even compact-layout policies previously could receive false tracks. Pre-change synthetic policies with and without masked depth reject under the new contract; explicitly rebound policies compile. Matched policies without object tracks retain their fingerprints. Rebinding only acknowledges the changed input contract; it does not qualify old learned behavior. Earlier fingerprint tests were updated to reflect this additional semantic change. Seven targeted native CTests passed.

## Provenance and limits

The measured baseline used main `419130e08` and the existing runtime/shader build; the pre-change source and prior build identities are recorded in `../tracker-binding-stride-20261108/final-identities.sha256`. The newly compiled baseline fixture executable was not separately hashed. Candidate source, executable, runtime and metallib identities are in `final-identities.sha256`. The first synthetic-test build failed from a missing type header and is retained compressed; it was fixed before any synthetic baseline run. Plans and complete native/synthetic outputs are retained.

The rendered object is selected by simulator identity and the small scene is not a real-camera calibration, deployment-valid detector or learned manipulation qualification. No timing conclusion follows. The visual callback can still finish before a failed submission or rejected physical step; mixed-environment acceptance and full sensor-state restoration remain unqualified. The experimental collision advancement and augmented-Lagrangian contact options remain disabled.
