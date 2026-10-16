# Source220 local projection state-certificate candidate

This is a default-off, opt-in diagnostic candidate for the captured bilinear15 paper material. It adds a stricter dimensionless local Newton-correction stopping certificate for the accurate projection policy: 2e-10 instead of 2e-8. The residual tolerance remains 2e-8; the selected test holds both local-iteration budgets at 32, with no outer Newton, FGMRES or acceptance-limit changes. The material equations and physical operator are unchanged.

The source change carries the flag through material compilation, validation, fingerprinting, CLI source-paper selection, and GPU policy. The targeted checker compares accurate32 with strict-state32 over the ten selected rows (center, duplicate center, and p22-p25 +/- endpoints) from the exact source547 1,472-row batch. It traces the center and p23-positive endpoint. The native route is diagnostic only and reports no physical transactions or qualification.

## Verification

- Clean CMake build succeeded for the projection-policy CPU checker, capture diagnostic checker, and cardboard probe. See build-validation.json and clean-build.log.
- The projection-policy test passed, including policy default/budget preservation, invalid-combination rejection, compiler fingerprint binding and package roundtrip.
- CPU admission passed for the exact source547 full input SHA a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e, source capture SHA c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2, and exact Haj-Ali liner and medium assets. It verifies 32 iterations in both policies and the strict threshold 2e-10.
- Earlier wrong-SHA and wrong-medium CPU admission attempts are retained in retained-failed-preflights.json.

## Pending native command

Root may run the following only under its single-worker GPU guard, with a new empty output directory:

<build>/numi-matter-cardboard-bilinear15-capture-diagnostic-check --native-source547-strict-projection <source547-input.json> a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e <source220>/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter <source220>/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter <fresh-output-directory>

No native or GPU dispatch has been performed by this overlay. A tighter local stopping certificate is a numerical experiment; it does not establish constitutive accuracy, physical behavior, or calibration.
