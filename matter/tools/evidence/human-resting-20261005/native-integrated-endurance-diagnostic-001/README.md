# Integrated native endurance diagnostic: 320 seconds

One whole-body Apple Metal scene completed 160,000 accepted 2 ms steps on the
SSH Mac mini (Apple M4 Pro). Excluding the first 10 simulated seconds leaves
310.000015 seconds of observation. This is an engineering endurance result,
**not final anatomical acceptance**: the BP diaphragm/lung intersections and
passive myocardial wall defects in this frozen asset set remain unresolved.
The later common respiratory-map candidate is not the implementation tested here.

The physical and physiological clocks matched; no root assistance was used.
The accepted state recorded 58 breaths and 373 complete filling/ejection cycles.
The complete-breath interval from 293.454014 to 314.712015 s yielded
6.138550 L/min ventilation and 11.289867 breaths/min. The last reported tidal
volume was approximately 544 mL; LV stroke volume was approximately 70 mL.

Native elapsed time was 3098.482929 s, giving **0.103276× real time**. The wrapper
took 3101.104700 s including setup/teardown. The 5,000 physical command buffers
reported 2950.519987 GPU seconds, 0.058484 s of host copy time, and one terminal
full-state readback. This run was not real time. Independent CPU anatomy audits
and a separate source build also ran on this Mac during part of the horizon;
it is complete-run timing under that recorded work, not an isolated benchmark.

All 5,001 displayed states passed the existing skin/bed and functional-volume
checks. The worst full-skin bed gap was −0.110552 mm; maximum rendered functional
volume relative error was 1.419980e-6. The endpoint's 118 enabled scalar joint
ranges all satisfy the unchanged 16-FP32-epsilon representation band (maximum
raw overrun 1.735026e-6). All 51 NHEQ1 rows were evaluated on the unmodified
terminal q; maximum absolute polynomial residual was 8.647821e-10.

Maximum blood-volume error was 0.099652 mL out of 5,150 mL. The terminal physical
delta ledger is +0.098480 mL and the normalized continuity-residual ledger is
+0.098461 mL. Their difference is −0.000018933 mL; the endpoint reduction adds
−0.000225555 mL relative to the physical-delta ledger. These values disclose
and account for numerical drift; they are not exact mass conservation.
Maximum gas-balance errors were 0.000233 mL O₂ and 0.000466 mL CO₂ (STPD).

The last 30-second retained window had mean PaCO₂ 39.764 mmHg, PaO₂ 99.571 mmHg,
SaO₂ 97.190%, mean aortic pressure 89.101 mmHg, and mean pulmonary pressure
14.760 mmHg. The source-linked comparisons in `endurance-analysis.json` keep
numerical consistency separate from physiological plausibility. In particular,
respiratory rate is below the selected MedlinePlus 12–18/min interval, and mean
pulmonary pressure is below the selected AACN 15–20 mmHg interval. The supine
cohort context is descriptive, not a replacement normal interval or a clinical
validation claim.

The continuous native AppKit/Metal recording remains on the Mac mini at:

`/Users/n/numi-human-resting-evidence-20261005/native-integrated-endurance-diagnostic-001/native-viewer.mov`

Its SHA-256 is retained in `endurance-analysis.json`. AVFoundation read all
5,001 image samples (plus five non-image timing markers), verified strictly
increasing presentation times, and matched the surface-row count. The first
and last image timestamps are 0.188333 and 3098.658333 wall seconds; maximum
presentation gap is 4.053333 wall seconds. The recording has not been retimed.
`frame-final.png` is its exact last image, corresponding to accepted physical
time 319.998015 s. The Mac remained at the login screen, so this verifies the
native renderer/recording path, not interactive desktop presentation.

The exact argv, environment, three-owner source revisions, source diff,
configuration and runtime/asset hashes are in `invocation.json`, `source.diff`,
and `run-metadata.json`. All consumed runtime inputs had unchanged hashes after
execution. Compact complete traces and the log are retained as deterministic
gzip files. The final accepted MRVPack and its state/asset receipt remain beside
the recording under `accepted-geometry/step-159999.*`.
