# Matched stationary toe traction screen: partial improvement, not qualified

The control reproduced all 4,400 recorded joint position and velocity states of the earlier probe exactly. Both arms completed without reported simulator errors. The passive contact observer covered every substep; its worst checked difference was 1.01e-9 N s. See `control-parity.json`, `high/observer-validation.json`, and the preserved logs.

The higher-traction arm changed only the foot and toe material. With the perch unchanged, effective static/dynamic coefficients increased from 0.8/0.6 to 1.2/0.9; these exploratory values are not crow measurements. All eight constructed starts are development data.

Neither arm passed the complete supported release and 100-interval bilateral regrip gate in any of the eight lanes. Both had two falling lanes. Non-foot contact intervals were 575 in the control and 547 at higher traction, concentrated in the fallen lanes. Thus the preregistered complete-fix prediction failed and this material is not promoted.

In lanes 2 and 3 the candidate did allow sustained release while the other foot supported and gripped the perch: 41 and 34 consecutive 20-ms intervals, respectively, followed by foot return. The baseline had no such qualified sustained release. In the final 100 intervals, these lanes retained toe opposition and whole-body support without non-foot contact, but one foot failed the individual upward-load threshold in 1 and 4 intervals respectively. The criterion remains unchanged. Four other candidate lanes did not recover left toe opposition; two fell. See `comparison.json` and `high/terminal-diagnostics.json`.

This is a stationary probe, not an autonomous walk, flight, landing, or beak-object manipulation result. It does not establish biological realism or generalization to unused layouts. The original observer host and Metal library are archived in `artifacts/crow-v15-hip-roll-staged-release-20261111/runtime.tar.gz`; `inputs.json` records their hashes. The probe binary is preserved here in compressed form; `probe.cpp` includes the earlier staged probe source.
