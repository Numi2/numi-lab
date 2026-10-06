# Respiratory observation consistency

The native study adapter now checks every retained observation against the
already bound respiratory configuration: lung volume equals FRC plus diaphragm
and rib displacement times their effective areas; airway pressure equals
negative resistance times airflow; pleural pressure follows lung compliance.
It also requires finite observations and excitation/activation within [0, 1].
The check runs after simulation on the retained CSV. It adds no per-step CPU
physics, state, or GPU readback.

These are algebraic telemetry checks with a declared Float32 rounding allowance
of 32 epsilons times the largest term (minimum scale 1 in the reported units).
They do not establish integration accuracy, physiological plausibility, or
causality independently. Window summaries expose actual excitation, activation,
motion, pressure, and both airflow directions for comparison with the coupled
gas measurements. Their means and ranges depend on the declared sample window;
complete-breath ventilation remains a separate existing measurement.

All 20 adapter tests passed on the SSH Mac mini. New regressions reject unit
errors, stale anatomical area, and invalid activation. The completed baseline100
trace independently passes with maximum residuals 0.000275 mL for volume,
0.00000636 Pa for airway pressure, and 0.0000678 Pa for compliance. The largest
residual uses 3.18% of its rounding allowance. Source, configuration and trace
hashes are in the retained result.

This is a post-run descriptive check of baseline100. The frozen study100
instrument, primary outcome and active treatment run are unchanged. Full inputs
remain under `/Users/n/numi-human-resting-evidence-20261005/` on the Mini, in
`native-drive-study-100/`, `costal-current003-full-chain-003/`, and
`respiratory-mechanical-consistency-117/`. Baseline100's known anatomical
defects and the unfinished five-minute acceptance remain explicit.
