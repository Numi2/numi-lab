# Integrated reference sensitivity 926

This is a 24-second native transient sensitivity check using the predeclared current anatomy and anatomy-derived diaphragm parameter. It is not a steady-state, anatomy, clinical, or 310-second endurance qualification.

The primary response is the unweighted mean inspired-volume ledger across all complete breaths whose starts are at or after 10 s and whose ends are at or before 24 s, compared with the baseline arm mean. Every breath remains listed. A breath crossing the 10 s initialization boundary is descriptive only and excluded. The owner ledger is authoritative; sparse sampled airflow is not integrated.

All seven arms passed exact native completion checks at 12000 accepted 2 ms roots (24 s). Protocol SHA256: a3d23d6d5d16545d2d6e563d5c3ba8d92c4e0395f0b30b22d303976368086b61.
The analysis does not claim steady state. It reports complete breath and cardiac-cycle coverage, per-breath and across-breath dose directions, forward ejection, and maximum absolute running inventory residuals. Cumulative oxygen/CO2 residual fields are not summed.

The baseline comparison is limited to common accepted steps with the validated 925 capture; 925's physical/physiological trace was byte-identical to914 and its exact body/respiration states were verified. No input, root transaction, or program-fingerprint equality is claimed. The 926 anatomy receipt/source differs from the older template, and the current moving skin remains unqualified.

The prior standalone diaphragm area of 0.025 m2 is not transferred. This protocol uses 0.018687047064304352 m2; any response is an observed short transient for these declared parameter perturbations, not subject-specific physiology.

## Arm summaries

- baseline: complete post-init breath count=2; mean ledger=527.361 ml; mean VT=0.527361 L; mean RR=11.1577/min; mean VE=5.88431 L/min; mean PaO2=102.555 mmHg; mean PaCO2=38.9871 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.018-10.176s crosses the 10s cutoff and is descriptive only.
- lung-compliance-0p8: complete post-init breath count=2; mean ledger=492.274 ml; mean VT=0.492274 L; mean RR=11.3985/min; mean VE=5.61121 L/min; mean PaO2=101.719 mmHg; mean PaCO2=39.3165 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.016-10.144s crosses the 10s cutoff and is descriptive only.
- lung-compliance-1p2: complete post-init breath count=2; mean ledger=553.672 ml; mean VT=0.553672 L; mean RR=10.9673/min; mean VE=6.07261 L/min; mean PaO2=103.176 mmHg; mean PaCO2=38.7417 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.018-10.198s crosses the 10s cutoff and is descriptive only.
- airway-resistance-0p5: complete post-init breath count=2; mean ledger=532.005 ml; mean VT=0.532005 L; mean RR=11.1083/min; mean VE=5.90988 L/min; mean PaO2=102.717 mmHg; mean PaCO2=38.9336 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.018-10.188s crosses the 10s cutoff and is descriptive only.
- airway-resistance-2: complete post-init breath count=2; mean ledger=513.746 ml; mean VT=0.513746 L; mean RR=11.2747/min; mean VE=5.79244 L/min; mean PaO2=102.152 mmHg; mean PaCO2=39.1302 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.016-10.154s crosses the 10s cutoff and is descriptive only.
- muscle-force-0p8: complete post-init breath count=2; mean ledger=464.893 ml; mean VT=0.464893 L; mean RR=11.5875/min; mean VE=5.38695 L/min; mean PaO2=101.038 mmHg; mean PaCO2=39.5799 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.014-10.118s crosses the 10s cutoff and is descriptive only.
- muscle-force-1p2: complete post-init breath count=2; mean ledger=583.09 ml; mean VT=0.58309 L; mean RR=10.7453/min; mean VE=6.26617 L/min; mean PaO2=103.893 mmHg; mean PaCO2=38.4617 mmHg. Positive LV/aortic/pulmonary ejection cycles=15/15/15.
  Breath 5.020-10.230s crosses the 10s cutoff and is descriptive only.
- lung-compliance-0p8 vs baseline: complete-breath mean ledger difference=-35.087 ml; observed direction=decrease; predeclared direction=decrease; match=True.
- lung-compliance-1p2 vs baseline: complete-breath mean ledger difference=26.3104 ml; observed direction=increase; predeclared direction=increase; match=True.
- airway-resistance-0p5 vs baseline: complete-breath mean ledger difference=4.64395 ml; observed direction=increase; predeclared direction=increase; match=True.
- airway-resistance-2 vs baseline: complete-breath mean ledger difference=-13.6154 ml; observed direction=decrease; predeclared direction=decrease; match=True.
- muscle-force-0p8 vs baseline: complete-breath mean ledger difference=-62.4684 ml; observed direction=decrease; predeclared direction=decrease; match=True.
- muscle-force-1p2 vs baseline: complete-breath mean ledger difference=55.7291 ml; observed direction=increase; predeclared direction=increase; match=True.

## Interpretation limits

Short finite-time response directions can be modified by initialization, feedback, and cycle phase. Preserve contrary or unavailable effects as observed; do not infer steady-state behavior from the two-breath window. Gas changes are observations, not a calibrated clinical prediction. The 925 comparison covers only shared first-20-second samples and checks output fields, not input or fingerprint identity. Pressure columns remain instantaneous unless explicitly labeled cycle means.

Gas and blood balance fields are cumulative inventory residual high-watermarks; they are not summed across rows. Current skin and complete anatomy remain unqualified.
