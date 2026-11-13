# Ground contact observer: development diagnostic

The logging-only observer reproduced all 3,208 endpoint samples and 5,888 contact substeps of the in-range knee trial. It captured 3,584 beginning-of-substep states and 48,903 positive ground-contact entries. Native kinematics agreed with replay positions to 1.31e-8 m and joint positions to 4.8e-9; this verifies reconstruction at the sampled state, not its timing against the solver.

For lanes 2–7, 30,274 contacts with normal impulse above 1e-7 Ns had median, 90th percentile and maximum tangential impulse / (0.8 * normal impulse) of 0.247, 0.742 and 0.750. Against the configured dynamic coefficient 0.6 the corresponding median, 90th and 99th percentiles were approximately 0.330, 0.989 and 1.0; 7.758% of normal impulse was associated with ratios at least 0.95. These are solver impulse ratios, not evidence of biological friction calibration or actual slip.

The reconstructed normal contact-point velocity from beginning-of-substep state did not match the recorded solver pre-solve velocity closely enough: best-sign RMSE was 0.029329 m/s, against solver RMS 0.04453 m/s. We therefore do not infer sliding or attribute the failed leg push to friction. The next observer samples the candidate velocity actually handed to contact evaluation.

An intermediate archive commit contains a partial compressed run snapshot copied while the process was still writing. That historical snapshot is not a completed run. The files in this commit supersede it with the completed run; preserve both history and this qualification. See analysis.json and parity.json for calculations and checks.
