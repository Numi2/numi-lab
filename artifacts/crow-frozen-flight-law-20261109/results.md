# v15 flight controller V1 freeze

The standalone V1 source reproduced the teacher-off candidate exactly at all 2,560 sampled q/v pairs in the matched two-seed perturbed ground-start rerun. All 160 first episodes passed the preregistered straight/left/right/climb/descent screen, with no physical failures, no failed simulator steps, and no teacher labels. The frozen source identity is recorded in freeze.json. Its antecedent law also passed the matched teacher-off airborne recovery and combined-command studies; the extraction check here directly covers the ground-start commands.

This is a flight-only qualification for the sampled conditions and bounds described in the plan. The source remains a low-level law requiring a fingerprinted command/feedback adapter; it is not a learned navigator, collision-clearance proof, landing controller or supported-landing result. No final layout has been consumed.
