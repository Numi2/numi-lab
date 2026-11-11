# Stationary perch: separated pitch-response arms

Four native arms used the same v15-derived toe body, stationary perch, contact observer, 32 substeps and matched +0.04/-0.04 rad initial pitch states. There are only two distinct starts in each arm, each duplicated across four lanes. All 80 logged q/v samples through the first ten intervals matched zero gain in each nonzero arm. This constructed bench is not a physical landing or a biological measurement.

| Added hip gain | +0.04 rad start | -0.04 rad start | Decision |
| --- | --- | --- | --- |
| 0 | All duplicate lanes completed and met final support/pitch gate; final pitch 0.143 rad. | All fell below 0.1 m root height at interval 136; each had 147 toe-speed-cap substeps. | Baseline has asymmetric recovery. |
| +0.15 | No height-defined fall but final pitch 0.321 rad, outside 0.25-rad gate. | Fell at interval 141; 121 toe-cap substeps. | Reject. |
| -0.15 | Native status 11 rolled back attempted step 32, substep 9; only 31 intervals accepted in the arm. | Censored by the same batch rollback after 31 accepted intervals. | Reject; no conclusion about later physical outcomes. |
| +0.30 | Fell at interval 472; 111 toe-cap substeps. | Fell at interval 151; 117 toe-cap substeps. | Reject. |

The full accepted traces, failed attempt diagnostics, source and inputs are retained; per-lane details are in the analysis JSON. A rollback ends that arm and attempted substeps are excluded from accepted-state statistics. All accepted substep contact and toe-velocity records were present; no contact-capacity overflow was observed. Pitch and fall are sampled at 20 ms endpoints; toe speed uses 32 substeps. A below-height failure is not automatically a ground contact, as shown by the zero nonfoot-contact intervals in the +0.30 arm. These gains are not integrated into the journey. The smaller ramped response is a separate preregistered hypothesis and remains untested.
