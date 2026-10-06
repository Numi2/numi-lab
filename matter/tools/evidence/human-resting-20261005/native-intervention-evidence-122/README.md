# Coupled native respiratory intervention, 160 seconds per arm

The frozen study100 ran the same supported whole-body scene twice on the SSH
Mac mini (Apple M4 Pro), with 80,000 accepted 2 ms steps per arm. Treatment
halved delivered respiratory excitation on [30, 50) seconds. The comparison
uses preregistered pre-dose [15, 30), dose [35, 50), and recovery [145, 160)
windows. The 468 retained pre-intervention rows match in every exported field.

| Measurement | Control dose | Treatment dose | Treatment minus control |
| --- | ---: | ---: | ---: |
| Inspiratory ventilation, L/min | 6.2800 | 4.1387 | -2.1413 |
| PaCO2, mmHg | 39.2338 | 40.3739 | +1.1400 |
| PaO2, mmHg | 103.1687 | 100.6107 | -2.5581 |
| Oxygen saturation, fraction | 0.974406 | 0.972626 | -0.001781 |

The measured chain includes lower mean diaphragm activation (0.04034 to
0.02168), lower maximum diaphragm excursion (20.334 to 16.018 mm), smaller
lung expansion (maximum 3.02335 to 2.84911 L), and reduced pressure-driven
airflow. The existing pressure-flow, pressure-compliance and volume
decomposition checks pass on all 2,500 retained observations in each arm.
These checks establish numerical consistency separately from the paired
response. The curves were produced by the coupled GPU state, not prescribed.

After normal regulation resumed, the final-window treatment/control
differences were -0.01827 mmHg PaCO2, +1.04368 mmHg PaO2, and +0.05301 L/min
ventilation (0.839%). All three are within the predeclared descriptive
recovery margins. These are deterministic comparison tolerances, not clinical
cutoffs. The formal primary difference-in-differences is +1.14003 mmHg PaCO2;
`numi science analyze` reports supported, and `numi science verify` reports
verified declared-artifact integrity. Full observations also retain
complete-breath ventilation, physiological reference comparisons and outliers;
the fixed-window integral can contain partial breaths.

Control/treatment completed 29/30 breaths and 187 filling/ejection cycles each.
Native real-time factors were 0.067843 and 0.068067. The maximum blood-volume
error was 0.06566 mL against the 5,150 mL reference; maximum O2/CO2 ledger
residuals were 0.000233/0.000466 STPD mL. Represented body mass stayed 72 kg;
late centre-of-mass drift was 0.00406 mm/s. Drift is a descriptive numerical
measurement, not an inferred stationary-rest certificate.

Each continuous native movie has 2,501 image frames matching every retained
displayed state, monotonic presentation times, and no retiming. Recorded
wall durations are 2,361.285 and 2,353.532 seconds. The maximum presentation
gaps are 2.360 and 2.397 seconds. Final displayed physical time is 159.998 s,
the documented one-step presentation lag; terminal physics reaches 160 s.
The movies and accepted geometry packs remain on the Mini at manifest-bound
paths and hashes. Native initial/final frames and complete compact traces are
included here. Source revisions, exact code hashes, parameters, asset identities,
commands, registration, original instrument and post-run analysis are retained.

This remains exploratory evidence, not final acceptance. The old anatomy has
known taenia, lung/rib and cardiac-neighbor defects. The manual study receipt
also omitted preregistration of some implicitly loaded runtime libraries;
the subsequently hardened adapter rejects that incomplete receipt (see
`../runtime-dependency-admission-124`). No historical binding was rewritten.
The original study098 capture-index failures remain at their registered Mini
paths, as described in study100's limitations. The required five continuous
minutes on the corrected anatomy and the real-time target remain unmet.
Anatomical source rights and mixed-source/reference status remain those of
the exact bound source receipts; no measured-person or clinical claim follows.
