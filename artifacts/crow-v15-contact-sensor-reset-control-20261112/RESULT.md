# Staggered-reset contact sensor comparison

On one used seed, the runner reported nine host-requested resets. The separate-sensor and predecessor implementations had identical logged state and toe contact samples over 300 steps and identical recorded q and v substeps in all eight related lanes. The lane-5 replay matched q, v, reward, observations, accepted actions, body states, terminal fields, tracking score and outcomes in all 300 frames. This does not establish identical unrecorded GPU state or all-lane rewards.

For lane 7, the separated-sensor values match the same-seed no-scheduled-reset run through trace submission 256. In the reset run the sensor reads invalid zero at submissions 257–260, while the no-reset control remains active. The exact no-reset input is identified by commit and hash in `sensor-no-reset-comparison.json`. This is consistent with clearing stale contact feedback around a scheduled restart; the logger does not independently record the applied reset mask or prove behavior on every reset timing.

This is a single correlated development batch, not qualification. The host changes the persistent-buffer/checkpoint schema; current top-level life-checkpoint tooling admits v13/v14 rather than v15. Checkpoint compatibility, restoration, rollback, the biological gait and the autonomous mission remain untested. No new controller is promoted.
