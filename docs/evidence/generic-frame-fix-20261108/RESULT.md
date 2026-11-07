# Result: second isolated frame/depth candidate

All three preregistered native processes exited 0 on 2026-11-07 UTC, from 23:52:45.650613 through 23:52:45.781728. Full per-process times and matching preflight/postflight identities are retained. These are correctness checks, not performance measurements.

The original synthetic invalid-depth regression and its baseline passed: all three depth-invalid arms produced 0/8 false near values. The proposed mixed-frame gate passed: measured and frame-present neighbor-filled pixels were 0.0816327, while the adjacent no-frame pixel and geometry-only no-neighbor pixel remained 1.0; a fully missing frame was all far. This is a specific proposed edge-corruption contract, not proof of a prior documented semantic.

In the actual four-environment 160×120 Franka renderer, the candidate matched the independent CPU reference at every pixel with normal and 100%-dropout sensors, with edge flicker disabled and forced to one. The ordinary frame had 3,708 valid-depth pixels and the dropout frame had none. The edge-on candidate filled 805 measured-neighbor border pixels by the chosen rule; the original differed on those 805, so integration would change the observation distribution.

This is an isolated shader, not a product change. Direct-winner sensor dropout is covered synthetically, but direct-winner missing-frame detection is not: that shortcut does not expose an independent per-pixel FRAME signal in this fixture. Other renderer modes, full task/policy and failed-physics transactions, real-sensor calibration and exact replay remain unqualified. The changed observation contract needs a version/fingerprint and old-policy compatibility rejection before integration.

