# Captured VF witness audit: native contact-budget64-028

The script uses contact feature_start as the base configuration, uses per-alpha trial_feature_endpoints as trial configurations, rounds all recorded trace coordinates to IEEE-754 binary32, then independently computes the closest triangle point in float64 by checking an admissible face projection and all three clamped edges. No GPU/build was run.

- Trace SHA-256: `c763beef9cbd6fe90f9ac08643f7a8bbfbed69905d9dcc661e70cba5e079029b`
- stderr SHA-256: `a3fbf0bd5d8fc6e5b4d9e6b7bdc5aebfb28118e9720a5307e3a9f58938a75d17`
- Capture root/tick/iteration/slot: 5/0/22/2941; identity `[19, 19, 0, 1]`.
- Base feature/bary: edge_ca / `[0.8553639268728757, 0.0, 0.14463607312712434]`.
- Base separation: FP64 `8.102128180416e-06 m`; trace `8.102128050000e-06 m`.
- Old emitted normal cosine against FP64 closest-feature normal: `0.999994265592` (angle `0.194036 deg`).
- Base exact signed distance-to-thickness gap: `-1.897871566963e-06 m`; old-normal projection on the independent closest point: `-1.897918027871e-06 m` (error `-0.0464609 nm`). Serialized solver plane-gap field is `-1.897329639178e-06 m`; it is retained separately because its reference offset and float accumulation are not the Euclidean gap.

| trial | alpha | feature | FP64 Δdistance (nm) | FP64 Δenergy (J) | old-normal linearized Δgap (nm) | FP64-normal linearized Δgap (nm) | old normal·true normal | fixed-normal gap error (nm) | trace separation (µm) | trace energy (J) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | edge_ca | -0.189654 | 2.61037e-12 | 0.00723824 | -0.193338 | 0.999990673 | -0.0755667 | 8.10193887 | 8.99901e-09 |
| 1 | 0.5 | edge_ca | -11.1896 | 1.54848e-10 | -11.0236 | -11.2206 | 0.99999895 | -0.00849847 | 8.09093854 | 9.15125e-09 |
| 2 | 0.25 | edge_ca | -5.142 | 7.09464e-11 | -5.03312 | -5.23294 | 0.999997613 | -0.0193238 | 8.09698577 | 9.06736e-09 |
| 3 | 0.125 | edge_ca | -4.1881 | 5.7758e-11 | -4.11344 | -4.31382 | 0.99999588 | -0.0333648 | 8.09793983 | 9.05417e-09 |
| 4 | 0.0625 | edge_ca | -5.38937 | 7.43685e-11 | -5.39284 | -5.39055 | 0.999992463 | -0.061026 | 8.09673838 | 9.07078e-09 |
| 5 | 0.03125 | edge_ca | -3.50218 | 4.82821e-11 | -3.50417 | -3.50268 | 0.999993128 | -0.0556554 | 8.09862559 | 9.04469e-09 |
| 6 | 0.015625 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |
| 7 | 0.0078125 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |
| 8 | 0.00390625 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |
| 9 | 0.001953125 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |
| 10 | 0.0009765625 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |
| 11 | 0.00048828125 | edge_ca | -1.88772 | 2.6004e-11 | -1.88867 | -1.88787 | 0.999993667 | -0.0512961 | 8.10024085 | 9.0224e-09 |

The base IPC energy recomputed in float64 with thickness `9.99999975e-06 m` and stiffness `1.81084021e+13` is `8.996406824027e-09 J` (trace `8.996410730000e-09 J`); this is diagnostic only because feature mollifier is not serialized separately.

Interpretation is limited to the captured VF states. The FP64-normal linearized gap uses the recorded alpha-scaled relative displacement vector dotted with the independently computed base closest-feature normal. If exact FP64 Δdistance follows that prediction but disagrees with the recorded old-normal directional gap, that demonstrates a force-normal/closest-distance derivative mismatch for that trial. Do not treat this offline check as proof that the repaired mapped path or full native trial passes.
