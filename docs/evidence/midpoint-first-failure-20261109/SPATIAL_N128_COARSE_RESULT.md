# 128-division coarse reference completed

The frozen independent FP64 reference completed the 10 ms, 128-division
trajectory at a 0.5 µs step after 8,405.8 s, with no supervisor stop. It used
the same material and area-weighted contact law as the earlier refinement.
All 2,146,689 nodal area weights were checked against the intended patch, not
just their sum; the total was 0.0003999999999999661 m². Minimum reported
clearance was 96.2771 µm. The sampled peak was 0.1016194 N and the integrated
impulse was 0.0004106810 N s. The full trajectory, execution record and
reproducible coarse validation are retained.

The 0.25 µs run has started under the preregistered guard. Until that run
completes and passes the temporal gates, the coarse trace alone cannot qualify
the 128-division response or the 64-to-128 spatial comparison. In particular,
the sampled peak is not a 1 ms-binned force. These are CPU reference
measurements, not native performance results.
