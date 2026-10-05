# Geometry observations from the same accepted state

The native surface audit now records all four target chamber volumes and both
respiratory swept volumes beside the rendered cardiac coordinates. These are
read from the same accepted presentation buffer used by the GPU deformation
and volume audit. Offline checks no longer need to interpolate the separately
sampled physiological trace to associate geometry with its targets.

The SSH Mac mini native diagnostic `native-accepted-geometry-001` completed
3,000 accepted 2 ms steps, six simulated seconds, with no root assistance.
Its 95 retained geometry frames have finite target fields, a maximum frame
clock discrepancy of 2.85e-7 s from `step * 0.002`, and maximum GPU functional
volume error 1.003e-6 relative to the actual target. The lung target minus FRC
and the two swept volumes closes within 0.000289 mL (FP32 arithmetic). Native
wall time was 63.14875 s, RTF 0.09501; wrapper exit status was zero.

The integrated profile covers all 94 bounded command buffers: 56.515 s of GPU
update work and 5.420 ms total host snapshot copies. Median GPU time per
physical step was 18.871 ms; full physical-state readback occurred only at the
terminal segment. This identifies a GPU update bottleneck at the current
resolution. It is a six-second profile, not the final five-minute measurement.

This tests observation binding and numerical volume consistency. The retained
atlas still has known phase intersections; it is not anatomical or endurance
qualification. Exact invocation, source/binary/library/asset identities and
compact surface trace are included here. The full native output and continuous
recording remain at
`/Users/n/numi-human-resting-evidence-20261005/native-accepted-geometry-001`.
