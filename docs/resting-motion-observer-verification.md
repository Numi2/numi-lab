# Resting motion observer reconciliation

The existing full-skin resting observer is enabled with
`NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT=1`,
`NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS=1`, and
`NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT=1` on an ordinary admitted native
resting invocation. The COM observer remains required. Body rows use accepted
kinematics; support point and selected-skin records explicitly retain pre-step
kinematics and source/query identity. These clocks must not be conflated.

At source `441aec7719c4acc764224e6f1067f6f26bb0c6a3`, a matched M4 Pro Metal
four-step control/candidate pair used the same source-bound construction posture,
2 ms timestep and COM cadence. Enabling body/support observation preserved the
terminal state, COM diagnostic, support impulse and coupled trace byte exactly.
All 628 body rows, 128 support-point rows and 128 selected-skin rows passed
independent mass/COM/momentum, orientation, timing and source/query checks.
The original pure helper tests also pass; the helper and test source match both
historical observer prototypes exactly. The later q-audit window/cadence owner
retains this observer at normal samples; profile time excludes the separately
recorded body CSV work.

The paired receipt is `resting-motion-observer-native-verification.json`.
This qualifies an observation instrument on a bounded fixture. It does not
qualify sustained drift, recovery, physiology, or a performance improvement.
